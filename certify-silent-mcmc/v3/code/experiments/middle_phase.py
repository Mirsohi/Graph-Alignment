"""Characterise an intermediate (partially aligned) phase: run a chain restricted
to the middle overlap basin [lo, hi] at beta, record per-residue agreement
frequency with the graph-only witness, and test whether the aligned residues
form a rigid body between the two conformations (Kabsch superposition of the
aligned set; RMSD of aligned vs remaining residues).  Structure coordinates
and PSCDB displacements are used ONLY for this post-hoc interpretation.
Usage: python middle_phase.py case beta lo hi"""
import json, os, sys
import numpy as np
import math
import numba as nb
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import _start_with_overlap
from gacert2.phase import rmh
from gacert.core import score, delta


@nb.njit(cache=True)
def agree_freq(A, B, p0, w, beta, kmin, kmax, T, thin, seed):
    np.random.seed(seed)
    n = len(p0)
    p = p0.copy()
    k = 0
    for i in range(n):
        if p[i] == w[i]:
            k += 1
    F = np.zeros(n)
    c = 0
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        pu = p[u]; pv = p[v]
        k2 = k - (1 if pu == w[u] else 0) - (1 if pv == w[v] else 0) + (1 if pv == w[u] else 0) + (1 if pu == w[v] else 0)
        if k2 >= kmin and k2 <= kmax:
            d = delta(A, B, p, u, v)
            if d >= 0 or np.random.random() < math.exp(beta * d):
                p[u] = pv; p[v] = pu; k = k2
        if t > T // 4 and t % thin == 0:
            c += 1
            for i in range(n):
                if p[i] == w[i]:
                    F[i] += 1
    return F / c


def kabsch_rmsd(X, Y, sel):
    Xc = X[sel] - X[sel].mean(0)
    Yc = Y[sel] - Y[sel].mean(0)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    dd = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, dd])
    Rm = U @ D @ Vt
    Xall = (X - X[sel].mean(0)) @ Rm + Y[sel].mean(0)
    dev = np.linalg.norm(Xall - Y, axis=1)
    return dev


def segments(mask):
    segs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


case, beta, lo, hi = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
A, B, pt, w, meta = load_instance(case)
n = len(w)
z = np.load(os.path.join(ROOT, 'data', 'cases', case + '.npz'))
X, Y, disp = z['x'], z['y'], z['displacement']
Fs = []
for s in range(3):
    p0 = _start_with_overlap(w, (lo + hi) // 2, 100 + s)
    Fs.append(agree_freq(A, B, p0, w, beta, lo, hi, 2 ** 24, 256, 200 + s))
F = np.mean(Fs, 0)
aligned = F > 0.5
wt = (w == pt)                       # witness agrees with the known correspondence (evaluation)
dev = kabsch_rmsd(X, Y, aligned & wt)
glob = kabsch_rmsd(X, Y, np.ones(n, bool))
segs = segments(aligned)
rec = dict(case=case, beta=beta, window=[lo, hi], n=n, n_aligned=int(aligned.sum()),
           replicate_corr=float(np.corrcoef(Fs[0], Fs[1])[0, 1]),
           segments=[(int(a) + 1, int(b) + 1) for a, b in segs if b - a >= 2],
           rmsd_aligned_after_fit_on_aligned=float(np.sqrt(np.mean(dev[aligned] ** 2))),
           rmsd_rest_after_fit_on_aligned=float(np.sqrt(np.mean(dev[~aligned] ** 2))),
           rmsd_global_fit=float(np.sqrt(np.mean(glob ** 2))),
           pscdb_displacement_aligned=float(np.median(disp[aligned])),
           pscdb_displacement_rest=float(np.median(disp[~aligned])),
           agree_freq=F.tolist())
print(json.dumps({k: v for k, v in rec.items() if k != 'agree_freq'}))
os.makedirs(os.path.join(V2, 'results', 'middle_phase'), exist_ok=True)
json.dump(rec, open(os.path.join(V2, 'results', 'middle_phase', f'{case}_{beta}.json'), 'w'))
