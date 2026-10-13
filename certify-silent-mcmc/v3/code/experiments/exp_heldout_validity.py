"""Held-out test of the validity rules (thresholds frozen after O.16).
Instances with KNOWN phase structure:
  multi : two disjoint correlated components (n1 = n2 = n/2), component 1 with low
          flip noise q1, component 2 with high noise q2 -> a partially aligned phase
          (component 1 aligned) between the diffuse and the fully aligned phase;
  single: one correlated component of size n (control).
For each: graph-only witness, one two-phase run (k0 = 0.3n), then the validity rules
at the estimated beta_c exactly as in validity.py/validity_c.py (edge < 1%;
D- and C- two-start |dS| < 3 SE + 2 and |dk| < 0.05 n; window agreement
|dlogit| < 1; middle basin with mass > 5% and peak in (0.1n, k0 + 0.05n)),
plus an extra rule not used before: any basin with mass > 5% strictly between the
diffuse peak and the witness peak.  Ground truth structure: window profile around
the planted pi*.  Usage: python exp_heldout_validity.py"""
import json, os, sys, time
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import find_witness
from gacert2.phase import estimate_two_phase, nested_reference, log_W_sum, rmh
from gacert2.windows import overlap_profile, basins, _start_with_overlap
from gacert.core import score

OUT = os.path.join(V2, 'results', 'heldout')
os.makedirs(OUT, exist_ok=True)


def er_pair(n, deg, q, rng):
    u, v = np.triu_indices(n, 1)
    A = np.zeros((n, n), dtype=np.int64)
    A[u, v] = rng.random(len(u)) < deg / (n - 1)
    A[v, u] = A[u, v]
    Bp = A.copy()
    f = (rng.random(len(u)) < q).astype(np.int64)
    Bp[u, v] ^= f
    Bp[v, u] = Bp[u, v]
    return A, Bp


def instance(kind, n, seed):
    rng = np.random.default_rng(7000 + seed + (0 if kind == 'single' else 100))
    if kind == 'single':
        A, Bp = er_pair(n, 7.0, 0.03, rng)
    else:
        h = n // 2
        A1, B1 = er_pair(h, 7.0, 0.03, rng)
        A2, B2 = er_pair(n - h, 7.0, 0.12, rng)
        A = np.zeros((n, n), dtype=np.int64); Bp = np.zeros_like(A)
        A[:h, :h] = A1; A[h:, h:] = A2
        Bp[:h, :h] = B1; Bp[h:, h:] = B2
    pt = rng.permutation(n)
    B = np.zeros_like(A)
    B[np.ix_(pt, pt)] = Bp
    return A, B, pt.astype(np.int64)


def two_start(A, B, w, beta, lo, hi, p_a, p_b, seed, T=2 ** 22):
    rng = np.random.default_rng(seed)
    res = []
    for p0 in (p_a, p_b):
        r = rmh(A, B, p0, w, beta, T, T // 4, 64, lo, hi, int(rng.integers(2 ** 31)), False)
        st, kt = r[0], r[1]
        L = len(st) // 20
        res.append((st.mean(), st[:20 * L].reshape(20, L).mean(1).std(ddof=1) / np.sqrt(20), kt.mean()))
    dS = abs(res[0][0] - res[1][0]); se = np.hypot(res[0][1], res[1][1])
    return bool(dS < 3 * se + 2 and abs(res[0][2] - res[1][2]) < 0.05 * len(w)), float(dS)


def run(kind, n, seed):
    tag = f'{kind}_n{n}_s{seed}'
    p = os.path.join(OUT, tag + '.json')
    if os.path.exists(p):
        return json.load(open(p))
    t0 = time.time()
    A, B, ps = instance(kind, n, seed)
    w = find_witness(A, B, restarts=4, T=2 ** 21)
    k0 = int(round(0.3 * n))
    gD = np.round(np.arange(0, 3.0001, 0.02), 4)
    gC = np.round(np.arange(1.0, 6.0001, 0.02), 4)[::-1]
    tp, pd, pc = estimate_two_phase(A, B, w, k0, gD, gC, 2 ** 16, seed)
    Sw = int(score(A, B, w))
    lm, *_ = nested_reference(A, B, w, k0, 6.0, 2 ** 20, seed + 3)
    tp.FC = tp.FC - tp.ref + (6.0 * Sw + log_W_sum(A, B, w, 6.0) - lm)
    bc = tp.beta_c()
    rec = dict(tag=tag, kind=kind, n=n, seed=seed, Sw=Sw, S_planted=int(score(A, B, ps)),
               witness_agree=float(np.mean(w == ps)), beta_c=bc)
    if not np.isfinite(bc):
        rec['note'] = 'no crossing on grid'
        json.dump(rec, open(p, 'w'))
        print(tag, rec, flush=True)
        return rec
    # rules
    iD = int(np.argmin(np.abs(np.array(pd['beta']) - bc)))
    iC = int(np.argmin(np.abs(np.array(pc['beta']) - bc)))
    hD, hC = pd['k_hist'][iD], pc['k_hist'][iC]
    edge = max(hD[k0 - 1] / max(1, sum(hD)), hC[k0] / max(1, sum(hC)))
    rng = np.random.default_rng(seed)
    pr = rng.permutation(n)
    while np.sum(pr == w) >= k0:
        pr = rng.permutation(n)
    meltD, dSD = two_start(A, B, w, bc, 0, k0 - 1, _start_with_overlap(w, k0 - 1, 1), pr, 11)
    meltC, dSC = two_start(A, B, w, bc, k0, n, w.copy(), _start_with_overlap(w, k0 + 1, 2), 12)
    lp = overlap_profile(A, B, w, bc, 2 ** 21, 1)['logp']
    wwin = float(np.exp(logsumexp(lp[k0:][np.isfinite(lp[k0:])])))
    wE = float(tp.wcond(bc))
    prof_ok = abs(np.log(wwin / (1 - wwin + 1e-300) + 1e-300) - np.log(wE / (1 - wE))) < 1.0
    bas, mins = basins(lp, 2.0)
    bas = [b for b in bas if b['mass'] >= 0.01]
    mid = any(b['mass'] > 0.05 and 0.1 * n < b['k_peak'] < k0 + 0.05 * n for b in bas)
    flagged = (edge >= 0.01) or (not meltD) or (not meltC) or (not prof_ok) or mid
    # ground-truth structure: profile around the planted permutation at the same beta
    lpg = overlap_profile(A, B, ps, bc, 2 ** 21, 2)['logp']
    basg = [b for b in basins(lpg, 2.0)[0] if b['mass'] >= 0.01]
    true_multi = sum(1 for b in basg if b['mass'] > 0.05) >= 3 or any(
        b['mass'] > 0.05 and 0.15 * n < b['k_peak'] < 0.85 * n for b in basg)
    rec.update(edge=float(edge), meltD_ok=meltD, dSD=dSD, meltC_ok=meltC, dSC=dSC, w_window=wwin, w_est=wE,
               profile_ok=bool(prof_ok), middle_basin=bool(mid), basins=bas, flagged=bool(flagged),
               gt_basins=basg, true_multi=bool(true_multi), sec=time.time() - t0)
    json.dump(rec, open(p, 'w'))
    print(tag, f"bc={bc:.3f} agree={rec['witness_agree']:.2f} edge={edge:.3f} meltD={meltD}({dSD:.1f}) meltC={meltC}({dSC:.1f}) "
          f"win={wwin:.2f}/est={wE:.2f} mid={mid} FLAG={flagged} | GT basins {[(b['k_peak'], round(b['mass'], 2)) for b in basg]} "
          f"true_multi={true_multi} {time.time()-t0:.0f}s", flush=True)
    return rec


if __name__ == '__main__':
    seeds = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(4)
    for s in seeds:
        run('multi', 60, s)
        run('single', 60, s)
