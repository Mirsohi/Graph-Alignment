"""R15 S1 unit tests for gacert2.variational (coset ELBO).
 1. n=9 enumeration: ELBO <= log Z for 4 block structures x 3 beta.
 2. One block covering all n=7 positions: ELBO = exact log Z (to 1e-9).
 3. Monte Carlo check of the exact E_q S formula (n=30, 6 blocks of 5).
Usage: python r15_test_variational.py  (exit code 1 on failure; writes results/r15/mass_floor/unit_tests.json)"""
import itertools, json, os, sys, time
import numpy as np
from scipy.special import logsumexp
from r15_common import RES, prov
from gacert2.instances import synth_pair
from gacert2.variational import coset_elbo, PERMS
from gacert.core import score

t0 = time.time()
res = dict(t1=[], t2=[], t3={})
ok = True
A, B, pt = synth_pair(9, 4, 0.1, 3)
w = pt.copy()
allp = np.array(list(itertools.permutations(range(9))))
S = np.array([score(A, B, p) for p in allp])
for beta in (0.5, 1.0, 2.0):
    lz = float(logsumexp(beta * S))
    for bl in ([[i] for i in range(9)], [[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 2, 3, 4, 5], [6, 7, 8]],
               [[0, 1, 2, 3, 4, 5, 6], [7], [8]]):
        L = coset_elbo(A, B, w, beta, bl, sweeps=20)['elbo']
        good = L <= lz + 1e-9
        ok &= good
        res['t1'].append(dict(beta=beta, blocks=[len(b) for b in bl], elbo=L, logZ=lz, ok=bool(good)))
A, B, pt = synth_pair(7, 3, 0.1, 5)
w = pt.copy()
allp = np.array(list(itertools.permutations(range(7))))
S = np.array([score(A, B, p) for p in allp])
for beta in (0.5, 1.0, 2.0):
    lz = float(logsumexp(beta * S))
    L = coset_elbo(A, B, w, beta, [list(range(7))], sweeps=1)['elbo']
    good = abs(L - lz) < 1e-9
    ok &= good
    res['t2'].append(dict(beta=beta, elbo=L, logZ=lz, diff=L - lz, ok=bool(good)))
# 3. MC check: rebuild q_b from the fitted marginals is not possible in general; refit and sample
A, B, pt = synth_pair(30, 5, 0.1, 2)
w = pt.copy(); n = 30
blocks = [list(range(i, i + 5)) for i in range(0, n, 5)]
from gacert2 import variational as V
# instrumented copy of the update to keep q_b
Af, Bf = A.astype(float), B.astype(float)
P = np.zeros((n, n)); P[np.arange(n), w] = 1.0; qs = []; pre = []
for b in blocks:
    J = np.array(b); img = w[J][PERMS[len(b)]]; Sw_ = np.zeros(len(img))
    for a in range(len(b)):
        for c in range(a + 1, len(b)):
            if A[J[a], J[c]]: Sw_ += Bf[img[:, a], img[:, c]]
    pre.append((J, img, Sw_)); qs.append(None)
for _ in range(5):
    for bi, (J, img, Sw_) in enumerate(pre):
        mask = np.ones(n, bool); mask[J] = False
        h = (Af[np.ix_(J, mask)] @ P[mask]) @ Bf.T
        E = Sw_ + sum(h[a, img[:, a]] for a in range(len(J)))
        q = np.exp(E - logsumexp(E)); qs[bi] = q
        Pn = np.zeros((len(J), n))
        for a in range(len(J)): np.add.at(Pn[a], img[:, a], q)
        P[J] = Pn
ref = coset_elbo(A, B, w, 1.0, blocks, sweeps=5)
rng = np.random.default_rng(0)
vals = []
for _ in range(20000):
    p = w.copy()
    for q, (J, img, _) in zip(qs, pre):
        p[J] = img[rng.choice(len(q), p=q)]
    vals.append(score(A, B, p))
mc, se = float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))
good = abs(ref['ES'] - mc) < 4 * se
ok &= good
res['t3'] = dict(formula=ref['ES'], mc=mc, se=se, ok=bool(good))
res['all_ok'] = bool(ok)
res['prov'] = prov(__file__, t0)
os.makedirs(os.path.join(RES, 'mass_floor'), exist_ok=True)
json.dump(res, open(os.path.join(RES, 'mass_floor', 'unit_tests.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'prov'}, indent=1))
sys.exit(0 if ok else 1)
