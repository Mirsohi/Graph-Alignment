"""Exact check of the nested multilevel reference: compare the estimate of
log mu_{b|C}(W), W = {k >= n-2}, with exact enumeration at n = 10 for several
b (small b makes mu(W) << 1, which exercises several levels).
Usage: python exact_nested.py"""
import json, os, sys
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'tests'))
import gacert2
from gacert2.phase import nested_reference
from test_core import correlated_pair, all_perms, all_scores
out = []
for (n, p, q, seed) in ((10, .5, .05, 5), (10, .4, .02, 6), (10, .6, .15, 7)):
    rng = np.random.default_rng(seed)
    A, B, pt = correlated_pair(n, p, q, rng)
    P = all_perms(n)
    S = all_scores(A, B, P).astype(float)
    w = P[int(np.argmax(S))].copy()
    K = (P == w[None, :]).sum(1)
    k0 = n // 2
    for b in (0.5, 1.0, 2.0, 4.0):
        C = K >= k0
        ex = float(logsumexp(b * S[C & (K >= n - 2)]) - logsumexp(b * S[C]))
        ests = []
        for r in range(5):
            lm, lev, fr, se, bs, bp, fb = nested_reference(A, B, w, k0, b, 2 ** 18, 100 * seed + r)
            ests.append((lm, se, lev, fb))
        e = np.array([x[0] for x in ests])
        rec = dict(n=n, seed=seed, beta=b, exact=ex, est_mean=float(e.mean()), est_sd=float(e.std(ddof=1)),
                   mean_reported_se=float(np.mean([x[1] for x in ests])), levels=ests[0][2],
                   floor_bound=any(x[3] for x in ests), err=float(e.mean() - ex))
        out.append(rec)
        print(json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(V2, 'results', 'exact_nested.json'), 'w'), indent=1)
