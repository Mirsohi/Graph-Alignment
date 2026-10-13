"""Exact-enumeration check (n = 9, 10) of the two-phase estimator identities:
Z = Z_C + Z_D, F_D by upward restricted TI, F_C by downward restricted TI with
the W-fraction reference, phase weights, mixture marginals.
Run: python v2/code/experiments/exact_twophase.py"""
import json, os, sys, time
import numpy as np
from scipy.special import logsumexp, gammaln
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'tests'))
from gacert2.phase import estimate_two_phase, log_region_sizes
from test_core import correlated_pair, all_perms, all_scores

out = {}
for (n, p, q, seed) in ((9, .5, .05, 3), (9, .45, .1, 4), (10, .5, .05, 5), (10, .4, .02, 6)):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    A, B, pt = correlated_pair(n, p, q, rng)
    P = all_perms(n)
    S = all_scores(A, B, P).astype(float)
    w = P[int(np.argmax(S))].copy()
    K = (P == w[None, :]).sum(1)
    k0 = n // 2
    C = K >= k0
    bgrid = np.round(np.arange(0, 6.0001, 0.05), 4)
    exD = np.array([logsumexp(b * S[~C]) for b in bgrid])
    exC = np.array([logsumexp(b * S[C]) for b in bgrid])
    logD, logC = log_region_sizes(n, k0)
    assert abs(logD - np.log((~C).sum())) < 1e-9 and abs(logC - np.log(C.sum())) < 1e-9
    tp, pd, pc = estimate_two_phase(A, B, w, k0, bgrid, bgrid[::-1], 2 ** 17, seed,
                                    marg_at=(1.0, 2.0, 3.0))
    est_D, est_C = tp.fD(bgrid), tp.fC(bgrid)
    wc_ex = 1 / (1 + np.exp(-(exC - exD)))
    wc_est = tp.wcond(bgrid)
    # exact marginals at a few betas vs mixture estimate
    mt = {}
    for b in (1.0, 2.0, 3.0):
        lw = b * S - logsumexp(b * S)
        pr = np.exp(lw)
        Mex = np.zeros((n, n))
        for i in range(n):
            np.add.at(Mex[i], P[:, i], pr)
        wc = float(tp.wcond(b))
        Mmix = wc * pc['marg'][b] + (1 - wc) * pd['marg'][b]
        mt[str(b)] = dict(rowTV_mix=float(np.abs(Mmix - Mex).sum() / (2 * n)),
                          w_exact=float(np.interp(b, bgrid, wc_ex)), w_est=wc)
    rec = dict(n=n, p=p, q=q, k0=k0, S_max=float(S.max()), S_true=float(S[np.all(P == pt, 1)][0]),
               logC_exact=float(np.log(C.sum())),
               maxerr_FD=float(np.max(np.abs(est_D - exD))),
               maxerr_FC=float(np.max(np.abs(est_C - exC))),
               maxerr_logZ=float(np.max(np.abs(np.logaddexp(est_C, est_D) - np.logaddexp(exC, exD)))),
               maxerr_wcond=float(np.max(np.abs(wc_est - wc_ex))),
               beta_c_exact=float(np.interp(0, (exC - exD), bgrid)) if np.any(exC > exD) else None,
               beta_c_est=tp.beta_c(), fracW_hi=tp.fracW_hi,
               ref_err=float(tp.ref - exC[-1]), marginals=mt, sec=time.time() - t0)
    out[f'n{n}_s{seed}'] = rec
    print(json.dumps(rec, indent=1), flush=True)
os.makedirs(os.path.join(V2, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(V2, 'results', 'exact_twophase.json'), 'w'), indent=1)
