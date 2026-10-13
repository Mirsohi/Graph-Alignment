"""Verifier: exact enumeration check at n=9 (362,880 permutations).
Exact Lambda vs TI; exact mu_beta(S<=s) vs crude/tilted bounds; false-certificate
rate on exact IID draws.  Run: python code/tests/exact_n9.py  (~1-2 min, 1 CPU)."""
import json, os, sys
import numpy as np
from scipy.special import logsumexp, gammaln
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gacert import core as C
from test_core import correlated_pair, all_perms, all_scores

n = 9
lnf = float(gammaln(n + 1))
P = all_perms(n)
LG = np.round(np.arange(0, 1.2001, 0.05), 4)        # TI grid used in the paper
LF = np.linspace(0, 4, 401)                          # fine grid for exact Lambda
out = {}
for cid, (p, q, seed) in enumerate(((.5, .10, 1), (.5, .25, 2), (.3, .05, 3))):
    rng = np.random.default_rng(seed)
    A, B, pt = correlated_pair(n, p, q, rng)
    S = all_scores(A, B, P)
    Lex = np.array([logsumexp(l * S) - lnf for l in LF])
    Lex_g = np.array([logsumexp(l * S) - lnf for l in LG])
    Lti, ms, sd = C.log_mgf_ti(A, B, LG, T=2 ** 17, seeds=(11, 12), rng_seed=1)
    ti_err = float(np.max(np.abs(Lti - Lex_g)))
    sw_best, sw_true = int(S.max()), int(C.score(A, B, pt))
    rec = dict(p=p, q=q, edges=int(A.sum() // 2), S_true=sw_true, S_max=sw_best,
               S_mean_uniform=float(S.mean()), ti_max_abs_err=ti_err,
               ti_err_at_1=float(Lti[20] - Lex_g[20]), ti_err_at_1_2=float(Lti[-1] - Lex_g[-1]))
    for beta in (1.0, 2.0, 4.0):
        lw = beta * S - logsumexp(beta * S)
        rows = []
        for s in np.unique(S):
            ex = float(logsumexp(lw[S <= s]))
            cr = C.crude_log_bound(n, beta, s, sw_best)
            tf, lamf = C.tilted_log_bound(n, beta, s, sw_best, LF, Lex)
            tg, lamg = C.tilted_log_bound(n, beta, s, sw_best, LG, Lti)
            assert ex <= cr + 1e-9 and ex <= tf + 1e-9
            rows.append((int(s), ex, cr, tf, float(lamf), tg))
        R = np.array(rows)
        # tightness where the exact tail is small but not astronomically so
        sel = (R[:, 1] < np.log(0.05)) & (R[:, 1] > np.log(1e-12))
        # false certificates with exact IID draws: K=4 chains x N=64 draws, statistic M=max
        cum = np.cumsum(np.exp(lw))
        fc = {}
        for name, col in (('crude', 2), ('tilted_exact', 3), ('tilted_ti', 5)):
            bmap = dict(zip(R[:, 0].astype(int), R[:, col]))
            hits = 0; reps = 4000
            for _ in range(reps):
                idx = np.searchsorted(cum, rng.random(256) * cum[-1])
                M = int(S[np.minimum(idx, len(S) - 1)].max())
                hits += bmap[M] < np.log(0.01)
            fc[name] = hits / reps
        # super-uniformity check of the exact law of M:  P(mu(S<=M) <= a) <= a
        rec['beta=%g' % beta] = dict(
            exact_tail_at_Suniform_mean=float(logsumexp(lw[S <= S.mean()]) / np.log(10)),
            n_s_values=int(len(R)),
            n_s_certified_crude=int((R[:, 2] < np.log(.01)).sum()),
            n_s_certified_tilted=int((R[:, 3] < np.log(.01)).sum()),
            n_s_certified_tilted_TIgrid=int((R[:, 5] < np.log(.01)).sum()),
            n_s_exact_below_001=int((R[:, 1] < np.log(.01)).sum()),
            median_gap_crude_nats=float(np.median(R[sel, 2] - R[sel, 1])) if sel.any() else None,
            median_gap_tilted_nats=float(np.median(R[sel, 3] - R[sel, 1])) if sel.any() else None,
            median_gain_tilted_vs_crude_nats=float(np.median(R[sel, 2] - R[sel, 3])) if sel.any() else None,
            false_cert_rate_alpha001=fc)
    out['case%d' % cid] = rec
    print(json.dumps(rec, indent=1), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exact_n9_results.json'), 'w'), indent=1)
