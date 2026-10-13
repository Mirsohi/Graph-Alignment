"""Numerical check of Proposition 3 (anatomy of stuck TI) on the exact BVS runs and
of the Riemann bracketing (Proposition 4) on exact n<=10 permutation instances and
on the protein two-phase runs.  Writes results/d1_check.json."""
import json, os, sys, glob
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
R = os.path.join(V2, 'results')
out = dict(bvs=[], bracket_exact=[], bracket_protein={})
for f in sorted(glob.glob(os.path.join(R, 'bvs', '*.json'))):
    d = json.load(open(f))
    if 'exact' not in d:
        continue
    lams = np.array(d['ti']['lams'])
    FD = np.array(d['ti']['FD'])
    FC = np.array(d['ti']['FC'])
    Lex = np.array(d['exact']['Lambda'])
    Lfp = np.array(d['ti']['Lambda_fp'])
    floor = np.array(d['ti']['witness_floor'])
    true_err = Lex[-1] - Lfp[-1]
    cert = floor[-1] - Lfp[-1]
    gap = Lex[-1] - floor[-1]
    # decomposition of the gap at lambda = 1: [F_C - floor] + log(1 + e^{F_D - F_C})
    if np.isfinite(FD[-1]):
        dec1 = FC[-1] - floor[-1]
        dec2 = float(np.log1p(np.exp(FD[-1] - FC[-1])))
    else:
        dec1, dec2 = float('nan'), float('nan')
    out['bvs'].append(dict(name=d['name'], true_err=float(true_err), certified=float(cert), gap=float(gap),
                           gap_2p_decomp=[float(dec1), float(dec2)], FD_minus_FP=float(FD[-1] - Lfp[-1]) if np.isfinite(FD[-1]) else None,
                           detect=bool(cert > 0), FD_below_floor=bool(FD[-1] < floor[-1]) if np.isfinite(FD[-1]) else None))
# Riemann bracket on exact n <= 10 (exact restricted means from enumeration)
sys.path.insert(0, os.path.join(ROOT, 'code', 'tests'))
from test_core import correlated_pair, all_perms, all_scores
for (n, p, q, seed) in ((9, .5, .05, 3), (10, .4, .02, 6)):
    rng = np.random.default_rng(seed)
    A, B, pt = correlated_pair(n, p, q, rng)
    P = all_perms(n)
    S = all_scores(A, B, P).astype(float)
    w = P[int(np.argmax(S))]
    K = (P == w).sum(1)
    for reg, sel in (('D', K < n // 2), ('C', K >= n // 2)):
        for h in (0.05, 0.2):
            g = np.arange(0, 3 + 1e-9, h)
            f = np.array([np.sum(S[sel] * np.exp(b * S[sel] - logsumexp(b * S[sel]))) for b in g])
            left = np.sum(h * f[:-1]); right = np.sum(h * f[1:]); trap = 0.5 * (left + right)
            exact = logsumexp(3 * S[sel]) - logsumexp(0 * S[sel])
            out['bracket_exact'].append(dict(inst=f'n{n}s{seed}', region=reg, h=h, left=float(left), right=float(right),
                                             exact=float(exact), trap_err=float(trap - exact),
                                             inside=bool(left - 1e-9 <= exact <= right + 1e-9)))
# bracket widths of the protein runs (with estimated means): width = sum h (f_{i+1} - f_i)
for f in sorted(glob.glob(os.path.join(R, 'twophase', '*_gt.json'))):
    d = json.load(open(f))
    r = d['reps'][0]
    gD, gC = np.array(d['gridD']), np.array(d['gridC'])
    bc = r['beta_c']
    def width(g, m, lo, hi):
        sel = (g >= lo - 1e-9) & (g <= hi + 1e-9)
        gg, mm = g[sel], np.array(m)[sel]
        return float(np.sum(np.diff(gg) * np.maximum(np.diff(mm), 0)))
    wD = width(gD, r['meanS_D'], 0, bc)
    wC = width(gC, r['meanS_C'], bc, d['bhi'])
    out['bracket_protein'][d['case']] = dict(beta_c=bc, width_D=wD, width_C=wC,
                                             logit_halfwidth=0.5 * (wD + wC))
json.dump(out, open(os.path.join(R, 'd1_check.json'), 'w'), indent=1)
for r in out['bvs']:
    print(r['name'], 'true %.2f cert %.2f gap %.3f decomp %s FD<floor %s' % (r['true_err'], r['certified'], r['gap'], np.round(r['gap_2p_decomp'], 3), r['FD_below_floor']))
for r in out['bracket_exact']:
    print(r)
print(out['bracket_protein'])
