"""R15 M2 reference: stratified overlap-window profile around the PLANTED pi* (no method's witness)
at each beta in G, want_marg=True, two seeds (9001, 9002), T = 2^20 per window.
a_ref(beta) = sum_k k p_k / n averaged over seeds; SE_ref = |difference| / 2.
Writes results/r15/p1_heldout/ref/<inst>.json (+ _marg.npz, float32 marginals per seed).
Usage: python r15_p1_reference.py <inst> [logT]"""
import os, sys, time
import numpy as np
from r15_common import prov, dump
from r15_p1_common import P1, G, make, row_tv
from gacert2.windows import overlap_profile

SEEDS = (9001, 9002)


def run(inst, logT=20, outdir=None):
    outdir = outdir or os.path.join(P1, 'ref')
    path = os.path.join(outdir, inst + '.json')
    if os.path.exists(path):
        return None
    t0 = time.time()
    A, B, ps = make(inst)
    n = len(ps)
    rows, margs = [], {}
    prop = 0
    for b in G:
        a_s, lp_s = [], []
        for s in SEEDS:
            o = overlap_profile(A, B, ps, float(b), 2 ** logT, s * 1000 + int(round(b * 100)), want_marg=True)
            lp = o['logp']
            p = np.exp(np.where(np.isfinite(lp), lp, -np.inf))
            a_s.append(float((np.arange(n + 1) * p).sum() / n))
            lp_s.append(lp)
            margs[f'{b}_s{s}'] = o['marg'].astype(np.float32)
            prop += len(range(0, n - 1, 2)) * 2 ** logT   # windows a = 0, 2, ..., <= n-2 (exact for even n)
        rows.append(dict(beta=b, a_seeds=a_s, a_ref=float(np.mean(a_s)), se_ref=abs(a_s[0] - a_s[1]) / 2,
                         rowtv_between_seeds=row_tv(margs[f'{b}_s{SEEDS[0]}'], margs[f'{b}_s{SEEDS[1]}']),
                         logp_seeds=[l.tolist() for l in lp_s]))
    rec = dict(inst=inst, n=n, G=G, seeds=SEEDS, logT_per_window=logT, rows=rows,
               max_se_ref=max(r['se_ref'] for r in rows), reference_uncertain=bool(max(r['se_ref'] for r in rows) > 0.02),
               proposals_approx=int(prop), prov=prov(__file__, t0))
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, inst + '_marg.npz'), **margs)
    dump(rec, path)
    print(inst, 'a_ref', [round(r['a_ref'], 3) for r in rows], 'max se', round(rec['max_se_ref'], 4),
          f"{rec['prov']['wall_seconds']:.0f}s", flush=True)
    return rec


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20,
        outdir=(sys.argv[3] if len(sys.argv) > 3 else None))
