"""Ground truth by stratified overlap windows (fixed beta, chains restricted to
{a,a+1,a+2}; ratio-chained profile; per-level marginals mixed by the profile).
Independent of the two-phase TI (shares only the witness).
Usage: python gt_windows.py case betas logT seeds"""
import json, os, sys, time
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import overlap_profile
case, betas, logT, ns = sys.argv[1], [float(x) for x in sys.argv[2].split(',')], int(sys.argv[3]), int(sys.argv[4])
A, B, pt, w, meta = load_instance(case)
n = len(w); k0 = int(round(0.3 * n))
res = dict(case=case, n=n, k0=k0, logT=logT, betas={})
margs = {}
for b in betas:
    rows = []
    for s in range(ns):
        t0 = time.time()
        o = overlap_profile(A, B, w, b, 2 ** logT, 500 + s + 1000 * int(os.environ.get('SEEDOFF', '0')), want_marg=True)
        lp = o['logp']
        fc = float(np.exp(logsumexp(lp[k0:][np.isfinite(lp[k0:])])))
        rows.append(dict(fracC=fc, logp=lp.tolist(), traversals=-1, sec=time.time() - t0))
        margs[f'{b}_s{s}'] = o['marg']
        print(case, b, s, round(fc, 4), f'{time.time()-t0:.0f}s', flush=True)
    res['betas'][str(b)] = rows
    margs[str(b)] = np.mean([margs[f'{b}_s{s}'] for s in range(ns)], 0)
od = os.path.join(V2, 'results', 'gt')
tag = os.environ.get('GTTAG', '')
json.dump(res, open(os.path.join(od, f'gt_win_{case}{tag}.json'), 'w'))
np.savez_compressed(os.path.join(od, f'gt_win_{case}{tag}_marg.npz'), **margs)
