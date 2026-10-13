"""Budget convergence of the stratified-window profile: w(k >= k0) at a fixed
beta for window lengths 2^T, 2 seeds each.  Usage: python window_budget.py case beta k0 T1,T2,..."""
import json, os, sys, time
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import overlap_profile, basins
case, beta, k0 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
Ts = [int(x) for x in sys.argv[4].split(',')]
A, B, pt, w, meta = load_instance(case)
p = os.path.join(V2, 'results', 'validity', f'budget_{case}_{beta}.json')
res = json.load(open(p)) if os.path.exists(p) else {}
for T in Ts:
    for s in (11, 12):
        key = f'T{T}_s{s}'
        if key in res:
            continue
        t0 = time.time()
        o = overlap_profile(A, B, w, beta, 2 ** T, s)
        lp = o['logp']
        wc = float(np.exp(logsumexp(lp[k0:][np.isfinite(lp[k0:])])))
        res[key] = dict(wC=wc, logp=lp.tolist(), basins=basins(lp, 2.0)[0])
        print(case, beta, key, round(wc, 4), [(b['k_peak'], round(b['mass'], 3)) for b in res[key]['basins'] if b['mass'] > .01],
              f'{time.time()-t0:.0f}s', flush=True)
        json.dump(res, open(p, 'w'))
