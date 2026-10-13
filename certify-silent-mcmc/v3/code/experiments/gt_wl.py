"""Witness-free ground truth: long Wang-Landau (multicanonical) run in the score S,
reweighted to each beta (phase weights w.r.t. the witness overlap are computed
only for evaluation).  Usage: python gt_wl.py case betas logT seeds"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.baselines2 import wl_score, wl_prod, wl_reweight
case, betas, logT, nseeds = sys.argv[1], [float(x) for x in sys.argv[2].split(',')], int(sys.argv[3]), int(sys.argv[4])
A, B, pt, w, meta = load_instance(case)
n = len(w)
k0 = int(round(0.3 * n))
est = json.load(open(os.path.join(V2, 'results', 'twophase', f'{case}_gt.json')))
gD = np.array(est['gridD'])
r0 = est['reps'][0]
i = int(np.argmin(np.abs(gD - min(betas))))
slo = int(max(0, np.floor(r0['meanS_D'][i] - 5 * np.sqrt(max(r0['var_D'][i], 1.0)))))
shi = int(meta['Sw'])
Btot = 2 ** logT
res = dict(case=case, n=n, k0=k0, slo=slo, shi=shi, logT=logT, seeds=[])
margs = {}
for s in range(nseeds):
    t0 = time.time()
    rng = np.random.default_rng(1000 + s)
    g, p, f, st = wl_score(A, B, rng.permutation(n), slo, shi, Btot // 128, 1.0, 1e-5, 0.5, 64, int(rng.integers(2 ** 31)))
    Tp = Btot - st * (Btot // 128)
    H, Hk, M, Hm = wl_prod(A, B, p, w, slo, shi, g, Tp, Tp // 16, 16, int(rng.integers(2 ** 31)), True, 1)
    row = dict(stages=int(st), final_f=float(f), prod=int(Tp), visited=int((H > 0).sum()), nbins=len(H), w={}, ess={})
    for b in betas:
        o = wl_reweight(H, Hk, M, Hm, g, slo, b, k0, 1)
        row['w'][str(b)] = o['wC']
        row['ess'][str(b)] = o['ess']
        margs[f'{b}_s{s}'] = o['marg']
    row['sec'] = time.time() - t0
    res['seeds'].append(row)
    print(case, 'seed', s, {b: round(v, 4) for b, v in row['w'].items()}, 'f', f, 'visited', row['visited'], '/', len(H),
          f'{row["sec"]:.0f}s', flush=True)
for b in betas:
    margs[str(b)] = np.mean([margs[f'{b}_s{s}'] for s in range(nseeds)], 0)
od = os.path.join(V2, 'results', 'gt')
json.dump(res, open(os.path.join(od, f'gt_wl_{case}.json'), 'w'))
np.savez_compressed(os.path.join(od, f'gt_wl_{case}_marg.npz'), **margs)
