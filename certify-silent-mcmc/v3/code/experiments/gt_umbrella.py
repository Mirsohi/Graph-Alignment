"""Ground truth by flat-histogram umbrella sampling in the witness overlap
(independent of the two-phase TI estimator: a single unrestricted biased chain
that traverses the overlap range, reweighted exactly).
Usage: python gt_umbrella.py --case pscdb_O_19 --betas 1.44,1.46 --logT 28 --seeds 2"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.phase import rmh
from gacert2.umbrella import umbrella_gt


def run(case, betas, logT, seeds, k0frac, tag=''):
    A, B, pt, w, meta = load_instance(case)
    n = len(w)
    k0 = int(round(k0frac * n))
    res = dict(case=case, n=n, k0=k0, logT=logT, meta=meta, betas={})
    margs = {}
    for b in betas:
        r = rmh(A, B, w.copy(), w, float(b), 2 ** 22, 2 ** 20, 64, k0, n, 7, False)
        kcap = int(min(n - 2, np.argmax(np.bincount(r[1], minlength=n + 1))))
        rows = []
        Ms = []
        for s in range(seeds):
            t0 = time.time()
            o = umbrella_gt(A, B, w, float(b), k0, kcap, 2 ** 23, 2 ** 26, 2 ** logT, 100 * s + 3)
            rows.append(dict(fracC=o['fracC'], meanS=o['meanS'], traversals=o['traversals'],
                             kcap=kcap, pk=o['pk'].tolist(), sec=time.time() - t0))
            Ms.append(o['marg'])
            print(case, b, s, f"fracC={o['fracC']:.4f} trav={o['traversals']} S={o['meanS']:.1f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        res['betas'][str(b)] = rows
        margs[str(b)] = np.mean(Ms, 0)
        for s, M in enumerate(Ms):
            margs[f'{b}_s{s}'] = M
    od = os.path.join(V2, 'results', 'gt')
    os.makedirs(od, exist_ok=True)
    name = f'gt_umb_{case}' + (('_' + tag) if tag else '')
    json.dump(res, open(os.path.join(od, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(od, name + '_marg.npz'), **margs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--betas', required=True)
    ap.add_argument('--logT', type=int, default=28)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--k0frac', type=float, default=0.3)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    run(a.case, [float(x) for x in a.betas.split(',')], a.logT, a.seeds, a.k0frac, a.tag)
