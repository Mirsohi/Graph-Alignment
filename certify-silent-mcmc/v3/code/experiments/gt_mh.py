"""Ground truth near the transition by very long unrestricted local MH that
demonstrably switches phase many times (random start and witness start).
Usage: python gt_mh.py --case pscdb_N_74 --betas 1.4,1.5 --logT 29
   or  python gt_mh.py --synth n40_d6_s0 ... (see synth_instance)."""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.phase import rmh, switch_count
from gacert2.instances import load_instance
from gacert.core import score


def run(case, betas, logT, k0frac, seed, outdir):
    A, B, pt, w, meta = load_instance(case)
    n = len(w)
    k0 = int(round(k0frac * n))
    T = 2 ** logT
    retain = 16384
    thin = max(1, (T - T // 8) // retain)
    rng = np.random.default_rng(seed)
    res = dict(case=case, n=n, k0=k0, T=T, Sw=int(score(A, B, w)), betas={}, meta=meta)
    margs = {}
    for b in betas:
        rows = []
        for start in ('random', 'witness'):
            p0 = rng.permutation(n) if start == 'random' else w.copy()
            t0 = time.time()
            st, kt, M, p, smax, best, acc, brej, fW, m1, m2 = rmh(
                A, B, p0, w, float(b), T, T // 8, thin, 0, n, int(rng.integers(2 ** 31)), True)
            sw, _ = switch_count(A, B, p0, w, float(b), T, int(0.15 * n), int(0.5 * n),
                                 int(rng.integers(2 ** 31)))
            fc = float(np.mean(kt >= k0))
            # batch-means SE of the phase fraction (20 batches)
            L = len(kt) // 20
            bm = (kt[:20 * L].reshape(20, L) >= k0).mean(1)
            rows.append(dict(start=start, fracC=fc, fracC_se=float(bm.std(ddof=1) / np.sqrt(20)),
                             meanS=float(m1), k_mean=float(kt.mean()), switches=int(sw),
                             smax=int(smax), sec=time.time() - t0,
                             k_hist=np.bincount(kt, minlength=n + 1).tolist()))
            margs[f'{b}_{start}'] = M
            print(case, b, start, f'fracC={fc:.3f} sw={sw} S={m1:.1f} t={time.time()-t0:.0f}s', flush=True)
        res['betas'][str(b)] = rows
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, f'gt_mh_{case}.json'), 'w'))
    np.savez_compressed(os.path.join(outdir, f'gt_mh_{case}_marg.npz'), **margs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--betas', required=True)
    ap.add_argument('--logT', type=int, default=29)
    ap.add_argument('--k0frac', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run(a.case, [float(x) for x in a.betas.split(',')], a.logT, a.k0frac, a.seed,
        os.path.join(V2, 'results', 'gt'))
