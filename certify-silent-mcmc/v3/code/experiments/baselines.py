"""Matched-budget baselines at fixed beta near the transition: random-start MH,
witness-start MH, annealed MH and geometric PT.  Each reports the estimated
condensed-phase weight (fraction of retained draws with overlap >= k0) and
pooled marginals.  Total budget per method and beta = 2^logB proposals.
Usage: python baselines.py --case pscdb_N_74 --betas 1.5,1.52 --logB 27"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert.core import mh_chain
from gacert2.phase import rmh, pt_k


def run(case, betas, logB, k0frac, K=4, seed=0):
    A, B, pt, w, meta = load_instance(case)
    n = len(w)
    k0 = int(round(k0frac * n))
    rng = np.random.default_rng(seed)
    per = 2 ** logB // K
    retain = 4096
    out = dict(case=case, n=n, k0=k0, logB=logB, K=K, betas={})
    margs = {}
    for b in betas:
        row = {}
        for name in ('mh_random', 'mh_witness', 'annealed', 'pt'):
            Ms, fr = [], []
            extra = []
            for c in range(K):
                sd = int(rng.integers(2 ** 31))
                if name == 'pt':
                    ladder = np.geomspace(0.5, b, 16)
                    sweeps = per // 16
                    P0 = np.array([rng.permutation(n) for _ in range(16)])
                    M, kt, st, rate = pt_k(A, B, P0, ladder, w, sweeps, sweeps // 4,
                                           max(1, (sweeps - sweeps // 4) // retain), sd)
                    extra.append(float(rate))
                else:
                    p0 = w.copy() if name == 'mh_witness' else rng.permutation(n)
                    T = per
                    if name == 'annealed':
                        ramp = np.linspace(0.5, b, per // 2)
                        r0 = mh_chain(A, B, p0, ramp, len(ramp) - 1, 1, sd)
                        p0 = r0[2].copy()
                        T = per - per // 2
                    burn = T // 4 if name != 'annealed' else 0
                    r = rmh(A, B, p0, w, float(b), T, burn, max(1, (T - burn) // retain), 0, n, sd + 1, True)
                    M, kt = r[2], r[1]
                Ms.append(M)
                fr.append(float(np.mean(kt >= k0)))
            margs[f'{b}_{name}'] = np.mean(Ms, 0)
            row[name] = dict(wC_chain=fr, wC_est=float(np.mean(fr)))
            if extra:
                row[name]['min_swap'] = extra
        out['betas'][str(b)] = row
        print(case, b, {k: round(v['wC_est'], 3) for k, v in row.items()}, flush=True)
    od = os.path.join(V2, 'results', 'baselines')
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, case + '.json'), 'w'))
    np.savez_compressed(os.path.join(od, case + '_marg.npz'), **margs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--betas', required=True)
    ap.add_argument('--logB', type=int, default=27)
    ap.add_argument('--k0frac', type=float, default=0.3)
    a = ap.parse_args()
    run(a.case, [float(x) for x in a.betas.split(',')], a.logB, a.k0frac)
