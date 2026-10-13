"""Round-5/6 comparators at a matched TOTAL budget per run (default 2^29 proposals, at
least the cost of the full validated two-phase pipeline: witness search + estimator
+ nested reference + validity checks; see analyze_r5.py):
  pt_broad   : PT, geometric ladder of 32 rungs from beta=0.5 (random walk mixes
               freely) to max(eval betas)+0.5, plus the evaluation betas; random init.
  pt_broad_w : same ladder, every replica initialised at the witness (witness-seeded PT).
  wl_ext     : Wang-Landau in S over [S_lo, min(|E_A|,|E_B|)] (the trivial maximum),
               so states scoring above the witness are not excluded.
  win_direct : direct stratified-window estimation (windows of three overlap levels,
               ratio chaining) at each evaluation beta, budget split evenly.
Seed 0 writes results/baselines3/<case>.json (round 5); seeds >= 1 write
<case>_s<seed>.json (round-6 replication).  Wall-clock seconds are recorded per method.
Usage: python baselines3.py --case pscdb_O_19 [--logB 29] [--seeds 0] [--methods pt_broad,pt_broad_w,wl_ext,win_direct]"""
import argparse, json, os, sys, time
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.baselines2 import wl_score, wl_prod, wl_reweight, pt_all
from gacert2.windows import overlap_profile

OUT = os.path.join(V2, 'results', 'baselines3')
os.makedirs(OUT, exist_ok=True)
ALL = ('pt_broad', 'pt_broad_w', 'wl_ext', 'win_direct')


def run(case, logB, seed=0, methods=ALL):
    A, B, pt, w, meta = load_instance(case)
    n = len(w)
    k0 = int(round(0.3 * n))
    Btot = 2 ** logB
    gt = json.load(open(os.path.join(V2, 'results', 'gt_comparison.json')))[case]
    betas = [r['beta'] for r in gt['rows']]
    rng = np.random.default_rng(seed + 5151)
    out = dict(case=case, n=n, k0=k0, logB=logB, seed=seed, betas=betas, methods={})
    margs = {}
    lad = np.geomspace(0.5, max(betas) + 0.5, 32)
    ladder = np.unique(np.round(np.concatenate((lad, betas)), 6))
    idx = np.array([int(np.argmin(np.abs(ladder - b))) for b in betas], dtype=np.int64)
    sweeps = Btot // len(ladder)
    for name, init in (('pt_broad', 'random'), ('pt_broad_w', 'witness')):
        if name not in methods:
            continue
        t0 = time.time()
        P0 = (np.array([rng.permutation(n) for _ in range(len(ladder))]) if init == 'random'
              else np.array([w.copy() for _ in range(len(ladder))]))
        fc, M, sw = pt_all(A, B, P0, ladder.astype(np.float64), w, k0, sweeps, sweeps // 4, 16,
                           int(rng.integers(2 ** 31)), idx)
        out['methods'][name] = dict(w={str(b): float(fc[i]) for b, i in zip(betas, idx)}, ladder=ladder.tolist(),
                                    swap=sw.tolist(), steps_per_rung=int(sweeps), sec=time.time() - t0)
        for j, b in enumerate(betas):
            margs[f'{b}_{name}'] = M[j]
        print(case, seed, name, {b: round(float(fc[i]), 3) for b, i in zip(betas, idx)}, 'swap min/med',
              round(float(sw.min()), 3), round(float(np.median(sw)), 3), f'{time.time()-t0:.0f}s', flush=True)
    if 'wl_ext' in methods:
        t0 = time.time()
        est = json.load(open(os.path.join(V2, 'results', 'twophase', f'{case}_gt.json')))
        gD = np.array(est['gridD'])
        r0 = est['reps'][0]
        i = int(np.argmin(np.abs(gD - min(betas))))
        slo = int(max(0, np.floor(r0['meanS_D'][i] - 5 * np.sqrt(max(r0['var_D'][i], 1.0)))))
        shi = int(min(A.sum() // 2, B.sum() // 2))
        stage_T = Btot // 128
        g, p, f, st = wl_score(A, B, rng.permutation(n), slo, shi, stage_T, 1.0, 1e-4, 0.5, 48, int(rng.integers(2 ** 31)))
        used = st * stage_T
        Tp = max(Btot - used, Btot // 4)
        H, Hk, Mw, Hm = wl_prod(A, B, p, w, slo, shi, g, Tp, Tp // 16, 16, int(rng.integers(2 ** 31)), True, 1)
        vis = np.flatnonzero(H > 0)
        wl = dict(slo=slo, shi=shi, Sw=int(meta['Sw']), max_visited_S=int(slo + vis.max()) if len(vis) else None,
                  stages=int(st), final_f=float(f), visited_bins=int(len(vis)), nbins=int(len(H)), w={},
                  proposals=int(used + Tp))
        for b in betas:
            o = wl_reweight(H, Hk, Mw, Hm, g, slo, b, k0, 1)
            wl['w'][str(b)] = o['wC']
            margs[f'{b}_wl_ext'] = o['marg']
        wl['sec'] = time.time() - t0
        out['methods']['wl_ext'] = wl
        print(case, seed, 'wl_ext', {b: round(v, 3) for b, v in wl['w'].items()}, 'max visited S', wl['max_visited_S'],
              'Sw', meta['Sw'], 'shi', shi, f'{time.time()-t0:.0f}s', flush=True)
    if 'win_direct' in methods:
        t0 = time.time()
        nwin = n // 2 + 2
        Tw = int(Btot // (len(betas) * nwin))
        wd = dict(T_per_window=Tw, w={})
        for b in betas:
            o = overlap_profile(A, B, w, float(b), Tw, int(rng.integers(2 ** 31)), want_marg=True)
            lp = o['logp']
            f_ = np.isfinite(lp[k0:])
            wd['w'][str(b)] = float(np.exp(logsumexp(lp[k0:][f_])))
            margs[f'{b}_win_direct'] = o['marg']
        wd['sec'] = time.time() - t0
        out['methods']['win_direct'] = wd
        print(case, seed, 'win_direct', {b: round(v, 3) for b, v in wd['w'].items()}, f'T/window 2^{np.log2(Tw):.1f}',
              f'{time.time()-t0:.0f}s', flush=True)
    tag = case if seed == 0 else f'{case}_s{seed}'
    json.dump(out, open(os.path.join(OUT, tag + '.json'), 'w'))
    np.savez_compressed(os.path.join(OUT, tag + '_marg.npz'), **margs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--logB', type=int, default=29)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--methods', default=','.join(ALL))
    a = ap.parse_args()
    for s in [int(x) for x in a.seeds.split(',')]:
        for c in a.case.split(','):
            if os.path.exists(os.path.join(OUT, (c if s == 0 else f'{c}_s{s}') + '.json')):
                continue
            run(c, a.logB, s, tuple(a.methods.split(',')))
