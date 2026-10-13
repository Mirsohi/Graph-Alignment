"""Strong baselines at a matched WHOLE-CURVE budget (default 2^28 proposals,
the cost of one two-phase replicate is 2^27.9):
  pt_tuned : PT with 24 rungs concentrated on [beta_c-0.3, beta_c+0.3] incl. the
             evaluation betas; every rung recorded.
  wl_score : Wang-Landau in S (witness-free), reweighted to every beta.
  umb_k    : umbrella/WL in the witness overlap, budget split evenly over betas.
Writes results/baselines2/<case>.json and _marg.npz.
Usage: python baselines2.py --case pscdb_O_19 --betas 1.40,... [--logB 28]"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.baselines2 import wl_score, wl_prod, wl_reweight, pt_all
from gacert2.umbrella import umbrella_gt
from gacert2.phase import rmh


def run(case, betas, logB, k0frac=0.3, seed=0, tag=''):
    A, B, pt, w, meta = load_instance(case)
    n = len(w)
    k0 = int(round(k0frac * n))
    Btot = 2 ** logB
    rng = np.random.default_rng(seed + 4242)
    est = json.load(open(os.path.join(V2, 'results', 'twophase', f'{case}_gt.json')))
    bc = float(np.mean([r['beta_c'] for r in est['reps']]))
    out = dict(case=case, n=n, k0=k0, logB=logB, beta_c_used=bc, methods={})
    margs = {}
    # ---- tuned PT
    t0 = time.time()
    R = 24
    lin = np.linspace(bc - 0.3, bc + 0.3, R - len(betas))
    ladder = np.unique(np.round(np.concatenate((lin, betas)), 6))
    ladder = np.sort(ladder)
    idx = np.array([int(np.argmin(np.abs(ladder - b))) for b in betas], dtype=np.int64)
    sweeps = Btot // len(ladder)
    P0 = np.array([rng.permutation(n) for _ in range(len(ladder))])
    fc, M, sw = pt_all(A, B, P0, ladder.astype(np.float64), w, k0, sweeps, sweeps // 4, 16,
                       int(rng.integers(2 ** 31)), idx)
    out['methods']['pt_tuned'] = dict(w={str(b): float(fc[i]) for b, i in zip(betas, idx)},
                                      ladder=ladder.tolist(), swap=sw.tolist(), cold_steps_per_rung=int(sweeps),
                                      sec=time.time() - t0)
    for j, b in enumerate(betas):
        margs[f'{b}_pt_tuned'] = M[j]
    print(case, 'pt_tuned', {b: round(float(fc[i]), 3) for b, i in zip(betas, idx)}, 'min swap', round(float(sw.min()), 3),
          f'{time.time()-t0:.0f}s', flush=True)
    # ---- Wang-Landau in S (witness-free)
    t0 = time.time()
    gD = np.array(est['gridD'])
    r0 = est['reps'][0]
    i = int(np.argmin(np.abs(gD - min(betas))))
    slo = int(max(0, np.floor(r0['meanS_D'][i] - 5 * np.sqrt(max(r0['var_D'][i], 1.0)))))
    shi = int(meta['Sw'])
    stage_T = Btot // 128
    g, p, f, st = wl_score(A, B, rng.permutation(n), slo, shi, stage_T, 1.0, 1e-4, 0.5, 48,
                           int(rng.integers(2 ** 31)))
    used = st * stage_T
    Tp = max(Btot - used, Btot // 4)
    H, Hk, Mw, Hm = wl_prod(A, B, p, w, slo, shi, g, Tp, Tp // 16, 16, int(rng.integers(2 ** 31)), True, 1)
    wl = dict(slo=slo, shi=shi, stages=int(st), final_f=float(f), prod_steps=int(Tp), wl_steps=int(used),
              visited_bins=int((H > 0).sum()), nbins=int(len(H)), w={}, ess={})
    for b in betas:
        o = wl_reweight(H, Hk, Mw, Hm, g, slo, b, k0, 1)
        wl['w'][str(b)] = o['wC']
        wl['ess'][str(b)] = o['ess']
        margs[f'{b}_wl_score'] = o['marg']
    wl['sec'] = time.time() - t0
    out['methods']['wl_score'] = wl
    print(case, 'wl_score', {b: round(v, 3) for b, v in wl['w'].items()}, 'stages', st, 'f', f,
          'visited', wl['visited_bins'], '/', wl['nbins'], f'{time.time()-t0:.0f}s', flush=True)
    # ---- umbrella in witness overlap, per beta budget Btot/len(betas)
    t0 = time.time()
    per = Btot // len(betas)
    um = dict(w={}, traversals={}, per_beta=int(per))
    for b in betas:
        r = rmh(A, B, w.copy(), w, float(b), per // 16, per // 64, 64, k0, n, int(rng.integers(2 ** 31)), False)
        kcap = int(min(n - 2, np.argmax(np.bincount(r[1], minlength=n + 1))))
        o = umbrella_gt(A, B, w, float(b), k0, kcap, per // 64, per // 16, per // 2, int(rng.integers(2 ** 31)),
                        True, 64)
        um['w'][str(b)] = o['fracC']
        um['traversals'][str(b)] = o['traversals']
        margs[f'{b}_umb_k'] = o['marg']
    um['sec'] = time.time() - t0
    out['methods']['umb_k'] = um
    print(case, 'umb_k', {b: round(v, 3) for b, v in um['w'].items()}, f'{time.time()-t0:.0f}s', flush=True)
    od = os.path.join(V2, 'results', 'baselines2')
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, case + tag + '.json'), 'w'))
    np.savez_compressed(os.path.join(od, case + tag + '_marg.npz'), **margs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--betas', required=True)
    ap.add_argument('--logB', type=int, default=28)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run(a.case, [float(x) for x in a.betas.split(',')], a.logB, seed=a.seed)
