"""Mesh-halving check of TI discretisation at large n (round 6).
Graphs (O.19, N.141): the two-phase estimator with the beta step halved on BOTH
branches (0.02 -> 0.01, same chain length per grid point, 4x boost on [1,2.5]),
cached graph-only witness, fresh nested reference; three seeds.  Compare with the
standard-mesh end-to-end replicates in results/pipeline_reps.
Variable selection (suppressor p=200, seed-1 instance 'p200'): two-phase TI with the
lambda schedule (i/40)^4 (41 points) vs (i/80)^4 (81 points), 2 replicates each;
witness = data-only backward elimination from the full model (equals the recorded
witness).  Round 6 final: 4 replicates per schedule.  Writes results/mesh_half/*.json.
Usage: python mesh_half.py graphs|bvs"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, HERE)
OUT = os.path.join(V2, 'results', 'mesh_half')
os.makedirs(OUT, exist_ok=True)


def graphs():
    from gacert2.instances import load_instance
    from gacert2.phase import estimate_two_phase, nested_reference, log_W_sum, TwoPhase
    from gacert.core import score
    from run_twophase import grids
    gtc = json.load(open(os.path.join(V2, 'results', 'gt_comparison.json')))
    for case in ('pscdb_O_19', 'pscdb_N_141'):
        A, B, pt, w, meta = load_instance(case)
        n = len(w)
        k0 = int(round(0.3 * n))
        betas = [r['beta'] for r in gtc[case]['rows']]
        bcg = gtc[case]['beta_c_gt']
        bstar = min(betas, key=lambda b: abs(b - bcg))
        for seed in (1, 2, 3):
            path = os.path.join(OUT, f'{case}_h001_s{seed}.json')
            if os.path.exists(path):
                continue
            t0 = time.time()
            gD, gC = grids(betas, 6.0, step=0.01)
            tp, pd, pc = estimate_two_phase(A, B, w, k0, gD, gC, 2 ** 18, 81000 + 31 * seed)
            Sw = int(score(A, B, w))
            lm, lev, fr, se, bs, bp, fb = nested_reference(A, B, w, k0, 6.0, 2 ** 22, 82000 + seed)
            FC = np.array(tp.FC) - tp.ref + (6.0 * Sw + log_W_sum(A, B, w, 6.0) - lm)
            tp2 = TwoPhase(n, Sw, tp.logD, tp.logC, tp.bD, tp.FD, tp.bC, FC)
            rec = dict(case=case, seed=seed, step=0.01, beta_c=tp2.beta_c(), beta_star=bstar,
                       wC={str(b): float(tp2.wcond(b)) for b in betas},
                       logit_star=float(tp2.fC(bstar) - tp2.fD(bstar)),
                       logZ_star=float(np.logaddexp(tp2.fC(bstar), tp2.fD(bstar))),
                       best_seen=int(tp.best_seen[0]), Sw=Sw, proposals=int(tp.proposals), sec=time.time() - t0)
            json.dump(rec, open(path, 'w'))
            print(case, seed, f"bc={rec['beta_c']:.4f} logit*={rec['logit_star']:.3f} logZ*={rec['logZ_star']:.3f}",
                  {b: round(v, 3) for b, v in rec['wC'].items()}, f"{rec['sec']:.0f}s", flush=True)


def bvs():
    from gacert2.bvs import make_suppressor_design, ell_of, log_prior, backward_from_full
    from exp_bvs import power_ti
    d = json.load(open(os.path.join(V2, 'results', 'bvs', 'p200.json')))
    N, p, blocks, k, eta, a, h, seed = (d[x] for x in ('N', 'p', 'blocks', 'k', 'eta', 'a', 'h', 'seed'))
    X, y, tr = make_suppressor_design(N, p, blocks, k, eta, a, 1.0, seed)
    G, c, yy, g = X.T @ X, X.T @ y, float(y @ y), float(N)
    w = backward_from_full(G, c, yy, N, g, h, p)
    assert abs(ell_of(G, c, yy, N, g, w) - d['witness']['ell']) < 1e-6
    wmask = w.astype(np.int64)
    kw = int(w.sum())
    lrC = kw * np.log(h)
    lrD = float(np.log1p(-np.exp(lrC)))
    rng = np.random.default_rng(777)
    res = {}
    for m in (40, 80):
        lams = (np.arange(m + 1) / float(m)) ** 4
        vals = []
        for r in range(4):
            t0 = time.time()
            while True:
                g0 = (rng.random(p) < h).astype(np.int64)
                if (g0 & wmask).sum() < kw:
                    break
            LD, mD, _ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, 9100 + 10 * m + r, 60, g0, wmask, 0, kw - 1)
            g1 = (rng.random(p) < h).astype(np.int64) | wmask
            LC, mC, _ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, 9200 + 10 * m + r, 60, g1, wmask, kw, kw)
            L2 = float(np.logaddexp(lrD + LD[-1], lrC + LC[-1]))
            vals.append(L2)
            print('bvs p200 m', m, 'rep', r, f'logZ_2P={L2:.4f}', f'{time.time()-t0:.0f}s', flush=True)
        res[str(m)] = vals
    rec = dict(name='p200', logZ_2P=res, recorded_2P=d['evidence']['two_phase'], fp_ti=d['evidence']['fp_ti'],
               floor=d['evidence']['witness_lower_bound'])
    json.dump(rec, open(os.path.join(OUT, 'bvs_p200_4rep.json'), 'w'), indent=1)
    print(rec, flush=True)


if __name__ == '__main__':
    {'graphs': graphs, 'bvs': bvs}[sys.argv[1]]()
