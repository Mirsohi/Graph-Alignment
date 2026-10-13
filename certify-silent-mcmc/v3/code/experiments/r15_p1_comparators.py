"""R15 M2 comparator arm at matched TOTAL compute (wrapper around the baselines3 methods).
Budget per instance B = max over the two pipeline replicates of total proposals (witness search +
estimator incl. refinements + nested reference + diagnostics + retry).  The comparator reuses
replicate 1's final witness (stored in its JSON), and the witness-search cost of replicate 1 is charged inside B, so the sampler
gets B - witness_cost proposals.
  pt_broad_w : PT, ladder = geomspace(0.5, max(G)+0.5, 32) U G, all replicas started at the witness,
               sweeps = floor(B_s / R) (one proposal per rung per sweep), neighbour swaps after each
               sweep (gacert2.baselines2.pt_all, burn sweeps/4, thin 16).
  wl_ext     : Wang-Landau in S over [S_lo, S_hi], S_hi = min(|E_A|, |E_B|) (trivial maximum),
               S_lo = floor(|E_A||E_B| / C(n,2)) = E_uniform[S] (witness-free, deterministic; baselines3
               used a released two-phase run for S_lo, which test instances do not have), stage_T = B_s/128,
               at most 48 stages (<= 0.375 B_s), production for the remaining B_s - used proposals
               (so the total is exactly B_s), reweighted to each beta in G.
The witness enters wl_ext only through the recorded overlap (phase weight, evaluation only).
a(beta) = (1/n) sum_i M[i, pi*(i)] from the method's marginals (pi* used ONLY for evaluation).
Seed: default_rng(5151 + idx).  Writes results/r15/p1_heldout/comp/<inst>_<method>.json.
Usage: python r15_p1_comparators.py <inst> <method> [pipe_dir] [outdir]"""
import json, os, sys, time
import numpy as np
from r15_common import prov, dump
from r15_p1_common import P1, G, make, a_from_marg
from gacert2.baselines2 import wl_score, wl_prod, wl_reweight, pt_all


def budget(inst, pipe_dir):
    recs = [json.load(open(os.path.join(pipe_dir, f'{inst}_r{r}.json'))) for r in (1, 2)]
    return max(r['cost_total'] for r in recs), recs[0]


def run(inst, method, pipe_dir=None, outdir=None):
    pipe_dir = pipe_dir or os.path.join(P1, 'pipe')
    outdir = outdir or os.path.join(P1, 'comp')
    path = os.path.join(outdir, f'{inst}_{method}.json')
    if os.path.exists(path):
        return None
    t0 = time.time()
    A, B, ps = make(inst)
    n = A.shape[0]
    k0 = int(round(0.3 * n))
    Btot, r1 = budget(inst, pipe_dir)
    wcost = int(r1['cost']['witness'])
    Bs = Btot - wcost
    idx = int(inst.split('_i')[1]) if '_i' in inst else 0
    rng = np.random.default_rng(5151 + idx)
    # replicate 1's final witness (after any refinement), stored by r15_p1_pipeline.py
    w = np.array(r1['witness'], dtype=np.int64)
    from gacert.core import score
    Sw = int(score(A, B, w))
    assert Sw == r1['Sw']
    wsec = 0.0
    out = dict(inst=inst, method=method, n=n, k0=k0, budget_total=Btot, witness_cost=wcost, sampler_budget=Bs,
               witness_S=Sw, witness_S_rep1_final=r1['Sw'], rep1_refinements=r1['refinements'], seed=5151 + idx, G=G,
               witness_recompute_seconds=wsec)
    ts = time.time()
    if method == 'pt_broad_w':
        lad = np.geomspace(0.5, max(G) + 0.5, 32)
        ladder = np.unique(np.round(np.concatenate((lad, G)), 6))
        ix = np.array([int(np.argmin(np.abs(ladder - b))) for b in G], dtype=np.int64)
        sweeps = Bs // len(ladder)
        P0 = np.array([w.copy() for _ in range(len(ladder))])
        fc, M, sw = pt_all(A, B, P0, ladder.astype(np.float64), w, k0, sweeps, sweeps // 4, 16,
                           int(rng.integers(2 ** 31)), ix)
        out.update(ladder=ladder.tolist(), swap=sw.tolist(), sweeps=int(sweeps), proposals=int(sweeps * len(ladder)),
                   rows=[dict(beta=b, wC=float(fc[i]), a=a_from_marg(M[j], ps)) for j, (b, i) in enumerate(zip(G, ix))])
        margs = {f'{b}': M[j].astype(np.float32) for j, b in enumerate(G)}
    elif method == 'wl_ext':
        eA, eB = int(A.sum() // 2), int(B.sum() // 2)
        slo = int(np.floor(eA * eB / (n * (n - 1) / 2)))
        shi = int(min(eA, eB))
        stage_T = Bs // 128
        g, p, f, st = wl_score(A, B, rng.permutation(n), slo, shi, stage_T, 1.0, 1e-4, 0.5, 48, int(rng.integers(2 ** 31)))
        used = int(st * stage_T)
        Tp = int(Bs - used)
        H, Hk, Mw, Hm = wl_prod(A, B, p, w, slo, shi, g, Tp, Tp // 16, 16, int(rng.integers(2 ** 31)), True, 1)
        vis = np.flatnonzero(H > 0)
        rows, margs = [], {}
        for b in G:
            o = wl_reweight(H, Hk, Mw, Hm, g, slo, b, k0, 1)
            rows.append(dict(beta=b, wC=o['wC'], ess=o['ess'], a=a_from_marg(o['marg'], ps)))
            margs[f'{b}'] = o['marg'].astype(np.float32)
        out.update(slo=slo, shi=shi, stages=int(st), final_f=float(f), visited_bins=int(len(vis)), nbins=int(len(H)),
                   max_visited_S=int(slo + vis.max()) if len(vis) else None, proposals=int(used + Tp),
                   proposals_wl_stages=used, proposals_production=Tp, rows=rows)
    else:
        raise ValueError(method)
    out['sampler_seconds'] = time.time() - ts
    out['charged_proposals'] = int(out['proposals'] + wcost)
    out['prov'] = prov(__file__, t0)
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f'{inst}_{method}_marg.npz'), **margs)
    dump(out, path)
    print(inst, method, 'a', [round(r['a'], 3) for r in out['rows']], f'log2B={np.log2(Btot):.2f}',
          f"{out['prov']['wall_seconds']:.0f}s", flush=True)
    return out


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], pipe_dir=(sys.argv[3] if len(sys.argv) > 3 else None),
        outdir=(sys.argv[4] if len(sys.argv) > 4 else None))
