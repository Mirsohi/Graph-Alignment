"""Synthetic correlated graphs at the model-consistent inverse temperature.

A ~ ER(n, d/(n-1)) or a two-block SBM with the same expected degree; B is a
uniformly relabelled copy of A with every pair flipped independently w.p. q,
so the exact posterior is mu_beta with beta = 2 log((1-q)/q).
Per instance: TI estimate of Lambda, random-start MH, oracle-start MH,
annealed MH and geometric PT (equal 2^23-proposal budget for PT).
Usage: python exp_synth.py [--workers 2]
"""
import argparse, json, os, sys, time, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from gacert.core import (score, mh_chain, pt_chain, mean_score, d_marg, split_rhat,
                         iid_floor_plugin, log_mgf_ti, log_nfact)

OUT = os.path.join(ROOT, 'results', 'synth')
N = 120
DEG = 6.0
QS = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
SEEDS = [0, 1, 2]
FAMILIES = ['er', 'sbm']
LAMS = np.round(np.arange(0, 1.0001, 0.05), 4)
K = 4
RETAIN = 1024


def make_instance(family, q, seed):
    rng = np.random.default_rng(10_000 * seed + int(q * 1e5) + (0 if family == 'er' else 7))
    n = N
    if family == 'er':
        prob = np.full((n, n), DEG / (n - 1))
    else:
        lab = np.arange(n) >= int(0.4 * n)
        same = lab[:, None] == lab[None, :]
        # within/between ratio 6:1, scaled to expected degree DEG
        base = np.where(same, 6.0, 1.0)
        np.fill_diagonal(base, 0)
        prob = base * DEG / base.sum(1).mean()
    u, v = np.triu_indices(n, 1)
    A = np.zeros((n, n), dtype=np.int64)
    A[u, v] = rng.random(len(u)) < prob[u, v]
    A[v, u] = A[u, v]
    clean = A.copy()
    flip = (rng.random(len(u)) < q).astype(np.int64)
    clean[u, v] ^= flip
    clean[v, u] = clean[u, v]
    pi = rng.permutation(n)            # vertex i of A corresponds to vertex pi[i] of B
    B = np.zeros_like(A)
    B[np.ix_(pi, pi)] = clean
    return A, B, pi.astype(np.int64), rng


def summarise(res, pt, extra=None):
    Ms = [r[0] for r in res]
    sc = [r[1] for r in res]
    L = min(len(s) for s in sc)
    scm = np.array([s[:L] for s in sc])
    P = np.mean(Ms, 0)
    n = len(pt)
    out = dict(rhat=split_rhat(scm), d_marg=d_marg(Ms), floor_plugin=iid_floor_plugin(P, L),
               match=float(P[np.arange(n), pt].mean()),
               match_chain=[float(M[np.arange(n), pt].mean()) for M in Ms],
               chain_max_retained=scm.max(1).tolist(), chain_mean_score=scm.mean(1).tolist(),
               max_visited=[int(r[3]) for r in res], retained=int(L))
    if extra:
        out.update(extra)
    return out, P


def run(job):
    family, q, seed = job
    tag = f'{family}_q{q}_s{seed}'
    path = os.path.join(OUT, tag + '.json')
    if os.path.exists(path):
        return tag, 'cached'
    t0 = time.time()
    A, B, pt, rng = make_instance(family, q, seed)
    n = N
    beta = float(2 * np.log((1 - q) / q))
    sw = int(score(A, B, pt))
    rec = dict(tag=tag, family=family, q=q, seed=seed, n=n, beta=beta, known_score=sw,
               edges_a=int(A.sum() // 2), edges_b=int(B.sum() // 2), log_nfact=log_nfact(n))
    Lam, ms, sd = log_mgf_ti(A, B, LAMS, T=2 ** 19, seeds=(11, 12), rng_seed=seed + 5)
    # diffuse-branch check at the top of the TI grid: oracle start must melt
    melt = mean_score(A, B, pt.copy(), float(LAMS[-1]), 2 ** 19, 2 ** 17, 99)
    rec['ti'] = dict(lams=LAMS.tolist(), Lambda=Lam.tolist(), mean_S=ms.tolist(), sd_S=sd.tolist(),
                     oracle_mean_S_at_top=float(melt))
    T = 2 ** 21
    thin = max(1, (T - T // 4) // RETAIN)
    groups = {}
    margs = {}
    best = -1
    for start in ('uniform', 'oracle'):
        res = [mh_chain(A, B, rng.permutation(n) if start == 'uniform' else pt.copy(),
                        np.full(T, beta), T // 4, thin, int(rng.integers(2 ** 31))) for _ in range(K)]
        groups['mh_' + start], margs['mh_' + start] = summarise(res, pt, dict(work_per_chain=T))
        if start == 'uniform':
            best = max(best, max(int(r[3]) for r in res))
    ramp = np.concatenate((np.linspace(0.5, beta, 2 ** 22), np.full(2 ** 20, beta)))
    res = [mh_chain(A, B, rng.permutation(n), ramp, 2 ** 22, max(1, 2 ** 20 // RETAIN),
                    int(rng.integers(2 ** 31))) for _ in range(K)]
    groups['annealed_mh'], margs['annealed_mh'] = summarise(res, pt, dict(work_per_chain=len(ramp)))
    best = max(best, max(int(r[3]) for r in res))
    ladder = np.geomspace(1.0, beta, 16)
    sweeps = 2 ** 23 // 16
    res = []
    for _ in range(K):
        P0 = np.array([rng.permutation(n) for _ in range(16)])
        M, sc, smax, bst, rate = pt_chain(A, B, P0, ladder, sweeps, sweeps // 4,
                                          max(1, (sweeps - sweeps // 4) // RETAIN), int(rng.integers(2 ** 31)))
        res.append((M, sc, None, smax, bst, rate))
    groups['pt_geometric16'], margs['pt_geometric16'] = summarise(
        res, pt, dict(work_per_chain=sweeps * 16, min_swap_rate=float(np.mean([r[5].min() for r in res]))))
    best = max(best, max(int(r[3]) for r in res))
    # agreement of each method's pooled marginals with the oracle-start reference
    for k in groups:
        groups[k]['tv_to_oracle'] = float(np.abs(margs[k] - margs['mh_oracle']).sum() / (2 * n))
    rec['groups'] = groups
    rec['best_graph_only_score'] = int(best)
    rec['seconds'] = time.time() - t0
    json.dump(rec, open(path + '.tmp', 'w'))
    os.replace(path + '.tmp', path)
    return tag, rec['seconds']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = list(itertools.product(FAMILIES, QS, SEEDS))
    with Pool(a.workers) as pool:
        for tag, sec in pool.imap_unordered(run, jobs):
            print(tag, sec, flush=True)
