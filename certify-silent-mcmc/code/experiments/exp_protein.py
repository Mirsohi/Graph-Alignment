"""Protein contact-network experiments (22 structure pairs).

For each case: (1) thermodynamic-integration estimate of Lambda(lam);
(2) beta sweep with random-start and oracle-start local MH (hysteresis);
(3) beta = 4 with annealed MH and two parallel-tempering ladders.
Raw per-group summaries are written to results/protein/<case>.json;
certificates are computed in analyze.py.
Usage: python exp_protein.py [--workers 2] [--cases a,b]
"""
import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from gacert.core import (score, mh_chain, pt_chain, d_marg, split_rhat, iid_floor_uniform,
                         iid_floor_plugin, log_mgf_ti, log_nfact)

CASES = os.path.join(ROOT, 'data', 'cases')
OUT = os.path.join(ROOT, 'results', 'protein')
BETAS = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
LAMS = np.round(np.arange(0, 1.2001, 0.05), 4)
T_MH = 2 ** 21
RETAIN = 1024
K = 4


def group_summary(A, B, pt, res, label, beta, extra=None):
    Ms = [r[0] for r in res]
    sc = [r[1] for r in res]
    L = min(len(s) for s in sc)
    scm = np.array([s[:L] for s in sc])
    P = np.mean(Ms, 0)
    n = len(pt)
    out = dict(label=label, beta=beta,
               rhat=split_rhat(scm), d_marg=d_marg(Ms),
               floor_uniform=iid_floor_uniform(n, L), floor_plugin=iid_floor_plugin(P, L),
               match=float(P[np.arange(n), pt].mean()),
               match_chain=[float(M[np.arange(n), pt].mean()) for M in Ms],
               mean_score=float(scm.mean()), chain_mean_score=scm.mean(1).tolist(),
               chain_max_retained=scm.max(1).tolist(), chain_min_retained=scm.min(1).tolist(),
               chain_median_retained=np.median(scm, 1).tolist(),
               max_visited=[int(r[3]) for r in res], retained=int(L))
    if extra:
        out.update(extra)
    return out, [r[4] for r in res]


def run_case(cid):
    path = os.path.join(OUT, cid + '.json')
    if os.path.exists(path):
        return cid, 'cached'
    t0 = time.time()
    z = np.load(os.path.join(CASES, cid + '.npz'))
    A = z['A_c8'].astype(np.int64)
    B = z['B_c8'].astype(np.int64)
    pt = z['pi_true'].astype(np.int64)
    n = len(pt)
    sw = int(score(A, B, pt))
    rng = np.random.default_rng(zlib.crc32(cid.encode()))
    rec = dict(case=cid, n=n, known_score=sw, edges_a=int(A.sum() // 2), edges_b=int(B.sum() // 2),
               log_nfact=log_nfact(n))
    # (1) Lambda by thermodynamic integration (diffuse branch, lam <= 1.2)
    Lam, ms, sd = log_mgf_ti(A, B, LAMS, T=2 ** 19, seeds=(11, 12), rng_seed=3)
    rec['ti'] = dict(lams=LAMS.tolist(), Lambda=Lam.tolist(), mean_S=ms.tolist(), sd_S=sd.tolist())
    best_graph_only = -1
    best_perm = None
    groups = []
    thin = max(1, (T_MH - T_MH // 4) // RETAIN)
    # (2) beta sweep, random and oracle starts
    for beta in BETAS:
        betas = np.full(T_MH, float(beta))
        for start in ('uniform', 'oracle'):
            res = []
            for c in range(K):
                p0 = rng.permutation(n) if start == 'uniform' else pt.copy()
                res.append(mh_chain(A, B, p0, betas, T_MH // 4, thin, int(rng.integers(2 ** 31))))
            g, bests = group_summary(A, B, pt, res, 'mh_' + start, beta)
            groups.append(g)
            if start == 'uniform':
                for r in res:
                    if r[3] > best_graph_only:
                        best_graph_only, best_perm = int(r[3]), r[4].copy()
    # (3) beta = 4: annealed MH and parallel tempering
    beta = 4.0
    ramp = np.concatenate((np.linspace(0.5, beta, 2 ** 22), np.full(2 ** 20, beta)))
    res = [mh_chain(A, B, rng.permutation(n), ramp, 2 ** 22, max(1, 2 ** 20 // RETAIN),
                    int(rng.integers(2 ** 31))) for _ in range(K)]
    g, _ = group_summary(A, B, pt, res, 'annealed_mh', beta,
                         dict(work_per_chain=int(len(ramp))))
    groups.append(g)
    for r in res:
        if r[3] > best_graph_only:
            best_graph_only, best_perm = int(r[3]), r[4].copy()
    for name, ladder in (('pt_quadratic8', beta * (np.arange(8) / 7.0) ** 2),
                         ('pt_geometric16', np.geomspace(1.0, beta, 16))):
        R = len(ladder)
        sweeps = 2 ** 23 // R
        thin_pt = max(1, (sweeps - sweeps // 4) // RETAIN)
        res = []
        for c in range(K):
            P0 = np.array([rng.permutation(n) for _ in range(R)])
            M, sc, smax, bst, sw_rate = pt_chain(A, B, P0, ladder.astype(np.float64), sweeps,
                                                  sweeps // 4, thin_pt, int(rng.integers(2 ** 31)))
            res.append((M, sc, None, smax, bst, sw_rate))
        g, _ = group_summary(A, B, pt, res, name, beta,
                             dict(work_per_chain=int(sweeps * R),
                                  min_swap_rate=float(np.mean([r[5].min() for r in res]))))
        groups.append(g)
        for r in res:
            if r[3] > best_graph_only:
                best_graph_only, best_perm = int(r[3]), r[4].copy()
    rec['groups'] = groups
    rec['best_graph_only_score'] = best_graph_only
    rec['best_graph_only_match'] = float(np.mean(best_perm == pt))
    rec['seconds'] = time.time() - t0
    tmp = path + '.tmp'
    json.dump(rec, open(tmp, 'w'))
    os.replace(tmp, path)
    return cid, rec['seconds']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cases', default='')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cases = sorted(f[:-4] for f in os.listdir(CASES) if f.endswith('.npz'))
    if a.cases:
        cases = a.cases.split(',')
    # smallest cases first to compile quickly
    sizes = {c: np.load(os.path.join(CASES, c + '.npz'))['pi_true'].shape[0] for c in cases}
    cases.sort(key=lambda c: sizes[c])
    with Pool(a.workers) as pool:
        for cid, sec in pool.imap_unordered(run_case, cases):
            print(cid, sec, flush=True)
