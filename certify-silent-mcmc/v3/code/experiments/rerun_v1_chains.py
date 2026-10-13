"""Re-run the v1 local-MH protocol (4 chains x 2^21 random-transposition
proposals, quarter burn-in, ~1024 retained draws) with the phase of every
retained draw measured by overlap with the GRAPH-ONLY witness (no pi_true).
Starts: uniform random, and witness start (replaces v1's oracle start).
Stores per chain: nu(D) = fraction of retained draws with k < k0, max/mean
retained S, the retained S trace (for R-hat), mean k.
Usage: python rerun_v1_chains.py [cases|all]"""
import json, os, sys, time, glob, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.phase import rmh

BETAS = [1.25, 1.5, 1.75, 2.0, 2.5]
T = 2 ** 21
K = 4
od = os.path.join(V2, 'results', 'v1chains')
os.makedirs(od, exist_ok=True)
cases = sorted(os.path.basename(f)[:-9] for f in glob.glob(os.path.join(V2, 'results', 'twophase', '*_all.json')))
if len(sys.argv) > 1 and sys.argv[1] != 'all':
    cases = sys.argv[1].split(',')
for case in cases:
    p = os.path.join(od, case + '.json')
    if os.path.exists(p):
        continue
    t0 = time.time()
    A, B, pt, w, meta = load_instance(case)
    n = len(w)
    k0 = int(round(0.3 * n))
    rng = np.random.default_rng(zlib.crc32(case.encode()) + 7)
    rec = dict(case=case, n=n, k0=k0, Sw=meta['Sw'], T=T, groups=[])
    thin = (T - T // 4) // 1024
    for b in BETAS:
        for start in ('random', 'witness'):
            ch = []
            for c in range(K):
                p0 = rng.permutation(n) if start == 'random' else w.copy()
                st, kt, M, pf, smax, best, acc, brej, fW, m1, m2 = rmh(A, B, p0, w, b, T, T // 4, thin, 0, n,
                                                                      int(rng.integers(2 ** 31)), False)
                ch.append(dict(nuD=float(np.mean(kt < k0)), maxS=float(st.max()), meanS=float(st.mean()),
                               meank=float(kt.mean()), S=st.astype(int).tolist()))
            rec['groups'].append(dict(beta=b, start=start, chains=ch))
    json.dump(rec, open(p, 'w'))
    print(case, f'{time.time()-t0:.0f}s', flush=True)
