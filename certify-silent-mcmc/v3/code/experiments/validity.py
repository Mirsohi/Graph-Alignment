"""Phase-validity diagnostic of the two-phase estimator at its beta_c.
For each pair: (a) edge occupancy of the D/C restricted chains at beta_c
(from the saved run); (b) stratified-window overlap profile p_k at beta_c:
w_C(win) = sum_{k>=k0} p_k vs the estimator's w_C, basins (local maxima with
mass >= 1%) and dividing minima; (c) D-chain melt test: D-restricted chain
started at overlap k0-1 (partially aligned) vs random start, mean S and mean k.
Usage: python validity.py tag [cases] [--T 21] [--seed 1]"""
import argparse, json, os, sys, time, glob
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import overlap_profile, basins, _start_with_overlap
from gacert2.phase import rmh, TwoPhase

ap = argparse.ArgumentParser()
ap.add_argument('tag')
ap.add_argument('cases', nargs='?', default='all')
ap.add_argument('--T', type=int, default=21)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='validity')
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(V2, 'results', 'twophase', f'*_{a.tag}.json')))
if a.cases != 'all':
    want = set(a.cases.split(','))
    files = [f for f in files if os.path.basename(f)[:-len(a.tag) - 6] in want]
od = os.path.join(V2, 'results', a.out)
os.makedirs(od, exist_ok=True)
for f in files:
    d = json.load(open(f))
    case = d['case']
    outp = os.path.join(od, f'{case}_s{a.seed}.json')
    if os.path.exists(outp):
        continue
    t0 = time.time()
    A, B, pt, w, meta = load_instance(case)
    n, k0 = d['n'], d['k0']
    bcs = [r['beta_c'] for r in d['reps']]
    bc = float(np.nanmean(bcs))
    gD, gC = np.array(d['gridD']), np.array(d['gridC'])
    wE = []
    for r in d['reps']:
        tp = TwoPhase(n, 0, 0, 0, gD, r['FD'], gC, r['FC'])
        wE.append(float(tp.wcond(bc)))
    iD = int(np.argmin(np.abs(gD - bc)))
    iC = int(np.argmin(np.abs(gC - bc)))
    edge = dict(edge_D=[r['edge_D'][iD] for r in d['reps']], edge_C=[r['edge_C'][iC] for r in d['reps']],
                kmean_D=[r['kmean_D'][iD] for r in d['reps']], kmean_C=[r['kmean_C'][iC] for r in d['reps']])
    prof = overlap_profile(A, B, w, bc, 2 ** a.T, a.seed)
    lp = prof['logp']
    fin = np.isfinite(lp)
    w_win = float(np.exp(logsumexp(lp[k0:][fin[k0:]])))
    bas, mins = basins(lp, 2.0)
    bas = [b for b in bas if b['mass'] >= 0.01]
    # D melt test
    T = 2 ** 22
    rng = np.random.default_rng(a.seed + 100)
    p_part = _start_with_overlap(w, k0 - 1, int(rng.integers(2 ** 31)))
    p_rand = rng.permutation(n)
    while np.sum(p_rand == w) >= k0:
        p_rand = rng.permutation(n)
    melt = {}
    for nm, p0 in (('partial_start', p_part), ('random_start', p_rand)):
        r = rmh(A, B, p0, w, bc, T, T // 4, 64, 0, k0 - 1, int(rng.integers(2 ** 31)), False)
        st, kt = r[0], r[1]
        L = len(st) // 20
        melt[nm] = dict(meanS=float(st.mean()), seS=float(st[:20 * L].reshape(20, L).mean(1).std(ddof=1) / np.sqrt(20)),
                        meank=float(kt.mean()), edge=float(np.mean(kt == k0 - 1)))
    dS = abs(melt['partial_start']['meanS'] - melt['random_start']['meanS'])
    seS = np.hypot(melt['partial_start']['seS'], melt['random_start']['seS'])
    checks = dict(
        edge_ok=bool(max(edge['edge_D'] + edge['edge_C']) < 0.01),
        profile_ok=bool(abs(np.log(w_win / (1 - w_win + 1e-300) + 1e-300) - np.log(np.mean(wE) / (1 - np.mean(wE)))) < 1.0),
        melt_ok=bool(dS < 3 * seS + 2.0 and abs(melt['partial_start']['meank'] - melt['random_start']['meank']) < 0.05 * n),
        two_basins=bool(len(bas) <= 2))
    rec = dict(case=case, n=n, k0=k0, beta_c=bc, wC_est=wE, wC_window=w_win, logp=lp.tolist(),
               meanS_by_k=prof['meanS'].tolist(), basins=bas, minima=mins, edge=edge, melt=melt,
               checks=checks, valid=bool(all(checks.values())), T=a.T, seed=a.seed, sec=time.time() - t0)
    json.dump(rec, open(outp, 'w'))
    print(f"{case:22s} bc={bc:.3f} wE={np.mean(wE):.2f} wWin={w_win:.3f} basins={[(b['k_peak'], round(b['mass'], 3)) for b in bas]} "
          f"mins={mins} edge={max(edge['edge_D'] + edge['edge_C']):.3f} melt dS={dS:.1f}±{seS:.1f} "
          f"k {melt['partial_start']['meank']:.0f}/{melt['random_start']['meank']:.0f} VALID={rec['valid']} {time.time()-t0:.0f}s",
          flush=True)
