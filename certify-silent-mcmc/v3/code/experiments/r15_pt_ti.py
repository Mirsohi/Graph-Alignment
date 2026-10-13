"""R15 S3: population-MCMC (parallel tempering) TI as a stronger standard TI, checked by the witness floor.
PT with 64 rungs uniform on beta in [0, 3] (rung 0 at beta=0 samples uniformly), every replica started at a
seeded uniform random permutation; one random-transposition MH proposal per rung per sweep, then neighbour
swaps (alternating even/odd pairs, as gacert2.baselines2.pt_all); total 2^29 proposals (sweeps = 2^29/64),
burn-in 1/4 of the sweeps.  Lambda_PT(beta) = log n! + trapezoid integral of the per-rung mean score.
RIGOROUS: E_cert(beta) = beta S* - Lambda_PT(beta), S* = max(S(cached witness), best S seen by the PT run).
Writes results/r15/pt_ti/<case>.json.  Usage: python r15_pt_ti.py worker <wid> | one <case> <logB> <seed>"""
import json, math, os, sys, time
import numpy as np
import numba as nb
from scipy.special import gammaln
from r15_common import RES, V3, prov, dump, claim, append_jsonl
from gacert2.instances import load_instance
from gacert2.phase import cumtrapz0
from gacert.core import score, delta

OUT = os.path.join(RES, 'pt_ti')


@nb.njit(cache=True)
def pt_ti_kernel(A, B, P0, ladder, sweeps, burn, nbatch, seed):
    np.random.seed(seed)
    R = len(ladder)
    n = P0.shape[1]
    P = P0.copy()
    S = np.zeros(R, dtype=np.int64)
    for r in range(R):
        S[r] = score(A, B, P[r])
    tot = np.zeros(R)
    bsum = np.zeros((nbatch, R))
    m = 0
    L = max(1, (sweeps - burn) // nbatch)
    sacc = np.zeros(R - 1)
    stry = np.zeros(R - 1)
    best = -1
    bestp = P[0].copy()
    for t in range(sweeps):
        for r in range(R):
            u = np.random.randint(n)
            v = np.random.randint(n - 1)
            if v >= u:
                v += 1
            d = delta(A, B, P[r], u, v)
            if d >= 0 or np.random.random() < math.exp(ladder[r] * d):
                a = P[r, u]
                P[r, u] = P[r, v]
                P[r, v] = a
                S[r] += d
                if S[r] > best:
                    best = S[r]
                    bestp[:] = P[r]
        for r in range(t % 2, R - 1, 2):
            stry[r] += 1
            la = (ladder[r + 1] - ladder[r]) * (S[r] - S[r + 1])
            if la >= 0 or np.random.random() < math.exp(la):
                tmp = P[r].copy()
                P[r] = P[r + 1]
                P[r + 1] = tmp
                ts = S[r]
                S[r] = S[r + 1]
                S[r + 1] = ts
                sacc[r] += 1
        if t >= burn:
            j = (t - burn) // L
            for r in range(R):
                tot[r] += S[r]
                if j < nbatch:
                    bsum[j, r] += S[r]
            m += 1
    return tot / max(m, 1), bsum / L, sacc / np.maximum(stry, 1), best, bestp


def run(case, logB=29, seed=1, R=64):
    path = os.path.join(OUT, f'{case}_B{logB}_s{seed}.json')
    if os.path.exists(path):
        return None
    t0 = time.time()
    A, B, pt, w, meta = load_instance(case)
    n = len(w)
    Sw = int(score(A, B, w))
    ladder = np.linspace(0.0, 3.0, R)
    sweeps = 2 ** logB // R
    rng = np.random.default_rng(seed)
    P0 = np.array([rng.permutation(n) for _ in range(R)])
    mS, bm, sw, best, bestp = pt_ti_kernel(A, B, P0, ladder, sweeps, sweeps // 4, 20, int(rng.integers(2 ** 31)))
    se = bm.std(0, ddof=1) / np.sqrt(20)
    assert int(score(A, B, bestp)) == int(best)
    Sstar = max(Sw, int(best))
    L = gammaln(n + 1) + cumtrapz0(ladder, mS)
    cert = ladder * Sstar - L
    ev = {}
    for b in (2.0, 2.5, 3.0):
        Lb = float(np.interp(b, ladder, L)) if b in ladder else None
        # exact trapezoid to b with linear interpolation of the mean score
        i = int(np.searchsorted(ladder, b, side='right') - 1)
        mb = np.interp(b, ladder, mS)
        Lb = float(L[i] + 0.5 * (mS[i] + mb) * (b - ladder[i]))
        ev[str(b)] = dict(Lambda_PT=Lb, E_cert_point=float(b * Sstar - Lb))
    first = float(ladder[np.argmax(cert > 0)]) if (cert > 0).any() else None
    rec = dict(case=case, n=n, seed=seed, logB=logB, rungs=R, sweeps=int(sweeps), proposals=int(sweeps * R),
               Sw_cached=Sw, best_S=int(best), S_star=Sstar, S_star_source='pt_best' if best > Sw else 'cached_witness',
               ladder=ladder, meanS=mS, se=se, swap=sw, Lambda_PT=L,
               RIGOROUS=dict(E_cert_point=cert, first_cert_beta=first, eval=ev),
               swap_min=float(sw.min()), swap_median=float(np.median(sw)))
    rec['prov'] = prov(__file__, t0)
    dump(rec, path)
    print(json.dumps(dict(case=case, n=n, best=int(best), Sw=Sw, E3=round(ev['3.0']['E_cert_point'], 2),
                          swap_min=round(rec['swap_min'], 3), secs=round(rec['prov']['wall_seconds']))), flush=True)
    return rec


if __name__ == '__main__':
    if sys.argv[1] == 'one':
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        wid = sys.argv[2]
        cases = sorted((json.load(open(f))['n'], os.path.basename(f)[:-9]) for f in
                       __import__('glob').glob(os.path.join(V3, 'results', 'twophase', '*_all.json')))
        for n, c in cases:
            nm = f'{c}_B29_s1'
            if os.path.exists(os.path.join(OUT, nm + '.json')) or not claim(os.path.join(OUT, 'locks'), nm):
                continue
            t = time.time()
            try:
                run(c, 29, 1)
                append_jsonl(dict(job=nm, worker=wid, secs=time.time() - t, ok=True), os.path.join(OUT, 'jobs_done.jsonl'))
            except Exception as e:
                import traceback; traceback.print_exc()
                append_jsonl(dict(job=nm, worker=wid, ok=False, err=repr(e)), os.path.join(OUT, 'jobs_done.jsonl'))
