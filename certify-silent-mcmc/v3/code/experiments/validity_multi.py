"""Validity rules at temperatures AWAY from beta_c (round 6): for each passing protein
pair and each GT instance, the frozen rules are evaluated at beta_c + {-0.25, -0.1,
+0.1, +0.25} (and at beta_c for reference), using the saved two-phase run:
  edge      : < 1% of restricted draws on the region boundary (nearest grid point);
  D two-start: D-restricted chains from overlap k0-1 and from a random start (2^22 each);
  C two-start: C-restricted chains from w and from overlap k0+1 (2^23 each);
               tolerance |dS| < 3 SE + 2 and |dk| < 0.05 n;
  window    : stratified-window profile (2^21 per window);  passes if
               |logit w_win - logit w_hat| < 1, or both < 0.01, or both > 0.99
               (the logit rule alone is ill-posed deep inside one phase);
  middle basin: no basin of mass > 5% with peak in (0.1 n, k0 + 0.05 n).
Writes results/validity_multi/<case>.json.
Usage: python validity_multi.py case1,case2,..."""
import json, os, sys, time
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import overlap_profile, basins, _start_with_overlap
from gacert2.phase import rmh, TwoPhase

OUT = os.path.join(V2, 'results', 'validity_multi')
os.makedirs(OUT, exist_ok=True)
OFFS = (-0.25, -0.1, 0.0, 0.1, 0.25)


def two_start(A, B, w, beta, lo, hi, p_a, p_b, T, rng):
    res = []
    for p0 in (p_a, p_b):
        r = rmh(A, B, p0, w, beta, T, T // 4, 64, lo, hi, int(rng.integers(2 ** 31)), False)
        st, kt = r[0], r[1]
        L = len(st) // 20
        res.append((float(st.mean()), float(st[:20 * L].reshape(20, L).mean(1).std(ddof=1) / np.sqrt(20)), float(kt.mean())))
    dS = abs(res[0][0] - res[1][0])
    se = float(np.hypot(res[0][1], res[1][1]))
    return bool(dS < 3 * se + 2 and abs(res[0][2] - res[1][2]) < 0.05 * len(w)), float(dS), se


def run(case):
    path = os.path.join(OUT, case + '.json')
    if os.path.exists(path):
        return
    fn = os.path.join(V2, 'results', 'twophase', f'{case}_gt.json')
    if not os.path.exists(fn):
        fn = os.path.join(V2, 'results', 'twophase', f'{case}_all.json')
    d = json.load(open(fn))
    A, B, pt, w, meta = load_instance(case)
    n, k0 = d['n'], d['k0']
    gD, gC = np.array(d['gridD']), np.array(d['gridC'])
    bc = float(np.nanmean([r['beta_c'] for r in d['reps']]))
    rng = np.random.default_rng(606)
    rows = []
    for off in OFFS:
        t0 = time.time()
        b = round(bc + off, 4)
        wE = float(np.mean([TwoPhase(n, 0, 0, 0, gD, r['FD'], gC, r['FC']).wcond(b) for r in d['reps']]))
        iD, iC = int(np.argmin(np.abs(gD - b))), int(np.argmin(np.abs(gC - b)))
        edge = max(max(r['edge_D'][iD] for r in d['reps']), max(r['edge_C'][iC] for r in d['reps']))
        pr = rng.permutation(n)
        while np.sum(pr == w) >= k0:
            pr = rng.permutation(n)
        okD, dSD, seD = two_start(A, B, w, b, 0, k0 - 1, _start_with_overlap(w, k0 - 1, int(rng.integers(2 ** 31))), pr, 2 ** 22, rng)
        okC, dSC, seC = two_start(A, B, w, b, k0, n, w.copy(), _start_with_overlap(w, k0 + 1, int(rng.integers(2 ** 31))), 2 ** 23, rng)
        lp = overlap_profile(A, B, w, b, 2 ** 21, int(rng.integers(2 ** 31)))['logp']
        fin = np.isfinite(lp[k0:])
        ww = float(np.exp(logsumexp(lp[k0:][fin])))
        lg = lambda x: np.log(max(x, 1e-300) / max(1 - x, 1e-300))
        prof_ok = bool(abs(lg(ww) - lg(wE)) < 1 or (ww < 0.01 and wE < 0.01) or (ww > 0.99 and wE > 0.99))
        bas = [x for x in basins(lp, 2.0)[0] if x['mass'] >= 0.01]
        mid = any(x['mass'] > 0.05 and 0.1 * n < x['k_peak'] < k0 + 0.05 * n for x in bas)
        rules = dict(edge=bool(edge < 0.01), D_two_start=okD, C_two_start=okC, window=prof_ok, middle_basin=not mid)
        rows.append(dict(offset=off, beta=b, w_est=wE, w_window=ww, edge=float(edge), dS_D=dSD, se_D=seD, dS_C=dSC,
                         se_C=seC, basins=bas, rules=rules, passed=bool(all(rules.values())), sec=time.time() - t0))
        print(case, f'b={b:.3f} (bc{off:+.2f}) wE={wE:.3f} wWin={ww:.3f}', {k: int(v) for k, v in rules.items()},
              'PASS' if rows[-1]['passed'] else 'FAIL', f'{time.time()-t0:.0f}s', flush=True)
    json.dump(dict(case=case, n=n, k0=k0, beta_c=bc, source=os.path.basename(fn), rows=rows), open(path, 'w'), indent=1)


if __name__ == '__main__':
    for c in sys.argv[1].split(','):
        run(c)
