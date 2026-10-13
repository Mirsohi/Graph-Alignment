"""GT-labelled false-alarm / detection table on synthetic correlated graphs.
Model-consistent beta(q) = 2 log((1-q)/q): the planted pi* is an EXACT posterior
draw, so chains started at pi* are exactly stationary (BDMC-style reference).
Labels (GT side, may use pi*): posterior weight w*_GT of {overlap with pi* >= 0.3n}
from the stratified-window profile around pi* (2 seeds, independent of the two-phase
TI); a random-start chain is WRONG if |1{mean overlap with pi* >= 0.3} - w*_GT| > 0.5,
CORRECT if < 0.2 (else excluded).  Group marginal TV to window GT marginals is also
reported.  Planted-start
chains are CORRECT by construction.
Diagnostics (all graph-only; pi* never used by a diagnostic):
  group level (4 chains): split-R-hat(S) >= 1.01; R* (random-forest chain
  classifier on (S, k_w)) >= 1.05; D_marg / D_iid >= 1.5.
  chain level: rigorous certificate (lambda = 0, graph-only witness);
  calibrated tilted certificate (v1 TI Lambda-hat with witness floor);
  calibrated two-phase TV bound (nu_chain(R) - mu-hat(R) >= 0.5).
Usage: python exp_alarm.py"""
import json, os, sys, time, itertools
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, HERE)
from gacert2.instances import synth_pair, find_witness
from gacert2.windows import overlap_profile
from gacert2.phase import rmh, estimate_two_phase, nested_reference, log_W_sum, TwoPhase
from gacert.core import score, split_rhat, d_marg, iid_floor_plugin, log_mgf_ti, log_nfact
from sklearn.ensemble import RandomForestClassifier

LN01 = np.log(0.01)
OUT = os.path.join(V2, 'results', 'alarm')
os.makedirs(OUT, exist_ok=True)


def rstar(chains_feats, seed=0):
    """R* (Lambert & Vehtari 2022): K x held-out accuracy of a classifier
    predicting the chain index from the draw."""
    X = np.concatenate(chains_feats)
    y = np.concatenate([np.full(len(f), i) for i, f in enumerate(chains_feats)])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    ntr = int(0.7 * len(y))
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    clf.fit(X[idx[:ntr]], y[idx[:ntr]])
    acc = (clf.predict(X[idx[ntr:]]) == y[idx[ntr:]]).mean()
    return float(len(chains_feats) * acc)


def run_instance(n, deg, q, seed, T=2 ** 20, K=4):
    tag = f'n{n}_d{deg}_q{q}_s{seed}'
    p = os.path.join(OUT, tag + '.json')
    if os.path.exists(p):
        return json.load(open(p))
    t0 = time.time()
    A, B, pstar = synth_pair(n, deg, q, seed)
    beta = float(2 * np.log((1 - q) / q))
    w = find_witness(A, B, restarts=4, T=2 ** 20)
    Sw = int(score(A, B, w))
    k0 = int(round(0.3 * n))
    # ground-truth phase weight (independent stratified windows, 2 seeds)
    wg = []
    Pgt = []
    for s in (1, 2):
        o = overlap_profile(A, B, pstar, beta, 2 ** 21, s, want_marg=True)   # GT side: reference = pi*
        lp = o['logp']
        wg.append(float(np.exp(logsumexp(lp[k0:][np.isfinite(lp[k0:])]))))
        Pgt.append(o['marg'])
    Pgt_floor = float(np.abs(Pgt[0] - Pgt[1]).sum() / (2 * n))
    Pgt = np.mean(Pgt, 0)
    wGT = float(np.mean(wg))
    # two-phase estimate (1 replicate, nested reference), grids up to beta
    gD = np.round(np.arange(0, max(2.6, beta) + 1e-9, 0.02), 4)
    gC = np.round(np.arange(1.0, max(6.0, beta + 1) + 1e-9, 0.02), 4)[::-1]
    tp, pd, pc = estimate_two_phase(A, B, w, k0, gD, gC, 2 ** 15, seed + 7)
    lm, lev, fr, se, bs, bp, fb = nested_reference(A, B, w, k0, float(gC[0]), 2 ** 20, seed + 9)
    tp.FC = tp.FC - tp.ref + (float(gC[0]) * Sw + log_W_sum(A, B, w, float(gC[0])) - lm)
    wE = float(tp.wcond(beta))
    # v1 TI for the tilted certificate (lambda <= 1) with witness floor
    lams = np.round(np.arange(0, 1.0001, 0.05), 4)
    Lam, ms, sd = log_mgf_ti(A, B, lams, T=2 ** 18, seeds=(11, 12), rng_seed=seed)
    lnf = log_nfact(n)
    Lam = np.maximum(Lam, lams * Sw - lnf)            # Lambda(lam) >= lam S(w) - log n!
    rng = np.random.default_rng(seed + 99)
    groups = []
    for start in ('random', 'planted'):
        ch, Ms, feats, Ss = [], [], [], []
        for c in range(K):
            p0 = rng.permutation(n) if start == 'random' else pstar.copy()
            st, kt, M, pf, smax, best, acc, brej, fW, m1, m2 = rmh(A, B, p0, w, beta, T, T // 4, (T - T // 4) // 1024,
                                                                   0, n, int(rng.integers(2 ** 31)), True)
            nuC = float(np.mean(kt >= k0))                    # phase w.r.t. graph-only witness (diagnostic)
            ov_star = float(M[np.arange(n), pstar].mean())    # mean overlap with pi* (label side only)
            Mx = float(st.max())
            crude = lnf + beta * (Mx - Sw)
            tilt = float(np.min(lnf + Lam + (beta - lams) * Mx - beta * Sw))
            tv_est = max(nuC - wE, (1 - nuC) - (1 - wE))
            if start == 'planted':
                lab = 'correct'
            else:
                dv = abs(float(ov_star >= 0.3) - wGT)
                lab = 'wrong' if dv > 0.5 else ('correct' if dv < 0.2 else 'ambiguous')
            ch.append(dict(nuC=nuC, ov_star=ov_star, maxS=Mx, label=lab, crude_cert=bool(crude < LN01),
                           tilted_cert=bool(tilt < LN01), twophase_tv_alarm=bool(tv_est >= 0.5),
                           tv_est=float(tv_est)))
            Ms.append(M)
            Ss.append(st)
            feats.append(np.column_stack((st, kt)))
        L = min(len(s) for s in Ss)
        P = np.mean(Ms, 0)
        rh = split_rhat(np.array([s[:L] for s in Ss]))
        rs = rstar([f[:L] for f in feats])
        dr = d_marg(Ms) / max(iid_floor_plugin(P, L), 1e-12)
        tv_gt = float(np.abs(P - Pgt).sum() / (2 * n))
        groups.append(dict(start=start, chains=ch, rhat=float(rh), rstar=rs, d_ratio=float(dr), tv_to_gt=tv_gt,
                           rhat_alarm=bool(not rh < 1.01), rstar_alarm=bool(rs >= 1.05), d_alarm=bool(dr >= 1.5)))
    rec = dict(tag=tag, n=n, deg=deg, q=q, seed=seed, beta=beta, k0=k0, Sw=Sw, S_planted=int(score(A, B, pstar)),
               witness_agree_planted=float(np.mean(w == pstar)), w_GT=wGT, gt_marg_floor=Pgt_floor, w_GT_seeds=wg, w_twophase=wE,
               groups=groups, sec=time.time() - t0)
    json.dump(rec, open(p, 'w'))
    print(tag, f'beta={beta:.2f} wGT={wGT:.3f} ({wg[0]:.3f},{wg[1]:.3f}) w2ph={wE:.3f}',
          f"Sw={Sw} Spl={rec['S_planted']} agree={rec['witness_agree_planted']:.2f}",
          [(g['start'], [c['label'][0] for c in g['chains']], round(g['rhat'], 3), round(g['rstar'], 2), round(g['tv_to_gt'], 2),
            [int(c['crude_cert']) for c in g['chains']], [int(c['tilted_cert']) for c in g['chains']])
           for g in groups], f'{rec["sec"]:.0f}s', flush=True)
    return rec


if __name__ == '__main__':
    cfgs = [(n, 8, q, s) for n in (40, 50) for q in (0.03, 0.06, 0.09, 0.12) for s in (0, 1)]
    for c in cfgs:
        run_instance(*c)
