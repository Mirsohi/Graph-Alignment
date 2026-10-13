"""GT-labelled diagnostic table on exactly enumerable variable-selection posteriors
(p = 20).  Labels come from exact enumeration.  Chains started from EXACT posterior
draws are exactly stationary (false-alarm population); chains started empty /
from a prior draw / at the greedy solution are labelled WRONG if the posterior
mass of the witness phase C = {gamma ⊇ w} differs from the chain's occupancy of C
by more than 0.5, CORRECT if by less than 0.2.
Diagnostics (data-only; exact quantities are never used by a diagnostic):
  group: split-R-hat of ell and of model size (alarm if either >= 1.01), R* (RF on
         (ell, size, k_w)) >= 1.05;
  chain: rigorous certificate (lambda = 0); floor-tilted bound with FP-TI Lambda-hat;
         two-phase TV bound nu(R) - mu-hat(R) >= 0.5 with mu-hat from two-phase TI.
Usage: python exp_silent_bvs.py"""
import json, os, sys, time
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, HERE)
from gacert2.bvs import (make_design, make_suppressor_design, bvs_mh, ell_of, log_prior, greedy_forward,
                         enumerate_all, backward_from_full)
from gacert.core import split_rhat
from exp_bvs import backward_eliminate, power_ti
from exp_alarm import rstar
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

LN01 = np.log(0.01)
OUT = os.path.join(V2, 'results', 'silent_bvs')
os.makedirs(OUT, exist_ok=True)


def run(name, design, N, p, blocks, k, eta, a, h, seed, K=4, T=2 ** 20, mmax=40):
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    if design == 'decoy':
        X, y, tr, dec = make_design(N, p, blocks, k, eta, a, 1.0, seed)
    else:
        X, y, tr = make_suppressor_design(N, p, blocks, k, eta, a, 1.0, seed)
    G, c, yy, g = X.T @ X, X.T @ y, float(y @ y), float(N)
    rng = np.random.default_rng(seed + 123)
    lp = lambda gm: ell_of(G, c, yy, N, g, gm) + log_prior(gm, h)
    cand = [greedy_forward(G, c, yy, N, g, h, p)]
    las = LassoCV(cv=5, random_state=0, max_iter=20000).fit(X, y)
    cand.append(backward_eliminate(G, c, yy, N, g, h, (np.abs(las.coef_) > 1e-8).astype(np.int64)))
    cand.append(backward_from_full(G, c, yy, N, g, h, p))
    w = max(cand, key=lp)
    ellw, lrw = ell_of(G, c, yy, N, g, w), log_prior(w, h)
    wmask = w.astype(np.int64)
    kw = int(w.sum())
    # exact posterior
    Lall = enumerate_all(G, c, yy, N, g, p)
    masks = np.arange(1 << p)
    sizes = np.array([bin(m).count('1') for m in masks])
    lpost = Lall + sizes * np.log(h) + (p - sizes) * np.log(1 - h)
    Z = logsumexp(lpost)
    pr = np.exp(lpost - Z)
    wm = int(sum(1 << j for j in np.flatnonzero(w)))
    inC = (masks & wm) == wm
    wC_exact = float(pr[inC].sum())
    # FP-TI and two-phase Lambda (as in exp_bvs)
    lams = (np.arange(41) / 40.0) ** 4
    Lfp, _, _ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, seed * 10, mmax)
    Lfl = np.maximum(Lfp, lams * ellw + lrw)
    if kw > 0:
        lrC = kw * np.log(h)
        lrD = float(np.log1p(-np.exp(lrC)))
        while True:
            g0 = (rng.random(p) < h).astype(np.int64)
            if (g0 & wmask).sum() < kw:
                break
        LD, _, _ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, seed * 10 + 5, mmax, g0, wmask, 0, kw - 1)
        g0 = (rng.random(p) < h).astype(np.int64) | wmask
        LC, _, _ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, seed * 10 + 7, mmax, g0, wmask, kw, kw)
        wC_2p = float(1 / (1 + np.exp((lrD + LD[-1]) - (lrC + LC[-1]))))
    else:
        wC_2p = 1.0
    cum = np.cumsum(pr)
    groups = []
    starts = dict(exact=lambda: np.array([(int(m) >> j) & 1 for j in range(p)], dtype=np.int64),
                  empty=lambda: np.zeros(p, np.int64),
                  prior=lambda: (rng.random(p) < h).astype(np.int64),
                  greedy=lambda: cand[0].copy())
    for sname, f in starts.items():
        ch, ets, mts, feats = [], [], [], []
        for r in range(K):
            if sname == 'exact':
                m = int(np.searchsorted(cum, rng.random() * cum[-1]))
                g0 = np.array([(m >> j) & 1 for j in range(p)], dtype=np.int64)
            else:
                g0 = f()
            et, mt, inc, gam, emax, best, kt = bvs_mh(G, c, yy, N, g, h, g0, 1.0, T, T // 4, (T - T // 4) // 2048,
                                                      int(rng.integers(2 ** 31)), mmax, wmask, 0, p)
            nuC = float(np.mean(kt == kw)) if kw > 0 else 1.0
            Mx = float(et.max())
            crude = min(0.0, Mx - ellw - lrw)
            tilt = float(np.min(Lfl + (1 - lams) * Mx - ellw - lrw))
            tv = max(nuC - wC_2p, (1 - nuC) - (1 - wC_2p))
            if sname == 'exact':
                lab = 'correct'
            else:
                dv = abs(nuC - wC_exact)
                lab = 'wrong' if dv > 0.5 else ('correct' if dv < 0.2 else 'ambiguous')
            ch.append(dict(label=lab, nuC=nuC, crude_cert=bool(crude < LN01), tilted_cert=bool(tilt < LN01),
                           twophase_tv_alarm=bool(tv >= 0.5)))
            ets.append(et)
            mts.append(mt.astype(float))
            feats.append(np.column_stack((et, mt, kt)))
        L = min(len(e) for e in ets)
        rh1 = split_rhat(np.array([e[:L] for e in ets]))
        rh2 = split_rhat(np.array([m[:L] for m in mts]))
        rh = np.nanmax([rh1, rh2]) if not (np.isnan(rh1) and np.isnan(rh2)) else np.nan
        rs = rstar([ff[:L] for ff in feats])
        groups.append(dict(start=sname, chains=ch, rhat=float(rh), rstar=rs,
                           rhat_alarm=bool(rh >= 1.01), rstar_alarm=bool(rs >= 1.05)))
    rec = dict(name=name, design=design, a=a, seed=seed, wC_exact=wC_exact, wC_2p=wC_2p, witness_size=kw,
               logZ_exact=float(Z), fp=float(Lfp[-1]), floor=ellw + lrw, groups=groups, sec=time.time() - t0)
    json.dump(rec, open(path, 'w'))
    print(name, f"wC exact {wC_exact:.3f} 2P {wC_2p:.3f}",
          [(g_['start'][0], ''.join(c_['label'][0] for c_ in g_['chains']), round(g_['rhat'], 3), round(g_['rstar'], 2),
            ''.join(str(int(c_['crude_cert'])) for c_ in g_['chains'])) for g_ in groups], f"{rec['sec']:.0f}s", flush=True)
    return rec


if __name__ == '__main__':
    jobs = []
    for a in (0.6, 0.8, 1.0):
        for s in range(4):
            jobs.append((f'supp_a{a}_s{s}', 'suppressor', 200, 20, 1, 6, 0.3, a, 0.05, s))
    for s in range(6):
        jobs.append((f'decoy_s{s}', 'decoy', 300, 20, 1, 6, 0.4, 0.7, 0.05, s))
    for j in jobs:
        run(*j)
