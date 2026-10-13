"""I2 demonstration: power-posterior witness certificates for Bayesian variable
selection (g-prior, Bernoulli(h) inclusion prior) on suppressor-block designs.

Instances
  exact20_strong : p=20, 1 block of k=6, strong signal  (exact enumeration)
  exact20_weak   : p=20, 1 block of k=6, weak signal    (exact; chains mix)
  p200           : p=200, 3 blocks of k=6               (no enumeration)
For each: witness (greedy forward; lasso support + backward elimination; best
visited), K chains per start type (empty, prior draw, greedy, oracle), power-
posterior TI (Friel & Pettitt 2008) from the prior, crude and tilted
certificates per chain, split-R-hat of ell and model size, and the TI evidence
estimate versus the witness lower bound log p(y) >= ell(w) + log rho(w)
(and the exact value when enumerable).  Exact IID draws give the false-alarm
rate of the certificate.
Usage: python exp_bvs.py"""
import json, os, sys, time
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.bvs import (capped_anchors, make_design, make_suppressor_design, bvs_mh, ell_of, log_prior, greedy_forward,
                         enumerate_all, backward_from_full)
from gacert.core import split_rhat
from sklearn.linear_model import LassoCV

LN01 = np.log(0.01)
OUT = os.path.join(V2, 'results', 'bvs')


def backward_eliminate(G, c, yy, N, g, h, gam):
    gam = gam.copy()
    cur = ell_of(G, c, yy, N, g, gam) + log_prior(gam, h)
    while gam.sum() > 0:
        best, bj = cur, -1
        for j in np.flatnonzero(gam):
            gam[j] = 0
            v = ell_of(G, c, yy, N, g, gam) + log_prior(gam, h)
            gam[j] = 1
            if v > best:
                best, bj = v, j
        if bj < 0:
            break
        gam[bj] = 0
        cur = best
    return gam


def power_ti(G, c, yy, N, g, h, p, lams, T, seed, mmax, start=None, wmask=None, kmin=0, kmax=10 ** 6):
    """Friel-Pettitt TI: E_{mu_lam}[ell] on a grid, sequential warm starts from
    a prior draw (lam=0 is exact prior sampling).  Returns Lambda-hat."""
    rng = np.random.default_rng(seed)
    gam = (rng.random(p) < h).astype(np.int64) if start is None else start.copy()
    if wmask is None:
        wmask = np.zeros(p, np.int64)
    m, se = [], []
    for lam in lams:
        et, mt, inc, gam, emax, best, kt = bvs_mh(G, c, yy, N, g, h, gam, float(lam), T, T // 4,
                                                  max(1, (T - T // 4) // 2048), int(rng.integers(2 ** 31)),
                                                  mmax, wmask, kmin, kmax)
        m.append(float(et.mean()))
        se.append(float(et.std() / np.sqrt(max(1, len(et) / 50))))
    m = np.array(m)
    L = np.concatenate(([0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(lams))))
    return L, m, np.array(se)


def certify(M, ellw, lrw, lams, L):
    """log bound on mu(ell <= M): min over lam of Lambda(lam) + (1-lam) M - ell(w) - log rho(w)."""
    crude = min(0.0, M - ellw - lrw)
    vals = L + (1 - lams) * M - ellw - lrw
    j = int(np.argmin(vals))
    return crude, min(0.0, float(vals[j])), float(lams[j])


def run_instance(name, N, p, blocks, k, eta, a, h, seed, K=4, T=2 ** 20, mmax=60, design='suppressor'):
    t0 = time.time()
    if design == 'decoy':
        X, y, tr, dec = make_design(N, p, blocks, k, eta, a, 1.0, seed)
    else:
        X, y, tr = make_suppressor_design(N, p, blocks, k, eta, a, 1.0, seed)
    G = X.T @ X
    c = X.T @ y
    yy = float(y @ y)
    g = float(N)
    rng = np.random.default_rng(seed + 99)
    gt = np.zeros(p, np.int64)
    gt[tr] = 1
    lp = lambda gm: ell_of(G, c, yy, N, g, gm) + log_prior(gm, h)
    cand = {}
    cand['greedy'] = greedy_forward(G, c, yy, N, g, h, p)
    las = LassoCV(cv=5, random_state=0, max_iter=20000).fit(X, y)
    gl = (np.abs(las.coef_) > 1e-8).astype(np.int64)
    cand['lasso_backward'] = backward_eliminate(G, c, yy, N, g, h, gl)
    if p < N:
        cand['backward_full'] = backward_from_full(G, c, yy, N, g, h, p)
    rec = dict(name=name, design=design, N=N, p=p, blocks=blocks, k=k, eta=eta, a=a, h=h, seed=seed,
               truth=tr.tolist(), logpost_truth=lp(gt), lasso_size=int(gl.sum()),
               lasso_covers_truth=bool(gl[tr].all()),
               candidates={kk: dict(logpost=lp(v), ell=ell_of(G, c, yy, N, g, v), size=int(v.sum()),
                                    equals_truth=bool(np.array_equal(v, gt))) for kk, v in cand.items()})
    # chains
    starts = dict(empty=lambda: np.zeros(p, np.int64),
                  prior=lambda: (rng.random(p) < h).astype(np.int64),
                  greedy=lambda: cand['greedy'].copy(),
                  oracle=lambda: gt.copy())
    chains = {}
    best_state, best_lp = None, -np.inf
    for sname, f in starts.items():
        ets, mts, incs, rows = [], [], [], []
        for r in range(K):
            et, mt, inc, gam, emax, bst, _ = bvs_mh(G, c, yy, N, g, h, f(), 1.0, T, T // 4,
                                                    max(1, (T - T // 4) // 2048), int(rng.integers(2 ** 31)),
                                                    mmax, np.zeros(p, np.int64), 0, p)
            ets.append(et)
            mts.append(mt)
            incs.append(inc)
            rows.append(dict(max_ell=float(et.max()), mean_ell=float(et.mean()),
                             truth_incl=float(inc[tr].mean())))
            if sname != 'oracle' and lp(bst) > best_lp:
                best_lp, best_state = lp(bst), bst.copy()
        L = min(len(e) for e in ets)
        chains[sname] = dict(rows=rows, rhat_ell=split_rhat(np.array([e[:L] for e in ets])),
                             rhat_size=split_rhat(np.array([m[:L] for m in mts]).astype(float)),
                             incl=np.mean(incs, 0).tolist())
    cand['best_visited'] = best_state
    rec['candidates']['best_visited'] = dict(logpost=lp(best_state), size=int(best_state.sum()),
                                             equals_truth=bool(np.array_equal(best_state, gt)))
    # data-only witness: best log posterior among candidates (no truth used)
    wname = max(cand, key=lambda kk: lp(cand[kk]))
    w = cand[wname]
    ellw, lrw = ell_of(G, c, yy, N, g, w), log_prior(w, h)
    rec['witness'] = dict(source=wname, ell=ellw, log_prior=lrw, equals_truth=bool(np.array_equal(w, gt)))
    # (1) standard power-posterior TI (Friel-Pettitt schedule) from prior draws, 2 replicates
    lams = (np.arange(41) / 40.0) ** 4
    Ls, ms, ses = [], [], []
    for r in range(2):
        L_, m_, se_ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, seed * 10 + r, mmax)
        Ls.append(L_); ms.append(m_); ses.append(se_)
    Lfp = np.mean(Ls, 0)
    if mmax < p:
        Lfp = Lfp + float(np.log1p(-binom.sf(mmax, p, h)))   # exact Lambda(0) of the truncated measure
    mfp = np.mean(ms, 0)
    # (2) melt test: chains started at the witness at each lam; lam_max = last grid
    #     point before the first disagreement with the prior-start means
    wm = []
    for j, lam in enumerate(lams):
        et, *_ = bvs_mh(G, c, yy, N, g, h, w.copy(), float(lam), 2 ** 16, 2 ** 14, 32,
                        int(rng.integers(2 ** 31)), mmax, np.zeros(p, np.int64), 0, p)
        wm.append(float(et.mean()))
    wm = np.array(wm)
    tol = 1.0 + 3 * np.maximum(ses[0], ses[1])
    bad = np.flatnonzero(np.abs(wm - mfp) > tol)
    jmax = (bad[0] - 1) if len(bad) else len(lams) - 1
    lam_max = float(lams[max(jmax, 0)])
    # (3) two-phase power-posterior TI: C = {gamma ⊇ w}, D = complement; both
    #     anchored exactly at lam = 0: rho(C) = h^{|w|}
    wmask = w.astype(np.int64)
    kw = int(w.sum())
    # exact anchors for the Bernoulli(h) measure restricted to {|gamma| <= mmax} (unnormalised):
    #   rho(X) = Pr[Bin(p,h) <= mmax],  rho(C) = h^kw Pr[Bin(p-kw,h) <= mmax-kw],  rho(D) = rho(X) - rho(C).
    # (Round 6; the reported runs used 0, kw log h, log(1-h^kw): the tails are < 1e-31 here.)
    lr0, lrC, lrD = capped_anchors(p, h, mmax, kw)
    FD_, FC_ = [], []
    for r in range(2 if kw > 0 else 0):
        # D: prior draw conditioned on D (rejection), upward
        while True:
            g0 = (rng.random(p) < h).astype(np.int64)
            if (g0 & wmask).sum() < kw:
                break
        LD, mD, _ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, seed * 10 + 5 + r, mmax, g0, wmask, 0, kw - 1)
        g0 = (rng.random(p) < h).astype(np.int64) | wmask
        LC, mC, _ = power_ti(G, c, yy, N, g, h, p, lams, 2 ** 16, seed * 10 + 7 + r, mmax, g0, wmask, kw, kw)
        FD_.append(lrD + LD); FC_.append(lrC + LC)
    if kw == 0:   # witness is the empty model: C = everything, single phase
        FD_, FC_ = [np.full(len(lams), -np.inf)], [Lfp.copy()]
    FD = np.mean(FD_, 0); FC = np.mean(FC_, 0)
    L2 = np.logaddexp(FD, FC)
    wfloor = lams * ellw + lrw
    rec['ti'] = dict(lams=lams.tolist(), Lambda_fp=Lfp.tolist(), Lambda_fp_reps=[x.tolist() for x in Ls],
                     mean_ell_fp=mfp.tolist(), mean_ell_witness_start=wm.tolist(), lam_max_melt=lam_max,
                     FD=FD.tolist(), FC=FC.tolist(), FD_reps=[x.tolist() for x in FD_],
                     FC_reps=[x.tolist() for x in FC_], Lambda_2ph=L2.tolist(), witness_floor=wfloor.tolist(),
                     w_signal_at_1=float(1 / (1 + np.exp(FD[-1] - FC[-1]))))
    rec['evidence'] = dict(fp_ti=float(Lfp[-1]), two_phase=float(L2[-1]), witness_lower_bound=ellw + lrw,
                           fp_certified_too_low=bool(Lfp[-1] < ellw + lrw))
    Lcap = np.where(lams <= lam_max, Lfp, np.inf)            # FP-TI, capped at melt-test lam_max
    Lfl = np.maximum(Lfp, wfloor)                            # FP-TI with witness floor (uncapped)
    # certificates per chain: crude (rigorous), tilted with (a) naive FP-TI,
    # (b) FP-TI + witness floor, (c) FP-TI capped at melt lam_max, (d) two-phase Lambda
    variants = dict(naive=Lfp, floor=Lfl, capped=Lcap, two_phase=np.maximum(L2, wfloor))
    for sname in chains:
        cnt = {v: 0 for v in variants}
        cnt['crude'] = 0
        for row in chains[sname]['rows']:
            for v, Lv in variants.items():
                cr, tl, lam = certify(row['max_ell'], ellw, lrw, lams, Lv)
                row['tilted_' + v] = tl
                cnt[v] += tl < LN01
            row['crude'] = cr
            cnt['crude'] += cr < LN01
        chains[sname]['n_cert'] = cnt
    rec['chains'] = chains
    # exact enumeration
    if p <= 22:
        Lall = enumerate_all(G, c, yy, N, g, p)
        sizes = np.array([bin(mm).count('1') for mm in range(1 << p)])
        lpri = sizes * np.log(h) + (p - sizes) * np.log(1 - h)
        lpost = Lall + lpri
        Z = logsumexp(lpost)
        pr = np.exp(lpost - Z)
        # exact Lambda on the grid
        Lex = np.array([logsumexp(lam * Lall + lpri) for lam in lams])
        rec['exact'] = dict(log_evidence=float(Z), Lambda=Lex.tolist(),
                            fp_ti_max_abs_err=float(np.max(np.abs(Lfp - Lex))),
                            fp_ti_err_below_lam_max=float(np.max(np.abs(Lfp - Lex)[lams <= lam_max])),
                            two_phase_max_abs_err=float(np.max(np.abs(L2 - Lex))),
                            w_signal_exact_at_1=float(np.exp(logsumexp(lpost[[mm for mm in range(1 << p) if all((mm >> j) & 1 for j in np.flatnonzero(w))]]) - Z)),
                            P_truth=float(np.exp(lp(gt) - Z)),
                            incl=[float(pr[((np.arange(1 << p) >> j) & 1) == 1].sum()) for j in range(p)])
        # exact posterior mass of {ell <= s} vs bounds; false alarms with IID draws
        cum = np.cumsum(pr)
        fa = {}
        for draws in (1, 64, 1024):
            hits_c = hits_t = 0
            reps = 2000
            for _ in range(reps):
                idx = np.minimum(np.searchsorted(cum, rng.random(draws) * cum[-1]), len(pr) - 1)
                M = float(Lall[idx].max())
                cr, tl, _ = certify(M, ellw, lrw, lams, Lex)
                hits_c += cr < LN01
                hits_t += tl < LN01
            fa[str(draws)] = dict(crude=hits_c / reps, tilted_exactLambda=hits_t / reps)
        rec['exact']['false_alarm_iid'] = fa
        # exact incl vs chains' inclusion
        for sname in chains:
            chains[sname]['incl_maxabs_err'] = float(np.max(np.abs(np.array(chains[sname]['incl']) -
                                                                   np.array(rec['exact']['incl']))))
    rec['seconds'] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, name + '.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(name, 'witness', wname, round(ellw, 1), 'evidence', rec['evidence'], 'lam_max', lam_max,
          {s: (chains[s]['n_cert'], round(chains[s]['rhat_ell'], 3)) for s in chains},
          rec.get('exact', {}).get('log_evidence'), f'{rec["seconds"]:.0f}s', flush=True)
    return rec


if __name__ == '__main__':
    which = sys.argv[1:] or ['exact20_strong', 'exact20_weak', 'p200']
    cfg = dict(exact20_strong=dict(N=200, p=20, blocks=1, k=6, eta=0.3, a=1.0, h=0.05, seed=0),
               exact20_weak=dict(N=200, p=20, blocks=1, k=6, eta=0.3, a=0.12, h=0.05, seed=0),
               p200=dict(N=300, p=200, blocks=3, k=6, eta=0.3, a=1.0, h=0.02, seed=1),
               decoy20=dict(N=300, p=20, blocks=1, k=6, eta=0.4, a=0.7, h=0.05, seed=0, design='decoy'),
               decoy200=dict(N=400, p=200, blocks=3, k=6, eta=0.4, a=0.7, h=0.02, seed=0, design='decoy'))
    import re
    for nm in which:
        m = re.match(r'(exact20_strong|exact20_weak|p200|decoy20|decoy200)_s(\d+)$', nm)
        if m:
            c = dict(cfg[m[1]])
            c['seed'] = int(m[2])
            run_instance(nm, **c)
        else:
            run_instance(nm, **cfg[nm])
