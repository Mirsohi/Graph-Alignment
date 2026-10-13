"""Validity diagnostic of the two-phase estimator: the released procedure (round 7).

Default (rules='transition'): the five frozen rules, evaluated at the estimated
transition beta_c only, with the thresholds fixed after O.16:
  edge        : < 1% of restricted draws on the region boundary (D and C);
  D two-start : D-restricted chains from overlap k0-1 and from a random start (2^22 each):
                |dS| < 3 SE + 2 and |dk| < 0.05 n;
  C two-start : C-restricted chains from w and from overlap k0+1 (2^23 each), same tolerance;
  window      : stratified-window profile (2^21 per window): |logit w_win - logit w_hat| < 1;
  middle basin: no basin of mass > 5% with peak in (0.1 n, k0 + 0.05 n).
Two seeds (default): the whole check is run with two independent seeds.
  status = 'pass'        if every rule passes in both;
           'multi-phase' only if the middle-basin rule fires in both seeds (a basin of mass
                          above 5% with peak in (0.1 n, k0 + 0.05 n));
           'unresolved'  otherwise (a failed rule, or the two repeats disagree).
A failed rule is a diagnostic failure, not proof of an extra phase.
Bounded retry (round 10; diagnose_bounded, the released operational policy): if the first
two-seed attempt is 'unresolved', the two-seed check is repeated ONCE with two fresh seeds
on the same estimator output (same beta_c, witness and estimate; the estimator is not
re-run and the budget is not increased).  The final status is that of the retry, reported
as 'pass (retry)' when it passes; there is no further retry, so an output that fails twice
stays 'unresolved'.  Passing is a screen, not an accuracy guarantee, and a retry raises the
chance of accepting a favourable fluctuation, which is why it is bounded and recorded.
There is no 'repeat until pass' mode.
rules='extended' additionally evaluates the checks at beta_c +- 0.1, +- 0.25 with the two
round-6 EXTENSIONS: region rules count only for regions carrying >= 1% of the estimated
mass, and the window rule also passes when both weights are < 0.01 or both > 0.99.
The extensions are a sensitivity analysis; mass gating cannot rule out a region whose
mass the estimator underestimates.
The edge occupancy is taken from the estimator's own restricted chains when they are
supplied (edge_D, edge_C); otherwise it is re-measured with dedicated restricted chains
(2^20 steps each, D from a random start, C from the witness)."""
import numpy as np
from scipy.special import logsumexp
from .windows import overlap_profile, basins, _start_with_overlap
from .phase import rmh


def _two_start(A, B, w, beta, lo, hi, p_a, p_b, T, rng):
    res = []
    for p0 in (p_a, p_b):
        r = rmh(A, B, p0, w, beta, T, T // 4, 64, lo, hi, int(rng.integers(2 ** 31)), False)
        st, kt = r[0], r[1]
        L = len(st) // 20
        res.append((float(st.mean()), float(st[:20 * L].reshape(20, L).mean(1).std(ddof=1) / np.sqrt(20)), float(kt.mean())))
    dS = abs(res[0][0] - res[1][0])
    se = float(np.hypot(res[0][1], res[1][1]))
    return bool(dS < 3 * se + 2 and abs(res[0][2] - res[1][2]) < 0.05 * len(w)), float(dS), se


def _edges(A, B, w, beta, k0, rng, T=2 ** 20):
    n = len(w)
    pr = rng.permutation(n)
    while np.sum(pr == w) >= k0:
        pr = rng.permutation(n)
    rD = rmh(A, B, pr, w, beta, T, T // 4, 16, 0, k0 - 1, int(rng.integers(2 ** 31)), False)
    rC = rmh(A, B, w.copy(), w, beta, T, T // 4, 16, k0, n, int(rng.integers(2 ** 31)), False)
    return float(np.mean(rD[1] == k0 - 1)), float(np.mean(rC[1] == k0))


def check_once(A, B, w, k0, beta, w_hat, seed, edge_D=None, edge_C=None, extended_window=False):
    n = len(w)
    rng = np.random.default_rng(seed)
    remeasured = edge_D is None or edge_C is None
    if remeasured:
        edge_D, edge_C = _edges(A, B, w, beta, k0, rng)
    pr = rng.permutation(n)
    while np.sum(pr == w) >= k0:
        pr = rng.permutation(n)
    okD, dSD, seD = _two_start(A, B, w, beta, 0, k0 - 1, _start_with_overlap(w, k0 - 1, int(rng.integers(2 ** 31))), pr, 2 ** 22, rng)
    okC, dSC, seC = _two_start(A, B, w, beta, k0, n, w.copy(), _start_with_overlap(w, k0 + 1, int(rng.integers(2 ** 31))), 2 ** 23, rng)
    lp = overlap_profile(A, B, w, beta, 2 ** 21, int(rng.integers(2 ** 31)))['logp']
    fin = np.isfinite(lp[k0:])
    ww = float(np.exp(logsumexp(lp[k0:][fin])))
    lg = lambda x: np.log(max(x, 1e-300) / max(1 - x, 1e-300))
    win_ok = abs(lg(ww) - lg(w_hat)) < 1
    if extended_window:
        win_ok = win_ok or (ww < 0.01 and w_hat < 0.01) or (ww > 0.99 and w_hat > 0.99)
    bas = [x for x in basins(lp, 2.0)[0] if x['mass'] >= 0.01]
    mid = any(x['mass'] > 0.05 and 0.1 * n < x['k_peak'] < k0 + 0.05 * n for x in bas)
    rules = dict(edge=bool(max(edge_D, edge_C) < 0.01), D_two_start=okD, C_two_start=okC, window=bool(win_ok),
                 middle_basin=not mid)
    cost = int(2 * 2 ** 22 + 2 * 2 ** 23 + (n // 2 + 2) * 2 ** 21 + (2 * 2 ** 20 if remeasured else 0))
    return dict(rules=rules, passed=bool(all(rules.values())), edge_D=edge_D, edge_C=edge_C, w_window=ww,
                dS_D=dSD, se_D=seD, dS_C=dSC, se_C=seC, basins=bas, proposals=cost)


def diagnose(A, B, w, k0, beta_c, w_hat, seeds=(1, 2), edges=None):
    """Released default: transition-only frozen rules, two independent seeds."""
    runs = [check_once(A, B, w, k0, beta_c, w_hat, s, *(edges if edges else (None, None))) for s in seeds]
    if all(r['passed'] for r in runs):
        status = 'pass'
    elif all(not r['rules']['middle_basin'] for r in runs):
        status = 'multi-phase'
    else:
        status = 'unresolved'
    return dict(status=status, runs=runs)


def diagnose_bounded(A, B, w, k0, beta_c, w_hat, seeds=(1, 2), retry_seeds=(3, 4), edges=None):
    """Released operational policy: one two-seed attempt, at most one two-seed retry with
    fresh seeds on the same estimator output if the first attempt is 'unresolved'."""
    first = diagnose(A, B, w, k0, beta_c, w_hat, seeds=seeds, edges=edges)
    if first['status'] != 'unresolved':
        return dict(status=first['status'], attempts=[first], retried=False)
    second = diagnose(A, B, w, k0, beta_c, w_hat, seeds=retry_seeds, edges=edges)
    status = 'pass (retry)' if second['status'] == 'pass' else second['status']
    return dict(status=status, attempts=[first, second], retried=True)
