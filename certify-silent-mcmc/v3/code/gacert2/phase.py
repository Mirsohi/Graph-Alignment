"""Two-phase free-energy estimation for the permutation Gibbs posterior
mu_beta(pi) ∝ exp(beta S(pi)) (v2).

Phases are defined by the overlap k(pi) = #{i : pi(i) = w(i)} with a witness
w:  C = {k >= k0} (condensed),  D = {k < k0} (diffuse).  Z = Z_C + Z_D exactly.
Each phase's log normaliser F_R(beta) = log sum_{pi in R} e^{beta S(pi)} is
estimated by thermodynamic integration (TI) of *region-restricted* MH chains
(moves leaving R are rejected, so each chain is reversible w.r.t. mu_beta|R):

  F_D(beta) = log|D| + int_0^beta E_{mu_t|D}[S] dt          (upward from 0)
  F_C(beta) = F_C(b_hi) - int_beta^{b_hi} E_{mu_t|C}[S] dt   (downward)
  F_C(b_hi) = log sum_{pi in W} e^{b_hi S(pi)} - log mu_{b_hi|C}(W),

with W = {pi : d_H(pi, w) <= 2} (w and its transposition neighbours), whose
score sum is exact and whose restricted mass is estimated by the fraction of
time the C-chain spends in W.  |D| and |C| are exact (rencontres numbers).
"""
import math
import numpy as np
import numba as nb
from scipy.special import gammaln, logsumexp

import os, sys
_V1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'code')
sys.path.insert(0, os.path.abspath(_V1))
from gacert.core import score, delta  # noqa: E402  (numba functions, reused)


# ------------------------------------------------------------ kernels ---
@nb.njit(cache=True)
def rmh(A, B, p0, w, beta, T, burn, thin, kmin, kmax, seed, want_marg):
    """Region-restricted random-transposition MH at fixed beta.
    Region = {pi: kmin <= overlap(pi, w) <= kmax}.  p0 must lie in the region.
    Returns (S_trace, k_trace, marg, p_final, smax, best, acc_rate,
             boundary_reject_rate, frac_in_W, meanS_all, meanS2_all)."""
    np.random.seed(seed)
    n = len(p0)
    p = p0.copy()
    s = score(A, B, p)
    k = 0
    for i in range(n):
        if p[i] == w[i]:
            k += 1
    cnt = (T - burn + thin - 1) // thin
    st = np.zeros(cnt)
    kt = np.zeros(cnt, dtype=np.int64)
    M = np.zeros((n, n)) if want_marg else np.zeros((1, 1))
    j = 0
    smax = s
    best = p.copy()
    acc = 0
    brej = 0
    inW = 0
    tot = 0.0
    tot2 = 0.0
    m = 0
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        pu = p[u]
        pv = p[v]
        k2 = k - (1 if pu == w[u] else 0) - (1 if pv == w[v] else 0) \
            + (1 if pv == w[u] else 0) + (1 if pu == w[v] else 0)
        if k2 < kmin or k2 > kmax:
            brej += 1
        else:
            d = delta(A, B, p, u, v)
            if d >= 0 or np.random.random() < math.exp(beta * d):
                p[u] = pv
                p[v] = pu
                s += d
                k = k2
                acc += 1
                if s > smax:
                    smax = s
                    best[:] = p
        if t >= burn:
            tot += s
            tot2 += s * s
            m += 1
            if k >= n - 2:
                inW += 1
            if (t - burn) % thin == 0 and j < cnt:
                st[j] = s
                kt[j] = k
                if want_marg:
                    for i in range(n):
                        M[i, p[i]] += 1
                j += 1
    if want_marg:
        M /= max(j, 1)
    return (st[:j], kt[:j], M, p, smax, best, acc / T, brej / T,
            inW / max(m, 1), tot / max(m, 1), tot2 / max(m, 1))


@nb.njit(cache=True)
def switch_count(A, B, p0, w, beta, T, klo, khi, seed):
    """Unrestricted MH; counts phase switches with hysteresis thresholds
    (enter C when k >= khi, enter D when k <= klo).  Returns
    (n_switches, frac_time_k_ge_mid, final p) where mid=(klo+khi)//2."""
    np.random.seed(seed)
    n = len(p0)
    p = p0.copy()
    k = 0
    for i in range(n):
        if p[i] == w[i]:
            k += 1
    state = 1 if k >= khi else 0
    sw = 0
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        pu = p[u]
        pv = p[v]
        d = delta(A, B, p, u, v)
        if d >= 0 or np.random.random() < math.exp(beta * d):
            k += - (1 if pu == w[u] else 0) - (1 if pv == w[v] else 0) \
                + (1 if pv == w[u] else 0) + (1 if pu == w[v] else 0)
            p[u] = pv
            p[v] = pu
            if state == 0 and k >= khi:
                state = 1
                sw += 1
            elif state == 1 and k <= klo:
                state = 0
                sw += 1
    return sw, p


@nb.njit(cache=True)
def transposition_deltas(A, B, w):
    n = len(w)
    out = np.zeros(n * (n - 1) // 2)
    c = 0
    for u in range(n):
        for v in range(u + 1, n):
            out[c] = delta(A, B, w, u, v)
            c += 1
    return out


# ------------------------------------------------------ exact counting ---
def log_derangements(m):
    """log D_m, D_m = number of derangements of m items (exact via recursion in
    log-space-safe floats; D_m / m! -> 1/e)."""
    if m == 0:
        return 0.0
    # D_m = m! * sum_{j=0}^m (-1)^j / j!
    j = np.arange(m + 1)
    terms = (-1.0) ** j * np.exp(-gammaln(j + 1))
    return float(gammaln(m + 1) + np.log(terms.sum()))


def log_count_overlap(n):
    """log #{pi in S_n : exactly k agreements with a fixed w}, k = 0..n."""
    out = np.full(n + 1, -np.inf)
    for k in range(n + 1):
        if n - k == 1:
            continue  # exactly n-1 agreements impossible
        out[k] = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + log_derangements(n - k)
    return out


def log_region_sizes(n, k0):
    lc = log_count_overlap(n)
    return float(logsumexp(lc[:k0])), float(logsumexp(lc[k0:]))  # (log|D|, log|C|)


# ------------------------------------------------------ TI along a path ---
def _batch_se(x, nb_=20):
    x = np.asarray(x, float)
    if len(x) < 2 * nb_:
        return float(np.std(x) / max(1, np.sqrt(len(x))))
    L = len(x) // nb_
    bm = x[:L * nb_].reshape(nb_, L).mean(1)
    return float(bm.std(ddof=1) / np.sqrt(nb_))


def ti_path(A, B, w, betas, p_start, kmin, kmax, T, seed, burn_frac=0.25,
            retain=4096, marg_at=(), boost=(1.0, 2.5, 4)):
    """Run restricted chains along the ordered grid `betas`, warm-starting
    each from the previous final state.  Returns per-grid dict arrays."""
    n = len(w)
    p = p_start.copy()
    rng = np.random.default_rng(seed)
    thin = max(1, (T - int(T * burn_frac)) // retain)
    out = dict(beta=[], meanS=[], se=[], varS=[], k_mean=[], acc=[], brej=[], fracW=[],
               smax=[], k_hist=[], marg={})
    best_s, best_p = -1, None
    marg_at = set(round(float(b), 6) for b in marg_at)
    for b in betas:
        want = round(float(b), 6) in marg_at
        Tb = T * boost[2] if boost[0] <= b <= boost[1] else T
        thin = max(1, (Tb - int(Tb * burn_frac)) // retain)
        r = rmh(A, B, p, w, float(b), Tb, int(Tb * burn_frac), thin, kmin, kmax,
                int(rng.integers(2 ** 31)), want)
        st, kt, M, p, smax, bst, acc, brej, fW, m1, m2 = r
        out['T'] = out.get('T', 0) + Tb
        out['beta'].append(float(b))
        out['meanS'].append(float(m1))
        out['se'].append(_batch_se(st))
        out['varS'].append(float(m2 - m1 * m1))
        out['k_mean'].append(float(kt.mean()))
        out['acc'].append(float(acc))
        out['brej'].append(float(brej))
        out['fracW'].append(float(fW))
        out['smax'].append(int(smax))
        out['k_hist'].append(np.bincount(kt, minlength=n + 1).tolist())
        if want:
            out['marg'][round(float(b), 6)] = M
        if smax > best_s:
            best_s, best_p = int(smax), bst.copy()
    for key in ('beta', 'meanS', 'se', 'varS', 'k_mean', 'acc', 'brej', 'fracW', 'smax'):
        out[key] = np.array(out[key])
    out['best_s'] = best_s
    out['best_p'] = best_p
    return out


def cumtrapz0(x, y):
    return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))


def log_W_sum(A, B, w, beta):
    """log sum_{pi : d_H(pi,w) <= 2} e^{beta (S(pi) - S(w))}  (exact)."""
    d = transposition_deltas(A, B, w)
    return float(logsumexp(np.concatenate(([0.0], beta * d))))


class TwoPhase:
    """Holds F_D and F_C on grids and derived quantities."""

    def __init__(self, n, Sw, logD, logC, bD, FD, bC, FC, seD=None, seC=None):
        self.n, self.Sw = n, Sw
        self.logD, self.logC = logD, logC
        self.bD, self.FD = np.asarray(bD), np.asarray(FD)
        self.bC, self.FC = np.asarray(bC), np.asarray(FC)
        self.seD = seD
        self.seC = seC

    def fD(self, b):
        return np.interp(b, self.bD, self.FD)

    def fC(self, b):
        return np.interp(b, self.bC, self.FC)

    def wcond(self, b):
        return 1.0 / (1.0 + np.exp(-(self.fC(b) - self.fD(b))))

    def beta_c(self):
        lo, hi = max(self.bD[0], self.bC[0]), min(self.bD[-1], self.bC[-1])
        g = np.linspace(lo, hi, 20001)
        d = self.fC(g) - self.fD(g)
        idx = np.where(np.diff(np.sign(d)) != 0)[0]
        if len(idx) == 0:
            return float('nan')
        i = idx[0]
        return float(g[i] - d[i] * (g[i + 1] - g[i]) / (d[i + 1] - d[i]))


def estimate_two_phase(A, B, w, k0, grid_D, grid_C, T, seed, marg_at=(), p_diff0=None, T_ref=2 ** 24):
    """Full estimator.  grid_D increasing from 0; grid_C decreasing from b_hi.
    Returns TwoPhase object plus raw path outputs."""
    n = len(w)
    rng = np.random.default_rng(seed)
    logD, logC = log_region_sizes(n, k0)
    # diffuse path: random start inside D
    p = rng.permutation(n)
    while np.sum(p == w) >= k0:
        p = rng.permutation(n)
    pd = ti_path(A, B, w, grid_D, p, 0, k0 - 1, T, int(rng.integers(2 ** 31)), marg_at=marg_at)
    FD = logD + cumtrapz0(pd['beta'], pd['meanS'])
    # condensed path: witness start, downward
    pc = ti_path(A, B, w, grid_C, w.copy(), k0, n, T, int(rng.integers(2 ** 31)), marg_at=marg_at)
    bhi = float(grid_C[0])
    Sw = int(score(A, B, w))
    # dedicated long reference chain at b_hi (restricted to C, started at w)
    rr = rmh(A, B, w.copy(), w, bhi, T_ref, T_ref // 8, max(1, T_ref // 8192), k0, n,
             int(rng.integers(2 ** 31)), False)
    fW = rr[8]
    inW = (rr[1] >= n - 2).astype(float)
    L = len(inW) // 32
    bm = inW[:32 * L].reshape(32, L).mean(1)
    se_fW = float(bm.std(ddof=1) / np.sqrt(32))
    ref = bhi * Sw + log_W_sum(A, B, w, bhi) - np.log(max(fW, 1e-300))
    b = pc['beta'][::-1]
    m = pc['meanS'][::-1]
    I = cumtrapz0(b, m)            # int_{b[0]}^{b} E_C S
    FC = ref - (I[-1] - I)         # F_C(b) = F_C(bhi) - int_b^{bhi}
    tp = TwoPhase(n, Sw, logD, logC, pd['beta'], FD, b, FC)
    tp.ref = ref
    tp.proposals = int(pd['T'] + pc['T'] + T_ref)
    tp.fracW_hi = float(fW)
    tp.se_log_fracW = se_fW / max(fW, 1e-300)
    tp.best_ref = int(rr[4])
    # best states seen anywhere (for witness refinement)
    cands = [(int(rr[4]), rr[5]), (int(pc['best_s']), pc['best_p']), (int(pd['best_s']), pd['best_p'])]
    tp.best_seen = max(cands, key=lambda t: t[0])
    return tp, pd, pc


@nb.njit(cache=True)
def pt_k(A, B, P0, ladder, w, sweeps, burn, thin, seed):
    """Parallel tempering (as v1 pt_chain) that also records the witness
    overlap of the cold (last) replica.  Returns (marg, k_trace, S_trace,
    min adjacent swap rate)."""
    np.random.seed(seed)
    R = len(ladder)
    n = P0.shape[1]
    P = P0.copy()
    S = np.zeros(R, dtype=np.int64)
    for r in range(R):
        S[r] = score(A, B, P[r])
    cnt = (sweeps - burn + thin - 1) // thin
    M = np.zeros((n, n))
    kt = np.zeros(cnt, dtype=np.int64)
    st = np.zeros(cnt)
    j = 0
    sacc = np.zeros(R - 1)
    stry = np.zeros(R - 1)
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
        if t >= burn and (t - burn) % thin == 0 and j < cnt:
            kk = 0
            for i in range(n):
                M[i, P[R - 1, i]] += 1
                if P[R - 1, i] == w[i]:
                    kk += 1
            kt[j] = kk
            st[j] = S[R - 1]
            j += 1
    return M / max(j, 1), kt[:j], st[:j], (sacc / np.maximum(stry, 1)).min()


def nested_reference(A, B, w, k0, bhi, T, seed, target=0.1):
    """log mu_{bhi|C}(W), W = {k >= n-2}, by a product of conditional fractions
    over overlap levels k0 = a_0 < a_1 < ... < a_m = n-2:
    log mu(W) = sum_j log mu(k >= a_{j+1} | k >= a_j).  At each stage a PILOT
    chain restricted to {k >= a_j} chooses a_{j+1} (largest level with
    estimated conditional mass >= target); an independent FRESH chain then
    estimates the fraction, so level selection does not bias the estimate.
    Returns (log mu(W), levels, fractions, SE of the log, best score seen,
    best state, floor_bound) where floor_bound says whether the 1/len floor
    was ever used (then the estimate is only an upper bound)."""
    n = len(w)
    rng = np.random.default_rng(seed)
    a = k0
    logmu, var = 0.0, 0.0
    levels, fracs = [a], []
    best_s, best_p = -1, None
    floor_bound = False
    while a < n - 2:
        pil = rmh(A, B, w.copy(), w, bhi, T // 4, T // 32, max(1, T // 32768), a, n, int(rng.integers(2 ** 31)), False)
        kt = pil[1]
        cand = [b for b in range(a + 1, n - 1) if np.mean(kt >= b) >= target]
        b = n - 2 if np.mean(kt >= n - 2) >= target else (max(cand) if cand else a + 1)
        r = rmh(A, B, w.copy(), w, bhi, T, T // 8, max(1, T // 8192), a, n, int(rng.integers(2 ** 31)), False)
        for rr in (pil, r):
            if rr[4] > best_s:
                best_s, best_p = int(rr[4]), rr[5].copy()
        ind = (r[1] >= b).astype(float)
        f = ind.mean()
        if f * len(ind) < 1:
            f = 1.0 / len(ind)
            floor_bound = True
        L = len(ind) // 32
        bm = ind[:32 * L].reshape(32, L).mean(1)
        var += (bm.std(ddof=1) / np.sqrt(32) / f) ** 2
        logmu += np.log(f)
        fracs.append(float(f))
        a = b
        levels.append(int(a))
    return float(logmu), levels, fracs, float(np.sqrt(var)), best_s, best_p, floor_bound
