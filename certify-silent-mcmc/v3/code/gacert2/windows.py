"""Stratified overlap windows (fixed beta): chains restricted to
k in {a, a+1, a+2} (k = overlap with witness w), a = 0, 2, 4, ..., give the
ratios p_{k+1}/p_k, which are chained into the full profile log p_k.
Used as (i) the phase-validity diagnostic of the two-phase estimator and
(ii) the K-phase fallback: phase weights are sums of p_k over basins.
Only needs mixing inside each 3-level window."""
import math
import numpy as np
import numba as nb
from scipy.special import logsumexp
from gacert.core import score, delta


@nb.njit(cache=True)
def _start_with_overlap(w, target, seed):
    """Permutation with exactly `target` agreements with w (target != n-1)."""
    np.random.seed(seed)
    n = len(w)
    p = w.copy()
    m = n - target
    if m == 0:
        return p
    idx = np.random.permutation(n)[:m]
    vals = p[idx].copy()
    for t in range(m):
        p[idx[t]] = vals[(t + 1) % m]      # cyclic shift = derangement of the chosen set
    return p


@nb.njit(cache=True)
def window_chain(A, B, w, beta, a, T, seed, want_marg):
    """Restricted MH on {a <= k <= a+2}.  Returns visit counts per level (3),
    per-level score sums, and (optionally) per-level marginal counts (3,n,n)
    accumulated every 64 steps."""
    n = len(w)
    tgt = a + 1 if a + 1 != n - 1 else a
    if tgt > n:
        tgt = n
    p = _start_with_overlap(w, tgt, seed)
    np.random.seed(seed + 1)
    s = score(A, B, p)
    k = 0
    for i in range(n):
        if p[i] == w[i]:
            k += 1
    H = np.zeros(3)
    SS = np.zeros(3)
    M = np.zeros((3, n, n)) if want_marg else np.zeros((1, 1, 1))
    burn = T // 8
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        pu = p[u]
        pv = p[v]
        k2 = k - (1 if pu == w[u] else 0) - (1 if pv == w[v] else 0) \
            + (1 if pv == w[u] else 0) + (1 if pu == w[v] else 0)
        if k2 >= a and k2 <= a + 2:
            d = delta(A, B, p, u, v)
            if d >= 0 or np.random.random() < math.exp(beta * d):
                p[u] = pv
                p[v] = pu
                s += d
                k = k2
        if t >= burn:
            H[k - a] += 1
            SS[k - a] += s
            if want_marg and (t & 63) == 0:
                for i in range(n):
                    M[k - a, i, p[i]] += 1
    return H, SS, M


def overlap_profile(A, B, w, beta, T, seed, want_marg=False):
    """Returns log p_k (normalised, k=0..n; p_{n-1} = 0), per-level mean S,
    and, if want_marg, the posterior marginal matrix (mixture over levels)."""
    n = len(w)
    lr = np.full(n + 1, np.nan)          # lr[k] = log p_k - log p_{k-1}
    meanS = np.full(n + 1, np.nan)
    Mk = {} if want_marg else None
    rng = np.random.default_rng(seed)
    a = 0
    while a <= n - 2:
        H, SS, M = window_chain(A, B, w, float(beta), a, T, int(rng.integers(2 ** 31)), want_marg)
        for j in range(3):
            k = a + j
            if k <= n and H[j] > 0:
                meanS[k] = SS[j] / H[j]
                if want_marg and M[j].sum() > 0:
                    Mk[k] = M[j] / M[j].sum(1, keepdims=True).clip(1e-300)
        H = np.maximum(H, 0.5)           # floor only matters for unvisited levels (reported)
        if a + 1 == n - 1:               # level n-1 impossible: ratio p_n / p_{n-2}
            lr[n] = np.log(H[2] / H[0])
        else:
            lr[a + 1] = np.log(H[1] / H[0])
            if a + 2 <= n:
                if a + 2 == n - 1:
                    pass                 # handled by the next window (a = n-2)
                else:
                    lr[a + 2] = np.log(H[2] / H[1])
        a += 2
    if np.isnan(lr[n]):
        H, SS, M = window_chain(A, B, w, float(beta), n - 2, T, int(rng.integers(2 ** 31)), False)
        H = np.maximum(H, 0.5)
        lr[n] = np.log(H[2] / H[0])
    lp = np.full(n + 1, -np.inf)
    lp[0] = 0.0
    for k in range(1, n + 1):
        if k == n - 1:
            continue
        prev = lp[k - 1] if k - 1 != n - 1 else lp[k - 2]
        if np.isnan(lr[k]):
            # fall back: missing ratio (should not happen); treat as equal
            lr[k] = 0.0
        lp[k] = prev + lr[k]
    lp -= logsumexp(lp[np.isfinite(lp)])
    out = dict(logp=lp, meanS=meanS)
    if want_marg:
        P = np.zeros((n, n))
        for k, Mm in Mk.items():
            P += np.exp(lp[k]) * Mm
        out['marg'] = P / max(sum(np.exp(lp[k]) for k in Mk), 1e-300)
    return out


def basins(lp, min_depth=1.0):
    """Local maxima of the log profile separated by dips >= min_depth nats;
    returns list of (argmax, mass) and the dividing minima."""
    x = np.where(np.isfinite(lp), lp, np.nan)
    ks = np.flatnonzero(np.isfinite(x))
    xs = x[ks]
    # smooth-free peak detection with depth criterion
    peaks = []
    mins = []
    cur_max_i = 0
    cur_min_i = None
    i = 0
    state_up = True
    best = xs[0]
    best_i = 0
    low = None
    low_i = None
    for i in range(1, len(xs)):
        if state_up:
            if xs[i] > best:
                best, best_i = xs[i], i
            elif best - xs[i] >= min_depth:
                peaks.append(best_i)
                state_up = False
                low, low_i = xs[i], i
        else:
            if xs[i] < low:
                low, low_i = xs[i], i
            elif xs[i] - low >= min_depth:
                mins.append(low_i)
                state_up = True
                best, best_i = xs[i], i
    if state_up:
        peaks.append(best_i)
    bounds = [0] + [ks[m] for m in mins] + [len(lp)]
    p = np.exp(np.where(np.isfinite(lp), lp, -np.inf))
    res = []
    for j, pk in enumerate(peaks):
        res.append(dict(k_peak=int(ks[pk]), mass=float(p[bounds[j]:bounds[j + 1]].sum()),
                        lo=int(bounds[j]), hi=int(bounds[j + 1] - 1)))
    return res, [int(ks[m]) for m in mins]


@nb.njit(cache=True)
def window_chain_w(A, B, w, beta, a, W, T, seed):
    """Restricted MH on {a <= k <= a+W-1}; returns visit counts per level."""
    n = len(w)
    tgt = a + W // 2
    if tgt == n - 1:
        tgt = n - 2
    if tgt > n:
        tgt = n
    p = _start_with_overlap(w, tgt, seed)
    np.random.seed(seed + 1)
    k = 0
    for i in range(n):
        if p[i] == w[i]:
            k += 1
    H = np.zeros(W)
    burn = T // 8
    hi = a + W - 1
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        pu = p[u]
        pv = p[v]
        k2 = k - (1 if pu == w[u] else 0) - (1 if pv == w[v] else 0) \
            + (1 if pv == w[u] else 0) + (1 if pu == w[v] else 0)
        if k2 >= a and k2 <= hi:
            d = delta(A, B, p, u, v)
            if d >= 0 or np.random.random() < math.exp(beta * d):
                p[u] = pv
                p[v] = pu
                k = k2
        if t >= burn:
            H[k - a] += 1
    return H


def overlap_profile_wide(A, B, w, beta, T, seed, W=8, stride=6):
    """Wide overlapping windows; consecutive windows are matched on their common
    levels (count-weighted mean of log-ratio differences).  Returns log p_k."""
    n = len(w)
    rng = np.random.default_rng(seed)
    lp = np.full(n + 1, np.nan)
    starts = list(range(0, n - W + 2, stride))
    if starts[-1] + W - 1 < n:
        starts.append(n - W + 1)
    for a in starts:
        H = window_chain_w(A, B, w, float(beta), a, W, T, int(rng.integers(2 ** 31)))
        ks = np.arange(a, a + W)
        with np.errstate(divide='ignore'):
            loc = np.where(H > 0, np.log(H), np.nan)
        loc[ks == n - 1] = np.nan
        ok = ~np.isnan(loc) & (ks <= n)
        if np.all(np.isnan(lp)):
            lp[ks[ok]] = loc[ok]
            continue
        common = ok & ~np.isnan(lp[np.minimum(ks, n)])
        if common.sum() == 0:
            raise RuntimeError('no overlap between windows')
        wts = H[common]
        shift = np.sum(wts * (lp[ks[common]] - loc[common])) / wts.sum()
        new = ok & np.isnan(lp[np.minimum(ks, n)])
        lp[ks[new]] = loc[new] + shift
    lp[n - 1] = -np.inf
    fin = np.isfinite(lp)
    lp[~fin & ~np.isneginf(lp)] = -np.inf
    lp -= logsumexp(lp[np.isfinite(lp)])
    return lp
