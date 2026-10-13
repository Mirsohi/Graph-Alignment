"""Strong competitors for posterior estimation across the transition.
 * wl_score / wl_prod: Wang-Landau (flat histogram) in the score S -- witness-free
   multicanonical sampling; production run with fixed log density of states,
   reweighted to any beta.  The overlap k with the witness is recorded ONLY for
   evaluation (phase weights), never used by the sampler.
 * pt_all: parallel tempering recording the phase indicator and marginals at
   every rung (whole beta-curve from one run).
"""
import math
import numpy as np
import numba as nb
from scipy.special import logsumexp
from gacert.core import score, delta


@nb.njit(cache=True)
def wl_score(A, B, p0, slo, shi, stage_T, f0, fmin, flat, max_stages, seed):
    """Wang-Landau on S in [slo, shi] (moves leaving the range rejected).
    Returns log g (length shi-slo+1), final state, final f, stages used."""
    np.random.seed(seed)
    n = len(p0)
    p = p0.copy()
    s = score(A, B, p)
    nb_ = shi - slo + 1
    g = np.zeros(nb_)
    Hcum = np.zeros(nb_)
    f = f0
    stages = 0
    # walk into range first (greedy/random) if needed
    while (s < slo or s > shi):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        d = delta(A, B, p, u, v)
        if (s < slo and d >= 0) or (s > shi and d <= 0):
            a = p[u]; p[u] = p[v]; p[v] = a
            s += d
    while f > fmin and stages < max_stages:
        H = np.zeros(nb_)
        for t in range(stage_T):
            u = np.random.randint(n)
            v = np.random.randint(n - 1)
            if v >= u:
                v += 1
            d = delta(A, B, p, u, v)
            s2 = s + d
            if s2 >= slo and s2 <= shi:
                la = g[s - slo] - g[s2 - slo]
                if la >= 0 or np.random.random() < math.exp(la):
                    a = p[u]; p[u] = p[v]; p[v] = a
                    s = s2
            g[s - slo] += f
            H[s - slo] += 1
            Hcum[s - slo] += 1
        stages += 1
        vis = H > 0
        if vis.sum() > 0:
            hm = H[vis].mean()
            if H[vis].min() > flat * hm:
                f *= 0.5
    g -= g.max()
    # never-visited bins are excluded from production (otherwise they act as sinks)
    for i in range(nb_):
        if Hcum[i] == 0:
            g[i] = 1e9
    return g, p, f, stages


@nb.njit(cache=True)
def wl_prod(A, B, p0, w, slo, shi, g, T, burn, thin, seed, want_marg, bin_w):
    """Fixed-weight multicanonical production (pi ∝ exp(-g(S))).  Records per
    S-bin counts H, joint counts with the witness overlap k (evaluation only),
    and per coarse S-bin marginal counts (bin width bin_w)."""
    np.random.seed(seed)
    n = len(p0)
    p = p0.copy()
    s = score(A, B, p)
    nb_ = shi - slo + 1
    H = np.zeros(nb_)
    Hk = np.zeros((nb_, n + 1))
    ncb = nb_ // bin_w + 1
    M = np.zeros((ncb, n, n)) if want_marg else np.zeros((1, 1, 1))
    Hm = np.zeros((nb_,))
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        d = delta(A, B, p, u, v)
        s2 = s + d
        if s2 >= slo and s2 <= shi:
            la = g[s - slo] - g[s2 - slo]
            if la >= 0 or np.random.random() < math.exp(la):
                a = p[u]; p[u] = p[v]; p[v] = a
                s = s2
        if t >= burn and (t - burn) % thin == 0:
            k = 0
            for i in range(n):
                if p[i] == w[i]:
                    k += 1
            H[s - slo] += 1
            Hk[s - slo, k] += 1
            if want_marg:
                cb = (s - slo) // bin_w
                Hm[s - slo] += 1
                for i in range(n):
                    M[cb, i, p[i]] += 1
    return H, Hk, M, Hm


def wl_reweight(H, Hk, M, Hm, g, slo, beta, k0, bin_w):
    """Phase weight mu_beta(k >= k0) and marginals from a multicanonical run."""
    S = np.arange(len(H)) + slo
    ok = H > 0
    lw = np.full(len(H), -np.inf)
    lw[ok] = np.log(H[ok]) + beta * S[ok] + g[ok]
    lw -= logsumexp(lw[ok])
    ps = np.exp(lw)
    wc = float(sum(ps[i] * Hk[i, k0:].sum() / H[i] for i in np.flatnonzero(ok)))
    out = dict(wC=wc)
    if M.shape[0] > 1:
        n = M.shape[1]
        P = np.zeros((n, n))
        cbs = np.arange(len(H)) // bin_w
        for cb in np.unique(cbs[ok]):
            sel = (cbs == cb) & ok
            mass = ps[sel].sum()
            cnt = M[cb].sum(1, keepdims=True)
            if cnt.max() > 0 and mass > 0:
                # within a coarse bin, draws are reweighted only across bins (approximation
                # exact when bin_w = 1)
                P += mass * M[cb] / np.maximum(cnt, 1e-300)
        out['marg'] = P / P.sum(1, keepdims=True)
    # effective sample size of the reweighted draws (per retained draw weights)
    wts = np.zeros(len(H))
    wts[ok] = np.exp(lw[ok] - np.log(H[ok]))
    ess = (H * wts).sum() ** 2 / max((H * wts ** 2).sum(), 1e-300)
    out['ess'] = float(ess)
    return out


@nb.njit(cache=True)
def pt_all(A, B, P0, ladder, w, k0, sweeps, burn, thin, seed, marg_rungs):
    """PT; records for every rung the fraction of retained draws with k >= k0
    and, for rungs listed in marg_rungs (indices), the marginal counts.
    Returns (fracC per rung, marginals (len(marg_rungs), n, n), swap rates)."""
    np.random.seed(seed)
    R = len(ladder)
    n = P0.shape[1]
    P = P0.copy()
    S = np.zeros(R, dtype=np.int64)
    for r in range(R):
        S[r] = score(A, B, P[r])
    fc = np.zeros(R)
    cnt = 0
    M = np.zeros((len(marg_rungs), n, n))
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
        if t >= burn and (t - burn) % thin == 0:
            cnt += 1
            for r in range(R):
                kk = 0
                for i in range(n):
                    if P[r, i] == w[i]:
                        kk += 1
                if kk >= k0:
                    fc[r] += 1
            for m in range(len(marg_rungs)):
                r = marg_rungs[m]
                for i in range(n):
                    M[m, i, P[r, i]] += 1
    return fc / max(cnt, 1), M / max(cnt, 1), sacc / np.maximum(stry, 1)
