"""Core routines: Gibbs posterior over permutations, samplers, certificates.

Target:  mu_beta(pi) ∝ exp(beta * S(pi)),  S(pi) = sum_{i<j} A_ij B_{pi(i) pi(j)},
uniform prior on the symmetric group S_n.  All samplers are numba-compiled.
"""
import math
import numpy as np
import numba as nb
from scipy.special import gammaln
from scipy.stats import norm, rankdata, binom


# ----------------------------------------------------------------- score ---
@nb.njit(cache=True)
def score(A, B, p):
    n = len(p)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += A[i, j] * B[p[i], p[j]]
    return s


@nb.njit(cache=True)
def delta(A, B, p, u, v):
    """Score change when the images of u and v are swapped."""
    n = len(p)
    d = 0
    pu = p[u]
    pv = p[v]
    for k in range(n):
        if k == u or k == v:
            continue
        d += (A[u, k] - A[v, k]) * (B[pv, p[k]] - B[pu, p[k]])
    return d


# ------------------------------------------------------- local MH sampler ---
@nb.njit(cache=True)
def mh_chain(A, B, p0, betas, burn, thin, seed):
    """Random-transposition Metropolis.  betas[t] is the inverse temperature at
    step t (constant array = ordinary MH; increasing array = annealing).
    Returns marginal counts over retained draws, retained scores, final state,
    maximum score visited and acceptance rate."""
    np.random.seed(seed)
    p = p0.copy()
    n = len(p)
    s = score(A, B, p)
    T = len(betas)
    cnt = (T - burn + thin - 1) // thin
    M = np.zeros((n, n))
    sc = np.zeros(cnt)
    k = 0
    smax = s
    best = p.copy()
    acc = 0
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        d = delta(A, B, p, u, v)
        if d >= 0 or np.random.random() < math.exp(betas[t] * d):
            p[u], p[v] = p[v], p[u]
            s += d
            acc += 1
            if s > smax:
                smax = s
                best[:] = p
        if t >= burn and (t - burn) % thin == 0 and k < cnt:
            for i in range(n):
                M[i, p[i]] += 1
            sc[k] = s
            k += 1
    return M[:, :] / max(k, 1), sc[:k], p, smax, best, acc / T


@nb.njit(cache=True)
def mean_score(A, B, p0, beta, T, burn, seed):
    np.random.seed(seed)
    p = p0.copy()
    n = len(p)
    s = score(A, B, p)
    tot = 0.0
    c = 0
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        d = delta(A, B, p, u, v)
        if d >= 0 or np.random.random() < math.exp(beta * d):
            p[u], p[v] = p[v], p[u]
            s += d
        if t >= burn:
            tot += s
            c += 1
    return tot / c


# ---------------------------------------------------- parallel tempering ---
@nb.njit(cache=True)
def pt_chain(A, B, P0, ladder, sweeps, burn, thin, seed):
    """Parallel tempering.  ladder is increasing; the last replica is the cold
    (target) chain.  One sweep = one transposition proposal per replica
    followed by one round of adjacent swap proposals (alternating parity).
    Total local proposals = sweeps * len(ladder)."""
    np.random.seed(seed)
    R = len(ladder)
    n = P0.shape[1]
    P = P0.copy()
    S = np.zeros(R, dtype=np.int64)
    for r in range(R):
        S[r] = score(A, B, P[r])
    cnt = (sweeps - burn + thin - 1) // thin
    M = np.zeros((n, n))
    sc = np.zeros(cnt)
    k = 0
    smax = S.max()
    best = P[np.argmax(S)].copy()
    swap_acc = np.zeros(R - 1)
    swap_try = np.zeros(R - 1)
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
                if S[r] > smax:
                    smax = S[r]
                    best[:] = P[r]
        start = t % 2
        for r in range(start, R - 1, 2):
            swap_try[r] += 1
            la = (ladder[r + 1] - ladder[r]) * (S[r] - S[r + 1])
            if la >= 0 or np.random.random() < math.exp(la):
                tmp = P[r].copy()
                P[r] = P[r + 1]
                P[r + 1] = tmp
                ts = S[r]
                S[r] = S[r + 1]
                S[r + 1] = ts
                swap_acc[r] += 1
        if t >= burn and (t - burn) % thin == 0 and k < cnt:
            for i in range(n):
                M[i, P[R - 1, i]] += 1
            sc[k] = S[R - 1]
            k += 1
    return M / max(k, 1), sc[:k], smax, best, swap_acc / np.maximum(swap_try, 1)


# ----------------------------------------------------------- diagnostics ---
def row_tv(P, Q):
    return float(np.abs(P - Q).sum() / (2 * P.shape[0]))


def d_marg(Ms):
    K = len(Ms)
    return float(np.mean([row_tv(Ms[a], Ms[b]) for a in range(K) for b in range(a + 1, K)]))


def _zscale(x):
    return norm.ppf((rankdata(x.ravel()) - .375) / (x.size + .25)).reshape(x.shape)


def _basic_rhat(x):
    m, n = x.shape
    W = x.var(axis=1, ddof=1).mean()
    Bv = n * x.mean(axis=1).var(ddof=1)
    if W <= 1e-300:
        return float('nan') if Bv <= 1e-300 else float('inf')
    return float(np.sqrt(((n - 1) * W / n + Bv / n) / W))


def split_rhat(chains):
    """Rank-normalised, folded split-R-hat (Vehtari et al. 2021) of a scalar."""
    x = np.asarray(chains, float)
    h = x.shape[1] // 2
    s = np.concatenate((x[:, :h], x[:, -h:]), axis=0)
    a = _basic_rhat(_zscale(s))
    b = _basic_rhat(_zscale(np.abs(s - np.median(s))))
    vals = [v for v in (a, b) if not np.isnan(v)]
    return max(vals) if vals else float('nan')


def iid_floor_uniform(n, N):
    """Exact E[rowTV] between two independent N-draw empirical marginal
    matrices when the target is uniform on S_n."""
    k = np.arange(N)
    F = binom.cdf(k, N, 1.0 / n)
    return float(n / N * np.sum(F * (1 - F)))


def iid_floor_plugin(P, N):
    """E[rowTV] between two independent N-draw estimates if draws were IID
    from a law with entry marginals P (entries only; exact per entry)."""
    tot = 0.0
    k = np.arange(N + 1)
    for p in P[P > 1e-12].ravel():
        if p >= 1 - 1e-12:
            continue
        F = binom.cdf(k[:-1], N, p)
        tot += 2.0 / N * np.sum(F * (1 - F))
    return tot / (2 * P.shape[0])


# ---------------------------------------------------------- certificates ---
def log_nfact(n):
    return float(gammaln(n + 1))


def log_mgf_ti(A, B, lams, T=2 ** 19, seeds=(11, 12), rng_seed=1):
    """Thermodynamic-integration estimate of Lambda(lam) = log E_U exp(lam S)
    = int_0^lam E_{mu_t}[S] dt on a grid lams (must start at 0).  Uses
    random-start local MH at each t; valid when every t in lams is in the
    diffuse phase (checked separately).  Returns Lambda on the grid, the mean
    scores and their between-seed spread."""
    n = A.shape[0]
    rng = np.random.default_rng(rng_seed)
    ms = np.zeros((len(seeds), len(lams)))
    for a, sd in enumerate(seeds):
        for j, t in enumerate(lams):
            ms[a, j] = mean_score(A, B, rng.permutation(n), float(t), T, T // 4, sd + 1000 * j)
    m = ms.mean(0)
    L = np.concatenate(([0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(lams))))
    return L, m, ms.std(0)


def crude_log_bound(n, beta, s, sw):
    """log of min(1, n! exp(beta (s - sw)))  -- witness bound."""
    return min(0.0, log_nfact(n) + beta * (s - sw))


def tilted_log_bound(n, beta, s, sw, lams, L):
    """log of min(1, min_{lam<=beta} n! exp((beta-lam) s + Lambda(lam) - beta sw))."""
    lams = np.asarray(lams)
    L = np.asarray(L)
    ok = lams <= beta + 1e-12
    val = log_nfact(n) + (beta - lams[ok]) * s + L[ok] - beta * sw
    j = int(np.argmin(val))
    return min(0.0, float(val[j])), float(lams[ok][j])
