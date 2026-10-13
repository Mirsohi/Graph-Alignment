"""Bayesian variable selection with Zellner's g-prior (I2 demonstration).

Model: y = 1 alpha + X_gamma b_gamma + eps, g-prior on b_gamma, flat on alpha,
Jeffreys on sigma^2 (Liang et al. 2008):
    log L(gamma) = ((N-1-|gamma|)/2) log(1+g) - ((N-1)/2) log(1 + g (1 - R^2_gamma)),
prior rho(gamma) = prod_j h^{gamma_j} (1-h)^{1-gamma_j}.
Posterior mu ∝ rho L.  Power posteriors mu_lam ∝ rho L^lam.
Sampler: add/delete/swap Metropolis-Hastings (symmetric proposal mixture),
numba-compiled; R^2 via Cholesky of the Gram submatrix."""
import math
import numpy as np
import numba as nb


def make_design(N, p, blocks, block_size, decoy_noise, coef, noise_sd, seed, rho_bg=0.0):
    """Correlated design with 'decoy' variables.  For each block b a set T_b of
    `block_size` true predictors; a decoy d_b = normalised sum_{i in T_b} x_i +
    decoy_noise * z.  y = sum_b coef * sum_{i in T_b} x_i + noise.  A single
    decoy explains almost all of its block's signal, so add/delete MCMC that
    adds the decoy first must pass through models that pay the prior cost of
    several true predictors before the decoy can be dropped."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, p))
    if rho_bg > 0:
        f = rng.standard_normal((N, 1))
        X = np.sqrt(1 - rho_bg) * X + np.sqrt(rho_bg) * f
    truth = []
    decoys = []
    j = 0
    y = np.zeros(N)
    for b in range(blocks):
        T = list(range(j, j + block_size))
        j += block_size
        d = j
        j += 1
        s = X[:, T].sum(1)
        X[:, d] = s / s.std() + decoy_noise * rng.standard_normal(N)
        y += coef * s
        truth += T
        decoys.append(d)
    y += noise_sd * rng.standard_normal(N)
    X = (X - X.mean(0)) / X.std(0)
    y = y - y.mean()
    return X, y, np.array(truth), np.array(decoys)


@nb.njit(cache=True)
def r2(G, c, yy, idx, m):
    """R^2 for columns idx[:m] (centred data): c_g' G_gg^{-1} c_g / y'y."""
    if m == 0:
        return 0.0
    S = np.empty((m, m))
    v = np.empty(m)
    for a in range(m):
        v[a] = c[idx[a]]
        for b in range(m):
            S[a, b] = G[idx[a], idx[b]]
    Lc = np.linalg.cholesky(S)
    # solve L z = v
    z = np.empty(m)
    for a in range(m):
        t = v[a]
        for b in range(a):
            t -= Lc[a, b] * z[b]
        z[a] = t / Lc[a, a]
    return min(np.dot(z, z) / yy, 1.0 - 1e-12)


@nb.njit(cache=True)
def loglik(G, c, yy, N, gpar, idx, m):
    R = r2(G, c, yy, idx, m)
    return 0.5 * (N - 1 - m) * math.log1p(gpar) - 0.5 * (N - 1) * math.log1p(gpar * (1.0 - R))


@nb.njit(cache=True)
def bvs_mh(G, c, yy, N, gpar, h, gamma0, lam, T, burn, thin, seed, mmax, wmask, kmin, kmax):
    """Add/delete/swap MH targeting rho * L^lam, optionally restricted to
    {kmin <= |gamma ∩ w| <= kmax} (wmask = indicator of w).  Returns (ell trace,
    model size trace, inclusion frequencies, final gamma, max ell, argmax gamma,
    overlap trace)."""
    np.random.seed(seed)
    p = len(gamma0)
    gam = gamma0.copy()
    idx = np.empty(p, dtype=np.int64)
    m = 0
    for j in range(p):
        if gam[j]:
            idx[m] = j
            m += 1
    ell = loglik(G, c, yy, N, gpar, idx, m)
    kw = 0
    for j in range(p):
        if gam[j] and wmask[j]:
            kw += 1
    kt = np.zeros((T - burn + thin - 1) // thin, dtype=np.int64)
    lph = math.log(h) - math.log(1 - h)
    cnt = (T - burn + thin - 1) // thin
    et = np.zeros(cnt)
    mt = np.zeros(cnt, dtype=np.int64)
    inc = np.zeros(p)
    k = 0
    emax = ell
    best = gam.copy()
    idx2 = np.empty(p, dtype=np.int64)
    for t in range(T):
        r = np.random.random()
        if r < 0.5:
            # toggle a uniformly chosen coordinate (add or delete); symmetric
            j = np.random.randint(p)
            kw2 = kw + (wmask[j] if gam[j] == 0 else -wmask[j])
            if (gam[j] == 0 and m >= mmax) or kw2 < kmin or kw2 > kmax:
                pass
            else:
                m2 = 0
                if gam[j]:
                    for a in range(m):
                        if idx[a] != j:
                            idx2[m2] = idx[a]
                            m2 += 1
                    dprior = -lph
                else:
                    for a in range(m):
                        idx2[a] = idx[a]
                    idx2[m] = j
                    m2 = m + 1
                    dprior = lph
                ell2 = loglik(G, c, yy, N, gpar, idx2, m2)
                la = lam * (ell2 - ell) + dprior
                if la >= 0 or np.random.random() < math.exp(la):
                    gam[j] = 1 - gam[j]
                    for a in range(m2):
                        idx[a] = idx2[a]
                    m = m2
                    ell = ell2
                    kw = kw2
        elif m > 0 and m < p:
            # swap: one included out, one excluded in (symmetric given m)
            a = np.random.randint(m)
            out = idx[a]
            j = np.random.randint(p)
            kw2 = kw - wmask[out] + wmask[j]
            if gam[j] == 0 and kw2 >= kmin and kw2 <= kmax:
                for b in range(m):
                    idx2[b] = idx[b]
                idx2[a] = j
                ell2 = loglik(G, c, yy, N, gpar, idx2, m)
                la = lam * (ell2 - ell)
                if la >= 0 or np.random.random() < math.exp(la):
                    gam[out] = 0
                    gam[j] = 1
                    idx[a] = j
                    ell = ell2
                    kw = kw2
        if ell > emax:
            emax = ell
            best[:] = gam
        if t >= burn and (t - burn) % thin == 0 and k < cnt:
            et[k] = ell
            mt[k] = m
            kt[k] = kw
            for b in range(m):
                inc[idx[b]] += 1
            k += 1
    return et[:k], mt[:k], inc / max(k, 1), gam, emax, best, kt[:k]


def log_prior(gamma, h):
    m = int(np.sum(gamma))
    return m * math.log(h) + (len(gamma) - m) * math.log(1 - h)


def ell_of(G, c, yy, N, gpar, gamma):
    idx = np.flatnonzero(gamma).astype(np.int64)
    idx = np.concatenate((idx, np.zeros(len(gamma) - len(idx), dtype=np.int64)))
    return loglik(G, c, yy, N, gpar, idx, int(np.sum(gamma)))


def greedy_forward(G, c, yy, N, gpar, h, p):
    """Forward stepwise on log posterior; returns best gamma found."""
    gam = np.zeros(p, dtype=np.int64)
    cur = ell_of(G, c, yy, N, gpar, gam) + log_prior(gam, h)
    while True:
        best, bj = cur, -1
        for j in np.flatnonzero(gam == 0):
            gam[j] = 1
            v = ell_of(G, c, yy, N, gpar, gam) + log_prior(gam, h)
            gam[j] = 0
            if v > best:
                best, bj = v, j
        if bj < 0:
            return gam
        gam[bj] = 1
        cur = best


@nb.njit(cache=True)
def enumerate_all(G, c, yy, N, gpar, p):
    """log L for all 2^p models (p <= 22), index = bitmask."""
    M = 1 << p
    out = np.empty(M)
    idx = np.empty(p, dtype=np.int64)
    for mask in range(M):
        m = 0
        for j in range(p):
            if (mask >> j) & 1:
                idx[m] = j
                m += 1
        out[mask] = loglik(G, c, yy, N, gpar, idx, m)
    return out


def make_suppressor_design(N, p, blocks, k, eta, a, noise_sd, seed):
    """Blocks of k predictors: x_i = z_i (i < k), x_k = -sum_{i<k} z_i + eta u;
    y = sum_b a u_b + noise.  Only the full block explains y (u_b = sum x / eta);
    every proper subset has R^2 ~ 0.  Remaining p - blocks*k columns are noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, p))
    y = noise_sd * rng.standard_normal(N)
    truth = []
    for b in range(blocks):
        cols = list(range(b * k, (b + 1) * k))
        u = rng.standard_normal(N)
        X[:, cols[-1]] = -X[:, cols[:-1]].sum(1) + eta * u
        y += a * u
        truth += cols
    perm = rng.permutation(p)            # hide block structure in column order
    X = X[:, perm]
    inv = np.argsort(perm)
    truth = np.sort(inv[np.array(truth)])
    X = (X - X.mean(0)) / X.std(0)
    y = y - y.mean()
    return X, y, truth


def backward_from_full(G, c, yy, N, gpar, h, p):
    """Backward elimination on the log posterior starting from the full model
    (requires p < N).  Uses the OLS drop-one identity
    R^2_{-j} = R^2 - b_j^2 / (yy * [G_gg^{-1}]_jj)."""
    act = list(range(p))
    lph = math.log(h) - math.log(1 - h)

    def lp_of(m, R2):
        return 0.5 * (N - 1 - m) * math.log1p(gpar) - 0.5 * (N - 1) * math.log1p(gpar * (1 - R2)) + m * lph

    while act:
        idx = np.array(act)
        Gi = np.linalg.inv(G[np.ix_(idx, idx)])
        b = Gi @ c[idx]
        R2 = float(c[idx] @ b / yy)
        m = len(act)
        cur = lp_of(m, R2)
        R2drop = R2 - b ** 2 / (yy * np.diag(Gi))
        cand = np.array([lp_of(m - 1, r) for r in R2drop])
        j = int(np.argmax(cand))
        if cand[j] <= cur:
            break
        act.pop(j)
    gam = np.zeros(p, dtype=np.int64)
    gam[act] = 1
    return gam


def capped_anchors(p, h, m, r):
    """Exact log masses of the Bernoulli(h) product measure restricted to {|gamma| <= m}
    (unnormalised): returns (log rho(X), log rho(C), log rho(D)) with C = {gamma ⊇ w},
    |w| = r, D = X \\ C.  rho(X) = Pr[Bin(p,h) <= m]; rho(C) = h^r Pr[Bin(p-r,h) <= m-r].
    For m >= p this reduces to (0, r log h, log(1 - h^r))."""
    import math
    from scipy.stats import binom
    lX = math.log1p(-binom.sf(m, p, h)) if m < p else 0.0
    lC = r * math.log(h) + (math.log1p(-binom.sf(m - r, p - r, h)) if m < p else 0.0)
    lD = lX + math.log1p(-math.exp(lC - lX))
    return lX, lC, lD
