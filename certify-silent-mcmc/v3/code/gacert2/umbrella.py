"""Flat-histogram (Wang-Landau) umbrella sampling in the witness overlap k,
used as an independent ground-truth sampler near the transition.

Biased target:  pi_g(p) ∝ exp(beta S(p) - g[min(k(p), kcap)]).
Stage 1 learns g by Wang-Landau; stage 2 runs with g fixed and reweights
exactly: mu_beta(k) ∝ H[k] exp(g[min(k,kcap)]).  Stage-2 estimates are
consistent for any fixed g as long as the biased chain mixes (it traverses
the overlap range, which we count)."""
import math
import numpy as np
import numba as nb
from scipy.special import logsumexp
from gacert.core import score, delta


@nb.njit(cache=True)
def _kchange(p, w, u, v):
    pu = p[u]
    pv = p[v]
    return (- (1 if pu == w[u] else 0) - (1 if pv == w[v] else 0)
            + (1 if pv == w[u] else 0) + (1 if pu == w[v] else 0))


@nb.njit(cache=True)
def wang_landau(A, B, p0, w, beta, kcap, stage_T, f0, fmin, flat, seed):
    np.random.seed(seed)
    n = len(p0)
    p = p0.copy()
    s = score(A, B, p)
    k = 0
    for i in range(n):
        if p[i] == w[i]:
            k += 1
    g = np.zeros(kcap + 1)
    f = f0
    stages = 0
    while f > fmin and stages < 60:
        H = np.zeros(kcap + 1)
        for t in range(stage_T):
            u = np.random.randint(n)
            v = np.random.randint(n - 1)
            if v >= u:
                v += 1
            k2 = k + _kchange(p, w, u, v)
            d = delta(A, B, p, u, v)
            la = beta * d - g[min(k2, kcap)] + g[min(k, kcap)]
            if la >= 0 or np.random.random() < math.exp(la):
                a = p[u]
                p[u] = p[v]
                p[v] = a
                s += d
                k = k2
            kb = min(k, kcap)
            g[kb] += f
            H[kb] += 1
        # flatness over bins 0..kcap (all reachable since kcap <= n-2)
        mn = H.min()
        mean = H.mean()
        stages += 1
        if mn > flat * mean:
            f *= 0.5
    g -= g.min()
    return g, p, f, stages


@nb.njit(cache=True)
def umbrella_prod(A, B, p0, w, beta, g, kcap, T, burn, thin, seed, want_marg):
    """Fixed-bias production run.  Returns per-k visit counts H, per-k score
    sums, per-k marginal counts (thinned; shape (n+1, n, n) if want_marg),
    and number of full traversals between k<=klo and k>=khi (klo = 0.15 n,
    khi = kcap)."""
    np.random.seed(seed)
    n = len(p0)
    p = p0.copy()
    s = score(A, B, p)
    k = 0
    for i in range(n):
        if p[i] == w[i]:
            k += 1
    H = np.zeros(n + 1)
    SS = np.zeros(n + 1)
    Hm = np.zeros(n + 1)
    Mk = np.zeros((n + 1, n, n)) if want_marg else np.zeros((1, 1, 1))
    klo = int(0.15 * n)
    khi = kcap
    state = 1 if k >= khi else 0
    trav = 0
    for t in range(T):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        k2 = k + _kchange(p, w, u, v)
        d = delta(A, B, p, u, v)
        la = beta * d - g[min(k2, kcap)] + g[min(k, kcap)]
        if la >= 0 or np.random.random() < math.exp(la):
            a = p[u]
            p[u] = p[v]
            p[v] = a
            s += d
            k = k2
            if state == 0 and k >= khi:
                state = 1
                trav += 1
            elif state == 1 and k <= klo:
                state = 0
                trav += 1
        if t >= burn:
            H[k] += 1
            SS[k] += s
            if want_marg and (t - burn) % thin == 0:
                Hm[k] += 1
                for i in range(n):
                    Mk[k, i, p[i]] += 1
    return H, SS, Hm, Mk, trav, p


def reweight(H, SS, Hm, Mk, g, kcap, k0):
    """Exact reweighting of a fixed-bias run to mu_beta."""
    n1 = len(H)
    gk = np.array([g[min(k, kcap)] for k in range(n1)])
    ok = H > 0
    lw = np.full(n1, -np.inf)
    lw[ok] = np.log(H[ok]) + gk[ok]
    lw -= logsumexp(lw[ok])
    pk = np.exp(lw)
    out = dict(pk=pk, fracC=float(pk[k0:].sum()),
               meanS=float(np.sum(pk[ok] * SS[ok] / H[ok])))
    if Mk.shape[0] == n1:
        n = Mk.shape[1]
        M = np.zeros((n, n))
        for k in range(n1):
            if Hm[k] > 0:
                M += pk[k] * Mk[k] / Hm[k]
        # renormalise for k-bins visited in H but not in thinned marginals
        out['marg'] = M / max(sum(pk[k] for k in range(n1) if Hm[k] > 0), 1e-300)
    return out


def refine_bias(A, B, p, w, beta, g, kcap, T, seed, iters=2):
    """Iterative flat-histogram correction g += log(H / mean H) on 0..kcap."""
    g = g.copy()
    for it in range(iters):
        H, SS, Hm, Mk, trav, p = umbrella_prod(A, B, p, w, beta, g, kcap, T, 0, T, seed + it, False)
        h = H[:kcap + 1] + 1.0
        # bins beyond kcap share the kcap bias: fold their counts into kcap
        h[kcap] += H[kcap + 1:].sum()
        g += np.log(h / h.mean())
        g -= g.min()
    return g, p


def umbrella_gt(A, B, w, beta, k0, kcap, T_wl, T_ref, T_prod, seed, want_marg=True, thin=512):
    rng = np.random.default_rng(seed)
    n = len(w)
    g, p, f, st = wang_landau(A, B, rng.permutation(n), w, beta, kcap, T_wl, 1.0, 1e-4, 0.6,
                              int(rng.integers(2 ** 31)))
    g, p = refine_bias(A, B, p, w, beta, g, kcap, T_ref, int(rng.integers(2 ** 31)))
    H, SS, Hm, Mk, trav, p = umbrella_prod(A, B, p, w, beta, g, kcap, T_prod, T_prod // 16, thin,
                                           int(rng.integers(2 ** 31)), want_marg)
    out = reweight(H, SS, Hm, Mk, g, kcap, k0)
    out.update(traversals=int(trav), g=g, H=H, kcap=kcap)
    return out
