"""Structured-mass floor (R15 S1): the coset ELBO.

q = product over disjoint position blocks J_b (|J_b| <= K) of distributions q_b on the bijections
J_b -> w(J_b) (the witness's images on the block).  Images are disjoint, so every tuple is a valid
permutation in the coset-like set {pi : pi(J_b) = w(J_b) for all b}, H(q) = sum_b H(q_b), and
E_q S = sum_{within-block pairs} (enumerated) + sum_{cross-block pairs i<j} A_ij (P B P^T)_ij,
with P the marginal matrix of q.  Both terms are computed exactly (deterministic, floating point), so
  ELBO(q) = beta E_q S + H(q) <= log sum_{pi in coset} e^{beta S(pi)} <= Lambda(beta)
is a rigorous lower bound (Gibbs variational principle -- textbook; the point floor is q = delta_w).
Blocks are built from co-exchanges observed in a C-restricted chain (k >= k0) started at w; any
block structure gives a valid bound, so the data-dependent construction does not affect rigour.
q is fitted by block-coordinate ascent (each block update is the exact conditional optimum)."""
import itertools
import numpy as np
from scipy.special import logsumexp
from .phase import rmh

PERMS = {K: np.array(list(itertools.permutations(range(K))), dtype=np.int64) for K in range(1, 8)}


def chain_samples(A, B, w, beta, k0, nsamp=150, T=2 ** 14, seed=7):
    """nsamp states of a C-restricted (k >= k0) MH chain started at w, T steps apart."""
    n = len(w)
    rng = np.random.default_rng(seed)
    p = w.copy()
    out = []
    for _ in range(nsamp):
        o = rmh(A, B, p, w, float(beta), T, 0, T, k0, n, int(rng.integers(2 ** 31)), False)
        p = o[3].copy()
        out.append(p.copy())
    return out, int(nsamp * T)


def make_blocks(w, samples, K=6):
    """Greedy merge of positions by co-exchange frequency (pairs (i, w^{-1}(pi(i))) with pi(i) != w(i)),
    block sizes capped at K."""
    n = len(w)
    winv = np.argsort(w)
    Wt = np.zeros((n, n))
    for p in samples:
        bad = np.where(p != w)[0]
        np.add.at(Wt, (bad, winv[p[bad]]), 1)
    Wt = Wt + Wt.T
    blocks = [[i] for i in range(n)]
    owner = list(range(n))
    order = np.dstack(np.unravel_index(np.argsort(-Wt, axis=None, kind='stable'), Wt.shape))[0]
    for a, b in order:
        if Wt[a, b] <= 0:
            break
        if a >= b:
            continue
        ba, bb = owner[a], owner[b]
        if ba == bb or len(blocks[ba]) + len(blocks[bb]) > K:
            continue
        blocks[ba] += blocks[bb]
        for x in blocks[bb]:
            owner[x] = ba
        blocks[bb] = []
    return [sorted(b) for b in blocks if b]


def coset_elbo(A, B, w, beta, blocks, sweeps=8):
    """Returns dict(elbo, ES, H, P, n_blocks_gt1)."""
    n = len(w)
    A = A.astype(float)
    B = B.astype(float)
    P = np.zeros((n, n))
    P[np.arange(n), w] = 1.0
    pre = []
    for b in blocks:
        K = len(b)
        J = np.array(b)
        img = w[J][PERMS[K]]                  # (K!, K): sigma maps J[a] -> img[s, a]
        Sw_ = np.zeros(len(img))
        for a in range(K):
            for c in range(a + 1, K):
                if A[J[a], J[c]]:
                    Sw_ += B[img[:, a], img[:, c]]
        pre.append((J, img, Sw_))
    Hs = np.zeros(len(blocks))
    Ews = np.zeros(len(blocks))
    for _ in range(sweeps):
        for bi, (J, img, Sw_) in enumerate(pre):
            if len(J) == 1:
                Ews[bi] = 0.0
                continue
            mask = np.ones(n, bool)
            mask[J] = False
            h = (A[np.ix_(J, mask)] @ P[mask]) @ B.T          # h[a, y]: expected cross score of J[a] -> y
            E = Sw_ + sum(h[a, img[:, a]] for a in range(len(J)))
            lq = beta * E - logsumexp(beta * E)
            q = np.exp(lq)
            Hs[bi] = -(q * lq).sum()
            Ews[bi] = (q * Sw_).sum()
            Pn = np.zeros((len(J), n))
            for a in range(len(J)):
                np.add.at(Pn[a], img[:, a], q)
            P[J] = Pn
    owner = np.empty(n, int)
    for bi, b in enumerate(blocks):
        owner[b] = bi
    E = P @ B @ P.T
    same = owner[:, None] == owner[None, :]
    ES = float(np.triu(A * E * (~same), 1).sum() + Ews.sum())
    H = float(Hs.sum())
    return dict(elbo=float(beta * ES + H), ES=ES, H=H, P=P, n_blocks_gt1=int(sum(len(b) > 1 for b in blocks)))
