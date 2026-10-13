"""Instances (protein pairs from v1 data, small synthetic correlated graphs)
and graph-only witnesses (best of several annealing runs; cached)."""
import os, re
import numpy as np
from gacert.core import mh_chain, score

V2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ROOT = os.path.dirname(V2)
CASES = os.path.join(ROOT, 'data', 'cases')
WDIR = os.path.join(V2, 'results', 'witness')


def synth_pair(n, deg, q, seed):
    """A ~ ER(n, deg/(n-1)); B = relabelled copy with each pair flipped w.p. q."""
    rng = np.random.default_rng(1000 + seed)
    u, v = np.triu_indices(n, 1)
    A = np.zeros((n, n), dtype=np.int64)
    A[u, v] = rng.random(len(u)) < deg / (n - 1)
    A[v, u] = A[u, v]
    Bp = A.copy()
    f = (rng.random(len(u)) < q).astype(np.int64)
    Bp[u, v] ^= f
    Bp[v, u] = Bp[u, v]
    pt = rng.permutation(n)
    B = np.zeros_like(A)
    B[np.ix_(pt, pt)] = Bp
    return A, B, pt.astype(np.int64)


def find_witness(A, B, restarts=4, T=None, bmax=5.0, seed=0):
    n = A.shape[0]
    if T is None:
        T = 2 ** 21 if n <= 110 else 2 ** 23
    rng = np.random.default_rng(seed)
    best, w = -1, None
    for r in range(restarts):
        ramp = np.concatenate((np.linspace(0.5, bmax, T), np.full(T // 8, bmax)))
        res = mh_chain(A, B, rng.permutation(n), ramp, len(ramp) - 1, 1, int(rng.integers(2 ** 31)))
        if res[3] > best:
            best, w = int(res[3]), res[4].copy()
    return w


def save_witness(case, w, cutoff='c8'):
    tag = case if cutoff == 'c8' else f'{case}_{cutoff}'
    np.save(os.path.join(WDIR, tag + '.npy'), w)


def load_instance(case, cutoff='c8'):
    """Returns A, B, pi_true, witness w (graph-only), meta."""
    m = re.match(r'syn_n(\d+)_d([\d.]+)_q([\d.]+)_s(\d+)$', case)
    if m:
        n, deg, q, s = int(m[1]), float(m[2]), float(m[3]), int(m[4])
        A, B, pt = synth_pair(n, deg, q, s)
        meta = dict(kind='synth', n=n, deg=deg, q=q, seed=s)
    else:
        z = np.load(os.path.join(CASES, case + '.npz'))
        A = z['A_' + cutoff].astype(np.int64)
        B = z['B_' + cutoff].astype(np.int64)
        pt = z['pi_true'].astype(np.int64)
        meta = dict(kind='protein', cutoff=cutoff)
    os.makedirs(WDIR, exist_ok=True)
    tag = case if cutoff == 'c8' else f'{case}_{cutoff}'
    wp = os.path.join(WDIR, tag + '.npy')
    if os.path.exists(wp):
        w = np.load(wp).astype(np.int64)
    else:
        w = find_witness(A, B)
        np.save(wp, w)
    meta.update(Sw=int(score(A, B, w)), S_true=int(score(A, B, pt)),
                witness_agree_true=float(np.mean(w == pt)))
    return A, B, pt, w, meta


def synth_channel(n, deg, s, r, seed):
    """A ~ ER(n, deg/(n-1)); B = relabelled channel output: an A-edge is kept
    w.p. s, a non-edge becomes an edge w.p. r (independently per pair).  By the
    edge-channel lemma the posterior of the planted relabelling is mu_beta with
    beta = log(s (1-r) / ((1-s) r)), so the planted pi* is an exact posterior draw."""
    rng = np.random.default_rng(5000 + seed)
    u, v = np.triu_indices(n, 1)
    a = rng.random(len(u)) < deg / (n - 1)
    x = rng.random(len(u))
    b = np.where(a, x < s, x < r)
    A = np.zeros((n, n), dtype=np.int64)
    A[u, v] = a
    A[v, u] = a
    Bp = np.zeros((n, n), dtype=np.int64)
    Bp[u, v] = b
    Bp[v, u] = b
    pt = rng.permutation(n)
    B = np.zeros_like(A)
    B[np.ix_(pt, pt)] = Bp
    beta = float(np.log(s * (1 - r) / ((1 - s) * r)))
    return A, B, pt.astype(np.int64), beta
