"""R15 M2 (P1 held-out accuracy-coverage): instances, evaluation grid, target quantity.

Test instances: 40 new synthetic pairs from a new seed namespace,
  rng = np.random.default_rng(np.random.SeedSequence([2027, 15, family, idx])), idx = 0..9,
  F1 single-component control  er_pair(n=60, deg 7, q=0.03)
  F2 hidden middle             components n1=n2=30, deg 7, q1=0.03, q2=0.12
  F3 shifted control           single component, n=50, deg 6, q=0.08
  F4 shifted hidden middle     components n1=24, n2=36, deg 7, q1=0.03, q2=0.10
then a random relabelling pt = rng.permutation(n), B[pt, pt] = B' (as exp_heldout_validity.instance);
the planted correspondence is pi* = pt (used ONLY for evaluation).
Dev instances (smoke tests only): exp_heldout_validity.instance('multi'|'single', 60, 0).
Processing order: interleaved (F1,0),(F2,0),(F3,0),(F4,0),(F1,1),...
Target: a(beta) = E_{mu_beta}[k(pi, pi*)] / n on G = {1.0, 1.1, ..., 1.9}."""
import numpy as np
from r15_common import RES
import os
from exp_heldout_validity import er_pair, instance as dev_instance

P1 = os.path.join(RES, 'p1_heldout')
G = [round(1.0 + 0.1 * i, 4) for i in range(10)]
FAMILIES = {1: dict(kind='single', n=60, deg=7.0, q=0.03),
            2: dict(kind='multi', n1=30, n2=30, deg=7.0, q1=0.03, q2=0.12),
            3: dict(kind='single', n=50, deg=6.0, q=0.08),
            4: dict(kind='multi', n1=24, n2=36, deg=7.0, q1=0.03, q2=0.10)}
TEST = [f'F{f}_i{i}' for i in range(10) for f in (1, 2, 3, 4)]       # interleaved processing order
DEV = ['dev_multi_n60_s0', 'dev_single_n60_s0']


def make(name):
    """Returns A, B, pi_star (int64)."""
    if name.startswith('dev_'):
        kind = name.split('_')[1]
        return dev_instance(kind, 60, 0)
    f, i = int(name[1]), int(name.split('_i')[1])
    c = FAMILIES[f]
    rng = np.random.default_rng(np.random.SeedSequence([2027, 15, f, i]))
    if c['kind'] == 'single':
        n = c['n']
        A, Bp = er_pair(n, c['deg'], c['q'], rng)
    else:
        n1, n2 = c['n1'], c['n2']
        n = n1 + n2
        A1, B1 = er_pair(n1, c['deg'], c['q1'], rng)
        A2, B2 = er_pair(n2, c['deg'], c['q2'], rng)
        A = np.zeros((n, n), dtype=np.int64)
        Bp = np.zeros_like(A)
        A[:n1, :n1] = A1; A[n1:, n1:] = A2
        Bp[:n1, :n1] = B1; Bp[n1:, n1:] = B2
    pt = rng.permutation(n)
    B = np.zeros_like(A)
    B[np.ix_(pt, pt)] = Bp
    return A, B, pt.astype(np.int64)


def a_from_marg(M, ps):
    """(1/n) sum_i M[i, pi*(i)]: expected fraction of correct correspondences under marginals M."""
    n = len(ps)
    return float(M[np.arange(n), ps].sum() / n)


def row_tv(M1, M2):
    return float(0.5 * np.abs(M1 - M2).sum(1).mean())
