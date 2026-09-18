"""Recompute the reviewer-requested modern rank-normalized/folded split-Rhat.

This uses exactly the local-MH chains/seeds/start states from budget_curve_experiment.py
at the maximum submitted budget, then joins to the checked-in exact row-TV and
cross-chain marginal disagreement. It does not recompute the exact posterior.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr

from budget_curve_experiment import (
    generate_paired_instances, likelihood_beta, run_local_compiled,
    seed_for, initial_local, warmup, split_rhat,
)

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / 'results' / 'budget_curves' / 'budget_curve_raw.csv'
OUT = ROOT / 'results' / 'rebuttal_modern_rhat'
BUDGET = 262144
CHAINS = 4
BURN_FRAC = 0.25
THIN = 5
P = 0.3
NOISE = 0.1


def basic_rhat_equal(chains: np.ndarray) -> float:
    """Basic Rhat on equal-length chains (chains x draws)."""
    m, n = chains.shape
    if m < 2 or n < 2:
        return float('nan')
    chain_means = chains.mean(axis=1)
    W = float(np.mean(np.var(chains, axis=1, ddof=1)))
    B = float(n * np.var(chain_means, ddof=1))
    if W == 0.0:
        return 1.0 if B == 0.0 else float('inf')
    var_hat = (n - 1.0) / n * W + B / n
    return float(math.sqrt(var_hat / W))


def split_chains(chains: list[np.ndarray]) -> np.ndarray:
    minimum = min(len(c) for c in chains)
    half = minimum // 2
    return np.asarray([part for c in chains for part in (c[:half], c[-half:])], dtype=float)


def z_scale(x: np.ndarray) -> np.ndarray:
    """Vehtari et al. rank normalization with average ranks for ties."""
    flat = x.ravel()
    ranks = rankdata(flat, method='average')
    z = norm.ppf((ranks - 3.0/8.0) / (len(flat) - 1.0/4.0))
    return z.reshape(x.shape)


def rank_folded_split_rhat(chains: list[np.ndarray]) -> tuple[float,float,float]:
    split = split_chains(chains)
    rank_rhat = basic_rhat_equal(z_scale(split))
    med = float(np.median(split))
    folded = np.abs(split - med)
    folded_rhat = basic_rhat_equal(z_scale(folded))
    return max(rank_rhat, folded_rhat), rank_rhat, folded_rhat


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    ranks = rankdata(scores, method='average')
    n1 = int(labels.sum()); n0 = len(labels)-n1
    return float((ranks[labels==1].sum() - n1*(n1+1)/2)/(n1*n0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    warmup()
    canonical = pd.read_csv(CANON)
    canonical = canonical[(canonical.method=='local') & (canonical.local_proposals_per_chain==BUDGET)].copy()
    lookup = canonical.set_index(['n','graph_seed'])
    rows=[]
    for n in (8,9,10):
        for graph_seed in range(20):
            inst = generate_paired_instances(n,P,[NOISE],graph_seed)[0]
            beta = likelihood_beta(P,NOISE)
            score_chains=[]
            for chain in range(CHAINS):
                seed = seed_for(n,graph_seed,0,chain)
                _, scores, _ = run_local_compiled(
                    inst.A, inst.B, beta, BUDGET, int(BUDGET*BURN_FRAC), THIN,
                    seed, initial_local(n,seed)
                )
                score_chains.append(scores.astype(float))
            classical = split_rhat(score_chains)
            modern, rank_rhat, folded_rhat = rank_folded_split_rhat(score_chains)
            old = lookup.loc[(n,graph_seed)]
            rows.append(dict(
                n=n, graph_seed=graph_seed,
                exact_mean_row_tv=float(old.mean_row_tv),
                chain_marginal_disagreement=float(old.chain_marginal_disagreement),
                canonical_classical_split_rhat=float(old.score_split_rhat),
                rerun_classical_split_rhat=classical,
                modern_rank_folded_rhat=modern,
                rank_normalized_rhat=rank_rhat,
                folded_rhat=folded_rhat,
            ))
            print(f'finished n={n}, seed={graph_seed}', flush=True)
    d=pd.DataFrame(rows)
    # Same deterministic sampler should reproduce old scalar split-Rhat exactly (floating tolerance).
    max_abs=float(np.max(np.abs(d.canonical_classical_split_rhat-d.rerun_classical_split_rhat)))
    print('max classical Rhat reproduction error:', max_abs)
    d['bad']=d.exact_mean_row_tv>0.1
    summary=[]
    for col,label in [
        ('canonical_classical_split_rhat','classical score split-Rhat'),
        ('modern_rank_folded_rhat','rank-normalized/folded score Rhat'),
        ('chain_marginal_disagreement','marginal disagreement')]:
        x=d[col].replace([np.inf,-np.inf],np.nan)
        ok=x.notna()
        rho,pval=spearmanr(x[ok],d.loc[ok,'exact_mean_row_tv'])
        auc=auc_rank(x[ok].to_numpy(),d.loc[ok,'bad'].astype(int).to_numpy())
        summary.append(dict(diagnostic=label,n_instances=int(ok.sum()),spearman_rho=float(rho),spearman_p=float(pval),auc_for_rowtv_gt_0_1=auc))
    s=pd.DataFrame(summary)
    d.to_csv(OUT/'modern_rhat_raw.csv',index=False)
    s.to_csv(OUT/'modern_rhat_summary.csv',index=False)
    print(s.to_string(index=False))

if __name__=='__main__':
    main()
