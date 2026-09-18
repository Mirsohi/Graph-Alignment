"""Reviewer-targeted robustness sweep for the LoG 2026 rebuttal.

Run from the supplement root:
    python code/rebuttal_robustness_sweep.py

The default reproduces the rebuttal experiment used by the authors:
  * n=9
  * 20 fixed graph seeds per (p, epsilon) cell
  * four independent local-MH chains and four PT runs
  * 2^18 within-temperature proposals per run
  * same burn-in, thinning, seeding, exact oracle, and PT ladder as the submitted paper

Outputs combined raw and summary CSVs in results/rebuttal_robustness/.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from budget_curve_experiment import (
    chain_disagreement,
    initial_local,
    initial_pt,
    marginals,
    row_tv,
    run_local_compiled,
    run_pt_compiled,
    seed_for,
    split_rhat,
    warmup,
)
from exact_posterior_benchmark import (
    exact_posterior_auto,
    generate_paired_instances,
    likelihood_beta,
)
from iid_initialization_analysis import iid_row_tv_replicates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "rebuttal_robustness"
CELLS = [
    (0.15, 0.10),
    (0.30, 0.05),
    (0.30, 0.10),
    (0.30, 0.20),
    (0.30, 0.30),
    (0.30, 0.50),
    (0.50, 0.10),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--budget", type=int, default=262144)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--burn-fraction", type=float, default=0.25)
    ap.add_argument("--local-thin", type=int, default=5)
    ap.add_argument("--pt-thin", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--iid-replicates", type=int, default=500)
    ap.add_argument("--quick", action="store_true", help="Use 3 seeds per cell for a fast smoke test")
    return ap.parse_args()


def one_method(n, instance, exact, beta, p, noise, graph_seed, method, args):
    samples_by_chain=[]; scores_by_chain=[]; accept=[]; swap_accept=[]; runtime=0.0
    replicas=8
    method_code = 0 if method == "local" else 1
    for chain in range(args.chains):
        seed = seed_for(n, graph_seed, method_code, chain)
        t0=time.perf_counter()
        if method == "local":
            samples,scores,accepted = run_local_compiled(
                instance.A, instance.B, beta,
                args.budget, int(args.budget*args.burn_fraction), args.local_thin,
                seed, initial_local(n,seed),
            )
            a=accepted/args.budget; sa=float('nan')
        else:
            if args.budget % replicas:
                raise ValueError("budget must be divisible by 8 for PT")
            sweeps=args.budget//replicas
            samples,scores,accepted,attempted,swapped = run_pt_compiled(
                instance.A, instance.B, beta,
                sweeps, int(sweeps*args.burn_fraction), args.pt_thin,
                seed, initial_pt(n,replicas,seed),
            )
            a=accepted/args.budget; sa=swapped/max(attempted,1)
        runtime += time.perf_counter()-t0
        samples_by_chain.append(samples)
        scores_by_chain.append(scores.astype(float))
        accept.append(a); swap_accept.append(sa)
    pooled=np.vstack(samples_by_chain)
    phat=marginals(pooled,n)
    return {
        'n':n,'p':p,'noise':noise,'graph_seed':graph_seed,'method':method,
        'beta_likelihood':beta,'local_proposals_per_chain':args.budget,
        'chains':args.chains,'runtime_seconds':runtime,
        'retained_samples_total':len(pooled),
        'mean_row_tv':row_tv(phat, exact.marginals),
        'score_split_rhat':split_rhat(scores_by_chain),
        'chain_marginal_disagreement':chain_disagreement(samples_by_chain,n),
        'mean_acceptance_rate':float(np.mean(accept)),
        'mean_swap_acceptance':float(np.nanmean(swap_accept)) if np.any(np.isfinite(swap_accept)) else float('nan'),
        'exact_effective_states':exact.effective_states,
        'exact_map_count':exact.map_count,
    }


def main():
    args=parse_args()
    if args.quick:
        args.instances=3
    args.output_dir.mkdir(parents=True,exist_ok=True)
    warmup()
    rows=[]
    for p,noise in CELLS:
        beta=likelihood_beta(p,noise)
        for graph_seed in range(args.instances):
            inst=generate_paired_instances(args.n,p,[noise],graph_seed)[0]
            exact=exact_posterior_auto(inst,beta)
            for method_code, method in enumerate(('local','parallel_tempering')):
                rec=one_method(args.n,inst,exact,beta,p,noise,graph_seed,method,args)
                floor_seed=610_000_000 + int(round(p*1000))*1_000_000 + int(round(noise*1000))*10_000 + graph_seed*10 + method_code
                floor=iid_row_tv_replicates(
                    exact.marginals, int(rec['retained_samples_total']), args.iid_replicates,
                    np.random.default_rng(floor_seed)
                )
                rec['iid_floor_mean']=float(floor.mean())
                rec['iid_floor_sd']=float(floor.std(ddof=1))
                rec['excess_row_tv']=float(rec['mean_row_tv']-floor.mean())
                rows.append(rec)
            print(f"finished p={p:.2f}, epsilon={noise:.2f}, seed={graph_seed}",flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(args.output_dir/'robustness_raw.csv',index=False)

    rng=np.random.default_rng(20260830)
    summary=[]
    for (p,noise),g in raw.groupby(['p','noise'],sort=True):
        loc=g[g.method=='local'].sort_values('graph_seed').reset_index(drop=True)
        pt=g[g.method=='parallel_tempering'].sort_values('graph_seed').reset_index(drop=True)
        diff=loc.mean_row_tv.to_numpy()-pt.mean_row_tv.to_numpy()
        boots=np.mean(rng.choice(diff,size=(args.bootstrap,len(diff)),replace=True),axis=1)
        summary.append({
            'p':p,'noise':noise,'beta_likelihood':likelihood_beta(p,noise),
            'instances':len(loc),
            'local_mean_row_tv':loc.mean_row_tv.mean(),'local_sem':loc.mean_row_tv.sem(),
            'pt_mean_row_tv':pt.mean_row_tv.mean(),'pt_sem':pt.mean_row_tv.sem(),
            'local_iid_floor':loc.iid_floor_mean.mean(),'local_excess_row_tv':loc.excess_row_tv.mean(),
            'pt_iid_floor':pt.iid_floor_mean.mean(),'pt_excess_row_tv':pt.excess_row_tv.mean(),
            'local_minus_pt':diff.mean(),
            'paired_bootstrap_95_low':np.quantile(boots,.025),
            'paired_bootstrap_95_high':np.quantile(boots,.975),
            'fraction_local_lower_error':float(np.mean(loc.mean_row_tv.to_numpy()<pt.mean_row_tv.to_numpy())),
            'median_exact_effective_states':float(np.median(loc.exact_effective_states)),
            'mean_log_exact_effective_states':float(np.mean(np.log(loc.exact_effective_states))),
            'mean_local_runtime_seconds':loc.runtime_seconds.mean(),
            'mean_pt_runtime_seconds':pt.runtime_seconds.mean(),
            'pt_over_local_runtime_ratio':pt.runtime_seconds.mean()/loc.runtime_seconds.mean(),
        })
    summary=pd.DataFrame(summary)
    summary.to_csv(args.output_dir/'robustness_summary.csv',index=False)
    print('\nSUMMARY\n',summary.to_string(index=False))

if __name__=='__main__':
    main()
