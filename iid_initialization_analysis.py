"""Matched i.i.d. row-TV floors and initialization sensitivity.

This script augments the budget-scaling experiment with two finite-sample controls:
  (1) the finite-sample row-TV expected from exact i.i.d. posterior draws,
      matched to each method's retained sample count; and
  (2) a common-start local-MH arm at the largest budget, compared with the
      independently uniform starts already used by the main budget experiment.

For the row-TV expectation it is sufficient to sample each row from its exact
categorical marginal: every row of an i.i.d. permutation sample has exactly
that multinomial distribution, and row-TV is the average of row-wise losses.
This reproduces the exact expectation without retaining all n! probabilities.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from exact_posterior_benchmark import (  # noqa: E402
    exact_posterior_auto,
    generate_paired_instances,
    likelihood_beta,
)
from budget_curve_experiment import (  # noqa: E402
    chain_disagreement,
    marginals,
    row_tv,
    run_local_compiled,
    seed_for,
    split_rhat,
    warmup,
)


def iid_row_tv_replicates(
    exact_marginals: np.ndarray,
    retained_samples: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte Carlo distribution of row-TV for exact i.i.d. posterior samples.

    Row i of an i.i.d. permutation sample is categorical with probability
    exact_marginals[i]. The expectation of average row-TV is therefore exactly
    reproduced by independent multinomial simulation row by row. Cross-row
    dependence is irrelevant for the mean, which is the primary floor used in
    the paper.
    """
    n = exact_marginals.shape[0]
    out = np.empty(replicates, dtype=float)
    for r in range(replicates):
        total = 0.0
        for i in range(n):
            counts = rng.multinomial(retained_samples, exact_marginals[i])
            empirical = counts / retained_samples
            total += 0.5 * np.abs(empirical - exact_marginals[i]).sum()
        out[r] = total / n
    return out


def common_initial_state(n: int, graph_seed: int) -> np.ndarray:
    """One deterministic random permutation shared by all four chains."""
    return np.random.default_rng(72_000_000 + 1_000 * n + graph_seed).permutation(n).astype(np.int64)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_raw = pd.read_csv(args.budget_raw)
    warmup()

    floor_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    instance_counts = {8: 20, 9: 20, 10: int(args.n10_instances)}

    for n in args.n_values:
        for graph_seed in range(args.seed_start, instance_counts[n]):
            instance = generate_paired_instances(n, args.p, [args.noise], graph_seed)[0]
            beta = likelihood_beta(args.p, args.noise)
            exact = exact_posterior_auto(instance, beta)

            instance_budget = budget_raw[
                (budget_raw.n == n) & (budget_raw.graph_seed == graph_seed)
            ]
            if len(instance_budget) == 0:
                raise RuntimeError(f"budget results missing n={n}, seed={graph_seed}")

            for _, row in instance_budget.iterrows():
                method = str(row.method)
                budget = int(row.local_proposals_per_chain)
                retained = int(row.retained_samples_total)
                seed = 310_000_000 + n * 1_000_000 + graph_seed * 10_000 + budget + (0 if method == "local" else 1)
                reps = iid_row_tv_replicates(
                    exact.marginals,
                    retained,
                    args.iid_replicates,
                    np.random.default_rng(seed),
                )
                floor_rows.append(
                    {
                        "n": n,
                        "graph_seed": graph_seed,
                        "method": method,
                        "local_proposals_per_chain": budget,
                        "retained_samples_total": retained,
                        "iid_replicates": args.iid_replicates,
                        "iid_floor_mean": float(reps.mean()),
                        "iid_floor_sd": float(reps.std(ddof=1)),
                        "iid_floor_q025": float(np.quantile(reps, 0.025)),
                        "iid_floor_q975": float(np.quantile(reps, 0.975)),
                        "observed_row_tv": float(row.mean_row_tv),
                        "excess_row_tv": float(row.mean_row_tv - reps.mean()),
                        "observed_over_floor_ratio": float(row.mean_row_tv / reps.mean()),
                    }
                )

            # Only the additional common-start arm needs new MCMC. Independent
            # uniform starts are already the local arm in budget_curve_raw.csv.
            budget = int(args.initialization_budget)
            initial = common_initial_state(n, graph_seed)
            chain_samples: list[np.ndarray] = []
            chain_scores: list[np.ndarray] = []
            acceptances: list[float] = []
            runtime = 0.0
            for chain in range(args.chains):
                seed = seed_for(n, graph_seed, 9, chain)
                start = time.perf_counter()
                samples, scores, accepted = run_local_compiled(
                    instance.A,
                    instance.B,
                    beta,
                    budget,
                    int(budget * args.burn_fraction),
                    args.local_thin,
                    seed,
                    initial,
                )
                runtime += time.perf_counter() - start
                chain_samples.append(samples)
                chain_scores.append(scores.astype(float))
                acceptances.append(accepted / budget)
            pooled = np.vstack(chain_samples)
            common_rows.append(
                {
                    "n": n,
                    "graph_seed": graph_seed,
                    "initialization": "common_random",
                    "local_proposals_per_chain": budget,
                    "retained_samples_total": len(pooled),
                    "mean_row_tv": row_tv(marginals(pooled, n), exact.marginals),
                    "score_split_rhat": split_rhat(chain_scores),
                    "chain_marginal_disagreement": chain_disagreement(chain_samples, n),
                    "mean_acceptance_rate": float(np.mean(acceptances)),
                    "runtime_seconds": runtime,
                }
            )
            print(f"finished floors/init n={n}, seed={graph_seed}", flush=True)

    floors = pd.DataFrame(floor_rows)
    floors.to_csv(args.output_dir / "iid_floor_raw.csv", index=False)

    floor_summary = (
        floors.groupby(["n", "method", "local_proposals_per_chain"], sort=True)
        .agg(
            instances=("graph_seed", "size"),
            observed_mean_row_tv=("observed_row_tv", "mean"),
            observed_sem_row_tv=("observed_row_tv", "sem"),
            iid_floor_mean=("iid_floor_mean", "mean"),
            iid_floor_sem=("iid_floor_mean", "sem"),
            mean_excess_row_tv=("excess_row_tv", "mean"),
            sem_excess_row_tv=("excess_row_tv", "sem"),
            mean_observed_over_floor_ratio=("observed_over_floor_ratio", "mean"),
        )
        .reset_index()
    )
    floor_summary.to_csv(args.output_dir / "iid_floor_summary.csv", index=False)

    common = pd.DataFrame(common_rows)
    independent = budget_raw[
        (budget_raw.method == "local")
        & (budget_raw.local_proposals_per_chain == args.initialization_budget)
        & (budget_raw.n.isin(args.n_values))
    ][
        [
            "n",
            "graph_seed",
            "local_proposals_per_chain",
            "retained_samples_total",
            "mean_row_tv",
            "score_split_rhat",
            "chain_marginal_disagreement",
            "mean_acceptance_rate",
            "runtime_seconds",
        ]
    ].copy()
    independent["initialization"] = "independent_uniform"
    initialization = pd.concat([independent, common], ignore_index=True)
    initialization.to_csv(args.output_dir / "initialization_raw.csv", index=False)
    init_summary = (
        initialization.groupby(["n", "initialization"], sort=True)
        .agg(
            instances=("graph_seed", "size"),
            mean_row_tv=("mean_row_tv", "mean"),
            sem_row_tv=("mean_row_tv", "sem"),
            median_row_tv=("mean_row_tv", "median"),
            mean_chain_disagreement=("chain_marginal_disagreement", "mean"),
            mean_score_rhat=("score_split_rhat", lambda x: np.nanmean(np.where(np.isfinite(x), x, np.nan))),
        )
        .reset_index()
    )
    init_summary.to_csv(args.output_dir / "initialization_summary.csv", index=False)

    paired = initialization.pivot_table(
        index=["n", "graph_seed"], columns="initialization", values="mean_row_tv"
    ).reset_index()
    paired["common_minus_independent"] = paired.common_random - paired.independent_uniform
    paired.to_csv(args.output_dir / "initialization_paired.csv", index=False)

    metadata = vars(args).copy()
    metadata["budget_raw"] = str(args.budget_raw)
    metadata["output_dir"] = str(args.output_dir)
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nI.I.D. floor summary\n", floor_summary.to_string(index=False))
    print("\nInitialization summary\n", init_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-raw",
        type=Path,
        default=ROOT / "results" / "budget_curves" / "budget_curve_raw.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "floor_initialization"
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument("--n10-instances", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--iid-replicates", type=int, default=250)
    parser.add_argument("--initialization-budget", type=int, default=262144)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--burn-fraction", type=float, default=0.25)
    parser.add_argument("--local-thin", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
