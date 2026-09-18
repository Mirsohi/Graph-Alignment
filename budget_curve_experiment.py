"""Budget-scaling experiment for exact posterior graph-alignment audits.

For the same 50 low-noise correlated Erdos-Renyi instances used in the paper,
compare local random-transposition Metropolis-Hastings with eight-replica
parallel tempering at matched local-proposal budgets. Exact posterior marginals
are recomputed once per instance. Four independent chains/runs are pooled at
all budgets.

The experiment uses Numba-compiled kernels for both methods. Reported proposal
counts, rather than runtime, are the implementation-independent primary budget.
Runtime is retained as a secondary within-implementation comparison.
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
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
from exact_posterior_benchmark import (  # noqa: E402
    exact_posterior_auto,
    generate_paired_instances,
    likelihood_beta,
)


@njit(cache=True)
def overlap_score_ab(A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> int:
    n = A.shape[0]
    score = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0 and B[pi[i], pi[j]] != 0:
                score += 1
    return score


@njit(cache=True)
def transposition_delta_ab(
    A: np.ndarray, B: np.ndarray, pi: np.ndarray, u: int, v: int
) -> int:
    a = pi[u]
    b = pi[v]
    delta = 0
    n = A.shape[0]
    for k in range(n):
        if k == u or k == v:
            continue
        pk = pi[k]
        delta += (A[u, k] - A[v, k]) * (B[b, pk] - B[a, pk])
    return int(delta)


@njit(cache=True)
def run_local_compiled(
    A: np.ndarray,
    B: np.ndarray,
    beta: float,
    iterations: int,
    burn: int,
    thin: int,
    seed: int,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    np.random.seed(seed)
    n = A.shape[0]
    pi = initial_state.copy()
    score = overlap_score_ab(A, B, pi)
    retained = (iterations - burn + thin - 1) // thin
    samples = np.empty((retained, n), dtype=np.int16)
    scores = np.empty(retained, dtype=np.int16)
    accepted = 0
    out = 0
    for t in range(iterations):
        u = np.random.randint(n)
        v = np.random.randint(n - 1)
        if v >= u:
            v += 1
        delta = transposition_delta_ab(A, B, pi, u, v)
        if delta >= 0 or math.log(np.random.random()) < beta * delta:
            tmp = pi[u]
            pi[u] = pi[v]
            pi[v] = tmp
            score += delta
            accepted += 1
        if t >= burn and (t - burn) % thin == 0:
            samples[out] = pi
            scores[out] = score
            out += 1
    return samples, scores, accepted


@njit(cache=True)
def run_pt_compiled(
    A: np.ndarray,
    B: np.ndarray,
    beta: float,
    sweeps: int,
    burn: int,
    thin: int,
    seed: int,
    initial_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    np.random.seed(seed)
    replicas = initial_states.shape[0]
    n = A.shape[0]
    states = initial_states.copy()
    state_scores = np.empty(replicas, dtype=np.int16)
    betas = np.empty(replicas, dtype=np.float64)
    for r in range(replicas):
        x = r / (replicas - 1)
        betas[r] = beta * x * x
        state_scores[r] = overlap_score_ab(A, B, states[r])

    retained = (sweeps - burn + thin - 1) // thin
    samples = np.empty((retained, n), dtype=np.int16)
    scores = np.empty(retained, dtype=np.int16)
    local_accepted = 0
    swap_attempted = 0
    swap_accepted = 0
    out = 0

    for t in range(sweeps):
        for r in range(replicas):
            u = np.random.randint(n)
            v = np.random.randint(n - 1)
            if v >= u:
                v += 1
            delta = transposition_delta_ab(A, B, states[r], u, v)
            if delta >= 0 or math.log(np.random.random()) < betas[r] * delta:
                tmp = states[r, u]
                states[r, u] = states[r, v]
                states[r, v] = tmp
                state_scores[r] += delta
                local_accepted += 1

        parity = t % 2
        for r in range(parity, replicas - 1, 2):
            log_ratio = (betas[r + 1] - betas[r]) * (
                state_scores[r] - state_scores[r + 1]
            )
            swap_attempted += 1
            if math.log(np.random.random()) < min(0.0, log_ratio):
                for i in range(n):
                    tmp = states[r, i]
                    states[r, i] = states[r + 1, i]
                    states[r + 1, i] = tmp
                tmp_score = state_scores[r]
                state_scores[r] = state_scores[r + 1]
                state_scores[r + 1] = tmp_score
                swap_accepted += 1

        if t >= burn and (t - burn) % thin == 0:
            samples[out] = states[replicas - 1]
            scores[out] = state_scores[replicas - 1]
            out += 1

    return samples, scores, local_accepted, swap_attempted, swap_accepted


def marginals(samples: np.ndarray, n: int) -> np.ndarray:
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        P[i] = np.bincount(samples[:, i], minlength=n) / len(samples)
    return P


def row_tv(P: np.ndarray, Q: np.ndarray) -> float:
    return float(0.5 * np.abs(P - Q).sum(axis=1).mean())


def split_rhat(chains: list[np.ndarray]) -> float:
    minimum = min(len(chain) for chain in chains)
    half = minimum // 2
    if half < 2:
        return float("nan")
    split = np.asarray(
        [part for chain in chains for part in (chain[:half], chain[-half:])],
        dtype=float,
    )
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = half * float(np.var(split.mean(axis=1), ddof=1))
    if within == 0.0:
        return 1.0 if between == 0.0 else float("inf")
    var_hat = (half - 1.0) / half * within + between / half
    return float(math.sqrt(var_hat / within))


def chain_disagreement(chain_samples: list[np.ndarray], n: int) -> float:
    Ps = [marginals(x, n) for x in chain_samples]
    values = []
    for i in range(len(Ps)):
        for j in range(i + 1, len(Ps)):
            values.append(row_tv(Ps[i], Ps[j]))
    return float(np.mean(values))


def seed_for(n: int, graph_seed: int, method_code: int, chain: int) -> int:
    return int(91_000_000 + n * 1_000_000 + graph_seed * 10_000 + method_code * 100 + chain)


def initial_local(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 17).permutation(n).astype(np.int64)


def initial_pt(n: int, replicas: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 31)
    return np.asarray([rng.permutation(n) for _ in range(replicas)], dtype=np.int64)


def warmup() -> None:
    A = np.zeros((4, 4), dtype=np.int8)
    A[0, 1] = A[1, 0] = 1
    B = A.copy()
    pi = np.arange(4, dtype=np.int64)
    run_local_compiled(A, B, 1.0, 4, 1, 1, 1, pi)
    states = np.vstack([pi.copy() for _ in range(8)])
    run_pt_compiled(A, B, 1.0, 2, 0, 1, 1, states)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warmup()
    rows: list[dict[str, object]] = []
    counts = {8: args.n8_instances, 9: args.n9_instances, 10: args.n10_instances}
    replicas = 8

    for n in args.n_values:
        for graph_seed in range(args.seed_start, counts[n]):
            instance = generate_paired_instances(n, args.p, [args.noise], graph_seed)[0]
            beta = likelihood_beta(args.p, args.noise)
            exact_start = time.perf_counter()
            exact = exact_posterior_auto(instance, beta)
            exact_runtime = time.perf_counter() - exact_start
            expected_score = float(np.dot(exact.probabilities, exact.scores))

            for budget in args.budgets:
                for method_code, method in enumerate(("local", "parallel_tempering")):
                    chain_samples: list[np.ndarray] = []
                    chain_scores: list[np.ndarray] = []
                    acceptances: list[float] = []
                    swap_acceptances: list[float] = []
                    runtime = 0.0
                    for chain in range(args.chains):
                        seed = seed_for(n, graph_seed, method_code, chain)
                        start = time.perf_counter()
                        if method == "local":
                            samples, scores, accepted = run_local_compiled(
                                instance.A,
                                instance.B,
                                beta,
                                int(budget),
                                int(budget * args.burn_fraction),
                                args.local_thin,
                                seed,
                                initial_local(n, seed),
                            )
                            acceptance = accepted / budget
                            swap_acceptance = float("nan")
                        else:
                            if budget % replicas:
                                raise ValueError("each budget must be divisible by replicas")
                            sweeps = budget // replicas
                            samples, scores, accepted, swap_attempted, swap_accepted = run_pt_compiled(
                                instance.A,
                                instance.B,
                                beta,
                                sweeps,
                                int(sweeps * args.burn_fraction),
                                args.pt_thin,
                                seed,
                                initial_pt(n, replicas, seed),
                            )
                            acceptance = accepted / budget
                            swap_acceptance = swap_accepted / max(swap_attempted, 1)
                        runtime += time.perf_counter() - start
                        chain_samples.append(samples)
                        chain_scores.append(scores.astype(float))
                        acceptances.append(float(acceptance))
                        swap_acceptances.append(float(swap_acceptance))

                    pooled = np.vstack(chain_samples)
                    P_hat = marginals(pooled, n)
                    scores_all = np.concatenate(chain_scores)
                    rows.append(
                        {
                            "n": n,
                            "graph_seed": graph_seed,
                            "noise": args.noise,
                            "p": args.p,
                            "beta_likelihood": beta,
                            "method": method,
                            "local_proposals_per_chain": budget,
                            "chains": args.chains,
                            "exact_runtime_seconds": exact_runtime,
                            "runtime_seconds": runtime,
                            "retained_samples_total": len(pooled),
                            "mean_row_tv": row_tv(P_hat, exact.marginals),
                            "score_split_rhat": split_rhat(chain_scores),
                            "chain_marginal_disagreement": chain_disagreement(chain_samples, n),
                            "mean_score_bias": float(scores_all.mean() - expected_score),
                            "mean_acceptance_rate": float(np.mean(acceptances)),
                            "mean_swap_acceptance": float(np.nanmean(swap_acceptances))
                            if np.any(np.isfinite(swap_acceptances))
                            else float("nan"),
                            "exact_expected_planted_accuracy": exact.planted_expected_accuracy,
                            "exact_map_count": exact.map_count,
                            "exact_effective_states": exact.effective_states,
                        }
                    )
            print(f"finished n={n}, seed={graph_seed}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "budget_curve_raw.csv", index=False)

    group_cols = ["n", "method", "local_proposals_per_chain"]
    summary_rows = []
    for key, df in raw.groupby(group_cols, sort=True):
        n, method, budget = key
        summary_rows.append(
            {
                "n": n,
                "method": method,
                "local_proposals_per_chain": budget,
                "instances": len(df),
                "mean_row_tv": df.mean_row_tv.mean(),
                "sem_row_tv": df.mean_row_tv.sem(),
                "median_row_tv": df.mean_row_tv.median(),
                "fraction_row_tv_below_0_1": (df.mean_row_tv < 0.1).mean(),
                "mean_score_split_rhat": df.score_split_rhat.replace([np.inf, -np.inf], np.nan).mean(),
                "mean_chain_marginal_disagreement": df.chain_marginal_disagreement.mean(),
                "mean_runtime_seconds": df.runtime_seconds.mean(),
                "sem_runtime_seconds": df.runtime_seconds.sem(),
                "mean_acceptance_rate": df.mean_acceptance_rate.mean(),
                "mean_swap_acceptance": df.mean_swap_acceptance.mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "budget_curve_summary.csv", index=False)

    pooled_rows = []
    for key, df in raw.groupby(["method", "local_proposals_per_chain"], sort=True):
        method, budget = key
        pooled_rows.append(
            {
                "method": method,
                "local_proposals_per_chain": budget,
                "instances": len(df),
                "mean_row_tv": df.mean_row_tv.mean(),
                "sem_row_tv": df.mean_row_tv.sem(),
                "fraction_row_tv_below_0_1": (df.mean_row_tv < 0.1).mean(),
                "mean_score_split_rhat": df.score_split_rhat.replace([np.inf, -np.inf], np.nan).mean(),
                "mean_chain_marginal_disagreement": df.chain_marginal_disagreement.mean(),
                "mean_runtime_seconds": df.runtime_seconds.mean(),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(args.output_dir / "budget_curve_pooled.csv", index=False)

    metadata = vars(args).copy()
    metadata["output_dir"] = str(metadata["output_dir"])
    with open(args.output_dir / "budget_curve_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return raw, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "budget_curves",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[1024, 4096, 16384, 65536, 262144]
    )
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--n8-instances", type=int, default=20)
    parser.add_argument("--n9-instances", type=int, default=20)
    parser.add_argument("--n10-instances", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--burn-fraction", type=float, default=0.25)
    parser.add_argument("--local-thin", type=int, default=5)
    parser.add_argument("--pt-thin", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raw, summary = run(parse_args())
    print(summary.to_string(index=False))
