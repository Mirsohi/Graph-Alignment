"""Connected rigid near-twin stress test for graph-alignment MCMC.

This reconstructs the experiment added in the current version of
"Good Scores, Wrong Marginals" from the paper's stated protocol and the older
reproducibility artifact.  The exact structural certificates are deterministic.
The original current-version chain-level random seeds were not available, so
this script fixes and records a new deterministic seed schedule for an
independent stochastic reproduction.

Construction
------------
For even block size m, sample H ~ G(m, 1/2) with NumPy's default_rng(seed),
then force e={0,1} present.  Let Hminus = H-e.  The 2m-vertex graph G consists
of disjoint copies of H and Hminus plus one bridge between corresponding
vertices.  The default bridge endpoint is vertex 1 in each block; it can be
changed with --bridge-vertex.  The block swap rho exchanges corresponding
vertices.

Protocol
--------
For each m in {6,8,10,12,14}, scan seeds from zero and retain the first seed for
which H and Hminus are connected, bridgeless, and rigid.  Run four chains from
opposing starts (two identity, two rho), with beta=2, 150,000 proposals, 20%
burn-in, thinning by 10.  Compare random transpositions against a mixture that
uses the exact rho jump with probability 0.05.

Outputs
-------
- structural_certificates.csv: exact graph and theorem certificates.
- sampler_chains.csv: one row per chain and method.
- sampler_summary.csv: opposing-start marginal TV and aggregate diagnostics.
- sampler_summary_paper_reported.csv: values transcribed from Table 5, clearly
  marked as paper-reported rather than regenerated.
- reproduction_comparison.csv: regenerated versus paper-reported values.

Numba is used to keep the 6 million-proposal experiment fast.  Randomness is
inside the compiled kernel and is fully determined by the recorded integer
chain seeds for the pinned environment.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from numba import njit


DEFAULT_BLOCK_SIZES = (6, 8, 10, 12, 14)
DELETED_EDGE = (0, 1)


@dataclass(frozen=True)
class Certificate:
    m: int
    seed: int
    H: np.ndarray
    Hminus: np.ndarray
    G: np.ndarray
    bridge_vertex: int
    h_H: int
    h_Hminus: int


# ---------------------------------------------------------------------------
# Graph construction and exact certificates
# ---------------------------------------------------------------------------


def sample_forced_edge_graph(
    m: int,
    seed: int,
    p: float = 0.5,
    edge: tuple[int, int] = DELETED_EDGE,
) -> np.ndarray:
    """Draw every upper-triangle Bernoulli edge, then force ``edge`` present.

    Drawing the full upper triangle before forcing the edge is important: this
    reproduces the seeds and edge counts reported in Table 5.
    """
    if m < 2:
        raise ValueError("m must be at least 2")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0,1)")
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(m, k=1)
    values = (rng.random(len(rows)) < p).astype(np.int8)
    A = np.zeros((m, m), dtype=np.int8)
    A[rows, cols] = values
    A[cols, rows] = values
    u, v = edge
    A[u, v] = A[v, u] = 1
    return A


def delete_edge(A: np.ndarray, edge: tuple[int, int] = DELETED_EDGE) -> np.ndarray:
    B = np.asarray(A, dtype=np.int8).copy()
    u, v = edge
    B[u, v] = B[v, u] = 0
    return B


def is_connected(A: np.ndarray) -> bool:
    return nx.is_connected(nx.from_numpy_array(A))


def is_bridgeless(A: np.ndarray) -> bool:
    return not any(nx.bridges(nx.from_numpy_array(A)))


def is_rigid(A: np.ndarray) -> bool:
    """Return whether the graph has only the identity automorphism."""
    graph = nx.from_numpy_array(A)
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, graph)
    automorphisms = matcher.isomorphisms_iter()
    first = next(automorphisms, None)
    if first is None:
        raise RuntimeError("self-isomorphism iterator unexpectedly empty")
    return next(automorphisms, None) is None


def exact_balanced_cut_width(A: np.ndarray) -> int:
    """Compute min |delta(U)| over all |U|=m/2 by exact enumeration."""
    m = A.shape[0]
    if m % 2:
        raise ValueError("balanced cut width is used here only for even m")
    best = math.inf
    vertices = range(m)
    # U and its complement define the same cut.  Fix vertex zero in U to halve
    # the enumeration without changing the optimum.
    for rest in itertools.combinations(range(1, m), m // 2 - 1):
        U = (0, *rest)
        mask = np.zeros(m, dtype=bool)
        mask[list(U)] = True
        cut = int(A[np.ix_(mask, ~mask)].sum())
        if cut < best:
            best = cut
    return int(best)


def build_near_twin_graph(
    H: np.ndarray,
    Hminus: np.ndarray,
    bridge_vertex: int = 1,
) -> np.ndarray:
    m = H.shape[0]
    if H.shape != Hminus.shape:
        raise ValueError("H and Hminus must have the same shape")
    if not 0 <= bridge_vertex < m:
        raise ValueError("bridge_vertex must be in 0,...,m-1")
    G = np.zeros((2 * m, 2 * m), dtype=np.int8)
    G[:m, :m] = H
    G[m:, m:] = Hminus
    G[bridge_vertex, m + bridge_vertex] = 1
    G[m + bridge_vertex, bridge_vertex] = 1
    return G


def block_swap(m: int) -> np.ndarray:
    return np.concatenate((np.arange(m, 2 * m), np.arange(m))).astype(np.int64)


def first_certificate(m: int, bridge_vertex: int = 1) -> Certificate:
    seed = 0
    while True:
        H = sample_forced_edge_graph(m, seed)
        Hminus = delete_edge(H)
        qualifies = (
            is_connected(H)
            and is_connected(Hminus)
            and is_bridgeless(H)
            and is_bridgeless(Hminus)
            and is_rigid(H)
            and is_rigid(Hminus)
        )
        if qualifies:
            G = build_near_twin_graph(H, Hminus, bridge_vertex)
            return Certificate(
                m=m,
                seed=seed,
                H=H,
                Hminus=Hminus,
                G=G,
                bridge_vertex=bridge_vertex,
                h_H=exact_balanced_cut_width(H),
                h_Hminus=exact_balanced_cut_width(Hminus),
            )
        seed += 1


# ---------------------------------------------------------------------------
# Fast exact-score MCMC
# ---------------------------------------------------------------------------


@njit(cache=True)
def overlap_score(A: np.ndarray, pi: np.ndarray) -> int:
    n = A.shape[0]
    score = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0 and A[pi[i], pi[j]] != 0:
                score += 1
    return score


@njit(cache=True)
def transposition_delta(A: np.ndarray, pi: np.ndarray, u: int, v: int) -> int:
    """Exact overlap-score change after swapping pi[u] and pi[v]."""
    n = A.shape[0]
    a = pi[u]
    b = pi[v]
    delta = 0
    for k in range(n):
        if k == u or k == v:
            continue
        pk = pi[k]
        delta += A[u, k] * (A[b, pk] - A[a, pk])
        delta += A[v, k] * (A[a, pk] - A[b, pk])
    return int(delta)


@njit(cache=True)
def run_chain_compiled(
    A: np.ndarray,
    beta: float,
    iterations: int,
    burn: int,
    thin: int,
    seed: int,
    initial_state: np.ndarray,
    jump_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Run one chain; Numba's RNG is seeded explicitly for reproducibility."""
    np.random.seed(seed)
    n = A.shape[0]
    m = n // 2
    pi = initial_state.copy()
    score = overlap_score(A, pi)
    retained = (iterations - burn + thin - 1) // thin
    samples = np.empty((retained, n), dtype=np.int16)
    scores = np.empty(retained, dtype=np.int16)
    block_counts = np.empty(retained, dtype=np.int16)

    accepted = 0
    jump_attempted = 0
    jump_accepted = 0
    out = 0

    for t in range(iterations):
        do_jump = jump_probability > 0.0 and np.random.random() < jump_probability
        if do_jump:
            jump_attempted += 1
            proposed = np.empty(n, dtype=np.int64)
            for i in range(n):
                x = pi[i]
                proposed[i] = x + m if x < m else x - m
            proposed_score = overlap_score(A, proposed)
            delta = proposed_score - score
            if delta >= 0 or math.log(np.random.random()) < beta * delta:
                pi = proposed
                score = proposed_score
                accepted += 1
                jump_accepted += 1
        else:
            u = np.random.randint(n)
            v = np.random.randint(n - 1)
            if v >= u:
                v += 1
            delta = transposition_delta(A, pi, u, v)
            if delta >= 0 or math.log(np.random.random()) < beta * delta:
                temp = pi[u]
                pi[u] = pi[v]
                pi[v] = temp
                score += delta
                accepted += 1

        if t >= burn and (t - burn) % thin == 0:
            crossing = 0
            for i in range(m):
                if pi[i] >= m:
                    crossing += 1
            samples[out] = pi
            scores[out] = score
            block_counts[out] = crossing
            out += 1

    return samples, scores, block_counts, accepted, jump_attempted, jump_accepted


def samples_to_marginals(samples: np.ndarray, n: int) -> np.ndarray:
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        P[i] = np.bincount(samples[:, i], minlength=n) / len(samples)
    return P


def mean_row_tv(P: np.ndarray, Q: np.ndarray) -> float:
    return float(0.5 * np.abs(P - Q).sum(axis=1).mean())


def split_rhat(chains: Iterable[np.ndarray]) -> float:
    split: list[np.ndarray] = []
    for values in chains:
        x = np.asarray(values, dtype=float)
        half = len(x) // 2
        if half < 2:
            return float("nan")
        split.extend((x[:half], x[-half:]))
    common = min(len(x) for x in split)
    matrix = np.asarray([x[:common] for x in split], dtype=float)
    chain_means = matrix.mean(axis=1)
    within = matrix.var(axis=1, ddof=1).mean()
    if within == 0.0:
        return 1.0 if np.allclose(chain_means, chain_means[0]) else float("inf")
    between = common * chain_means.var(ddof=1)
    variance = (common - 1) / common * within + between / common
    return float(math.sqrt(variance / within))


def orientation_switches(block_counts: np.ndarray, m: int) -> int:
    """Count changes between R<m/2 and R>m/2, ignoring midpoint samples."""
    previous = 0
    switches = 0
    for value in block_counts:
        current = -1 if value < m / 2 else (1 if value > m / 2 else 0)
        if current == 0:
            continue
        if previous != 0 and current != previous:
            switches += 1
        previous = current
    return switches


def chain_seed(master_seed: int, m: int, method_code: int, chain: int) -> int:
    """Simple recorded seed schedule used by the reconstructed experiment."""
    return int(master_seed + 100_000 * m + 10_000 * method_code + chain)


def warm_up_compiled_kernel(A: np.ndarray) -> None:
    """Compile the Numba kernel before any runtime measurement."""
    n = A.shape[0]
    initial = np.arange(n, dtype=np.int64)
    run_chain_compiled(A, 1.0, 2, 0, 1, 123, initial, 0.0)


def run_sampler_experiment(
    certificate: Certificate,
    beta: float,
    iterations: int,
    burn_fraction: float,
    thin: int,
    jump_probability: float,
    master_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    G = certificate.G
    warm_up_compiled_kernel(G)
    m = certificate.m
    n = 2 * m
    identity = np.arange(n, dtype=np.int64)
    rho = block_swap(m)

    chain_rows: list[dict[str, object]] = []
    retained_by_method: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for method_code, (method, jump_prob) in enumerate(
        (("local", 0.0), ("near_twin_jump", jump_probability))
    ):
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for chain in range(4):
            start_label = "identity" if chain < 2 else "block_swap"
            initial = identity if chain < 2 else rho
            seed = chain_seed(master_seed, m, method_code, chain)
            start = time.perf_counter()
            samples, scores, block_counts, accepted, j_attempted, j_accepted = (
                run_chain_compiled(
                    G,
                    beta,
                    iterations,
                    int(iterations * burn_fraction),
                    thin,
                    seed,
                    initial,
                    jump_prob,
                )
            )
            runtime = time.perf_counter() - start
            outputs.append((samples, scores, block_counts))
            visits_left = bool(np.any(block_counts < m / 2))
            visits_right = bool(np.any(block_counts > m / 2))
            chain_rows.append(
                {
                    "m": m,
                    "n": n,
                    "graph_seed": certificate.seed,
                    "bridge_vertex": certificate.bridge_vertex,
                    "method": method,
                    "chain": chain,
                    "start": start_label,
                    "chain_seed": seed,
                    "beta": beta,
                    "iterations": iterations,
                    "burn_fraction": burn_fraction,
                    "thin": thin,
                    "retained_samples": len(samples),
                    "acceptance_rate_all_moves": accepted / iterations,
                    "jump_attempts": j_attempted,
                    "jump_acceptance": j_accepted / j_attempted
                    if j_attempted
                    else float("nan"),
                    "mean_score": float(scores.mean()),
                    "score_split_rhat_single_chain": split_rhat([scores]),
                    "visits_left_orientation": visits_left,
                    "visits_right_orientation": visits_right,
                    "traverses_both_orientations": visits_left and visits_right,
                    "post_burn_mode_switches": orientation_switches(block_counts, m),
                    "min_R": int(block_counts.min()),
                    "max_R": int(block_counts.max()),
                    "runtime_seconds": runtime,
                }
            )
        retained_by_method[method] = outputs

    chains = pd.DataFrame(chain_rows)
    summary_rows: list[dict[str, object]] = []
    for method, outputs in retained_by_method.items():
        left_samples = np.vstack((outputs[0][0], outputs[1][0]))
        right_samples = np.vstack((outputs[2][0], outputs[3][0]))
        P_left = samples_to_marginals(left_samples, n)
        P_right = samples_to_marginals(right_samples, n)
        method_chains = chains[(chains.m == m) & (chains.method == method)]
        rhat_identity = split_rhat((outputs[0][1], outputs[1][1]))
        rhat_swap = split_rhat((outputs[2][1], outputs[3][1]))
        summary_rows.append(
            {
                "m": m,
                "n": n,
                "graph_seed": certificate.seed,
                "bridge_vertex": certificate.bridge_vertex,
                "method": method,
                "opposing_start_row_tv": mean_row_tv(P_left, P_right),
                "traverse_count": int(method_chains.traverses_both_orientations.sum()),
                "post_burn_mode_switches": int(
                    method_chains.post_burn_mode_switches.sum()
                ),
                "mean_within_start_score_split_rhat": float(
                    np.mean((rhat_identity, rhat_swap))
                ),
                "mean_acceptance_rate_all_moves": float(
                    method_chains.acceptance_rate_all_moves.mean()
                ),
                "mean_jump_acceptance": float(method_chains.jump_acceptance.mean()),
                "total_runtime_seconds": float(method_chains.runtime_seconds.sum()),
            }
        )
    return chains, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Output assembly and validation
# ---------------------------------------------------------------------------


def certificate_row(cert: Certificate, beta: float) -> dict[str, object]:
    H = cert.H
    Hminus = cert.Hminus
    G = cert.G
    m = cert.m
    n = 2 * m
    identity = np.arange(n, dtype=np.int64)
    rho = block_swap(m)
    graph = nx.from_numpy_array(G)
    bridges = list(nx.bridges(graph))
    M = int(np.triu(G, k=1).sum())
    delta_m = cert.h_H + cert.h_Hminus - 1
    state_count_midpoint = math.factorial(m) ** 2 * math.comb(m, m // 2) ** 2
    log_lower_bound = beta * (delta_m - 1) - math.log(4.0) - math.log(
        state_count_midpoint
    )
    return {
        "m": m,
        "n": n,
        "seed": cert.seed,
        "deleted_edge_u": DELETED_EDGE[0],
        "deleted_edge_v": DELETED_EDGE[1],
        "bridge_vertex": cert.bridge_vertex,
        "edges_H": int(np.triu(H, k=1).sum()),
        "edges_Hminus": int(np.triu(Hminus, k=1).sum()),
        "balanced_cut_H": cert.h_H,
        "balanced_cut_Hminus": cert.h_Hminus,
        "delta_m": delta_m,
        "random_family_threshold_m2_over_8_minus_3": m * m / 8.0 - 3.0,
        "H_connected": is_connected(H),
        "Hminus_connected": is_connected(Hminus),
        "H_bridgeless": is_bridgeless(H),
        "Hminus_bridgeless": is_bridgeless(Hminus),
        "H_rigid": is_rigid(H),
        "Hminus_rigid": is_rigid(Hminus),
        "G_connected": nx.is_connected(graph),
        "G_rigid": is_rigid(G),
        "G_bridge_count": len(bridges),
        "G_unique_bridge": len(bridges) == 1,
        "G_edges_M": M,
        "identity_score": overlap_score(G, identity),
        "block_swap_score": overlap_score(G, rho),
        "near_mode_gap": overlap_score(G, identity) - overlap_score(G, rho),
        "beta": beta,
        "log_mixing_lower_bound": log_lower_bound,
        "mixing_lower_bound": math.exp(log_lower_bound)
        if log_lower_bound < 700
        else float("inf"),
    }


def paper_reported_summary() -> pd.DataFrame:
    rows = [
        (6, 12, 228, 0.156, 0.029, 4, 4),
        (8, 16, 1, 0.999, 0.105, 0, 4),
        (10, 20, 0, 0.996, 0.042, 0, 4),
        (12, 24, 0, 1.000, 0.027, 0, 4),
        (14, 28, 0, 1.000, 0.013, 0, 4),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "m",
            "n",
            "graph_seed",
            "paper_local_tv",
            "paper_jump_tv",
            "paper_local_traverse_count",
            "paper_jump_traverse_count",
        ],
    )


def validate_certificates(frame: pd.DataFrame) -> None:
    expected = pd.DataFrame(
        {
            "m": [6, 8, 10, 12, 14],
            "seed": [228, 1, 0, 0, 0],
            "edges_H": [9, 16, 20, 33, 43],
            "edges_Hminus": [8, 15, 19, 32, 42],
            "balanced_cut_H": [4, 6, 7, 12, 16],
            "balanced_cut_Hminus": [3, 5, 7, 12, 15],
            "delta_m": [6, 10, 13, 23, 30],
            "near_mode_gap": [1, 1, 1, 1, 1],
        }
    )
    columns = list(expected.columns)
    actual = frame[columns].reset_index(drop=True)
    if not actual.equals(expected):
        raise AssertionError(
            "structural reconstruction does not match Table 5\n"
            f"expected:\n{expected}\nactual:\n{actual}"
        )
    boolean_columns = [
        "H_connected",
        "Hminus_connected",
        "H_bridgeless",
        "Hminus_bridgeless",
        "H_rigid",
        "Hminus_rigid",
        "G_connected",
        "G_rigid",
        "G_unique_bridge",
    ]
    if not frame[boolean_columns].to_numpy(dtype=bool).all():
        raise AssertionError("at least one structural certificate failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tmp/rigid_near_twin")
    )
    parser.add_argument(
        "--block-sizes", type=int, nargs="+", default=list(DEFAULT_BLOCK_SIZES)
    )
    parser.add_argument("--bridge-vertex", type=int, default=1)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=150_000)
    parser.add_argument("--burn-fraction", type=float, default=0.20)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--jump-probability", type=float, default=0.05)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument(
        "--certificates-only",
        action="store_true",
        help="skip MCMC and produce only exact structural certificates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    certificates = [
        first_certificate(m, bridge_vertex=args.bridge_vertex)
        for m in args.block_sizes
    ]
    structural = pd.DataFrame(
        [certificate_row(cert, beta=args.beta) for cert in certificates]
    )
    structural.to_csv(args.output_dir / "structural_certificates.csv", index=False)
    if tuple(args.block_sizes) == DEFAULT_BLOCK_SIZES:
        validate_certificates(structural)

    reported = paper_reported_summary()
    reported.to_csv(
        args.output_dir / "sampler_summary_paper_reported.csv", index=False
    )

    if args.certificates_only:
        print(structural.to_string(index=False))
        return

    chain_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for cert in certificates:
        chains, summary = run_sampler_experiment(
            cert,
            beta=args.beta,
            iterations=args.iterations,
            burn_fraction=args.burn_fraction,
            thin=args.thin,
            jump_probability=args.jump_probability,
            master_seed=args.master_seed,
        )
        chain_frames.append(chains)
        summary_frames.append(summary)

    chains = pd.concat(chain_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    chains.to_csv(args.output_dir / "sampler_chains.csv", index=False)
    summary.to_csv(args.output_dir / "sampler_summary.csv", index=False)

    wide = summary.pivot(
        index=["m", "n", "graph_seed"],
        columns="method",
        values=["opposing_start_row_tv", "traverse_count"],
    )
    wide.columns = ["_".join(reversed(column)) for column in wide.columns]
    wide = wide.reset_index()
    comparison = reported.merge(wide, on=["m", "n", "graph_seed"], how="left")
    comparison["local_tv_difference"] = (
        comparison["local_opposing_start_row_tv"] - comparison["paper_local_tv"]
    )
    comparison["jump_tv_difference"] = (
        comparison["near_twin_jump_opposing_start_row_tv"]
        - comparison["paper_jump_tv"]
    )
    comparison.to_csv(args.output_dir / "reproduction_comparison.csv", index=False)

    print("Exact structural certificates:")
    print(
        structural[
            [
                "n",
                "seed",
                "edges_H",
                "edges_Hminus",
                "balanced_cut_H",
                "balanced_cut_Hminus",
                "delta_m",
                "near_mode_gap",
            ]
        ].to_string(index=False)
    )
    print("\nIndependent stochastic reproduction:")
    print(
        summary[
            [
                "n",
                "method",
                "opposing_start_row_tv",
                "traverse_count",
                "post_burn_mode_switches",
                "mean_within_start_score_split_rhat",
                "mean_jump_acceptance",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
