"""Family-level replication of the connected rigid near-twin obstruction.

For each requested even block size m, scan integer graph seeds and retain the
first `instances_per_size` graphs whose H and H-e blocks are connected,
bridgeless, and rigid. Build G(H,e), certify balanced cut widths, and run the
same four opposing-start chains used in the paper for local transpositions and
for the 5% exact near-twin jump.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from rigid_near_twin import (  # noqa: E402
    DELETED_EDGE,
    block_swap,
    build_near_twin_graph,
    exact_balanced_cut_width,
    is_bridgeless,
    is_connected,
    is_rigid,
    mean_row_tv,
    orientation_switches,
    overlap_score,
    run_chain_compiled,
    sample_forced_edge_graph,
    samples_to_marginals,
    split_rhat,
    transposition_delta,
    delete_edge,
    warm_up_compiled_kernel,
)


def qualifying_graphs(m: int, count: int, bridge_vertex: int) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    seed = 0
    while len(found) < count:
        H = sample_forced_edge_graph(m, seed)
        Hminus = delete_edge(H)
        if (
            is_connected(H)
            and is_connected(Hminus)
            and is_bridgeless(H)
            and is_bridgeless(Hminus)
            and is_rigid(H)
            and is_rigid(Hminus)
        ):
            G = build_near_twin_graph(H, Hminus, bridge_vertex)
            graph = nx.from_numpy_array(G)
            if not nx.is_connected(graph) or not is_rigid(G):
                raise AssertionError("block certificate failed to make G connected and rigid")
            h_H = exact_balanced_cut_width(H)
            h_Hminus = exact_balanced_cut_width(Hminus)
            identity = np.arange(2 * m, dtype=np.int64)
            rho = block_swap(m)
            M = overlap_score(G, identity)
            near_score = overlap_score(G, rho)
            escape_losses = []
            for u in range(2 * m):
                for v in range(u + 1, 2 * m):
                    escape_losses.append(-transposition_delta(G, identity, u, v))
            positive_losses = [x for x in escape_losses if x > 0]
            found.append(
                {
                    "m": m,
                    "n": 2 * m,
                    "family_index": len(found),
                    "graph_seed": seed,
                    "H": H,
                    "Hminus": Hminus,
                    "G": G,
                    "graph_sha256": hashlib.sha256(G.tobytes()).hexdigest(),
                    "edges_H": int(np.triu(H, 1).sum()),
                    "edges_Hminus": int(np.triu(Hminus, 1).sum()),
                    "edges_G": int(np.triu(G, 1).sum()),
                    "edge_density_G": float(np.triu(G, 1).sum() / math.comb(2 * m, 2)),
                    "balanced_cut_H": h_H,
                    "balanced_cut_Hminus": h_Hminus,
                    "delta_m": h_H + h_Hminus - 1,
                    "identity_score": M,
                    "block_swap_score": near_score,
                    "near_mode_gap": M - near_score,
                    "minimum_positive_one_swap_loss_from_identity": min(positive_losses),
                }
            )
        seed += 1
    return found


def chain_seed(master_seed: int, m: int, family_index: int, method_code: int, chain: int) -> int:
    return int(master_seed + m * 10_000_000 + family_index * 100_000 + method_code * 10_000 + chain)


def first_switch_retained(block_counts: np.ndarray, m: int, thin: int) -> float:
    prior = 0
    for index, value in enumerate(block_counts):
        orientation = -1 if value < m / 2 else (1 if value > m / 2 else 0)
        if orientation == 0:
            continue
        if prior != 0 and orientation != prior:
            return float(index * thin)
        prior = orientation
    return float("nan")


def run_graph(
    record: dict[str, object],
    beta: float,
    iterations: int,
    burn_fraction: float,
    thin: int,
    jump_probability: float,
    master_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = int(record["m"])
    n = 2 * m
    family_index = int(record["family_index"])
    G = np.asarray(record["G"], dtype=np.int8)
    identity = np.arange(n, dtype=np.int64)
    rho = block_swap(m)
    chain_rows: list[dict[str, object]] = []
    outputs_by_method: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for method_code, (method, jump_prob) in enumerate(
        (("local", 0.0), ("near_twin_jump", jump_probability))
    ):
        outputs = []
        for chain in range(4):
            initial = identity if chain < 2 else rho
            start_label = "identity" if chain < 2 else "block_swap"
            seed = chain_seed(master_seed, m, family_index, method_code, chain)
            start = time.perf_counter()
            samples, scores, block_counts, accepted, jump_attempted, jump_accepted = run_chain_compiled(
                G,
                beta,
                iterations,
                int(iterations * burn_fraction),
                thin,
                seed,
                initial,
                jump_prob,
            )
            runtime = time.perf_counter() - start
            outputs.append((samples, scores, block_counts))
            left = bool(np.any(block_counts < m / 2))
            right = bool(np.any(block_counts > m / 2))
            chain_rows.append(
                {
                    "m": m,
                    "n": n,
                    "family_index": family_index,
                    "graph_seed": int(record["graph_seed"]),
                    "graph_sha256": record["graph_sha256"],
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
                    "jump_attempts": jump_attempted,
                    "jump_acceptance": jump_accepted / jump_attempted if jump_attempted else float("nan"),
                    "mean_score": float(scores.mean()),
                    "visits_left_orientation": left,
                    "visits_right_orientation": right,
                    "traverses_both_orientations": left and right,
                    "post_burn_mode_switches": orientation_switches(block_counts, m),
                    "first_switch_retained_proposals": first_switch_retained(block_counts, m, thin),
                    "min_R": int(block_counts.min()),
                    "max_R": int(block_counts.max()),
                    "runtime_seconds": runtime,
                }
            )
        outputs_by_method[method] = outputs

    chains = pd.DataFrame(chain_rows)
    summaries = []
    for method, outputs in outputs_by_method.items():
        P_identity = samples_to_marginals(np.vstack((outputs[0][0], outputs[1][0])), n)
        P_swap = samples_to_marginals(np.vstack((outputs[2][0], outputs[3][0])), n)
        selected = chains[chains.method == method]
        summaries.append(
            {
                "m": m,
                "n": n,
                "family_index": family_index,
                "graph_seed": int(record["graph_seed"]),
                "graph_sha256": record["graph_sha256"],
                "method": method,
                "opposing_start_row_tv": mean_row_tv(P_identity, P_swap),
                "traverse_count": int(selected.traverses_both_orientations.sum()),
                "post_burn_mode_switches": int(selected.post_burn_mode_switches.sum()),
                "mean_within_start_score_split_rhat": float(
                    np.mean(
                        [
                            split_rhat((outputs[0][1], outputs[1][1])),
                            split_rhat((outputs[2][1], outputs[3][1])),
                        ]
                    )
                ),
                "mean_acceptance_rate_all_moves": float(selected.acceptance_rate_all_moves.mean()),
                "mean_jump_acceptance": float(selected.jump_acceptance.mean()),
                "mean_runtime_seconds": float(selected.runtime_seconds.sum()),
                "delta_m": int(record["delta_m"]),
                "edge_density_G": float(record["edge_density_G"]),
                "one_swap_escape_loss": int(record["minimum_positive_one_swap_loss_from_identity"]),
            }
        )
    return chains, pd.DataFrame(summaries)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Compile once outside timing.
    test_H = sample_forced_edge_graph(args.m_values[0], 0)
    test_G = build_near_twin_graph(test_H, delete_edge(test_H), args.bridge_vertex)
    warm_up_compiled_kernel(test_G)

    structures = []
    chain_frames = []
    summary_frames = []
    for m in args.m_values:
        graphs = qualifying_graphs(m, args.instances_per_size, args.bridge_vertex)
        for record in graphs:
            structures.append({k: v for k, v in record.items() if k not in {"H", "Hminus", "G"}})
            chains, summary = run_graph(
                record,
                beta=args.beta,
                iterations=args.iterations,
                burn_fraction=args.burn_fraction,
                thin=args.thin,
                jump_probability=args.jump_probability,
                master_seed=args.master_seed,
            )
            chain_frames.append(chains)
            summary_frames.append(summary)
            print(
                f"finished m={m}, family_index={record['family_index']}, seed={record['graph_seed']}",
                flush=True,
            )

    structural = pd.DataFrame(structures)
    chains = pd.concat(chain_frames, ignore_index=True)
    per_graph = pd.concat(summary_frames, ignore_index=True)
    structural.to_csv(args.output_dir / "family_structural_certificates.csv", index=False)
    chains.to_csv(args.output_dir / "family_chain_results.csv", index=False)
    per_graph.to_csv(args.output_dir / "family_sampler_results.csv", index=False)

    aggregate_rows = []
    for (m, method), df in per_graph.groupby(["m", "method"], sort=True):
        graph_chains = chains[(chains.m == m) & (chains.method == method)]
        aggregate_rows.append(
            {
                "m": m,
                "n": 2 * m,
                "method": method,
                "graphs": len(df),
                "mean_opposing_start_row_tv": df.opposing_start_row_tv.mean(),
                "sem_opposing_start_row_tv": df.opposing_start_row_tv.sem(),
                "median_opposing_start_row_tv": df.opposing_start_row_tv.median(),
                "minimum_opposing_start_row_tv": df.opposing_start_row_tv.min(),
                "maximum_opposing_start_row_tv": df.opposing_start_row_tv.max(),
                "fraction_graphs_no_chain_traverses": (df.traverse_count == 0).mean(),
                "fraction_chains_traverse": graph_chains.traverses_both_orientations.mean(),
                "total_post_burn_mode_switches": graph_chains.post_burn_mode_switches.sum(),
                "mean_within_start_score_split_rhat": df.mean_within_start_score_split_rhat.mean(),
                "mean_jump_acceptance": df.mean_jump_acceptance.mean(),
                "mean_delta_m": df.delta_m.mean(),
                "sem_delta_m": df.delta_m.sem(),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.output_dir / "family_aggregate_summary.csv", index=False)
    print(aggregate.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "family_replication"
    )
    parser.add_argument("--m-values", type=int, nargs="+", default=[8, 10, 12, 14])
    parser.add_argument("--instances-per-size", type=int, default=20)
    parser.add_argument("--bridge-vertex", type=int, default=1)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=150_000)
    parser.add_argument("--burn-fraction", type=float, default=0.20)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--jump-probability", type=float, default=0.05)
    parser.add_argument("--master-seed", type=int, default=7_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
