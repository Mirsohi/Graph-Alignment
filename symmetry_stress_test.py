"""Evaluate automorphism-orbit averaging on the exact n=8 benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from exact_posterior_benchmark import (
    exact_posterior,
    generate_paired_instances,
    likelihood_beta,
    random_transposition_mh,
    samples_to_marginals,
)
from symmetry_averaging import automorphism_vertex_orbits, orbit_average_marginals


def mean_row_tv(P: np.ndarray, Q: np.ndarray) -> float:
    return float(0.5 * np.abs(P - Q).sum(axis=1).mean())


def main() -> None:
    rows = []
    for seed_index, seed in enumerate(range(20)):
        instance = generate_paired_instances(8, 0.3, [0.1], seed)[0]
        beta = likelihood_beta(instance.p, instance.noise)
        exact = exact_posterior(instance, beta)
        outputs = []
        for chain in range(4):
            chain_seed = (
                10_000_019 * (seed_index + 1)
                + 100_003 * 100
                + 997 * chain
            )
            outputs.append(
                random_transposition_mh(
                    instance.A,
                    instance.B,
                    beta,
                    20_000,
                    0.25,
                    np.random.default_rng(chain_seed),
                    sample_every=5,
                )
            )
        empirical = samples_to_marginals(
            np.vstack([output.samples for output in outputs]), instance.n
        )
        row_orbits = automorphism_vertex_orbits(instance.A)
        column_orbits = automorphism_vertex_orbits(instance.B)
        averaged = orbit_average_marginals(empirical, row_orbits, column_orbits)
        exact_averaged = orbit_average_marginals(
            exact.marginals, row_orbits, column_orbits
        )
        assert np.allclose(exact_averaged, exact.marginals, atol=1e-10)
        rows.append(
            {
                "seed": seed,
                "raw_tv": mean_row_tv(empirical, exact.marginals),
                "orbit_averaged_tv": mean_row_tv(averaged, exact.marginals),
                "a_orbit_count": len(row_orbits),
                "b_orbit_count": len(column_orbits),
                "a_largest_orbit": max(map(len, row_orbits)),
                "b_largest_orbit": max(map(len, column_orbits)),
                "exact_map_count": exact.map_count,
                "exact_expected_planted_accuracy": exact.planted_expected_accuracy,
            }
        )
        print(f"completed seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "results" / "symmetry_stress_test" / "symmetry_benchmark_n8.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False))
    print(frame[["raw_tv", "orbit_averaged_tv"]].mean())


if __name__ == "__main__":
    main()
