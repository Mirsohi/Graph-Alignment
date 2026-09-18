"""Regenerate the four figures used in the LoG 2026 extended abstract.

The script reads the checked-in canonical CSV outputs for Figures 1--2 and
reconstructs the fixed small-instance visualizations for Figures 3--4.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from exact_posterior_benchmark import (  # noqa: E402
    exact_posterior,
    generate_paired_instances,
    likelihood_beta,
    parallel_tempering,
    random_transposition_mh,
    samples_to_marginals,
)
from symmetry_averaging import (  # noqa: E402
    automorphism_vertex_orbits,
    orbit_average_marginals,
)

RESULTS = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2D5B8A"
ORANGE = "#D9782D"
GREEN = "#3A8D6D"
PURPLE = "#7156A5"
RED = "#B95043"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def budget_with_iid_floor() -> None:
    raw = pd.read_csv(RESULTS / "budget_curves" / "budget_curve_raw.csv")
    floors = pd.read_csv(RESULTS / "floor_initialization" / "iid_floor_raw.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.25), sharey=True)
    specs = [
        ("local", "local MH", "o", BLUE, ORANGE),
        ("parallel_tempering", "parallel tempering", "s", GREEN, RED),
    ]
    for ax, n in zip(axes, [8, 9, 10]):
        for method, label, marker, color, floor_color in specs:
            d = (
                raw[(raw.n == n) & (raw.method == method)]
                .groupby("local_proposals_per_chain")
                .mean_row_tv.agg(["mean", "sem"])
                .reset_index()
            )
            f = (
                floors[(floors.n == n) & (floors.method == method)]
                .groupby("local_proposals_per_chain")
                .iid_floor_mean.agg(["mean", "sem"])
                .reset_index()
            )
            ax.errorbar(
                d.local_proposals_per_chain,
                d["mean"],
                yerr=d["sem"],
                marker=marker,
                color=color,
                linewidth=1.6,
                markersize=4.5,
                capsize=2,
                label=label,
            )
            ax.plot(
                f.local_proposals_per_chain,
                f["mean"],
                linestyle="--",
                color=floor_color,
                linewidth=1.25,
                label=f"{label} i.i.d. floor",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(fr"$n={n}$")
        ax.set_xlabel("within-temperature proposals per run")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("mean row-TV (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "budget_with_iid_floor")


def near_twin_family() -> None:
    structural = pd.read_csv(RESULTS / "family_replication" / "family_structural_certificates.csv")
    aggregate = pd.read_csv(RESULTS / "family_replication" / "family_aggregate_summary.csv")
    chains = pd.read_csv(RESULTS / "family_replication" / "family_chain_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), constrained_layout=True)
    rng = np.random.default_rng(0)

    ax = axes[0]
    for n in sorted(structural.n.unique()):
        d = structural[structural.n == n]
        jitter = rng.uniform(-0.25, 0.25, len(d))
        ax.scatter(np.full(len(d), n) + jitter, d.delta_m, s=18, alpha=0.50, color=GREEN)
    means = structural.groupby("n").delta_m.mean()
    sems = structural.groupby("n").delta_m.sem()
    ns = means.index.to_numpy()
    ax.errorbar(
        ns,
        means.to_numpy(),
        yerr=sems.to_numpy(),
        marker="o",
        linewidth=1.8,
        capsize=3,
        color=BLUE,
        label="family mean",
    )
    threshold = np.array([(n / 2) ** 2 / 8 - 3 for n in ns])
    ax.plot(
        ns,
        threshold,
        linestyle="--",
        linewidth=1.4,
        color=ORANGE,
        label=r"theory: $\Delta_m\geq \frac{m^2}{8}-3$",
    )
    ax.set_xlabel(r"vertices $n=2m$")
    ax.set_ylabel(r"certified midpoint deficit $\Delta_m$")
    ax.set_title("80 connected rigid graphs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.1, loc="upper left")
    ax.text(-0.15, 1.08, "a", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    methods = [
        ("local", "local MH", "o", "-", BLUE),
        ("near_twin_jump", "5% near-twin jump", "s", "--", ORANGE),
    ]
    ax = axes[1]
    for method, label, marker, linestyle, color in methods:
        d = aggregate[aggregate.method == method].sort_values("n")
        ax.errorbar(
            d.n,
            d.mean_opposing_start_row_tv,
            yerr=d.sem_opposing_start_row_tv,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=1.6,
            capsize=3,
            label=label,
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"vertices $n=2m$")
    ax.set_ylabel("row-TV between opposing starts")
    ax.set_title("initialization sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.text(-0.15, 1.08, "b", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    ax = axes[2]
    ns_chain = sorted(chains.n.unique())
    for method, label, marker, linestyle, color in methods:
        values = []
        for n in ns_chain:
            d = chains[(chains.n == n) & (chains.method == method)]
            values.append(d.traverses_both_orientations.mean())
        ax.plot(
            ns_chain,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=1.6,
            label=label,
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"vertices $n=2m$")
    ax.set_ylabel("fraction of chains traversing")
    ax.set_title("mode traversal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.text(-0.15, 1.08, "c", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    save(fig, "near_twin_family")


def multimodal_marginals() -> None:
    seed_index = 10
    instance = generate_paired_instances(8, 0.3, [0.1], 10)[0]
    beta = likelihood_beta(instance.p, instance.noise)
    exact = exact_posterior(instance, beta)
    local_outputs = []
    pt_outputs = []
    for chain in range(4):
        base_seed = 10_000_019 * (seed_index + 1) + 100_003 * 100 + 997 * chain
        local_outputs.append(
            random_transposition_mh(
                instance.A,
                instance.B,
                beta,
                20_000,
                0.25,
                np.random.default_rng(base_seed),
                sample_every=5,
            )
        )
        pt_outputs.append(
            parallel_tempering(
                instance.A,
                instance.B,
                beta,
                2_000,
                0.25,
                np.random.default_rng(base_seed + 37 * 3),
                sample_every=1,
                replicas=8,
            )
        )
    local_p = samples_to_marginals(np.vstack([o.samples for o in local_outputs]), instance.n)
    pt_p = samples_to_marginals(np.vstack([o.samples for o in pt_outputs]), instance.n)
    orbit_p = orbit_average_marginals(
        local_p,
        automorphism_vertex_orbits(instance.A),
        automorphism_vertex_orbits(instance.B),
    )
    matrices = [exact.marginals, local_p, orbit_p, pt_p]
    titles = ["exact posterior", "local MCMC", "local + orbit average", "parallel tempering"]
    tvs = [
        0.0,
        0.5 * np.abs(local_p - exact.marginals).sum(axis=1).mean(),
        0.5 * np.abs(orbit_p - exact.marginals).sum(axis=1).mean(),
        0.5 * np.abs(pt_p - exact.marginals).sum(axis=1).mean(),
    ]

    fig, all_axes = plt.subplots(
        1,
        5,
        figsize=(7.25, 2.05),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.045]},
    )
    axes = all_axes[:4]
    cax = all_axes[4]
    image = None
    for ax, matrix, title, tv in zip(axes, matrices, titles, tvs):
        image = ax.imshow(matrix[:, instance.pi_true], vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"{title}\nrow-TV = {tv:.3f}", fontsize=7.9, linespacing=1.35)
        ax.set_xticks([0, 3, 7])
        ax.set_yticks([0, 3, 7])
    axes[0].set_ylabel("source vertex")
    for ax in axes:
        ax.set_xlabel("target rank")
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("match probability")
    fig.subplots_adjust(left=0.055, right=0.96, bottom=0.20, top=0.78, wspace=0.30)
    save(fig, "multimodal_marginals")


def temperature_mapping() -> None:
    epsilon = np.linspace(0.02, 1.0, 400)
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for p, color in [(0.15, PURPLE), (0.30, BLUE), (0.50, GREEN)]:
        beta = np.asarray([likelihood_beta(p, x) for x in epsilon])
        ax.plot(epsilon, beta, lw=1.8, label=fr"$p={p:.2f}$", color=color)
    ax.axhline(2.0, color=ORANGE, ls="--", lw=1.4, label=r"fixed $\beta=2$")
    ax.set(xlabel=r"resampling noise $\epsilon$", ylabel=r"likelihood $\beta_{\rm lik}$")
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "temperature_mapping")


def main() -> None:
    set_style()
    budget_with_iid_floor()
    near_twin_family()
    multimodal_marginals()
    temperature_mapping()
    print(f"Wrote submitted figures to {OUT}")


if __name__ == "__main__":
    main()
