"""Population-SMC and soft-assignment baselines for graph alignment.

Population annealing SMC targets the same Gibbs posterior as the MCMC audit.
An entropic Birkhoff relaxation (EBR) is included only as an interpretation
check: it is a soft optimization relaxation, not a posterior approximation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
from exact_posterior_benchmark import exact_posterior_auto, generate_paired_instances, likelihood_beta  # noqa: E402


def row_tv(P: np.ndarray, Q: np.ndarray) -> float:
    return float(0.5 * np.abs(P - Q).sum(axis=1).mean())


@njit(cache=True)
def overlap_score(A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> int:
    s = 0
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] and B[pi[i], pi[j]]:
                s += 1
    return s


@njit(cache=True)
def swap_delta(A: np.ndarray, B: np.ndarray, pi: np.ndarray, u: int, v: int) -> int:
    a, b = pi[u], pi[v]
    d = 0
    n = A.shape[0]
    for k in range(n):
        if k == u or k == v:
            continue
        pk = pi[k]
        d += (A[u, k] - A[v, k]) * (B[b, pk] - B[a, pk])
    return int(d)


@njit(cache=True)
def particle_scores(A: np.ndarray, B: np.ndarray, particles: np.ndarray) -> np.ndarray:
    out = np.empty(particles.shape[0], dtype=np.int16)
    for r in range(particles.shape[0]):
        out[r] = overlap_score(A, B, particles[r])
    return out


@njit(cache=True)
def rejuvenate(
    A: np.ndarray,
    B: np.ndarray,
    particles: np.ndarray,
    scores: np.ndarray,
    beta: float,
    moves_per_particle: int,
    seed: int,
) -> int:
    np.random.seed(seed)
    accepted = 0
    R, n = particles.shape
    for r in range(R):
        for _ in range(moves_per_particle):
            u = np.random.randint(n)
            v = np.random.randint(n - 1)
            if v >= u:
                v += 1
            d = swap_delta(A, B, particles[r], u, v)
            if d >= 0 or math.log(np.random.random()) < beta * d:
                tmp = particles[r, u]
                particles[r, u] = particles[r, v]
                particles[r, v] = tmp
                scores[r] += d
                accepted += 1
    return accepted


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def random_particles(count: int, n: int, rng: np.random.Generator) -> np.ndarray:
    # Sorting independent uniforms yields independent uniform permutations.
    return np.argsort(rng.random((count, n)), axis=1).astype(np.int16)


@dataclass
class PASMCResult:
    marginals: np.ndarray
    mean_ess_fraction: float
    min_ess_fraction: float
    acceptance_rate: float
    distinct_particles: int


def population_annealing_smc(
    A: np.ndarray,
    B: np.ndarray,
    beta: float,
    particles_count: int,
    stages: int,
    moves_per_stage: int,
    rng: np.random.Generator,
    seed_base: int,
) -> PASMCResult:
    n = A.shape[0]
    particles = random_particles(particles_count, n, rng)
    scores = particle_scores(A, B, particles)
    # A quadratic ladder spends more stages near the concentrated cold target.
    ladder = beta * np.square(np.linspace(0.0, 1.0, stages + 1))
    ess_fracs: list[float] = []
    accepted = 0
    attempted = particles_count * stages * moves_per_stage
    for k in range(1, stages + 1):
        db = float(ladder[k] - ladder[k - 1])
        logw = db * scores.astype(float)
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        ess_fracs.append(float(1.0 / np.square(w).sum() / particles_count))
        idx = systematic_resample(w, rng)
        particles = particles[idx].copy()
        scores = scores[idx].copy()
        accepted += rejuvenate(
            A, B, particles, scores, float(ladder[k]), moves_per_stage, seed_base + k
        )
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        P[i] = np.bincount(particles[:, i].astype(int), minlength=n) / particles_count
    distinct = np.unique(particles, axis=0).shape[0]
    return PASMCResult(
        marginals=P,
        mean_ess_fraction=float(np.mean(ess_fracs)),
        min_ess_fraction=float(np.min(ess_fracs)),
        acceptance_rate=float(accepted / max(attempted, 1)),
        distinct_particles=int(distinct),
    )


def sinkhorn(K: np.ndarray, iterations: int = 600, eps: float = 1e-100) -> np.ndarray:
    X = np.maximum(K, eps).astype(float)
    for _ in range(iterations):
        X /= np.maximum(X.sum(axis=1, keepdims=True), eps)
        X /= np.maximum(X.sum(axis=0, keepdims=True), eps)
    return X


def ebr_objective(A: np.ndarray, B: np.ndarray, X: np.ndarray, beta: float, tau: float) -> float:
    overlap = 0.5 * float(np.trace(A @ X @ B @ X.T))
    entropy = -float(np.sum(X * np.log(np.maximum(X, 1e-300))))
    return beta * overlap + tau * entropy


def entropic_birkhoff_relaxation(
    A: np.ndarray,
    B: np.ndarray,
    beta: float,
    tau: float,
    rng: np.random.Generator,
    restarts: int,
    iterations: int,
    step_size: float,
) -> tuple[np.ndarray, float]:
    n = A.shape[0]
    best_X, best_value = None, -np.inf
    for restart in range(restarts):
        X = np.full((n, n), 1.0 / n) if restart == 0 else sinkhorn(np.exp(rng.normal(scale=.25, size=(n, n))))
        value = ebr_objective(A, B, X, beta, tau)
        for t in range(iterations):
            grad = beta * (A @ X @ B) - tau * (np.log(np.maximum(X, 1e-300)) + 1.0)
            eta = step_size / math.sqrt(1.0 + t / 40.0)
            logits = np.log(np.maximum(X, 1e-300)) + eta * grad
            logits -= logits.max()
            candidate = sinkhorn(np.exp(logits))
            cval = ebr_objective(A, B, candidate, beta, tau)
            if cval + 1e-10 < value:
                logits = np.log(np.maximum(X, 1e-300)) + .2 * eta * grad
                logits -= logits.max()
                candidate = sinkhorn(np.exp(logits), iterations=900)
                cval = ebr_objective(A, B, candidate, beta, tau)
            X, value = candidate, cval
        if value > best_value:
            best_X, best_value = X.copy(), value
    assert best_X is not None
    return best_X, float(best_value)


def validate() -> None:
    inst = generate_paired_instances(6, .3, [.2], 11)[0]
    beta = likelihood_beta(.3, .2)
    exact = exact_posterior_auto(inst, beta)
    Ps = []
    for r in range(3):
        result = population_annealing_smc(inst.A, inst.B, beta, 4096, 24, 3, np.random.default_rng(r), 9000+r*100)
        Ps.append(result.marginals)
    if row_tv(np.mean(Ps, axis=0), exact.marginals) > .04:
        raise AssertionError("Population-SMC validation error is unexpectedly high")
    X, _ = entropic_birkhoff_relaxation(inst.A, inst.B, beta, 1.0, np.random.default_rng(4), 2, 40, .08)
    if not (np.allclose(X.sum(0), 1, atol=3e-4) and np.allclose(X.sum(1), 1, atol=3e-4)):
        raise AssertionError("EBR is not doubly stochastic")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate()
    rows: list[dict[str, object]] = []
    counts = {8: args.n8_instances, 9: args.n9_instances, 10: args.n10_instances}
    for n in args.n_values:
        for graph_seed in range(args.seed_start, counts[n]):
            inst = generate_paired_instances(n, args.p, [args.noise], graph_seed)[0]
            beta = likelihood_beta(args.p, args.noise)
            exact = exact_posterior_auto(inst, beta)
            for particles in args.particle_counts:
                estimates, ess, miness, acc, distinct = [], [], [], [], []
                start = time.perf_counter()
                for rep in range(args.smc_replicates):
                    seed = 410_000_000 + n*1_000_000 + graph_seed*10_000 + particles + rep
                    res = population_annealing_smc(
                        inst.A, inst.B, beta, particles, args.smc_stages, args.smc_moves,
                        np.random.default_rng(seed), seed + 1000,
                    )
                    estimates.append(res.marginals); ess.append(res.mean_ess_fraction); miness.append(res.min_ess_fraction); acc.append(res.acceptance_rate); distinct.append(res.distinct_particles)
                rows.append(dict(n=n,graph_seed=graph_seed,method="population_smc",configuration=f"particles={particles}",mean_row_tv=row_tv(np.mean(estimates,axis=0),exact.marginals),runtime_seconds=time.perf_counter()-start,mean_ess_fraction=np.mean(ess),min_ess_fraction=np.mean(miness),acceptance_rate=np.mean(acc),mean_distinct_particles=np.mean(distinct),objective=np.nan))
            for tau in ([] if args.skip_ebr else args.taus):
                seed = 510_000_000 + n*1_000_000 + graph_seed*10_000 + int(tau*100)
                start = time.perf_counter()
                X, obj = entropic_birkhoff_relaxation(inst.A, inst.B, beta, tau, np.random.default_rng(seed), args.ebr_restarts, args.ebr_iterations, args.ebr_step_size)
                rows.append(dict(n=n,graph_seed=graph_seed,method="entropic_birkhoff",configuration=f"tau={tau:g}",mean_row_tv=row_tv(X,exact.marginals),runtime_seconds=time.perf_counter()-start,mean_ess_fraction=np.nan,min_ess_fraction=np.nan,acceptance_rate=np.nan,mean_distinct_particles=np.nan,objective=obj))
            pd.DataFrame(rows).to_csv(args.output_dir / "inference_baselines_raw.partial.csv", index=False)
            print(f"finished baselines n={n}, seed={graph_seed}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "inference_baselines_raw.csv", index=False)
    summary = raw.groupby(["n","method","configuration"],sort=True).agg(instances=("graph_seed","size"),mean_row_tv=("mean_row_tv","mean"),sem_row_tv=("mean_row_tv","sem"),median_row_tv=("mean_row_tv","median"),mean_runtime_seconds=("runtime_seconds","mean"),mean_ess_fraction=("mean_ess_fraction","mean"),min_ess_fraction=("min_ess_fraction","mean"),acceptance_rate=("acceptance_rate","mean")).reset_index()
    summary.to_csv(args.output_dir / "inference_baselines_summary.csv", index=False)
    best = raw[raw.method=="entropic_birkhoff"].sort_values(["n","graph_seed","objective"],ascending=[True,True,False]).groupby(["n","graph_seed"],as_index=False).first()
    best.to_csv(args.output_dir / "ebr_best_objective.csv", index=False)
    metadata=vars(args).copy(); metadata["output_dir"]=str(args.output_dir)
    json.dump(metadata,open(args.output_dir/"metadata.json","w"),indent=2)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=ROOT/"results"/"inference_baselines")
    p.add_argument("--n-values",type=int,nargs="+",default=[8,9,10])
    p.add_argument("--n8-instances",type=int,default=20); p.add_argument("--n9-instances",type=int,default=20); p.add_argument("--n10-instances",type=int,default=20); p.add_argument("--seed-start",type=int,default=0)
    p.add_argument("--p",type=float,default=.3); p.add_argument("--noise",type=float,default=.1)
    p.add_argument("--particle-counts",type=int,nargs="+",default=[4096])
    p.add_argument("--smc-replicates",type=int,default=4); p.add_argument("--smc-stages",type=int,default=24); p.add_argument("--smc-moves",type=int,default=3)
    p.add_argument("--taus",type=float,nargs="+",default=[.5,1.,2.]); p.add_argument("--skip-ebr",action="store_true"); p.add_argument("--ebr-restarts",type=int,default=4); p.add_argument("--ebr-iterations",type=int,default=160); p.add_argument("--ebr-step-size",type=float,default=.08)
    return p.parse_args()

if __name__=="__main__":
    run(parse_args())
