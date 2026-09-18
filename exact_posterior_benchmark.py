"""Exact-posterior benchmark for Bayesian graph matching.

This script benchmarks permutation samplers against an exact posterior oracle on
small correlated Erdos-Renyi graph pairs.  It follows the data-generating model
used in the Graphical-Learning project:

    A ~ G(n, p)
    B_clean[pi(i), pi(j)] = A[i, j]
    B edge = B_clean edge with probability 1-epsilon, and otherwise a fresh
             Bernoulli(p) draw.

Under a uniform prior on pi, the posterior is proportional to

    exp(beta_lik * S(pi)),

where S(pi) is edge overlap and beta_lik is returned by likelihood_beta().

The implementation deliberately keeps all samplers in one file so their target,
metrics, seeding, and timing conventions are identical.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp


Array = np.ndarray


@dataclass(frozen=True)
class Instance:
    n: int
    p: float
    noise: float
    seed: int
    A: Array
    B: Array
    pi_true: Array


@dataclass
class ExactPosterior:
    beta: float
    permutations: Array
    scores: Array
    probabilities: Array
    marginals: Array
    map_score: float
    map_count: int
    map_permutation: Array
    max_state_probability: float
    entropy: float
    effective_states: float
    planted_probability: float
    planted_expected_accuracy: float
    bayes_permutation: Array
    bayes_accuracy: float


@dataclass
class SamplerOutput:
    samples: Array
    scores: Array
    acceptance_rate: float
    auxiliary_rate: float
    runtime_seconds: float
    iterations: int


def likelihood_beta(p: float, noise: float) -> float:
    """Inverse temperature induced by the resampling-noise likelihood."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie strictly between zero and one")
    if not (0.0 < noise <= 1.0):
        raise ValueError("noise must lie in (0, 1]")
    numerator = (1.0 - noise + noise * p) * (1.0 - noise * p)
    denominator = noise * noise * p * (1.0 - p)
    return float(math.log(numerator / denominator))


def sample_er_graph(n: int, p: float, rng: np.random.Generator) -> Array:
    rows, cols = np.triu_indices(n, k=1)
    A = np.zeros((n, n), dtype=np.int8)
    values = (rng.random(len(rows)) < p).astype(np.int8)
    A[rows, cols] = values
    A[cols, rows] = values
    return A


def permute_adjacency(A: Array, pi: Array) -> Array:
    B = np.zeros_like(A)
    B[np.ix_(pi, pi)] = A
    return B


def generate_paired_instances(
    n: int,
    p: float,
    noises: Sequence[float],
    seed: int,
) -> list[Instance]:
    """Generate nested-noise instances sharing A, pi_true, and noise uniforms."""
    rng = np.random.default_rng(seed)
    A = sample_er_graph(n, p, rng)
    pi_true = rng.permutation(n)
    B_clean = permute_adjacency(A, pi_true)
    rows, cols = np.triu_indices(n, k=1)
    resample_uniforms = rng.random(len(rows))
    new_edges = (rng.random(len(rows)) < p).astype(np.int8)

    result: list[Instance] = []
    for noise in noises:
        B = B_clean.copy()
        mask = resample_uniforms < noise
        B[rows[mask], cols[mask]] = new_edges[mask]
        B[cols[mask], rows[mask]] = new_edges[mask]
        result.append(
            Instance(
                n=n,
                p=p,
                noise=float(noise),
                seed=seed,
                A=A.copy(),
                B=B,
                pi_true=pi_true.copy(),
            )
        )
    return result


_PERMUTATION_CACHE: dict[int, Array] = {}


def all_permutations(n: int) -> Array:
    if n not in _PERMUTATION_CACHE:
        dtype = np.int8 if n < 128 else np.int16
        _PERMUTATION_CACHE[n] = np.asarray(
            list(itertools.permutations(range(n))), dtype=dtype
        )
    return _PERMUTATION_CACHE[n]


def scalar_overlap(A: Array, B: Array, pi: Array) -> float:
    aligned = B[np.ix_(pi, pi)]
    return float(np.sum(np.triu(A * aligned, k=1)))


def conditional_log_likelihood(instance: Instance, pi: Array) -> float:
    """Direct log P(B | A, pi, p, noise) under resampling noise."""
    aligned = instance.B[np.ix_(pi, pi)]
    rows, cols = np.triu_indices(instance.n, k=1)
    a = instance.A[rows, cols]
    b = aligned[rows, cols]
    p = instance.p
    noise = instance.noise
    probabilities = np.where(
        a == 1,
        np.where(b == 1, 1.0 - noise + noise * p, noise * (1.0 - p)),
        np.where(b == 1, noise * p, 1.0 - noise * p),
    )
    return float(np.log(probabilities).sum())


def vectorized_overlap_scores(A: Array, B: Array, permutations: Array) -> Array:
    """Compute S(pi) for every row of permutations without an n! Python loop."""
    scores = np.zeros(len(permutations), dtype=np.float64)
    edge_rows, edge_cols = np.where(np.triu(A, k=1) == 1)
    for i, j in zip(edge_rows, edge_cols):
        scores += B[permutations[:, i], permutations[:, j]]
    return scores


def posterior_marginals(permutations: Array, probabilities: Array, n: int) -> Array:
    P = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        P[i] = np.bincount(
            permutations[:, i], weights=probabilities, minlength=n
        )
    return P


def hamming_bayes_permutation(P: Array) -> Array:
    """Bayes action for vertex-wise Hamming loss: maximize sum_i P[i, pi(i)]."""
    rows, cols = linear_sum_assignment(-P)
    pi = np.empty(P.shape[0], dtype=int)
    pi[rows] = cols
    return pi


def exact_posterior(instance: Instance, beta: float) -> ExactPosterior:
    permutations = all_permutations(instance.n)
    scores = vectorized_overlap_scores(instance.A, instance.B, permutations)
    log_weights = beta * scores
    log_normalizer = logsumexp(log_weights)
    probabilities = np.exp(log_weights - log_normalizer)
    P = posterior_marginals(permutations, probabilities, instance.n)

    map_score = float(scores.max())
    map_mask = scores == map_score
    map_index = int(np.flatnonzero(map_mask)[0])
    planted_mask = np.all(permutations == instance.pi_true[None, :], axis=1)
    planted_probability = float(probabilities[planted_mask].sum())
    planted_expected_accuracy = float(
        np.mean(P[np.arange(instance.n), instance.pi_true])
    )
    bayes_pi = hamming_bayes_permutation(P)
    positive = probabilities > 0.0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))

    assert np.allclose(P.sum(axis=0), 1.0, atol=1e-9)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-10)

    return ExactPosterior(
        beta=float(beta),
        permutations=permutations,
        scores=scores,
        probabilities=probabilities,
        marginals=P,
        map_score=map_score,
        map_count=int(map_mask.sum()),
        map_permutation=permutations[map_index].astype(int),
        max_state_probability=float(probabilities.max()),
        entropy=entropy,
        effective_states=float(math.exp(entropy)),
        planted_probability=planted_probability,
        planted_expected_accuracy=planted_expected_accuracy,
        bayes_permutation=bayes_pi,
        bayes_accuracy=float(np.mean(bayes_pi == instance.pi_true)),
    )


def exact_posterior_streaming(
    instance: Instance,
    beta: float,
    chunk_size: int = 100_000,
) -> ExactPosterior:
    """Exact posterior without materializing all n! permutations.

    The sufficient statistics are a histogram of overlap scores and, for every
    possible vertex assignment, a histogram of scores among permutations that
    contain that assignment.  This keeps memory polynomial in n while retaining
    exact enumeration time.  It is intended for n=10--11, where a dense array of
    all permutations is unnecessarily memory hungry.
    """
    n = instance.n
    maximum_possible_score = int(np.triu(instance.A, k=1).sum())
    score_counts = np.zeros(maximum_possible_score + 1, dtype=np.int64)
    assignment_score_counts = np.zeros(
        (n, n, maximum_possible_score + 1), dtype=np.int64
    )
    iterator = itertools.permutations(range(n))
    first_map_permutation: Array | None = None
    observed_map_score = -1

    while True:
        chunk_list = list(itertools.islice(iterator, chunk_size))
        if not chunk_list:
            break
        permutations = np.asarray(chunk_list, dtype=np.int8 if n < 128 else np.int16)
        scores = vectorized_overlap_scores(instance.A, instance.B, permutations).astype(
            np.int64
        )
        score_counts += np.bincount(
            scores, minlength=maximum_possible_score + 1
        ).astype(np.int64)
        for i in range(n):
            combined = permutations[:, i].astype(np.int64) * (
                maximum_possible_score + 1
            ) + scores
            assignment_score_counts[i] += np.bincount(
                combined, minlength=n * (maximum_possible_score + 1)
            ).reshape(n, maximum_possible_score + 1)

        chunk_max = int(scores.max())
        if chunk_max > observed_map_score:
            observed_map_score = chunk_max
            first = int(np.flatnonzero(scores == chunk_max)[0])
            first_map_permutation = permutations[first].astype(int).copy()

    assert first_map_permutation is not None
    assert int(score_counts.sum()) == math.factorial(n)
    score_values = np.arange(maximum_possible_score + 1, dtype=float)
    nonempty = score_counts > 0
    shifted_weights = np.zeros_like(score_values)
    shifted_weights[nonempty] = np.exp(
        beta * (score_values[nonempty] - observed_map_score)
    )
    shifted_normalizer = float(np.dot(score_counts, shifted_weights))
    score_probabilities = score_counts * shifted_weights / shifted_normalizer
    log_normalizer = beta * observed_map_score + math.log(shifted_normalizer)

    marginals = np.tensordot(
        assignment_score_counts, shifted_weights, axes=([2], [0])
    ) / shifted_normalizer
    expected_score = float(np.dot(score_probabilities, score_values))
    bayes_pi = hamming_bayes_permutation(marginals)
    planted_score = scalar_overlap(instance.A, instance.B, instance.pi_true)
    planted_probability = float(math.exp(beta * planted_score - log_normalizer))
    entropy = float(log_normalizer - beta * expected_score)
    map_count = int(score_counts[observed_map_score])

    assert np.allclose(marginals.sum(axis=0), 1.0, atol=1e-9)
    assert np.allclose(marginals.sum(axis=1), 1.0, atol=1e-9)
    assert np.isclose(score_probabilities.sum(), 1.0, atol=1e-10)

    # Here scores/probabilities represent the exact *score* distribution rather
    # than one entry per state.  All downstream expectations remain identical.
    return ExactPosterior(
        beta=float(beta),
        permutations=np.empty((0, n), dtype=np.int8),
        scores=score_values,
        probabilities=score_probabilities,
        marginals=marginals,
        map_score=float(observed_map_score),
        map_count=map_count,
        map_permutation=first_map_permutation,
        max_state_probability=float(
            math.exp(beta * observed_map_score - log_normalizer)
        ),
        entropy=entropy,
        effective_states=float(math.exp(entropy)),
        planted_probability=planted_probability,
        planted_expected_accuracy=float(
            np.mean(marginals[np.arange(n), instance.pi_true])
        ),
        bayes_permutation=bayes_pi,
        bayes_accuracy=float(np.mean(bayes_pi == instance.pi_true)),
    )


def exact_posterior_auto(instance: Instance, beta: float) -> ExactPosterior:
    """Use the dense oracle through n=9 and the streaming oracle thereafter."""
    if instance.n <= 9:
        return exact_posterior(instance, beta)
    return exact_posterior_streaming(instance, beta)


def swap_delta(A: Array, B: Array, pi: Array, u: int, v: int) -> float:
    """Exact overlap change after swapping pi[u] and pi[v]."""
    a, b = int(pi[u]), int(pi[v])
    dA = (A[u] - A[v]).astype(np.float64)
    dB = (B[b, pi] - B[a, pi]).astype(np.float64)
    dA[u] = 0.0
    dA[v] = 0.0
    return float(dA @ dB)


def samples_to_marginals(samples: Array, n: int) -> Array:
    P = np.zeros((n, n), dtype=np.float64)
    if len(samples) == 0:
        return P
    for i in range(n):
        P[i] = np.bincount(samples[:, i], minlength=n) / len(samples)
    return P


def random_transposition_mh(
    A: Array,
    B: Array,
    beta: float,
    iterations: int,
    burn_fraction: float,
    rng: np.random.Generator,
    sample_every: int = 1,
    initial_state: Array | None = None,
) -> SamplerOutput:
    n = A.shape[0]
    pi = (
        rng.permutation(n)
        if initial_state is None
        else np.asarray(initial_state, dtype=int).copy()
    )
    if not np.array_equal(np.sort(pi), np.arange(n)):
        raise ValueError("initial_state must be a permutation of 0,...,n-1")
    score = scalar_overlap(A, B, pi)
    burn = int(iterations * burn_fraction)
    samples: list[Array] = []
    scores: list[float] = []
    accepted = 0
    start = time.perf_counter()
    for t in range(iterations):
        u = int(rng.integers(n))
        v = int(rng.integers(n - 1))
        if v >= u:
            v += 1
        delta = swap_delta(A, B, pi, u, v)
        if delta >= 0.0 or math.log(rng.random()) < beta * delta:
            pi[u], pi[v] = pi[v], pi[u]
            score += delta
            accepted += 1
        if t >= burn and (t - burn) % sample_every == 0:
            samples.append(pi.copy())
            scores.append(score)
    runtime = time.perf_counter() - start
    return SamplerOutput(
        samples=np.asarray(samples, dtype=np.int16),
        scores=np.asarray(scores, dtype=float),
        acceptance_rate=accepted / iterations,
        auxiliary_rate=float("nan"),
        runtime_seconds=runtime,
        iterations=iterations,
    )


def _all_swap_deltas(A: Array, B: Array, pi: Array) -> tuple[Array, Array, Array]:
    n = len(pi)
    us, vs = np.triu_indices(n, k=1)
    deltas = np.fromiter(
        (swap_delta(A, B, pi, int(u), int(v)) for u, v in zip(us, vs)),
        dtype=np.float64,
        count=len(us),
    )
    return us, vs, deltas


def locally_balanced_transposition_mh(
    A: Array,
    B: Array,
    beta: float,
    iterations: int,
    burn_fraction: float,
    rng: np.random.Generator,
    sample_every: int = 1,
) -> SamplerOutput:
    """Barker locally balanced proposal over every transposition."""
    n = A.shape[0]
    pi = rng.permutation(n)
    score = scalar_overlap(A, B, pi)
    burn = int(iterations * burn_fraction)
    samples: list[Array] = []
    scores: list[float] = []
    accepted = 0
    start = time.perf_counter()
    for t in range(iterations):
        us, vs, deltas = _all_swap_deltas(A, B, pi)
        log_weights = -np.logaddexp(0.0, -beta * deltas)
        log_z_current = float(logsumexp(log_weights))
        probabilities = np.exp(log_weights - log_z_current)
        move = int(rng.choice(len(deltas), p=probabilities))
        u, v, delta = int(us[move]), int(vs[move]), float(deltas[move])
        proposed = pi.copy()
        proposed[u], proposed[v] = proposed[v], proposed[u]
        _, _, reverse_deltas = _all_swap_deltas(A, B, proposed)
        reverse_log_weights = -np.logaddexp(0.0, -beta * reverse_deltas)
        log_z_proposed = float(logsumexp(reverse_log_weights))
        if math.log(rng.random()) < min(0.0, log_z_current - log_z_proposed):
            pi = proposed
            score += delta
            accepted += 1
        if t >= burn and (t - burn) % sample_every == 0:
            samples.append(pi.copy())
            scores.append(score)
    runtime = time.perf_counter() - start
    return SamplerOutput(
        samples=np.asarray(samples, dtype=np.int16),
        scores=np.asarray(scores, dtype=float),
        acceptance_rate=accepted / iterations,
        auxiliary_rate=float("nan"),
        runtime_seconds=runtime,
        iterations=iterations,
    )


def _cycle_reinsertion_candidates(pi: Array, v: int) -> Array:
    """All n permutations obtained by deleting v from its cycle and reinserting it."""
    n = len(pi)
    inverse = np.empty(n, dtype=int)
    inverse[pi] = np.arange(n)
    predecessor = int(inverse[v])
    successor = int(pi[v])

    reduced = pi.copy()
    if predecessor != v:
        reduced[predecessor] = successor

    candidates: list[Array] = []
    singleton = reduced.copy()
    singleton[v] = v
    candidates.append(singleton)

    reduced_inverse = np.full(n, -1, dtype=int)
    for u in range(n):
        if u != v:
            reduced_inverse[int(reduced[u])] = u
    for new_successor in range(n):
        if new_successor == v:
            continue
        pred = int(reduced_inverse[new_successor])
        candidate = reduced.copy()
        candidate[pred] = v
        candidate[v] = new_successor
        candidates.append(candidate)

    result = np.asarray(candidates, dtype=np.int16)
    assert result.shape == (n, n)
    assert all(np.array_equal(np.sort(row), np.arange(n)) for row in result)
    assert any(np.array_equal(row, pi) for row in result)
    return result


def node_reinsertion_gibbs(
    A: Array,
    B: Array,
    beta: float,
    iterations: int,
    burn_fraction: float,
    rng: np.random.Generator,
    sample_every: int = 1,
) -> SamplerOutput:
    n = A.shape[0]
    pi = rng.permutation(n)
    score = scalar_overlap(A, B, pi)
    burn = int(iterations * burn_fraction)
    samples: list[Array] = []
    scores: list[float] = []
    changed = 0
    start = time.perf_counter()
    for t in range(iterations):
        v = int(rng.integers(n))
        candidates = _cycle_reinsertion_candidates(pi, v)
        candidate_scores = np.fromiter(
            (scalar_overlap(A, B, candidate) for candidate in candidates),
            dtype=np.float64,
            count=n,
        )
        log_weights = beta * candidate_scores
        probabilities = np.exp(log_weights - logsumexp(log_weights))
        chosen = int(rng.choice(n, p=probabilities))
        new_pi = candidates[chosen].astype(int)
        if not np.array_equal(new_pi, pi):
            changed += 1
        pi = new_pi
        score = float(candidate_scores[chosen])
        if t >= burn and (t - burn) % sample_every == 0:
            samples.append(pi.copy())
            scores.append(score)
    runtime = time.perf_counter() - start
    return SamplerOutput(
        samples=np.asarray(samples, dtype=np.int16),
        scores=np.asarray(scores, dtype=float),
        acceptance_rate=1.0,
        auxiliary_rate=changed / iterations,
        runtime_seconds=runtime,
        iterations=iterations,
    )


def parallel_tempering(
    A: Array,
    B: Array,
    beta: float,
    iterations: int,
    burn_fraction: float,
    rng: np.random.Generator,
    sample_every: int = 1,
    replicas: int = 8,
) -> SamplerOutput:
    """Random-transposition parallel tempering; iterations count PT sweeps."""
    n = A.shape[0]
    betas = beta * np.linspace(0.0, 1.0, replicas) ** 2
    states = np.asarray([rng.permutation(n) for _ in range(replicas)], dtype=int)
    state_scores = np.asarray([scalar_overlap(A, B, pi) for pi in states])
    burn = int(iterations * burn_fraction)
    samples: list[Array] = []
    scores: list[float] = []
    local_accepted = 0
    swap_accepted = 0
    swap_attempted = 0
    start = time.perf_counter()
    for t in range(iterations):
        for r in range(replicas):
            u = int(rng.integers(n))
            v = int(rng.integers(n - 1))
            if v >= u:
                v += 1
            delta = swap_delta(A, B, states[r], u, v)
            if delta >= 0.0 or math.log(rng.random()) < betas[r] * delta:
                states[r, u], states[r, v] = states[r, v], states[r, u]
                state_scores[r] += delta
                local_accepted += 1

        parity = t % 2
        for r in range(parity, replicas - 1, 2):
            log_ratio = (betas[r + 1] - betas[r]) * (
                state_scores[r] - state_scores[r + 1]
            )
            swap_attempted += 1
            if math.log(rng.random()) < min(0.0, float(log_ratio)):
                states[[r, r + 1]] = states[[r + 1, r]]
                state_scores[[r, r + 1]] = state_scores[[r + 1, r]]
                swap_accepted += 1

        if t >= burn and (t - burn) % sample_every == 0:
            samples.append(states[-1].copy())
            scores.append(float(state_scores[-1]))
    runtime = time.perf_counter() - start
    return SamplerOutput(
        samples=np.asarray(samples, dtype=np.int16),
        scores=np.asarray(scores, dtype=float),
        acceptance_rate=local_accepted / (iterations * replicas),
        auxiliary_rate=swap_accepted / max(swap_attempted, 1),
        runtime_seconds=runtime,
        iterations=iterations,
    )


def _sequential_log_probability(
    A: Array,
    B: Array,
    target_pi: Array,
    order: Array,
    proposal_beta: float,
) -> float:
    n = len(target_pi)
    assigned = np.full(n, -1, dtype=int)
    available = np.ones(n, dtype=bool)
    log_q = 0.0
    previous_vertices: list[int] = []
    for u_raw in order:
        u = int(u_raw)
        candidates = np.flatnonzero(available)
        increments = np.zeros(len(candidates), dtype=float)
        for idx, candidate_v in enumerate(candidates):
            increment = 0.0
            for previous_u in previous_vertices:
                if A[u, previous_u]:
                    increment += B[candidate_v, assigned[previous_u]]
            increments[idx] = increment
        logits = proposal_beta * increments
        desired = int(target_pi[u])
        desired_index = int(np.flatnonzero(candidates == desired)[0])
        log_q += float(logits[desired_index] - logsumexp(logits))
        assigned[u] = desired
        available[desired] = False
        previous_vertices.append(u)
    return log_q


def _sequential_proposal(
    A: Array,
    B: Array,
    order: Array,
    proposal_beta: float,
    rng: np.random.Generator,
) -> tuple[Array, float]:
    n = len(order)
    proposed = np.full(n, -1, dtype=int)
    available = np.ones(n, dtype=bool)
    previous_vertices: list[int] = []
    log_q = 0.0
    for u_raw in order:
        u = int(u_raw)
        candidates = np.flatnonzero(available)
        increments = np.zeros(len(candidates), dtype=float)
        for idx, candidate_v in enumerate(candidates):
            increment = 0.0
            for previous_u in previous_vertices:
                if A[u, previous_u]:
                    increment += B[candidate_v, proposed[previous_u]]
            increments[idx] = increment
        logits = proposal_beta * increments
        probabilities = np.exp(logits - logsumexp(logits))
        chosen_index = int(rng.choice(len(candidates), p=probabilities))
        chosen = int(candidates[chosen_index])
        log_q += float(math.log(probabilities[chosen_index]))
        proposed[u] = chosen
        available[chosen] = False
        previous_vertices.append(u)
    return proposed, log_q


def sequential_global_mh(
    A: Array,
    B: Array,
    beta: float,
    iterations: int,
    burn_fraction: float,
    rng: np.random.Generator,
    sample_every: int = 1,
    proposal_temperature: float = 2.0,
) -> SamplerOutput:
    """State-dependent sequential-matching MH proposal of Volkovs and Zemel."""
    n = A.shape[0]
    pi = rng.permutation(n)
    score = scalar_overlap(A, B, pi)
    proposal_beta = beta / proposal_temperature
    burn = int(iterations * burn_fraction)
    samples: list[Array] = []
    scores: list[float] = []
    accepted = 0
    start = time.perf_counter()
    for t in range(iterations):
        order_forward = pi.copy()
        proposed, log_q_forward = _sequential_proposal(
            A, B, order_forward, proposal_beta, rng
        )
        proposed_score = scalar_overlap(A, B, proposed)
        log_q_reverse = _sequential_log_probability(
            A, B, pi, proposed, proposal_beta
        )
        log_ratio = (
            beta * (proposed_score - score) + log_q_reverse - log_q_forward
        )
        if math.log(rng.random()) < min(0.0, float(log_ratio)):
            pi = proposed
            score = proposed_score
            accepted += 1
        if t >= burn and (t - burn) % sample_every == 0:
            samples.append(pi.copy())
            scores.append(score)
    runtime = time.perf_counter() - start
    return SamplerOutput(
        samples=np.asarray(samples, dtype=np.int16),
        scores=np.asarray(scores, dtype=float),
        acceptance_rate=accepted / iterations,
        auxiliary_rate=float("nan"),
        runtime_seconds=runtime,
        iterations=iterations,
    )


SAMPLERS: dict[str, Callable[..., SamplerOutput]] = {
    "random_swap": random_transposition_mh,
    "locally_balanced": locally_balanced_transposition_mh,
    "reinsertion_gibbs": node_reinsertion_gibbs,
    "parallel_tempering": parallel_tempering,
    "sequential_global": sequential_global_mh,
}


def lag1_autocorrelation(values: Array) -> float:
    if len(values) < 3 or np.var(values) == 0.0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def finite_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def effective_sample_size(values: Array) -> float:
    """Initial-positive-sequence ESS for one scalar trace."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 4 or np.var(x) == 0.0:
        return float("nan")
    x = x - x.mean()
    variance = float(np.dot(x, x) / n)
    correlations: list[float] = []
    for lag in range(1, n):
        autocovariance = float(np.dot(x[:-lag], x[lag:]) / (n - lag))
        correlations.append(autocovariance / variance)
    positive_sum = 0.0
    for k in range(0, len(correlations) - 1, 2):
        pair_sum = correlations[k] + correlations[k + 1]
        if pair_sum <= 0.0:
            break
        positive_sum += pair_sum
    return float(n / (1.0 + 2.0 * positive_sum))


def split_rhat(chains: Sequence[Array]) -> float:
    """Basic split-Rhat for equal-length scalar chains."""
    if len(chains) < 2:
        return float("nan")
    minimum = min(len(chain) for chain in chains)
    half = minimum // 2
    if half < 2:
        return float("nan")
    split = np.asarray(
        [piece for chain in chains for piece in (chain[:half], chain[-half:])],
        dtype=float,
    )
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    if within == 0.0:
        return 1.0 if np.var(split.mean(axis=1)) == 0.0 else float("inf")
    between = half * float(np.var(split.mean(axis=1), ddof=1))
    variance_hat = (half - 1.0) / half * within + between / half
    return float(math.sqrt(variance_hat / within))


def sampler_metrics(
    outputs: Sequence[SamplerOutput],
    exact: ExactPosterior,
    instance: Instance,
) -> dict[str, float]:
    pooled = np.vstack([output.samples for output in outputs])
    P = samples_to_marginals(pooled, instance.n)
    differences = np.abs(P - exact.marginals)
    projected = hamming_bayes_permutation(P)
    per_chain_P = [samples_to_marginals(output.samples, instance.n) for output in outputs]
    disagreement = []
    for i in range(len(per_chain_P)):
        for j in range(i + 1, len(per_chain_P)):
            disagreement.append(
                0.5 * np.abs(per_chain_P[i] - per_chain_P[j]).sum(axis=1).mean()
            )
    all_scores = np.concatenate([output.scores for output in outputs])
    ess_values = [effective_sample_size(output.scores) for output in outputs]
    finite_ess = [value for value in ess_values if np.isfinite(value)]
    total_runtime = float(sum(output.runtime_seconds for output in outputs))
    return {
        "marginal_mae": float(differences.mean()),
        "marginal_max_error": float(differences.max()),
        "mean_row_tv": float(0.5 * differences.sum(axis=1).mean()),
        "bayes_action_accuracy": float(np.mean(projected == instance.pi_true)),
        "bayes_action_disagreement": float(np.mean(projected != exact.bayes_permutation)),
        "sample_expected_planted_accuracy": float(
            np.mean(P[np.arange(instance.n), instance.pi_true])
        ),
        "expected_accuracy_bias": float(
            np.mean(P[np.arange(instance.n), instance.pi_true])
            - exact.planted_expected_accuracy
        ),
        "mean_score": float(all_scores.mean()),
        "score_bias": float(all_scores.mean() - np.dot(exact.probabilities, exact.scores)),
        "acceptance_rate": float(np.mean([output.acceptance_rate for output in outputs])),
        "auxiliary_rate": float(np.nanmean([output.auxiliary_rate for output in outputs]))
        if any(np.isfinite(output.auxiliary_rate) for output in outputs)
        else float("nan"),
        "runtime_seconds": total_runtime,
        "iterations_per_chain": float(outputs[0].iterations),
        "samples": float(len(pooled)),
        "score_lag1": finite_mean(
            lag1_autocorrelation(output.scores) for output in outputs
        ),
        "score_ess_total": float(sum(finite_ess)) if finite_ess else float("nan"),
        "score_ess_per_second": float(sum(finite_ess) / total_runtime)
        if finite_ess and total_runtime > 0.0
        else float("nan"),
        "score_split_rhat": split_rhat([output.scores for output in outputs]),
        "chain_marginal_disagreement": float(np.mean(disagreement))
        if disagreement
        else float("nan"),
    }


def validate_implementation() -> None:
    instances = generate_paired_instances(5, 0.3, [0.3], seed=917)
    instance = instances[0]
    permutations = all_permutations(instance.n)
    vectorized = vectorized_overlap_scores(instance.A, instance.B, permutations)
    check_indices = [0, 1, 17, len(permutations) - 1]
    for index in check_indices:
        scalar = scalar_overlap(instance.A, instance.B, permutations[index])
        assert scalar == vectorized[index]

    rng = np.random.default_rng(123)
    pi = rng.permutation(instance.n)
    for _ in range(100):
        u, v = rng.choice(instance.n, size=2, replace=False)
        before = scalar_overlap(instance.A, instance.B, pi)
        delta = swap_delta(instance.A, instance.B, pi, int(u), int(v))
        proposed = pi.copy()
        proposed[u], proposed[v] = proposed[v], proposed[u]
        after = scalar_overlap(instance.A, instance.B, proposed)
        assert np.isclose(after - before, delta)
        pi = proposed

    exact = exact_posterior(instance, likelihood_beta(instance.p, instance.noise))
    assert exact.marginals.shape == (instance.n, instance.n)

    streaming = exact_posterior_streaming(
        instance, likelihood_beta(instance.p, instance.noise), chunk_size=37
    )
    assert np.allclose(streaming.marginals, exact.marginals, atol=1e-12)
    assert np.allclose(
        np.dot(streaming.probabilities, streaming.scores),
        np.dot(exact.probabilities, exact.scores),
        atol=1e-12,
    )
    assert streaming.map_count == exact.map_count
    assert np.isclose(streaming.entropy, exact.entropy, atol=1e-12)

    # Confirm that the overlap Gibbs law is exactly the generator's likelihood,
    # up to a pi-independent constant.
    for left, right in [(3, 19), (7, 71), (11, 103)]:
        log_likelihood_difference = conditional_log_likelihood(
            instance, permutations[left]
        ) - conditional_log_likelihood(instance, permutations[right])
        score_difference = vectorized[left] - vectorized[right]
        assert np.isclose(log_likelihood_difference, exact.beta * score_difference)

    for v in range(instance.n):
        candidates = _cycle_reinsertion_candidates(instance.pi_true, v)
        assert len({tuple(candidate) for candidate in candidates}) == instance.n

    # A long, tiny-state run catches detailed-balance or orientation bugs cheaply.
    output = random_transposition_mh(
        instance.A,
        instance.B,
        exact.beta,
        iterations=200_000,
        burn_fraction=0.2,
        rng=np.random.default_rng(991),
        sample_every=2,
    )
    empirical = samples_to_marginals(output.samples, instance.n)
    assert np.abs(empirical - exact.marginals).mean() < 0.02


def run_benchmark(
    n: int,
    p: float,
    noises: Sequence[float],
    seeds: Sequence[int],
    samplers: Sequence[str],
    chains: int,
    iterations: int,
    burn_fraction: float,
    sample_every: int,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed_index, seed in enumerate(seeds):
        instances = generate_paired_instances(n, p, noises, int(seed))
        for instance in instances:
            beta = likelihood_beta(instance.p, instance.noise)
            exact_start = time.perf_counter()
            exact = exact_posterior_auto(instance, beta)
            exact_runtime = time.perf_counter() - exact_start
            exact_beta2 = exact_posterior_auto(instance, 2.0)

            common = {
                "n": n,
                "p": p,
                "noise": instance.noise,
                "correlation": 1.0 - instance.noise,
                "seed": int(seed),
                "beta_likelihood": beta,
                "exact_runtime_seconds": exact_runtime,
                "exact_expected_planted_accuracy": exact.planted_expected_accuracy,
                "exact_bayes_accuracy": exact.bayes_accuracy,
                "exact_planted_probability": exact.planted_probability,
                "exact_max_state_probability": exact.max_state_probability,
                "exact_effective_states": exact.effective_states,
                "exact_map_count": exact.map_count,
            }

            target_configurations: list[tuple[str, float, ExactPosterior]] = [
                ("correct_target", beta, exact),
            ]
            if "random_swap" in samplers:
                # These two controls decompose temperature misspecification from mixing.
                target_configurations.extend(
                    [
                        ("report_beta2_vs_true", 2.0, exact),
                        ("report_beta2_own_target", 2.0, exact_beta2),
                    ]
                )

            for sampler_name in samplers:
                configurations = (
                    target_configurations
                    if sampler_name == "random_swap"
                    else target_configurations[:1]
                )
                for target_label, sampler_beta, comparison_exact in configurations:
                    outputs: list[SamplerOutput] = []
                    for chain in range(chains):
                        chain_seed = (
                            10_000_019 * (seed_index + 1)
                            + 100_003 * int(round(instance.noise * 1000))
                            + 997 * chain
                            + 37 * list(SAMPLERS).index(sampler_name)
                            + (1 if target_label.startswith("report") else 0)
                        )
                        kwargs = {}
                        if sampler_name == "parallel_tempering":
                            kwargs["replicas"] = 8
                        if sampler_name == "sequential_global":
                            kwargs["proposal_temperature"] = 2.0
                        output = SAMPLERS[sampler_name](
                            instance.A,
                            instance.B,
                            sampler_beta,
                            iterations=iterations,
                            burn_fraction=burn_fraction,
                            rng=np.random.default_rng(chain_seed),
                            sample_every=sample_every,
                            **kwargs,
                        )
                        outputs.append(output)
                    metrics = sampler_metrics(outputs, comparison_exact, instance)
                    rows.append(
                        {
                            **common,
                            "sampler": sampler_name,
                            "target_label": target_label,
                            "sampler_beta": sampler_beta,
                            **metrics,
                        }
                    )

            frame = pd.DataFrame(rows)
            frame.to_csv(output_dir / "benchmark_raw.csv", index=False)
            print(
                f"completed seed={seed} noise={instance.noise:.2f} "
                f"({len(rows)} rows total)",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    numeric = [
        "marginal_mae",
        "marginal_max_error",
        "mean_row_tv",
        "bayes_action_accuracy",
        "bayes_action_disagreement",
        "sample_expected_planted_accuracy",
        "expected_accuracy_bias",
        "acceptance_rate",
        "auxiliary_rate",
        "runtime_seconds",
        "score_ess_per_second",
        "score_split_rhat",
        "chain_marginal_disagreement",
        "exact_expected_planted_accuracy",
        "exact_bayes_accuracy",
        "exact_effective_states",
    ]
    summary = (
        frame.groupby(["noise", "sampler", "target_label"], as_index=False)[numeric]
        .agg(["mean", "std", "median"])
    )
    summary.to_csv(output_dir / "benchmark_summary.csv")
    metadata = {
        "n": n,
        "p": p,
        "noises": list(noises),
        "seeds": list(map(int, seeds)),
        "samplers": list(samplers),
        "chains": chains,
        "iterations": iterations,
        "burn_fraction": burn_fraction,
        "sample_every": sample_every,
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--noises", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=list(SAMPLERS),
        default=list(SAMPLERS),
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--burn-fraction", type=float, default=0.25)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tmp/benchmark_results")
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_validation:
        validate_implementation()
        print("implementation validation passed", flush=True)
    run_benchmark(
        n=args.n,
        p=args.p,
        noises=args.noises,
        seeds=args.seeds,
        samplers=args.samplers,
        chains=args.chains,
        iterations=args.iterations,
        burn_fraction=args.burn_fraction,
        sample_every=args.sample_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
