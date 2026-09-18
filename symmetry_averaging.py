"""Automorphism-aware marginal averaging for Bayesian graph alignment.

If sigma is an automorphism of A and tau is an automorphism of B, then the
overlap posterior assigns the same probability to pi and tau o pi o sigma^{-1}.
Consequently, averaging an estimated node-marginal matrix over the vertex
orbits of Aut(A) x Aut(B) is unbiased and cannot increase convex estimation
losses such as entrywise L1 error to the invariant exact marginal matrix.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def _has_automorphism_mapping(adjacency: Array, source: int, target: int) -> bool:
    """Small-graph backtracking test for an automorphism source -> target."""
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    if degrees[source] != degrees[target]:
        return False
    mapping = {source: target}
    used = {target}

    def feasible_targets(vertex: int) -> list[int]:
        options = []
        for candidate in range(n):
            if candidate in used or degrees[candidate] != degrees[vertex]:
                continue
            if all(
                adjacency[vertex, mapped_vertex]
                == adjacency[candidate, mapped_candidate]
                for mapped_vertex, mapped_candidate in mapping.items()
            ):
                options.append(candidate)
        return options

    def search() -> bool:
        if len(mapping) == n:
            return True
        choices = []
        for vertex in range(n):
            if vertex not in mapping:
                options = feasible_targets(vertex)
                if not options:
                    return False
                choices.append((len(options), -int(degrees[vertex]), vertex, options))
        _, _, vertex, options = min(choices, key=lambda item: item[:3])
        for candidate in options:
            mapping[vertex] = candidate
            used.add(candidate)
            if search():
                return True
            used.remove(candidate)
            del mapping[vertex]
        return False

    return search()


def automorphism_vertex_orbits(adjacency: Array) -> list[list[int]]:
    """Return vertex orbits using anchored graph-isomorphism queries.

    This implementation favors transparency over large-graph speed and is
    suitable for the small exact-oracle benchmark.  Each query asks whether an
    automorphism exists that maps one distinguished vertex to another.
    """
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    n = adjacency.shape[0]
    unseen = set(range(n))
    orbits: list[list[int]] = []

    while unseen:
        u = min(unseen)
        orbit = []
        for v in sorted(unseen):
            if _has_automorphism_mapping(adjacency, u, v):
                orbit.append(v)
        if not orbit:
            raise RuntimeError("failed to identify the trivial automorphism")
        orbits.append(orbit)
        unseen.difference_update(orbit)
    return orbits


def orbit_average_marginals(
    marginals: Array,
    row_orbits: list[list[int]],
    column_orbits: list[list[int]],
) -> Array:
    """Average a marginal matrix under independent row/column automorphisms."""
    P = np.asarray(marginals, dtype=float)
    result = np.empty_like(P)
    seen_rows = sorted(vertex for orbit in row_orbits for vertex in orbit)
    seen_columns = sorted(vertex for orbit in column_orbits for vertex in orbit)
    if seen_rows != list(range(P.shape[0])):
        raise ValueError("row_orbits must partition the rows")
    if seen_columns != list(range(P.shape[1])):
        raise ValueError("column_orbits must partition the columns")

    for row_orbit in row_orbits:
        for column_orbit in column_orbits:
            block = np.ix_(row_orbit, column_orbit)
            result[block] = float(P[block].mean())
    return result


def validate() -> None:
    # Two disjoint equal cliques have a transitive vertex automorphism action.
    m = 3
    A = np.zeros((2 * m, 2 * m), dtype=np.int8)
    for start in (0, m):
        A[start : start + m, start : start + m] = 1 - np.eye(m, dtype=np.int8)
    assert automorphism_vertex_orbits(A) == [list(range(2 * m))]

    rng = np.random.default_rng(3)
    P = rng.random((2 * m, 2 * m))
    P /= P.sum(axis=1, keepdims=True)
    averaged = orbit_average_marginals(
        P, automorphism_vertex_orbits(A), automorphism_vertex_orbits(A)
    )
    assert np.allclose(averaged, 1.0 / (2 * m))

    # A path has endpoint and interior-pair orbits.
    path = np.asarray(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=np.int8,
    )
    assert automorphism_vertex_orbits(path) == [[0, 3], [1, 2]]


if __name__ == "__main__":
    validate()
    print("symmetry averaging validation passed")
