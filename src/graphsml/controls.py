"""Classical reference starts for Experiment 1's controls (docs/Experiment1_Steps.md
sections 2.3-2.5 and 9): a uniformly random k-colouring and fixed-k DSATUR (deterministic
and seeded-randomised). Each is a hard colouring that is then fed to the same
strict_vertex_descent as PI-GNN's argmax, so the comparison is about the local minima
reached from different starts, not about which method is a better solver.

Self-contained: nothing is imported from geocolor or GraphExperiment.
"""

from __future__ import annotations

import numpy as np

from .endpoint import EndpointGraph


def random_colouring(n: int, k: int, seed: int) -> np.ndarray:
    """Each vertex gets a colour drawn uniformly from {0, ..., k-1}. The RNG stream is
    keyed on [seed, 1] so it is independent of the descent RNG, which uses default_rng(seed)."""
    rng = np.random.default_rng([seed, 1])
    return rng.integers(0, k, size=n, dtype=np.int64)


def dsatur_fixed_k(eg: EndpointGraph, k: int, seed: int | None = None,
                   return_order: bool = False):
    """DSATUR restricted to k colours (section 2.4); vertices are coloured one at a time.

    Vertex rule: pick an uncoloured vertex with the largest saturation (number of distinct
    colours among its already-coloured neighbours), then the largest degree, then break the
    remaining ties by the lowest vertex index (seed=None, fully deterministic) or uniformly
    at random with default_rng([seed, 2]) (randomised DSATUR, section 2.5).

    Colour rule: the colour that has the fewest already-coloured neighbours. A colour with
    count 0 is conflict-free, so this is "lowest available colour" when one exists and
    "least-conflict colour" when all k colours clash. Ties: lowest colour index (seed=None)
    or uniformly at random (seed given).

    Returns the colouring (int64, shape (n,)), plus the colouring order if return_order.
    """
    rng = None if seed is None else np.random.default_rng([seed, 2])
    n = eg.n
    colours = np.full(n, -1, dtype=np.int64)
    uncoloured = np.ones(n, dtype=bool)
    M = np.zeros((n, k), dtype=np.int64)  # M[u, c] = coloured neighbours of u with colour c
    sat = np.zeros(n, dtype=np.int64)  # sat[u] = number of colours c with M[u, c] > 0
    order = []
    for _ in range(n):
        s = np.where(uncoloured, sat, -1)
        cand = np.flatnonzero(s == s.max())
        cand = cand[eg.degree[cand] == eg.degree[cand].max()]
        v = int(cand[0] if rng is None else rng.choice(cand))
        row = M[v]
        best = np.flatnonzero(row == row.min())
        c = int(best[0] if rng is None else rng.choice(best))
        colours[v] = c
        uncoloured[v] = False
        order.append(v)
        for u in eg.neighbours[v]:
            if M[u, c] == 0:
                sat[u] += 1
            M[u, c] += 1
    return (colours, order) if return_order else colours
