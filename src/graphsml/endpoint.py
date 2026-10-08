"""Endpoint-analysis primitives for Experiment 1 (RQ1): conflict counting, the
single-flip optimality check, strict vertex descent, and the optional soft-space
(g_tilde, H_tilde) diagnostic, combined into evaluate_endpoint().
See docs/Experiment1_Steps.md sections 3-5.

The hard-colouring analysis (conflict counts, single-flip check, vertex descent)
works on an adjacency-list representation and numpy arrays, independent of how
the colouring was produced, so the same primitives serve PI-GNN, random-start
and DSATUR endpoints. The soft diagnostic works on the dense torch adjacency
used in training, since P is a torch tensor straight out of the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch


@dataclass
class EndpointGraph:
    """Graph structure precomputed once per graph and reused across seeds/checkpoints."""

    n: int
    neighbours: list[np.ndarray]  # neighbours[v]: sorted int64 array of v's neighbours
    degree: np.ndarray  # (n,) int64
    adj: torch.Tensor  # (n, n) dense 0/1 float, only used by soft_stationarity

    @classmethod
    def from_networkx(cls, g: nx.Graph) -> "EndpointGraph":
        n = g.number_of_nodes()
        neighbours = [np.array(sorted(g.neighbors(v)), dtype=np.int64) for v in range(n)]
        degree = np.array([len(nb) for nb in neighbours], dtype=np.int64)
        adj = torch.zeros(n, n)
        for u, v in g.edges():
            adj[u, v] = adj[v, u] = 1.0
        return cls(n=n, neighbours=neighbours, degree=degree, adj=adj)


def neighbour_colour_counts(eg: EndpointGraph, colours: np.ndarray, k: int) -> np.ndarray:
    """M[v, c] = number of neighbours of v coloured c (section 3.2). O(n + m)."""
    M = np.zeros((eg.n, k), dtype=np.int64)
    for v in range(eg.n):
        nb = eg.neighbours[v]
        if nb.size:
            M[v] = np.bincount(colours[nb], minlength=k)
    return M


def conflicts_from_counts(colours: np.ndarray, M: np.ndarray) -> int:
    """Total monochromatic edges from an M consistent with colours. Each conflicting
    edge contributes 1 to both endpoints' rows, so the row-sum is always even."""
    return int(M[np.arange(len(colours)), colours].sum() // 2)


def conflict_count(eg: EndpointGraph, colours: np.ndarray) -> int:
    """Conflict count recomputed from scratch, without M (section 3.1). Used to
    cross-check the M-based bookkeeping, and as a cheap one-off entry point."""
    total = 0
    for v in range(eg.n):
        nb = eg.neighbours[v]
        if nb.size:
            total += int(np.sum(colours[nb] == colours[v]))
    return total // 2


def apply_flip(eg: EndpointGraph, colours: np.ndarray, M: np.ndarray, v: int, new_c: int) -> None:
    """Recolour vertex v to new_c, updating colours and M in place (incremental
    update, section 3.2). M[v, :] itself is untouched: it counts v's neighbours'
    colours, not v's own."""
    old_c = int(colours[v])
    if old_c == new_c:
        return
    colours[v] = new_c
    for u in eg.neighbours[v]:
        M[u, old_c] -= 1
        M[u, new_c] += 1


@dataclass
class SingleFlipReport:
    is_optimal: bool
    n_conflicted_vertices: int
    n_improvable_vertices: int
    improvable_fraction: float  # of conflicted vertices; 0.0 if none conflicted
    best_gain: int  # largest single-vertex conflict reduction available; 0 if none
    improvable_mask: np.ndarray  # (n,) bool, True iff that vertex has a strict improving move


def single_flip_report(colours: np.ndarray, M: np.ndarray) -> SingleFlipReport:
    """Section 3.3. Only vertices with M[v, colours[v]] > 0 can have an improving
    move, so only those need checking."""
    n = len(colours)
    own = M[np.arange(n), colours]
    improvable_mask = np.zeros(n, dtype=bool)
    conflicted = np.flatnonzero(own > 0)
    if conflicted.size == 0:
        return SingleFlipReport(True, 0, 0, 0.0, 0, improvable_mask)
    best = M[conflicted].min(axis=1)
    gains = own[conflicted] - best
    improvable_mask[conflicted] = gains > 0
    n_improv = int(improvable_mask.sum())
    best_gain = int(gains.max()) if n_improv else 0
    return SingleFlipReport(
        is_optimal=n_improv == 0,
        n_conflicted_vertices=int(conflicted.size),
        n_improvable_vertices=n_improv,
        improvable_fraction=n_improv / conflicted.size,
        best_gain=best_gain,
        improvable_mask=improvable_mask,
    )


@dataclass
class DescentResult:
    colours: np.ndarray
    conflicts_before: int
    conflicts_after: int
    n_moves: int
    fraction_removed: float  # 0.0 if conflicts_before == 0
    seconds: float


def strict_vertex_descent(eg: EndpointGraph, colours: np.ndarray, k: int, seed: int,
                           M: np.ndarray | None = None) -> DescentResult:
    """Section 3.4: repeatedly apply the single largest strict one-vertex improvement,
    breaking ties with a seeded RNG, until no strict improvement remains (no sideways
    moves). Maintains M incrementally and only re-examines v and its neighbours after
    each move, not the whole graph, so cost scales with the number of conflicts touched
    rather than n on every iteration.
    """
    rng = np.random.default_rng(seed)
    colours = colours.copy()
    M = neighbour_colour_counts(eg, colours, k) if M is None else M.copy()
    t0 = time.perf_counter()
    conflicts_before = conflicts_from_counts(colours, M)

    own = M[np.arange(eg.n), colours]
    conflicted = set(np.flatnonzero(own > 0).tolist())
    n_moves = 0
    while conflicted:
        best_gain = 0
        best_vertices: list[int] = []
        for v in sorted(conflicted):
            row = M[v]
            gain = int(row[colours[v]] - row.min())
            if gain > best_gain:
                best_gain = gain
                best_vertices = [v]
            elif gain == best_gain and gain > 0:
                best_vertices.append(v)
        if best_gain == 0:
            break  # no conflicted vertex has a strict improving move left
        v = int(rng.choice(best_vertices))
        row = M[v]
        best_colours = np.flatnonzero(row == row.min())
        new_c = int(rng.choice(best_colours))

        apply_flip(eg, colours, M, v, new_c)
        n_moves += 1

        for u in eg.neighbours[v]:
            u = int(u)
            if M[u, colours[u]] > 0:
                conflicted.add(u)
            else:
                conflicted.discard(u)
        if M[v, new_c] > 0:
            conflicted.add(v)
        else:
            conflicted.discard(v)

    conflicts_after = conflicts_from_counts(colours, M)
    seconds = time.perf_counter() - t0
    fraction_removed = (conflicts_before - conflicts_after) / conflicts_before if conflicts_before else 0.0
    return DescentResult(colours, conflicts_before, conflicts_after, n_moves, fraction_removed, seconds)


def soft_stationarity(eg: EndpointGraph, P: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Section 4. For each vertex v and colour c, q(v,c) = sum_{u in N(v)} p(u,c).
    g(v) = sum_c p(v,c) q(v,c) - min_c q(v,c) >= 0 is a first-order stationarity gap
    in P-space (not a local/global optimality certificate); g_tilde = g / deg(v)
    (0 for isolated vertices, by convention). H_tilde is the row entropy normalised
    by log k. Returns (g_tilde, H_tilde) as numpy arrays of shape (n,)."""
    Q = eg.adj @ P
    current = (P * Q).sum(dim=1)
    best = Q.min(dim=1).values
    g = (current - best).clamp(min=0)  # guard tiny negative floating-point error
    deg = torch.as_tensor(eg.degree, dtype=P.dtype)
    g_tilde = torch.where(deg > 0, g / deg.clamp(min=1), torch.zeros_like(g))
    logp = torch.where(P > 0, torch.log(P), torch.zeros_like(P))
    H_tilde = -(P * logp).sum(dim=1) / np.log(k)
    return g_tilde.numpy(), H_tilde.numpy()


@dataclass
class EndpointResult:
    colours: np.ndarray  # argmax hard colouring
    conflicts: int  # before descent
    single_flip: SingleFlipReport
    descent: DescentResult
    g_tilde: np.ndarray | None
    H_tilde: np.ndarray | None
    seconds: float  # wall time for this whole evaluation


def evaluate_endpoint(eg: EndpointGraph, P: torch.Tensor, k: int, seed: int,
                       soft_diagnostics: bool = True) -> EndpointResult:
    """Section 5. All endpoint diagnostics for one saved soft output P. Knows nothing
    about how P was produced, so it is reusable across PI-GNN checkpoints."""
    t0 = time.perf_counter()
    colours = P.detach().argmax(dim=1).cpu().numpy()
    M = neighbour_colour_counts(eg, colours, k)
    conflicts = conflicts_from_counts(colours, M)
    flip = single_flip_report(colours, M)
    descent = strict_vertex_descent(eg, colours, k, seed, M=M)
    g_tilde = H_tilde = None
    if soft_diagnostics:
        g_tilde, H_tilde = soft_stationarity(eg, P, k)
    seconds = time.perf_counter() - t0
    return EndpointResult(colours, conflicts, flip, descent, g_tilde, H_tilde, seconds)
