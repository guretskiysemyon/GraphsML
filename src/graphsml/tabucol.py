"""TabuCol for fixed-k colouring (docs/Experiment1_Steps.md sections 2.6 and 15), implemented
from the published description (Hertz & de Werra 1987, with the dynamic tabu tenure of
Galinier & Hao 1999). Self-contained: nothing is imported from geocolor or GraphExperiment.

Role in Experiment 1: a *quality reference* conflict count at k = chi - 1, where the true
minimum is unknown. The difference to TabuCol is never a "gap to optimum".

Search: exactly k colours; the state is a full colouring with neighbour-colour counts
M[v, c] (section 3.2). Each iteration evaluates every move (v, c) with v a conflicted vertex
and c != colour(v); its change in total conflicts is M[v, c] - M[v, colour(v)]. The best
move is taken even if it is not improving. After v moves away from colour c_old, moving v
back to c_old is tabu for L + floor(alpha * |conflicted vertices|) iterations, L uniform in
{0, ..., L_max} (Galinier & Hao: alpha = 0.6, L_max = 9). Aspiration: a tabu move is allowed
if it gives fewer conflicts than the best colouring seen so far. If every move is tabu and
none aspirates, the best tabu move is taken. Ties between equally good moves are broken
uniformly at random with default_rng([seed, 4]). Stops at 0 conflicts, or when the
iteration or wall-clock budget runs out. Returns the best colouring seen (the solver's
answer), which is not the final state of the walk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .controls import random_colouring
from .endpoint import EndpointGraph, conflicts_from_counts, neighbour_colour_counts

_BIG = np.iinfo(np.int64).max // 4


@dataclass
class TabuResult:
    best_colours: np.ndarray
    best_conflicts: int
    start_conflicts: int
    iterations: int
    iter_at_best: int
    seconds: float  # search time (after the start colouring and M are built)
    seconds_at_best: float
    stopped_by: str  # "zero_conflicts" | "max_iters" | "time_budget"
    history: list = field(default_factory=list)  # (iteration, seconds, conflicts) at each new best


def tabucol(eg: EndpointGraph, k: int, seed: int, start: np.ndarray | None = None,
            max_iters: int | None = None, time_budget: float | None = None,
            alpha: float = 0.6, l_max: int = 9, time_check_every: int = 64) -> TabuResult:
    """Run TabuCol from `start` (default: random_colouring(n, k, seed), the same random start
    as the 'random' control) until 0 conflicts, `max_iters` iterations or `time_budget`
    seconds, whichever comes first. At least one budget must be given."""
    if max_iters is None and time_budget is None:
        raise ValueError("give max_iters and/or time_budget")
    rng = np.random.default_rng([seed, 4])
    n = eg.n
    colours = (random_colouring(n, k, seed) if start is None else np.asarray(start, dtype=np.int64)).copy()
    M = neighbour_colour_counts(eg, colours, k)
    f = conflicts_from_counts(colours, M)
    start_f = best_f = f
    best = colours.copy()
    tabu = np.zeros((n, k), dtype=np.int64)  # move (v, c) is tabu while iteration < tabu[v, c]
    vidx = np.arange(n)

    t0 = time.perf_counter()
    it = iter_at_best = 0
    seconds_at_best = 0.0
    history = [(0, 0.0, int(f))]
    stopped_by = "zero_conflicts"
    while best_f > 0:
        if max_iters is not None and it >= max_iters:
            stopped_by = "max_iters"
            break
        if time_budget is not None and it % time_check_every == 0 and time.perf_counter() - t0 >= time_budget:
            stopped_by = "time_budget"
            break
        own = M[vidx, colours]
        C = np.flatnonzero(own > 0)  # conflicted vertices; non-empty because f > 0 here
        delta = M[C] - own[C][:, None]
        delta[np.arange(C.size), colours[C]] = _BIG  # staying put is not a move
        allowed = (tabu[C] <= it) | (f + delta < best_f)  # not tabu, or aspiration
        cand = np.where(allowed, delta, _BIG)
        dmin = cand.min()
        if dmin >= _BIG:  # everything tabu and nothing aspirates: take the best tabu move
            cand = delta
            dmin = cand.min()
        rows, cols = np.nonzero(cand == dmin)
        j = int(rng.integers(rows.size))
        v, c = int(C[rows[j]]), int(cols[j])
        old = int(colours[v])
        nb = eg.neighbours[v]  # neighbours are distinct, so fancy-index updates are exact
        M[nb, old] -= 1  # same update as endpoint.apply_flip, vectorised
        M[nb, c] += 1
        colours[v] = c
        f += int(dmin)
        it += 1
        tabu[v, old] = it + int(rng.integers(0, l_max + 1)) + int(alpha * C.size)
        if f < best_f:
            best_f = f
            best = colours.copy()
            iter_at_best = it
            seconds_at_best = time.perf_counter() - t0
            history.append((it, seconds_at_best, int(f)))
    return TabuResult(best_colours=best, best_conflicts=int(best_f), start_conflicts=int(start_f),
                      iterations=it, iter_at_best=iter_at_best, seconds=time.perf_counter() - t0,
                      seconds_at_best=seconds_at_best, stopped_by=stopped_by, history=history)
