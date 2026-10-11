import itertools

import networkx as nx
import numpy as np

from graphsml.controls import dsatur_fixed_k
from graphsml.endpoint import (
    EndpointGraph,
    conflict_count,
    neighbour_colour_counts,
    single_flip_report,
)
from graphsml.tabucol import tabucol


def brute_force_min_conflicts(g: nx.Graph, k: int) -> int:
    n = g.number_of_nodes()
    edges = np.array(list(g.edges()))
    best = len(edges)
    for c in itertools.product(range(k), repeat=n):
        c = np.array(c)
        best = min(best, int(np.sum(c[edges[:, 0]] == c[edges[:, 1]])))
    return best


def test_tabucol_odd_cycle_k2_and_k4_k3_reach_the_minimum_of_one():
    for g, k in ((nx.cycle_graph(7), 2), (nx.complete_graph(4), 3)):
        res = tabucol(EndpointGraph.from_networkx(g), k, seed=0, max_iters=500)
        assert res.best_conflicts == 1


def test_tabucol_petersen_k2_reaches_three_conflicts():
    # Petersen graph: 15 edges, max cut 12, so 3 conflicts is the 2-colour minimum
    res = tabucol(EndpointGraph.from_networkx(nx.petersen_graph()), 2, seed=1, max_iters=2000)
    assert res.best_conflicts == 3


def test_tabucol_matches_brute_force_on_small_random_graphs():
    for i in range(10):
        g = nx.gnp_random_graph(8, 0.5, seed=100 + i)
        for k in (2, 3):
            res = tabucol(EndpointGraph.from_networkx(g), k, seed=i, max_iters=3000)
            assert res.best_conflicts == brute_force_min_conflicts(g, k), (i, k)


def test_tabucol_finds_proper_colouring_at_chi_and_stops_early():
    g = nx.random_regular_graph(3, 40, seed=0)
    eg = EndpointGraph.from_networkx(g)
    k = max(dsatur_fixed_k(eg, 4)) + 1  # DSATUR's colour count: a feasible k
    res = tabucol(eg, k, seed=0, max_iters=100_000)
    assert res.best_conflicts == 0 and res.stopped_by == "zero_conflicts"
    assert res.iterations < 100_000


def test_tabucol_bookkeeping_reproducibility_and_never_worse_than_start():
    g = nx.gnp_random_graph(120, 0.1, seed=7)
    eg = EndpointGraph.from_networkx(g)
    a = tabucol(eg, 4, seed=3, max_iters=3000)
    b = tabucol(eg, 4, seed=3, max_iters=3000)
    assert np.array_equal(a.best_colours, b.best_colours)
    assert [(h[0], h[2]) for h in a.history] == [(h[0], h[2]) for h in b.history]  # h[1] is wall-clock
    assert a.best_conflicts == conflict_count(eg, a.best_colours)  # M bookkeeping is exact
    assert a.best_conflicts <= a.start_conflicts
    assert a.best_colours.min() >= 0 and a.best_colours.max() <= 3
    bests = [h[2] for h in a.history]
    assert all(x > y for x, y in zip(bests, bests[1:]))  # history records strict improvements only
    assert a.history[-1][2] == a.best_conflicts and a.history[-1][0] == a.iter_at_best


def test_tabucol_best_colouring_is_single_flip_optimal():
    # by aspiration, any strictly improving move from the best state would be taken and
    # produce a new best, so a best colouring that was not the last state is single-flip optimal
    g = nx.gnp_random_graph(80, 0.2, seed=2)
    eg = EndpointGraph.from_networkx(g)
    res = tabucol(eg, 3, seed=0, max_iters=5000)
    assert res.iter_at_best < res.iterations
    M = neighbour_colour_counts(eg, res.best_colours, 3)
    assert single_flip_report(res.best_colours, M).is_optimal


def test_tabucol_time_budget_stops():
    g = nx.gnp_random_graph(200, 0.3, seed=1)
    res = tabucol(EndpointGraph.from_networkx(g), 3, seed=0, time_budget=0.5)
    assert res.stopped_by == "time_budget" and 0.5 <= res.seconds < 2.0
