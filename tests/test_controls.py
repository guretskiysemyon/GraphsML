import networkx as nx
import numpy as np

from graphsml.controls import dsatur_fixed_k, random_colouring
from graphsml.endpoint import (
    EndpointGraph,
    conflict_count,
    neighbour_colour_counts,
    single_flip_report,
    strict_vertex_descent,
)


def test_random_colouring_range_shape_and_reproducible():
    c = random_colouring(500, 7, seed=3)
    assert c.shape == (500,) and c.min() >= 0 and c.max() <= 6
    assert np.array_equal(c, random_colouring(500, 7, seed=3))
    assert not np.array_equal(c, random_colouring(500, 7, seed=4))


def test_random_colouring_is_roughly_uniform():
    counts = np.bincount(random_colouring(20000, 5, seed=0), minlength=5)
    assert np.all(np.abs(counts / 20000 - 0.2) < 0.02)


def test_dsatur_is_exact_on_bipartite_graphs_with_two_colours():
    for g in (nx.cycle_graph(6), nx.path_graph(9), nx.balanced_tree(2, 4)):
        eg = EndpointGraph.from_networkx(g)
        assert conflict_count(eg, dsatur_fixed_k(eg, 2)) == 0


def test_dsatur_k4_with_three_colours_has_exactly_one_conflict():
    eg = EndpointGraph.from_networkx(nx.complete_graph(4))
    colours = dsatur_fixed_k(eg, 3)
    assert conflict_count(eg, colours) == 1  # K4 cannot be 3-coloured; 1 is the minimum


def test_dsatur_odd_cycle_with_two_colours_hand_traced():
    eg = EndpointGraph.from_networkx(nx.cycle_graph(5))
    colours, order = dsatur_fixed_k(eg, 2, return_order=True)
    assert order == [0, 1, 2, 3, 4]
    assert colours.tolist() == [0, 1, 0, 1, 0]  # last vertex has both colours blocked -> lowest index
    assert conflict_count(eg, colours) == 1


def test_dsatur_colours_every_vertex_within_range():
    g = nx.gnp_random_graph(60, 0.3, seed=1)
    eg = EndpointGraph.from_networkx(g)
    for k in (3, 6, 12):
        colours = dsatur_fixed_k(eg, k)
        assert colours.shape == (60,) and colours.min() >= 0 and colours.max() < k


def test_dsatur_prefers_saturation_over_degree():
    # vertex 2 is a degree-4 hub, 3..6 are leaves, 0-1 is a separate edge with 1 of degree 2
    # (1 also touches 7). After the hub is coloured the leaves (saturation 1) must all come
    # before vertex 1 (higher degree but saturation 0).
    g = nx.Graph([(2, 3), (2, 4), (2, 5), (2, 6), (0, 1), (1, 7)])
    eg = EndpointGraph.from_networkx(g)
    _, order = dsatur_fixed_k(eg, 3, return_order=True)
    assert order[:5] == [2, 3, 4, 5, 6]


def test_dsatur_deterministic_and_seeded_randomised_reproducible():
    eg = EndpointGraph.from_networkx(nx.gnp_random_graph(40, 0.15, seed=2))
    assert np.array_equal(dsatur_fixed_k(eg, 5), dsatur_fixed_k(eg, 5))
    assert np.array_equal(dsatur_fixed_k(eg, 5, seed=7), dsatur_fixed_k(eg, 5, seed=7))
    variants = {tuple(dsatur_fixed_k(eg, 5, seed=s)) for s in range(10)}
    assert len(variants) > 1  # randomised tie-breaks actually change the colouring


def test_controls_feed_descent_to_a_single_flip_optimal_colouring():
    g = nx.gnp_random_graph(50, 0.25, seed=5)
    eg = EndpointGraph.from_networkx(g)
    k = 4
    for start in (random_colouring(50, k, seed=0), dsatur_fixed_k(eg, k), dsatur_fixed_k(eg, k, seed=0)):
        res = strict_vertex_descent(eg, start, k, seed=0)
        assert res.conflicts_after <= res.conflicts_before
        assert single_flip_report(res.colours, neighbour_colour_counts(eg, res.colours, k)).is_optimal


def test_dsatur_rand_v_random_vertex_ties_lowest_colour_rule():
    g = nx.gnp_random_graph(60, 0.15, seed=5)
    eg = EndpointGraph.from_networkx(g)
    k = 4
    colours, order = dsatur_fixed_k(eg, k, seed=3, return_order=True, random_colour_ties=False)
    assert np.array_equal(colours, dsatur_fixed_k(eg, k, seed=3, random_colour_ties=False))
    assert colours.min() >= 0 and colours.max() < k
    # replay the order: every vertex got the lowest-index colour among the least-conflict ones
    partial = np.full(eg.n, -1)
    for v in order:
        nb = eg.neighbours[v]
        counts = np.bincount(partial[nb][partial[nb] >= 0], minlength=k)
        assert colours[v] == int(np.flatnonzero(counts == counts.min())[0])
        partial[v] = colours[v]
    # vertex ties are random: different seeds give different orders on this graph
    orders = {tuple(dsatur_fixed_k(eg, k, seed=s, return_order=True, random_colour_ties=False)[1])
              for s in range(5)}
    assert len(orders) > 1


def test_dsatur_rand_v_is_exact_on_bipartite_graphs_and_differs_from_dsatur_rand():
    for g in (nx.cycle_graph(6), nx.path_graph(9), nx.balanced_tree(2, 4)):
        eg = EndpointGraph.from_networkx(g)
        for s in range(3):
            assert conflict_count(eg, dsatur_fixed_k(eg, 2, seed=s, random_colour_ties=False)) == 0
    eg = EndpointGraph.from_networkx(nx.gnp_random_graph(40, 0.2, seed=0))
    differs = [not np.array_equal(dsatur_fixed_k(eg, 3, seed=s), dsatur_fixed_k(eg, 3, seed=s, random_colour_ties=False))
               for s in range(5)]
    assert any(differs)
