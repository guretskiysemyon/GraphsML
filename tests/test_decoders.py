import networkx as nx
import numpy as np

from graphsml.controls import dsatur_fixed_k
from graphsml.decoders import (
    dce,
    permute_colours_within_rows,
    permute_rows,
    potts_loss_np,
    sample_colourings,
    softmax_np,
    uniform_noise_P,
)
from graphsml.endpoint import EndpointGraph, conflict_count


def _random_P(n, k, seed, scale=2.0):
    return softmax_np(scale * np.random.default_rng(seed).standard_normal((n, k)))


def _graphs():
    return [nx.gnp_random_graph(60, 0.15, seed=s) for s in range(4)] + [
        nx.cycle_graph(7), nx.complete_graph(6), nx.petersen_graph(), nx.empty_graph(5)]


def test_dce_conflicts_never_exceed_potts_loss():
    for gi, g in enumerate(_graphs()):
        eg = EndpointGraph.from_networkx(g)
        for k in (2, 3, 5):
            for s, scale in enumerate((0.1, 1.0, 5.0)):
                P = _random_P(eg.n, k, 100 * gi + s, scale)
                L = potts_loss_np(eg, P)
                for order, seed in (("saturation", None), ("random", s)):
                    c = dce(eg, P, k, order=order, seed=seed)
                    assert c.min() >= 0 and c.max() < k
                    assert conflict_count(eg, c) <= L + 1e-9


def test_dce_on_one_hot_P_never_worsens_that_colouring():
    # With one-hot rows L(P) is the colouring's own conflict count, and D-CE is a sweep of
    # least-conflict recolourings, so it can only keep or improve the colouring (not copy it).
    eg = EndpointGraph.from_networkx(nx.gnp_random_graph(50, 0.2, seed=1))
    target = np.random.default_rng(0).integers(0, 4, eg.n)
    for order, seed in (("saturation", None), ("random", 3)):
        c = dce(eg, np.eye(4)[target], 4, order=order, seed=seed)
        assert conflict_count(eg, c) <= conflict_count(eg, target)
    # A proper colouring stays proper.
    proper = dsatur_fixed_k(eg, 8)
    assert conflict_count(eg, proper) == 0
    for order, seed in (("saturation", None), ("random", 3)):
        assert conflict_count(eg, dce(eg, np.eye(8)[proper], 8, order=order, seed=seed)) == 0


def test_dce_saturation_on_uniform_P_equals_deterministic_dsatur():
    # With uniform rows the soft part of S[v] is the same for every colour, so D-CE picks the
    # colour with fewest fixed neighbours in DSATUR's vertex order: exactly dsatur_fixed_k.
    for g in _graphs():
        eg = EndpointGraph.from_networkx(g)
        for k in (2, 3, 4):
            P = np.full((eg.n, k), 1.0 / k)
            c, order = dce(eg, P, k, return_order=True)
            c_ds, order_ds = dsatur_fixed_k(eg, k, return_order=True)
            assert order == order_ds and np.array_equal(c, c_ds)


def test_dce_is_exact_on_bipartite_graphs_with_two_colours():
    for g in (nx.cycle_graph(8), nx.path_graph(9), nx.balanced_tree(2, 4)):
        eg = EndpointGraph.from_networkx(g)
        assert conflict_count(eg, dce(eg, np.full((eg.n, 2), 0.5), 2)) == 0


def test_dce_random_order_is_a_seeded_permutation():
    eg = EndpointGraph.from_networkx(nx.gnp_random_graph(40, 0.2, seed=2))
    P = _random_P(eg.n, 3, 0)
    _, o1 = dce(eg, P, 3, order="random", seed=1, return_order=True)
    _, o1b = dce(eg, P, 3, order="random", seed=1, return_order=True)
    _, o2 = dce(eg, P, 3, order="random", seed=2, return_order=True)
    assert sorted(o1) == list(range(eg.n)) and o1 == o1b and o1 != o2


def test_potts_loss_matches_hard_conflicts_on_one_hot_P():
    eg = EndpointGraph.from_networkx(nx.gnp_random_graph(50, 0.2, seed=3))
    c = np.random.default_rng(1).integers(0, 3, eg.n)
    assert abs(potts_loss_np(eg, np.eye(3)[c]) - conflict_count(eg, c)) < 1e-9


def test_sampling_shape_range_reproducible_and_frequencies():
    P = np.array([[0.7, 0.2, 0.1], [0.0, 0.0, 1.0], [1 / 3, 1 / 3, 1 / 3]])
    s = sample_colourings(P, 20000, seed=0)
    assert s.shape == (20000, 3) and s.min() >= 0 and s.max() <= 2
    assert np.array_equal(s, sample_colourings(P, 20000, seed=0))
    for v in range(3):
        assert np.all(np.abs(np.bincount(s[:, v], minlength=3) / 20000 - P[v]) < 0.02)


def test_sampling_one_hot_P_reproduces_argmax():
    P = np.eye(5)[np.random.default_rng(0).integers(0, 5, 30)]
    assert np.all(sample_colourings(P, 8, seed=1) == P.argmax(axis=1))


def test_uniform_noise_P_is_near_uniform_and_seeded():
    P = uniform_noise_P(200, 6, seed=0)
    assert np.allclose(P.sum(axis=1), 1) and np.all(np.abs(P - 1 / 6) < 0.1)
    assert np.array_equal(P, uniform_noise_P(200, 6, seed=0))
    assert not np.array_equal(P, uniform_noise_P(200, 6, seed=1))


def test_permute_rows_keeps_the_multiset_of_rows():
    P = _random_P(30, 4, 0)
    Q = permute_rows(P, seed=0)
    assert not np.array_equal(P, Q)
    assert np.array_equal(np.sort(P, axis=0), np.sort(Q, axis=0))
    assert sorted(map(tuple, P)) == sorted(map(tuple, Q))


def test_permute_colours_within_rows_keeps_each_rows_values():
    P = _random_P(30, 4, 0)
    Q = permute_colours_within_rows(P, seed=0)
    assert not np.array_equal(P, Q)
    assert np.array_equal(np.sort(P, axis=1), np.sort(Q, axis=1))
