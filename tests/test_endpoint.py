import networkx as nx
import numpy as np
import torch

from graphsml.endpoint import (
    EndpointGraph,
    apply_flip,
    conflict_count,
    conflicts_from_counts,
    evaluate_endpoint,
    neighbour_colour_counts,
    single_flip_report,
    soft_stationarity,
    strict_vertex_descent,
)


def odd_cycle(n=5):
    return nx.cycle_graph(n)


def k4():
    return nx.complete_graph(4)


# --- 1. conflict counter on hand-labelled small graphs ---------------------

def test_conflict_counter_odd_cycle_k2():
    eg = EndpointGraph.from_networkx(odd_cycle(5))
    colours = np.array([0, 1, 0, 1, 0])  # only edge (4,0) is monochromatic
    assert conflict_count(eg, colours) == 1
    M = neighbour_colour_counts(eg, colours, k=2)
    assert conflicts_from_counts(colours, M) == 1


def test_conflict_counter_k4_k3():
    eg = EndpointGraph.from_networkx(k4())
    colours = np.array([0, 1, 2, 0])  # only edge (0,3) is monochromatic
    assert conflict_count(eg, colours) == 1
    M = neighbour_colour_counts(eg, colours, k=3)
    assert conflicts_from_counts(colours, M) == 1


# --- 2. single-flip checker against exhaustive brute force -----------------

def brute_force_improvable_mask(eg: EndpointGraph, colours: np.ndarray, k: int) -> np.ndarray:
    base = conflict_count(eg, colours)
    mask = np.zeros(eg.n, dtype=bool)
    for v in range(eg.n):
        for c in range(k):
            if c == colours[v]:
                continue
            trial = colours.copy()
            trial[v] = c
            if conflict_count(eg, trial) < base:
                mask[v] = True
                break
    return mask


def test_single_flip_report_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(10):
        g = nx.gnp_random_graph(8, 0.4, seed=trial)
        eg = EndpointGraph.from_networkx(g)
        k = 3
        colours = rng.integers(0, k, size=eg.n)
        M = neighbour_colour_counts(eg, colours, k)
        report = single_flip_report(colours, M)
        expected_mask = brute_force_improvable_mask(eg, colours, k)
        assert np.array_equal(report.improvable_mask, expected_mask)
        assert report.is_optimal == (not expected_mask.any())


# --- 3. incremental M vs recomputation from scratch -------------------------

def test_incremental_M_matches_recompute_from_scratch():
    rng = np.random.default_rng(1)
    g = nx.gnp_random_graph(15, 0.3, seed=2)
    eg = EndpointGraph.from_networkx(g)
    k = 4
    colours = rng.integers(0, k, size=eg.n)
    M = neighbour_colour_counts(eg, colours, k)
    for v in rng.integers(0, eg.n, size=20):
        new_c = int(rng.integers(0, k))
        apply_flip(eg, colours, M, int(v), new_c)
        M_fresh = neighbour_colour_counts(eg, colours, k)
        assert np.array_equal(M, M_fresh)


# --- 4 & 5. vertex descent never increases conflicts, and always terminates
#            at a single-flip-optimal colouring ------------------------------

def test_vertex_descent_never_increases_conflicts_and_terminates_optimal():
    rng = np.random.default_rng(3)
    for trial in range(10):
        g = nx.gnp_random_graph(20, 0.3, seed=trial)
        eg = EndpointGraph.from_networkx(g)
        k = 3
        colours = rng.integers(0, k, size=eg.n)
        result = strict_vertex_descent(eg, colours, k, seed=trial)
        assert result.conflicts_after <= result.conflicts_before
        M_final = neighbour_colour_counts(eg, result.colours, k)
        assert conflicts_from_counts(result.colours, M_final) == result.conflicts_after
        assert single_flip_report(result.colours, M_final).is_optimal


# --- 6. fixed seeds reproduce tie-breaking ----------------------------------

def test_fixed_seed_reproduces_descent():
    g = k4()  # fully symmetric: lots of genuine ties at k=2
    eg = EndpointGraph.from_networkx(g)
    colours = np.array([0, 0, 1, 1])
    r1 = strict_vertex_descent(eg, colours, k=2, seed=42)
    r2 = strict_vertex_descent(eg, colours, k=2, seed=42)
    assert np.array_equal(r1.colours, r2.colours)
    assert r1.n_moves == r2.n_moves


# --- 7. one-hot P: soft gap agrees with the hard single-flip improvement ----

def test_soft_gap_matches_hard_gain_for_one_hot_P():
    rng = np.random.default_rng(4)
    g = nx.gnp_random_graph(10, 0.4, seed=5)
    eg = EndpointGraph.from_networkx(g)
    k = 3
    colours = rng.integers(0, k, size=eg.n)
    M = neighbour_colour_counts(eg, colours, k)
    P = torch.nn.functional.one_hot(torch.as_tensor(colours), num_classes=k).float()
    g_tilde, _ = soft_stationarity(eg, P, k)
    for v in range(eg.n):
        if eg.degree[v] == 0:
            continue
        hard_gain = int(M[v, colours[v]] - M[v].min())
        assert abs(g_tilde[v] * eg.degree[v] - hard_gain) < 1e-5


# --- 8. uniform P: g_tilde = 0, H_tilde = 1 ----------------------------------

def test_uniform_P_gives_zero_g_tilde_and_unit_entropy():
    g = nx.gnp_random_graph(12, 0.3, seed=6)
    eg = EndpointGraph.from_networkx(g)
    k = 4
    P = torch.full((eg.n, k), 1.0 / k)
    g_tilde, H_tilde = soft_stationarity(eg, P, k)
    assert np.allclose(g_tilde, 0.0, atol=1e-6)
    assert np.allclose(H_tilde, 1.0, atol=1e-6)


# --- evaluate_endpoint: smoke test tying everything together ---------------

def test_evaluate_endpoint_runs_and_is_internally_consistent():
    g = odd_cycle(5)
    eg = EndpointGraph.from_networkx(g)
    k = 2
    torch.manual_seed(0)
    Z = torch.randn(eg.n, k)
    P = torch.softmax(Z, dim=1)
    result = evaluate_endpoint(eg, P, k, seed=0)
    assert result.conflicts == conflict_count(eg, result.colours)
    assert result.descent.conflicts_before == result.conflicts
    assert result.descent.conflicts_after <= result.conflicts
    assert result.g_tilde.shape == (eg.n,)
    assert result.H_tilde.shape == (eg.n,)
