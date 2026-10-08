from graphsml.checkpoints import log_checkpoint_grid, per_decade_for_n


def test_per_decade_boundaries():
    assert per_decade_for_n(50) == 3
    assert per_decade_for_n(100) == 3
    assert per_decade_for_n(1000) == 6
    assert per_decade_for_n(5000) == 6


def test_per_decade_monotonic_in_between():
    values = [per_decade_for_n(n) for n in (100, 300, 550, 800, 1000)]
    assert values == sorted(values)
    assert all(3 <= v <= 6 for v in values)


def test_grid_always_includes_endpoints_and_is_sorted_within_bounds():
    for t_max in (1, 5, 100, 1000, 12345):
        grid = log_checkpoint_grid(t_max, n=100)
        assert grid[0] >= 1
        assert grid[-1] == t_max
        assert grid == sorted(set(grid))
        assert all(1 <= e <= t_max for e in grid)


def test_small_graph_base_grid_is_1_2_5_per_decade():
    assert log_checkpoint_grid(1000, n=100) == [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def test_bigger_graph_grid_is_superset_of_base_grid_for_same_t_max():
    t_max = 2000
    base = set(log_checkpoint_grid(t_max, n=100))
    dense = set(log_checkpoint_grid(t_max, n=1000))
    assert base.issubset(dense)
    assert len(dense) >= len(base)


def test_t_max_of_one_gives_single_checkpoint():
    assert log_checkpoint_grid(1, n=100) == [1]
