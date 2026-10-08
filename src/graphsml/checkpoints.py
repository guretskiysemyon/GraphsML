"""Log-spaced checkpoint grid (decision 8 in docs/progress_exp1.md): a base
1-2-5-per-decade grid shared by every graph for cross-graph comparability, with
extra points for larger graphs (up to 6 per decade at n~1000). Every larger grid
is a strict superset of the base {1, 2, 5} mantissas, so points always line up
across graphs on a shared log-budget axis.
"""

from __future__ import annotations

_PER_DECADE_MANTISSAS: dict[int, list[float]] = {
    3: [1, 2, 5],
    4: [1, 2, 3, 5],
    5: [1, 2, 3, 5, 7],
    6: [1, 1.5, 2, 3, 5, 7],
}

_N_LO, _N_HI = 100, 1000
_PER_DECADE_LO, _PER_DECADE_HI = 3, 6


def per_decade_for_n(n: int) -> int:
    """~3 points/decade at n<=100, ~6 at n>=1000, linearly interpolated in between."""
    if n <= _N_LO:
        return _PER_DECADE_LO
    if n >= _N_HI:
        return _PER_DECADE_HI
    frac = (n - _N_LO) / (_N_HI - _N_LO)
    return int(round(_PER_DECADE_LO + frac * (_PER_DECADE_HI - _PER_DECADE_LO)))


def log_checkpoint_grid(t_max: int, n: int) -> list[int]:
    """Epochs 1..t_max (inclusive) at which to evaluate and save a checkpoint.
    Always includes epoch 1 and epoch t_max exactly."""
    if t_max < 1:
        raise ValueError("t_max must be >= 1")
    mantissas = _PER_DECADE_MANTISSAS[per_decade_for_n(n)]
    epochs = {1, int(t_max)}
    decade = 0
    while mantissas[0] * 10**decade <= t_max:
        for m in mantissas:
            e = round(m * 10**decade)
            if 1 <= e <= t_max:
                epochs.add(e)
        decade += 1
    return sorted(epochs)
