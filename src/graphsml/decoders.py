"""Decoders and null-P controls for Experiment 2 (RQ3): how much of the final colouring
comes from the information in the learned soft output P, and how much from the discrete
procedure applied to it. See docs/Experiment2_Steps.md.

Decoders (all take a frozen P, never retrain):
  - argmax (already in endpoint.evaluate_endpoint, reused by the runner);
  - sample_colourings: K independent colourings with c_v ~ p_v;
  - dce: derandomised conditional-expectation decoding, in saturation order (DSATUR-like)
    or in a random vertex order (null control e).
Every decoder's output can then be fed to the same strict_vertex_descent as in Experiment 1.

Null P (same shape as the trained P, built per graph and seed):
  - uniform_noise_P: (a) softmax of small Gaussian logits, i.e. uniform plus noise;
  - untrained P: (b) softmax of the run's saved z0.pt, no code needed here;
  - permute_rows: (c) rows permuted across vertices;
  - permute_colours_within_rows: (d) each row's colour coordinates permuted independently.

Self-contained: nothing is imported from geocolor or GraphExperiment.
"""

from __future__ import annotations

import numpy as np

from .endpoint import EndpointGraph

# Colour ties in D-CE: expected conflicts within this of the minimum count as tied, so
# that exactly equal sums computed in a different order are not split by rounding.
_TIE_ATOL = 1e-9


def softmax_np(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def potts_loss_np(eg: EndpointGraph, P: np.ndarray) -> float:
    """L(P) = sum over edges of p_u . p_v, the expected conflicts under independent rows."""
    total = 0.0
    for v in range(eg.n):
        nb = eg.neighbours[v]
        if nb.size:
            total += float(P[nb].sum(axis=0) @ P[v])
    return total / 2


def sample_colourings(P: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """(n_samples, n) int64: each row is one colouring with c_v drawn from p_v independently.
    RNG stream default_rng([seed, 4]), independent of the descent and control streams."""
    rng = np.random.default_rng([seed, 4])
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0  # guard rounding: every draw u < 1 lands in some colour
    u = rng.random((n_samples, P.shape[0], 1))
    return (u >= cdf[None, :, :]).sum(axis=2).astype(np.int64)


def dce(eg: EndpointGraph, P: np.ndarray, k: int, order: str = "saturation",
        seed: int | None = None, return_order: bool = False):
    """Derandomised conditional-expectation decoding (plan section 4, D-CE).

    Vertices are fixed one at a time. Unfixed vertices keep their soft rows, so the expected
    number of conflicts at v for colour c is S[v, c] = sum over neighbours u of q_u(c), with
    q_u = one-hot of u's colour if u is fixed and p_u otherwise. v is fixed to argmin_c S[v, c]
    (ties: lowest colour index). Fixing v never raises the conditional expectation of the total
    conflict count, so the result has at most L(P) conflicts (method of conditional expectations).

    order="saturation": the next vertex is the unfixed one with the most distinct colours among
    its fixed neighbours, then the largest degree, then the lowest index, as in dsatur_fixed_k.
    order="random": a uniformly random permutation, default_rng([seed, 5]) (null control e).

    Deterministic given P (and seed for the random order). Cost O(n^2 + m k).
    """
    n = eg.n
    P = np.asarray(P, dtype=np.float64)
    S = np.zeros((n, k))
    for v in range(n):
        nb = eg.neighbours[v]
        if nb.size:
            S[v] = P[nb].sum(axis=0)
    colours = np.full(n, -1, dtype=np.int64)

    if order == "random":
        if seed is None:
            raise ValueError("order='random' needs a seed")
        sequence = np.random.default_rng([seed, 5]).permutation(n)
    elif order != "saturation":
        raise ValueError(order)

    unfixed = np.ones(n, dtype=bool)
    M = np.zeros((n, k), dtype=np.int64)  # fixed neighbours per colour, for the saturation
    sat = np.zeros(n, dtype=np.int64)
    fixed_order = []
    for step in range(n):
        if order == "random":
            v = int(sequence[step])
        else:
            s = np.where(unfixed, sat, -1)
            cand = np.flatnonzero(s == s.max())
            cand = cand[eg.degree[cand] == eg.degree[cand].max()]
            v = int(cand[0])
        row = S[v]
        c = int(np.flatnonzero(row <= row.min() + _TIE_ATOL)[0])
        colours[v] = c
        unfixed[v] = False
        fixed_order.append(v)
        nb = eg.neighbours[v]
        if nb.size:
            delta = -P[v]
            delta[c] += 1.0
            S[nb] += delta
            if order == "saturation":
                sat[nb] += M[nb, c] == 0
                M[nb, c] += 1
    return (colours, fixed_order) if return_order else colours


def uniform_noise_P(n: int, k: int, seed: int, sigma: float = 0.1) -> np.ndarray:
    """Null control (a): softmax of N(0, sigma^2) logits, near-uniform rows whose only role is
    to break ties at random. RNG stream default_rng([seed, 6])."""
    rng = np.random.default_rng([seed, 6])
    return softmax_np(sigma * rng.standard_normal((n, k)))


def permute_rows(P: np.ndarray, seed: int) -> np.ndarray:
    """Null control (c): the same multiset of rows, assigned to vertices by a random
    permutation, so P loses its alignment with the graph. RNG default_rng([seed, 7])."""
    rng = np.random.default_rng([seed, 7])
    return P[rng.permutation(P.shape[0])].copy()


def permute_colours_within_rows(P: np.ndarray, seed: int) -> np.ndarray:
    """Null control (d): each row's colour coordinates permuted independently. Every vertex
    keeps its confidence and entropy; colour identities are no longer coordinated across
    vertices. RNG default_rng([seed, 8])."""
    rng = np.random.default_rng([seed, 8])
    perm = np.argsort(rng.random(P.shape), axis=1)
    return np.take_along_axis(P, perm, axis=1)
