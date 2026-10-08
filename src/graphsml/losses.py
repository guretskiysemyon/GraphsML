"""Potts relaxation loss and basic hard-colouring utilities.

L_Potts(P) = sum_{(u,v) in E} p_u . p_v = 0.5 * sum(A * (P P^T))

(each unordered edge counted once; A is symmetric so sum_{u,v} A[u,v] p_u.p_v
double-counts every edge, hence the factor 1/2 — matches the reference
notebook's `sum(A * (P @ P.T)) / 2`).
"""

from __future__ import annotations

import torch


def potts_loss(p: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """p: (n, k) row-stochastic soft colouring. adj: (n, n) 0/1 symmetric adjacency."""
    return 0.5 * (adj * (p @ p.T)).sum()


def argmax_colouring(p: torch.Tensor) -> torch.Tensor:
    """Hard colouring c_v = argmax_j P[v, j]. Returns shape (n,) long tensor."""
    return p.argmax(dim=1)


def conflict_count(c: torch.Tensor, adj: torch.Tensor) -> int:
    """Number of monochromatic edges under hard colouring c, each counted once."""
    same_colour = (c.unsqueeze(0) == c.unsqueeze(1)).float()
    return int((0.5 * (adj * same_colour).sum()).item())
