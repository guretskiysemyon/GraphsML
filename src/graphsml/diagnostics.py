"""Always-logged Experiment 3 diagnostics (plan section 6, 4C and 4E), shared by every arm
(PI-GNN, free logits, random-start free logits) so they are computed identically:

  4C smoothness  Dirichlet energy of P, E(P) = sum_{(u,v) in E} ||p_u - p_v||^2 = tr(P^T L P),
                 next to the Potts loss, hard conflicts and row entropy;
  4E movement    distance of the logits Z (and, for PI-GNN, of the parameters theta) from
                 their values at initialisation.

For a one-hot P, ||p_u - p_v||^2 is 2 on a bichromatic edge and 0 on a conflict, so
E(P) = 2 (m - conflicts): at hard points the Dirichlet energy carries no information beyond
the conflict count. It is informative while P is still soft.
"""

from __future__ import annotations

import torch

from .losses import argmax_colouring, conflict_count, potts_loss


def dirichlet_energy(p: torch.Tensor, adj: torch.Tensor) -> float:
    """sum over unordered edges of ||p_u - p_v||^2 = sum_u deg(u) ||p_u||^2 - sum_{u,v} A_uv p_u.p_v."""
    deg = adj.sum(dim=1)
    return float((deg * (p * p).sum(dim=1)).sum() - (adj * (p @ p.T)).sum())


def logit_diagnostics(z: torch.Tensor, z0: torch.Tensor, adj: torch.Tensor) -> dict:
    """Everything about one saved logit matrix Z that does not need the hard-colouring harness
    (endpoint.evaluate_endpoint covers single-flip optimality and descent). Pass float64
    tensors for offline analysis."""
    p = torch.softmax(z, dim=1)
    c = argmax_colouring(p)
    top2 = z.topk(2, dim=1).values
    pmax = p.max(dim=1).values
    dz = z - z0
    # softmax ignores per-row shifts, so also report movement of the row-centred logits
    dzc = dz - dz.mean(dim=1, keepdim=True)
    z0c = z0 - z0.mean(dim=1, keepdim=True)
    return dict(
        eval_loss=float(potts_loss(p, adj)),
        eval_conflicts=conflict_count(c, adj),
        eval_entropy=float(-(p * torch.log(p.clamp_min(1e-300))).sum(dim=1).mean()),
        n_colours_used=int(torch.unique(c).numel()),
        dirichlet=dirichlet_energy(p, adj),
        logit_dist=float(dz.norm()),
        logit_dist_rel=float(dz.norm() / z0.norm().clamp_min(1e-300)),
        logit_dist_centred_rel=float(dzc.norm() / z0c.norm().clamp_min(1e-300)),
        logit_norm=float(z.norm()),
        margin_median=float((top2[:, 0] - top2[:, 1]).median()),
        frac_pmax_099=float((pmax >= 0.99).double().mean()),
    )


def param_distance(state: dict, state0: dict) -> dict:
    """||theta - theta0|| over all parameter tensors, absolute and relative to ||theta0||,
    plus the relative distance per tensor (keys as in the state dict)."""
    sq, sq0, per = 0.0, 0.0, {}
    for name, t0 in state0.items():
        d = (state[name].double() - t0.double()).norm()
        n0 = t0.double().norm()
        sq += float(d) ** 2
        sq0 += float(n0) ** 2
        per[f"param_dist_rel/{name}"] = float(d / n0) if n0 > 0 else None
    return dict(param_dist=sq ** 0.5, param_dist_rel=(sq / sq0) ** 0.5 if sq0 > 0 else None, **per)
