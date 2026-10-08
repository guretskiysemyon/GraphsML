"""Fixed-budget PI-GNN training loop, no early stopping (decision 2 in progress_exp1.md)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import argmax_colouring, conflict_count, potts_loss
from .model import PIGNN, build_mean_adjacency


@dataclass
class TrainConfig:
    k: int
    embed_dim: int
    hidden_dim: int
    dropout: float
    lr: float
    n_epochs: int
    seed: int = 0
    embed_init: str = "normal"
    patience: int = 500
    tolerance: float = 1e-4
    device: str = "cpu"


def train_pignn(adj: torch.Tensor, cfg: TrainConfig):
    """Train one PI-GNN instance to a fixed epoch budget.

    Returns a dict with: model, best_conflicts (min conflicts seen over training,
    eval-mode argmax), final_conflicts, best/final_train_conflicts (dropout-on argmax, as in the paper's code), loss_history (training-mode loss per epoch),
    paper_stop_epoch (the epoch at which the paper's early-stopping rule would have
    fired, or None if it never would have), eval_conflicts_history.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    adj = adj.to(device)
    mean_adj = build_mean_adjacency(adj)
    n = adj.shape[0]

    model = PIGNN(n, cfg.embed_dim, cfg.hidden_dim, cfg.k, cfg.dropout, cfg.embed_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    loss_history = []
    eval_conflicts_history = []
    train_conflicts_history = []
    eval_entropy_history = []
    best_conflicts = None
    best_train_conflicts = None
    paper_stop_epoch = None
    patience_counter = 0
    prev_loss = None

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(mean_adj)
        p = torch.softmax(z, dim=1)
        loss = potts_loss(p, adj)
        # paper-style measure: argmax of the same dropout-on forward pass used for the loss
        with torch.no_grad():
            n_train_conflicts = conflict_count(argmax_colouring(p), adj)
        train_conflicts_history.append(n_train_conflicts)
        if best_train_conflicts is None or n_train_conflicts < best_train_conflicts:
            best_train_conflicts = n_train_conflicts
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if prev_loss is not None:
            if loss_val > prev_loss or abs(loss_val - prev_loss) <= cfg.tolerance:
                patience_counter += 1
            else:
                patience_counter = 0
            if paper_stop_epoch is None and patience_counter >= cfg.patience:
                paper_stop_epoch = epoch
        prev_loss = loss_val

        model.eval()
        with torch.no_grad():
            z_eval = model(mean_adj)
            p_eval = torch.softmax(z_eval, dim=1)
            c_eval = argmax_colouring(p_eval)
            n_conflicts = conflict_count(c_eval, adj)
        eval_conflicts_history.append(n_conflicts)
        eval_entropy_history.append(float(-(p_eval * torch.log(p_eval.clamp_min(1e-12))).sum(dim=1).mean().item()))
        if best_conflicts is None or n_conflicts < best_conflicts:
            best_conflicts = n_conflicts

    return {
        "model": model,
        "mean_adj": mean_adj,
        "loss_history": loss_history,
        "eval_conflicts_history": eval_conflicts_history,
        "train_conflicts_history": train_conflicts_history,
        "eval_entropy_history": eval_entropy_history,
        "best_conflicts": best_conflicts,
        "final_conflicts": eval_conflicts_history[-1],
        "best_train_conflicts": best_train_conflicts,
        "final_train_conflicts": train_conflicts_history[-1],
        "paper_stop_epoch": paper_stop_epoch,
    }
