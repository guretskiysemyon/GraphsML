"""Lean PI-GNN trainer and config sampler for hyperparameter search.

Same model/loss/optimiser as train.py, but no per-epoch host syncs: the eval-mode
argmax is evaluated only every `eval_every` epochs. Not used for the main experiment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .losses import argmax_colouring, conflict_count, potts_loss
from .model import PIGNN, build_mean_adjacency


@dataclass
class TuneConfig:
    lr: float
    hidden_dim: int
    embed_dim: int
    dropout: float


# Table III configs used as reference points in the search (not tuned on our test graphs here).
REFERENCE_CONFIGS = [
    TuneConfig(lr=0.03507, hidden_dim=22, embed_dim=43, dropout=0.3298),   # anna
    TuneConfig(lr=0.02175, hidden_dim=12, embed_dim=67, dropout=0.2339),   # queen7_7
    TuneConfig(lr=0.01730, hidden_dim=23, embed_dim=107, dropout=0.1796),  # queen8_12
]


def sample_configs(n_random: int, seed: int = 0) -> list[TuneConfig]:
    """Reference configs first, then n_random log-uniform draws. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    cfgs = list(REFERENCE_CONFIGS)
    for _ in range(n_random):
        cfgs.append(TuneConfig(
            lr=float(np.exp(rng.uniform(np.log(3e-3), np.log(0.15)))),
            hidden_dim=int(round(np.exp(rng.uniform(np.log(16), np.log(256))))),
            embed_dim=int(round(np.exp(rng.uniform(np.log(16), np.log(128))))),
            dropout=float(rng.uniform(0.05, 0.5)),
        ))
    return cfgs


def train_lean(adj: torch.Tensor, k: int, cfg: TuneConfig, n_epochs: int, seed: int,
               device: str, eval_every: int = 500) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)
    adj = adj.to(dev)
    mean_adj = build_mean_adjacency(adj)
    model = PIGNN(adj.shape[0], cfg.embed_dim, cfg.hidden_dim, k, cfg.dropout, "normal").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    eval_epochs, eval_conflicts, eval_entropy, n_colours = [], [], [], []
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = potts_loss(torch.softmax(model(mean_adj), dim=1), adj)
        loss.backward()
        opt.step()
        if epoch % eval_every == 0 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(mean_adj), dim=1)
                c = argmax_colouring(p)
                eval_epochs.append(epoch)
                eval_conflicts.append(conflict_count(c, adj))
                eval_entropy.append(float(-(p * torch.log(p.clamp_min(1e-12))).sum(1).mean().item()))
                n_colours.append(int(c.unique().numel()))
    return dict(eval_epochs=eval_epochs, eval_conflicts=eval_conflicts, eval_entropy=eval_entropy,
                n_colours=n_colours, final_loss=float(loss.item()), seconds=time.time() - t0)
