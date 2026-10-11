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


# ---------------------------------------------------------------------------------------------
# Experiment 3, rung B (free logits): same search protocol and effort as PI-GNN's (fairness rule
# "tune each method separately on the same disjoint tuning set ... equal tuning effort"):
# stage 1 = 3 reference configs + 40 random draws, stage 2 = top 6 of stage 1 + 4 draws near them.
# Starting point during tuning = PI-GNN's Z0 for the same graph and seed (B's start on DIMACS).


@dataclass
class FreeTuneConfig:
    lr: float
    optimizer: str  # "adam" | "adamw"
    weight_decay: float  # 0 for adam


FREE_REFERENCE_CONFIGS = [
    FreeTuneConfig(lr=0.019615814221240105, optimizer="adamw", weight_decay=0.01),  # PI-GNN's frozen lr + optimiser
    FreeTuneConfig(lr=0.01, optimizer="adam", weight_decay=0.0),
    FreeTuneConfig(lr=0.1, optimizer="adam", weight_decay=0.0),
]

FREE_LR_RANGE = (1e-3, 1.0)
FREE_WD_RANGE = (1e-4, 1e-1)


def _draw_free(rng, lr_range, wd_range, optimizers=("adam", "adamw")) -> FreeTuneConfig:
    opt = str(rng.choice(optimizers))
    lr = float(np.exp(rng.uniform(np.log(lr_range[0]), np.log(lr_range[1]))))
    wd = float(np.exp(rng.uniform(np.log(wd_range[0]), np.log(wd_range[1])))) if opt == "adamw" else 0.0
    return FreeTuneConfig(lr=lr, optimizer=opt, weight_decay=wd)


def sample_free_logit_configs(n_random: int, seed: int = 0) -> list[FreeTuneConfig]:
    """Reference configs first, then n_random draws: optimiser uniform in {adam, adamw},
    lr log-uniform in FREE_LR_RANGE, AdamW weight decay log-uniform in FREE_WD_RANGE."""
    rng = np.random.default_rng(seed)
    return list(FREE_REFERENCE_CONFIGS) + [_draw_free(rng, FREE_LR_RANGE, FREE_WD_RANGE) for _ in range(n_random)]


def free_logit_stage2_configs(ranked: list[dict], n_top: int = 6, n_extra: int = 4, seed: int = 1) -> list[dict]:
    """Stage-2 input as for PI-GNN: the n_top best stage-1 configs (`ranked`, best first, each a
    dict with lr/optimizer/weight_decay and its stage-1 id under "config"), plus n_extra draws
    from the log-space bounding box of those top configs (optimiser drawn among theirs)."""
    top = ranked[:n_top]
    lrs = [c["lr"] for c in top]
    wds = [c["weight_decay"] for c in top if c["optimizer"] == "adamw"] or list(FREE_WD_RANGE)
    opts = sorted({c["optimizer"] for c in top})
    rng = np.random.default_rng(seed)
    out = [dict(lr=c["lr"], optimizer=c["optimizer"], weight_decay=c["weight_decay"], stage1_id=int(c["config"]))
           for c in top]
    for _ in range(n_extra):
        d = _draw_free(rng, (min(lrs), max(lrs)), (min(wds), max(wds)), optimizers=opts)
        out.append(dict(d.__dict__, stage1_id=None))
    return out
