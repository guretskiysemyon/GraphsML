"""Fixed-budget PI-GNN training loop, no early stopping (decision 2 in progress_exp1.md)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

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


def train_pignn_checkpointed(adj: torch.Tensor, cfg: TrainConfig, checkpoints: list[int],
                              out_dir: str | Path, graph_meta: dict | None = None,
                              resume: bool = True) -> dict:
    """Step B: fixed-budget training that saves a resumable, checkpointed record to
    `out_dir` for later offline analysis with `graphsml.endpoint.evaluate_endpoint`.

    Per checkpoint epoch (section/decision 4 in progress_exp1.md):
      - eval-mode Z and model weights -> out_dir/checkpoints/ckpt_<epoch>.pt;
      - one raw (not running-min) trajectory row -> out_dir/trajectory.jsonl,
        appended, so the file is readable mid-run and survives a kill.
    Once, before training starts: Z0 and the initial weights -> out_dir/z0.pt,
    out_dir/init_weights.pt, and static run metadata -> out_dir/meta.json.

    Resumable: out_dir/resume_state.pt holds the model, optimiser, RNG state and
    bookkeeping as of the last completed checkpoint; if `resume=True` and it
    exists, training continues from there instead of restarting at epoch 1.
    `checkpoints` must include `cfg.n_epochs` (T_max) -- see
    `graphsml.checkpoints.log_checkpoint_grid` for the standard grid, which is
    shared across graphs so budgets stay comparable; this function itself does
    not generate the grid, since that choice (and T_max itself) comes from the
    budget pilot (Step F).
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(set(checkpoints))
    if checkpoints[-1] != cfg.n_epochs:
        raise ValueError("checkpoints must include cfg.n_epochs (T_max)")

    device = torch.device(cfg.device)
    resume_path = out_dir / "resume_state.pt"
    traj_path = out_dir / "trajectory.jsonl"

    if resume and resume_path.exists():
        # map_location="cpu", not device: the saved RNG-state tensors must stay on
        # CPU (torch.set_rng_state/torch.cuda.get_rng_state always use CPU
        # ByteTensors, even for the CUDA generator's state); model/optimizer
        # tensors are moved to `device` below via load_state_dict, which copies
        # cross-device fine.
        state = torch.load(resume_path, map_location="cpu")
        adj = adj.to(device)
        mean_adj = build_mean_adjacency(adj)
        n = adj.shape[0]
        model = PIGNN(n, cfg.embed_dim, cfg.hidden_dim, cfg.k, cfg.dropout, cfg.embed_init).to(device)
        model.load_state_dict(state["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["torch_cuda_rng_state"], device)
        start_epoch = state["epoch"] + 1
        best_conflicts = state["best_conflicts"]
        paper_stop_epoch = state["paper_stop_epoch"]
        patience_counter = state["patience_counter"]
        prev_loss = state["prev_loss"]
        cumulative_seconds = state["cumulative_seconds"]
        done_checkpoints = set(state["done_checkpoints"])
        # A kill between appending a trajectory row and saving resume_state.pt leaves a
        # row for a checkpoint the resumed run will recompute; drop rows past the saved
        # epoch (and any torn last line) so each checkpoint appears exactly once.
        if traj_path.exists():
            kept = []
            for line in traj_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["epoch"] <= state["epoch"]:
                    kept.append(line)
            traj_path.write_text("".join(line + "\n" for line in kept))
    else:
        torch.manual_seed(cfg.seed)
        adj = adj.to(device)
        mean_adj = build_mean_adjacency(adj)
        n = adj.shape[0]
        model = PIGNN(n, cfg.embed_dim, cfg.hidden_dim, cfg.k, cfg.dropout, cfg.embed_init).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        start_epoch = 1
        best_conflicts = None
        paper_stop_epoch = None
        patience_counter = 0
        prev_loss = None
        cumulative_seconds = 0.0
        done_checkpoints: set[int] = set()

        model.eval()
        with torch.no_grad():
            z0 = model(mean_adj)
        torch.save(z0.cpu(), out_dir / "z0.pt")
        torch.save(model.state_dict(), out_dir / "init_weights.pt")
        meta = dict(graph_meta or {}, k=cfg.k, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
                    dropout=cfg.dropout, lr=cfg.lr, n_epochs=cfg.n_epochs, seed=cfg.seed,
                    embed_init=cfg.embed_init, patience=cfg.patience, tolerance=cfg.tolerance,
                    device=cfg.device, checkpoints=checkpoints, torch_version=torch.__version__)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    checkpoints_set = set(checkpoints)
    t0 = time.perf_counter()
    for epoch in range(start_epoch, cfg.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(mean_adj)
        p = torch.softmax(z, dim=1)
        loss = potts_loss(p, adj)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        if prev_loss is not None:
            if loss_val > prev_loss or abs(loss_val - prev_loss) <= cfg.tolerance:
                patience_counter += 1
            else:
                patience_counter = 0
            if paper_stop_epoch is None and patience_counter >= cfg.patience:
                paper_stop_epoch = epoch
        prev_loss = loss_val

        if epoch in checkpoints_set and epoch not in done_checkpoints:
            model.eval()
            with torch.no_grad():
                z_eval = model(mean_adj)
                p_eval = torch.softmax(z_eval, dim=1)
                c_eval = argmax_colouring(p_eval)
                n_conflicts = conflict_count(c_eval, adj)
                entropy = float(-(p_eval * torch.log(p_eval.clamp_min(1e-12))).sum(dim=1).mean().item())
                n_colours_used = int(torch.unique(c_eval).numel())
            if best_conflicts is None or n_conflicts < best_conflicts:
                best_conflicts = n_conflicts
            cumulative_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()

            torch.save({"epoch": epoch, "Z": z_eval.cpu(), "weights": model.state_dict()},
                       ckpt_dir / f"ckpt_{epoch:08d}.pt")
            with traj_path.open("a") as f:
                f.write(json.dumps(dict(
                    epoch=epoch, train_loss=loss_val, eval_conflicts=n_conflicts,
                    eval_entropy=entropy, n_colours_used=n_colours_used,
                    best_conflicts_so_far=best_conflicts, cumulative_seconds=cumulative_seconds,
                )) + "\n")
            done_checkpoints.add(epoch)

            # Write-then-rename so a kill mid-save (killable partition) can never leave a
            # half-written resume_state.pt that crashes every requeue.
            tmp_resume_path = resume_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "best_conflicts": best_conflicts, "paper_stop_epoch": paper_stop_epoch,
                "patience_counter": patience_counter, "prev_loss": prev_loss,
                "cumulative_seconds": cumulative_seconds, "done_checkpoints": sorted(done_checkpoints),
            }, tmp_resume_path)
            os.replace(tmp_resume_path, resume_path)

    last = json.loads(traj_path.read_text().splitlines()[-1])
    return {
        "final_conflicts": last["eval_conflicts"],
        "best_conflicts": last["best_conflicts_so_far"],
        "paper_stop_epoch": paper_stop_epoch,
        "checkpoints": checkpoints,
        "out_dir": str(out_dir),
    }
