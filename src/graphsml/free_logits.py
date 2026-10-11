"""Experiment 3 (RQ2), rung B: optimise one logit vector per vertex directly,
P = softmax(Z), Z in R^(n x k), on the same Potts loss as PI-GNN (plan section 5, Experiment 3).

Arms built from this module (rung A, PI-GNN, is the Experiment 1 sweep in results/main):
  B       Z starts from PI-GNN's exact initial eval-mode logits Z0 (z0.pt of the same graph
          and seed; fairness rule "start from the exact same initial logits");
  B-rand  Z starts from random logits with Z0's scale (`matched_random_logits`).
B against A isolates the dynamics, B against B-rand the starting point.

Same conventions as train.py so the arms are comparable: one full-batch gradient step per
"epoch", the same dense `potts_loss`, the same device, `loss.item()` every step (A does it
too, so per-step synchronisation costs are alike), the same log checkpoint grid, the same
run-directory layout (checkpoints/ckpt_<epoch>.pt holding "Z", trajectory.jsonl with
"epoch"/"eval_entropy", DONE with "t_max"), so scripts/run_soft.py and the endpoint
evaluation work unchanged on free-logit runs. The paper's patience rule is logged, not applied.

Matched wall-clock time (fairness rule "report both matched optimisation steps and matched
wall-clock time"): `time_targets` maps a label (e.g. "A@100000") to PI-GNN's cumulative
training seconds at that checkpoint; an extra checkpoint is saved at the first step whose
cumulative time reaches each target, and training continues past the step budget until
every target is reached (or `max_steps`). Saving extra checkpoints does not change the
optimisation trajectory, only which steps are recorded.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .losses import argmax_colouring, conflict_count, potts_loss
from .model import PIGNN, build_mean_adjacency


@dataclass
class FreeLogitConfig:
    k: int
    lr: float
    n_steps: int
    optimizer: str = "adam"  # "adam" | "adamw" | "sgd"
    weight_decay: float = 0.0  # L2 for adam/sgd, decoupled for adamw
    momentum: float = 0.0  # sgd only
    logit_noise: float = 0.0  # std of Gaussian noise added to Z in the training forward pass only (noise control)
    seed: int = 0
    patience: int = 500
    tolerance: float = 1e-4
    device: str = "cpu"


def make_optimizer(params, cfg: FreeLogitConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def pignn_initial_logits(adj: torch.Tensor, k: int, embed_dim: int, hidden_dim: int, dropout: float,
                         seed: int, device: str = "cpu", embed_init: str = "normal") -> torch.Tensor:
    """PI-GNN's initial eval-mode logits f_theta0(G, X), built exactly as
    train.train_pignn_checkpointed builds and saves z0.pt (same seeding and construction
    order). Used on the tuning graphs, which have no saved Z0; on DIMACS load z0.pt instead."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    adj = adj.to(dev)
    mean_adj = build_mean_adjacency(adj)
    model = PIGNN(adj.shape[0], embed_dim, hidden_dim, k, dropout, embed_init).to(dev)
    model.eval()
    with torch.no_grad():
        return model(mean_adj).cpu()


def matched_random_logits(z0: torch.Tensor, seed: int) -> torch.Tensor:
    """B-rand start: i.i.d. Gaussian logits, row-centred and rescaled so their entry-wise std
    equals that of Z0's row-centred logits (softmax ignores per-row shifts, so this is the
    scale that matters). RNG stream default_rng([seed, 5]) (streams 1-4 are used by the
    Experiment 1 controls and TabuCol)."""
    r = np.random.default_rng([seed, 5]).standard_normal(tuple(z0.shape))
    r -= r.mean(axis=1, keepdims=True)
    z0c = (z0.double() - z0.double().mean(dim=1, keepdim=True)).numpy()
    r *= z0c.std() / r.std()
    return torch.as_tensor(r, dtype=z0.dtype)


def _noise_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(np.random.default_rng([seed, 6]).integers(2**62)))
    return gen


def _train_forward(z: torch.Tensor, cfg: FreeLogitConfig, gen: torch.Generator | None) -> torch.Tensor:
    if cfg.logit_noise > 0:
        z = z + cfg.logit_noise * torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
    return torch.softmax(z, dim=1)


def train_free_logits_lean(adj: torch.Tensor, z_init: torch.Tensor, cfg: FreeLogitConfig,
                           eval_every: int = 500) -> dict:
    """Tuning counterpart of tune.train_lean: same outputs, no checkpoints."""
    dev = torch.device(cfg.device)
    adj = adj.to(dev)
    z = torch.nn.Parameter(z_init.clone().to(dev))
    opt = make_optimizer([z], cfg)
    gen = _noise_generator(cfg.seed, dev) if cfg.logit_noise > 0 else None
    eval_epochs, eval_conflicts, eval_entropy, n_colours = [], [], [], []
    t0 = time.time()
    for step in range(1, cfg.n_steps + 1):
        opt.zero_grad()
        loss = potts_loss(_train_forward(z, cfg, gen), adj)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == cfg.n_steps:
            with torch.no_grad():
                p = torch.softmax(z, dim=1)
                c = argmax_colouring(p)
                eval_epochs.append(step)
                eval_conflicts.append(conflict_count(c, adj))
                eval_entropy.append(float(-(p * torch.log(p.clamp_min(1e-12))).sum(1).mean().item()))
                n_colours.append(int(c.unique().numel()))
    return dict(eval_epochs=eval_epochs, eval_conflicts=eval_conflicts, eval_entropy=eval_entropy,
                n_colours=n_colours, final_loss=float(loss.item()), seconds=time.time() - t0)


def train_free_logits_checkpointed(adj: torch.Tensor, z_init: torch.Tensor, cfg: FreeLogitConfig,
                                   checkpoints: list[int], out_dir: str | Path,
                                   time_targets: dict[str, float] | None = None,
                                   max_steps: int | None = None, graph_meta: dict | None = None,
                                   resume: bool = True) -> dict:
    """Free-logit counterpart of train.train_pignn_checkpointed (same files, same resume
    semantics). Saves Z at every step in `checkpoints` (which must include cfg.n_steps), at the
    first step reaching each time target, and at the final step. Stops at the first step
    >= cfg.n_steps by which every time target is reached, or at `max_steps` (default:
    cfg.n_steps); checkpoints beyond the stopping step are not saved, so pass the log grid
    up to max_steps. Resumable from out_dir/resume_state.pt, and extendable in place to a
    larger n_steps / new targets.
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(set(checkpoints))
    max_steps = cfg.n_steps if max_steps is None else max_steps
    if cfg.n_steps not in checkpoints:
        raise ValueError("checkpoints must include cfg.n_steps")
    if max_steps < cfg.n_steps:
        raise ValueError("max_steps must be >= cfg.n_steps")
    targets = dict(time_targets or {})

    device = torch.device(cfg.device)
    adj = adj.to(device)
    resume_path = out_dir / "resume_state.pt"
    traj_path = out_dir / "trajectory.jsonl"
    z = torch.nn.Parameter(z_init.clone().to(device))
    optimizer = make_optimizer([z], cfg)
    gen = _noise_generator(cfg.seed, device) if cfg.logit_noise > 0 else None

    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        with torch.no_grad():
            z.copy_(state["Z"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        if gen is not None:
            gen.set_state(state["noise_rng_state"])
        last_step = state["epoch"]
        best_conflicts = state["best_conflicts"]
        paper_stop_epoch = state["paper_stop_epoch"]
        patience_counter = state["patience_counter"]
        prev_loss = state["prev_loss"]
        cumulative_seconds = state["cumulative_seconds"]
        done_checkpoints = set(state["done_checkpoints"])
        done_targets = set(state["done_targets"])
        if traj_path.exists():  # drop rows written after the saved state (see train.py)
            kept = []
            for line in traj_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["epoch"] <= last_step:
                    kept.append(line)
            traj_path.write_text("".join(line + "\n" for line in kept))
    else:
        last_step = 0
        best_conflicts = None
        paper_stop_epoch = None
        patience_counter = 0
        prev_loss = None
        cumulative_seconds = 0.0
        done_checkpoints: set[int] = set()
        done_targets: set[str] = set()
        torch.save(z_init.detach().cpu(), out_dir / "z0.pt")
        meta = dict(graph_meta or {}, **asdict(cfg), checkpoints=checkpoints, max_steps=max_steps,
                    time_targets=targets, torch_version=torch.__version__)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    def finished(step: int, pending: set[str]) -> bool:
        return step >= max_steps or (step >= cfg.n_steps and not pending)

    if last_step > 0 and finished(last_step, set(targets) - done_targets):
        start = max_steps + 1  # already complete for this budget and these targets: no-op
    else:
        start = last_step + 1

    stop_reason = "already_complete" if start > max_steps else None
    t0 = time.perf_counter()
    for step in range(start, max_steps + 1):
        optimizer.zero_grad()
        loss = potts_loss(_train_forward(z, cfg, gen), adj)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        if prev_loss is not None:
            if loss_val > prev_loss or abs(loss_val - prev_loss) <= cfg.tolerance:
                patience_counter += 1
            else:
                patience_counter = 0
            if paper_stop_epoch is None and patience_counter >= cfg.patience:
                paper_stop_epoch = step
        prev_loss = loss_val

        elapsed = cumulative_seconds + time.perf_counter() - t0
        crossed = sorted(lab for lab, sec in targets.items() if lab not in done_targets and elapsed >= sec)
        is_last = finished(step, set(targets) - done_targets - set(crossed))
        if (step in checkpoints and step not in done_checkpoints) or crossed or is_last:
            with torch.no_grad():
                p_eval = torch.softmax(z, dim=1)
                c_eval = argmax_colouring(p_eval)
                n_conflicts = conflict_count(c_eval, adj)
                eval_loss = float(potts_loss(p_eval, adj).item())
                entropy = float(-(p_eval * torch.log(p_eval.clamp_min(1e-12))).sum(dim=1).mean().item())
                n_colours_used = int(torch.unique(c_eval).numel())
            if best_conflicts is None or n_conflicts < best_conflicts:
                best_conflicts = n_conflicts
            cumulative_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()

            torch.save({"epoch": step, "Z": z.detach().cpu().clone(), "time_matched": crossed},
                       ckpt_dir / f"ckpt_{step:08d}.pt")
            with traj_path.open("a") as f:
                f.write(json.dumps(dict(
                    epoch=step, train_loss=loss_val, eval_loss=eval_loss, eval_conflicts=n_conflicts,
                    eval_entropy=entropy, n_colours_used=n_colours_used, best_conflicts_so_far=best_conflicts,
                    cumulative_seconds=cumulative_seconds, time_matched=crossed,
                )) + "\n")
            done_checkpoints.add(step)
            done_targets.update(crossed)

            tmp_resume_path = resume_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": step, "Z": z.detach().cpu().clone(), "optimizer": optimizer.state_dict(),
                "noise_rng_state": gen.get_state() if gen is not None else None,
                "best_conflicts": best_conflicts, "paper_stop_epoch": paper_stop_epoch,
                "patience_counter": patience_counter, "prev_loss": prev_loss,
                "cumulative_seconds": cumulative_seconds, "done_checkpoints": sorted(done_checkpoints),
                "done_targets": sorted(done_targets),
            }, tmp_resume_path)
            os.replace(tmp_resume_path, resume_path)
        if is_last:
            stop_reason = "max_steps_targets_missed" if set(targets) - done_targets else "complete"
            break

    rows = [json.loads(line) for line in traj_path.read_text().splitlines()]
    last = rows[-1]
    return {
        "final_step": last["epoch"],
        "final_conflicts": last["eval_conflicts"],
        "best_conflicts": last["best_conflicts_so_far"],
        "paper_stop_epoch": paper_stop_epoch,
        "missed_targets": sorted(set(targets) - {lab for r in rows for lab in r.get("time_matched", [])}),
        "stop_reason": stop_reason,
        "out_dir": str(out_dir),
    }
