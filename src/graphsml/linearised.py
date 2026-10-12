"""Experiment 3 (RQ2), rung D: fixed graph-preconditioned descent, implemented as PI-GNN linearised
at its initialisation (the "lazy-training" / tangent-kernel approximation, plan section 6, 4A):

    Z(delta) = Z0 + J0 delta,       J0 = dZ/dtheta at theta0 (eval mode, dropout off),

with delta (one tensor per PI-GNN parameter, starting at 0) trained on the Potts loss. With plain
gradient descent on delta this is exactly the plan's rung D update

    Z <- Z - eta * K0 * grad_Z L,   K0 = J0 J0^T,

i.e. PI-GNN's own induced preconditioner, frozen at initialisation: graph-structured (message
passing), coupled across vertices and colours, but not evolving. With Adam it is the plan's
"Delta Z ~ -eta J D_t J^T grad_Z L" with J frozen. Exact K0 (including the trainable embeddings,
both layers and the ReLU pattern at init), not the cruder A_hat X X^T A_hat^T approximation, and
never materialised: J0 delta is one forward-mode JVP, J0^T g one backward pass through it, so a
step costs about two PI-GNN forward/backward passes and works on the n = 1000 graphs.

Starts from Z0 = PI-GNN's initial eval-mode logits (same graph, k and seed), so D, A and B share
their starting point. Not included: weight decay (A's AdamW decays theta towards 0; delta has no
such term) and dropout (J0 is the deterministic eval-mode Jacobian).

The checkpointed trainer mirrors free_logits.train_free_logits_checkpointed (same files, resume
and time-matching semantics; that module is deliberately not modified while its runs are live).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call, jvp

from .free_logits import FreeLogitConfig, _noise_generator, _train_forward, make_optimizer
from .losses import argmax_colouring, conflict_count, potts_loss
from .model import PIGNN, build_mean_adjacency


class LinearisedPIGNN(nn.Module):
    """Z(delta) = Z0 + J0 delta for a PI-GNN `net` frozen at its current (initial) parameters."""

    def __init__(self, net: nn.Module, mean_adj: torch.Tensor):
        super().__init__()
        net.eval()  # dropout off: J0 is the deterministic Jacobian
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.register_buffer("mean_adj", mean_adj)
        self.names = [n for n, _ in net.named_parameters()]
        self.delta = nn.ParameterList([nn.Parameter(torch.zeros_like(p)) for _, p in net.named_parameters()])
        with torch.no_grad():
            self.register_buffer("z0", net(mean_adj))

    def _f(self, params: dict) -> torch.Tensor:
        return functional_call(self.net, params, (self.mean_adj,))

    def forward(self) -> torch.Tensor:
        primals = {n: p for n, p in self.net.named_parameters()}
        tangents = {n: d for n, d in zip(self.names, self.delta)}
        _, dz = jvp(self._f, (primals,), (tangents,))
        return self.z0 + dz


def build_linearised_pignn(adj: torch.Tensor, k: int, embed_dim: int, hidden_dim: int, dropout: float,
                           seed: int, device: str = "cpu", embed_init: str = "normal") -> LinearisedPIGNN:
    """Same seeding and construction order as train.train_pignn_checkpointed, so the network (and
    hence Z0 and J0) is PI-GNN's initial network for that graph, k and seed."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    adj = adj.to(dev)
    mean_adj = build_mean_adjacency(adj)
    net = PIGNN(adj.shape[0], embed_dim, hidden_dim, k, dropout, embed_init).to(dev)
    return LinearisedPIGNN(net, mean_adj).to(dev)


def train_linearised_lean(adj: torch.Tensor, model: LinearisedPIGNN, cfg: FreeLogitConfig,
                          eval_every: int = 500) -> dict:
    """Tuning counterpart (same outputs as tune.train_lean)."""
    dev = torch.device(cfg.device)
    adj = adj.to(dev)
    opt = make_optimizer(list(model.delta), cfg)
    gen = _noise_generator(cfg.seed, dev) if cfg.logit_noise > 0 else None
    eval_epochs, eval_conflicts, eval_entropy, n_colours = [], [], [], []
    t0 = time.time()
    for step in range(1, cfg.n_steps + 1):
        opt.zero_grad()
        loss = potts_loss(_train_forward(model(), cfg, gen), adj)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == cfg.n_steps:
            with torch.no_grad():
                p = torch.softmax(model(), dim=1)
                c = argmax_colouring(p)
                eval_epochs.append(step)
                eval_conflicts.append(conflict_count(c, adj))
                eval_entropy.append(float(-(p * torch.log(p.clamp_min(1e-12))).sum(1).mean().item()))
                n_colours.append(int(c.unique().numel()))
    return dict(eval_epochs=eval_epochs, eval_conflicts=eval_conflicts, eval_entropy=eval_entropy,
                n_colours=n_colours, final_loss=float(loss.item()), seconds=time.time() - t0)


def train_linearised_checkpointed(adj: torch.Tensor, model: LinearisedPIGNN, cfg: FreeLogitConfig,
                                  checkpoints: list[int], out_dir: str | Path,
                                  time_targets: dict[str, float] | None = None,
                                  max_steps: int | None = None, graph_meta: dict | None = None,
                                  resume: bool = True) -> dict:
    """Rung D counterpart of free_logits.train_free_logits_checkpointed: identical stopping,
    checkpoint, time-target and resume rules; the trained parameters are `model.delta`, and each
    checkpoint stores Z = Z0 + J0 delta (plus delta itself under "weights")."""
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
    optimizer = make_optimizer(list(model.delta), cfg)
    gen = _noise_generator(cfg.seed, device) if cfg.logit_noise > 0 else None

    def delta_state() -> dict:
        return {f"delta.{n}": d.detach().cpu().clone() for n, d in zip(model.names, model.delta)}

    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        with torch.no_grad():
            for n, d in zip(model.names, model.delta):
                d.copy_(state["delta"][f"delta.{n}"].to(device))
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
        if traj_path.exists():
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
        torch.save(model.z0.detach().cpu(), out_dir / "z0.pt")
        meta = dict(graph_meta or {}, **asdict(cfg), rung="D (PI-GNN linearised at init)", checkpoints=checkpoints,
                    max_steps=max_steps, time_targets=targets, torch_version=torch.__version__)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    def finished(step: int, pending: set[str]) -> bool:
        return step >= max_steps or (step >= cfg.n_steps and not pending)

    start = max_steps + 1 if last_step > 0 and finished(last_step, set(targets) - done_targets) else last_step + 1
    stop_reason = "already_complete" if start > max_steps else None
    t0 = time.perf_counter()
    for step in range(start, max_steps + 1):
        optimizer.zero_grad()
        loss = potts_loss(_train_forward(model(), cfg, gen), adj)
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
                z = model()
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

            torch.save({"epoch": step, "Z": z.detach().cpu().clone(), "time_matched": crossed,
                        "delta": delta_state()}, ckpt_dir / f"ckpt_{step:08d}.pt")
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
                "epoch": step, "delta": delta_state(), "optimizer": optimizer.state_dict(),
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
