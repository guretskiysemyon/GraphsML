"""Experiment 3 rung D main run on DIMACS at k = chi - 1: PI-GNN linearised at its initialisation
(graphsml.linearised), frozen config configs/linearised_frozen.json (tune_linearised.py). Same
layout and rules as run_free_logits.py: one graph per invocation, 5 seeds as parallel processes on
one GPU, step budget = A's (default capped by --t-max), checkpoints where A's cumulative seconds are
reached ("A@<epoch>"), endpoint evaluation at every checkpoint, idempotent and requeue-safe.

The linearised network is rebuilt from the frozen PI-GNN config and the run seed; its Z0 is checked
against A's saved z0.pt (same seeding; reported, must agree to float tolerance).

Output: <out>/D/<graph>_seed<s>/ (z0.pt, meta.json, trajectory.jsonl, checkpoints/ with "Z" and
"delta", resume_state.pt, endpoint_trajectory.jsonl, DONE).

    python scripts/run_linearised.py --task-id 0
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.checkpoints import log_checkpoint_grid  # noqa: E402
from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import EndpointGraph, evaluate_endpoint  # noqa: E402
from graphsml.free_logits import FreeLogitConfig  # noqa: E402
from graphsml.linearised import build_linearised_pignn, train_linearised_checkpointed  # noqa: E402

COL_DIR = ROOT / "data" / "dimacs" / "col"
CONFIG_KEYS = ("lr", "optimizer", "weight_decay", "momentum", "logit_noise")
ARM = "D"


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def graph_to_adj(graph):
    n = graph.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in graph.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def run_one(row: pd.Series, seed: int, args) -> int:
    name = row["name"]
    tag = f"{ARM}/{name}/seed{seed}"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log(tag, "CUDA requested but not available; refusing to fall back to CPU silently.")
        return 2
    a_dir = args.main / f"{name}_seed{seed}"
    if not (a_dir / "DONE").exists():
        log(tag, f"PI-GNN run {a_dir} has no DONE marker")
        return 4
    a_tmax = json.loads((a_dir / "DONE").read_text())["t_max"]
    t_max = min(args.t_max, a_tmax) if args.t_max else a_tmax

    out_dir = args.out / ARM / f"{name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        done = json.loads((out_dir / "DONE").read_text())
        if done["n_steps"] >= t_max and done["a_t_max"] >= a_tmax:
            log(tag, "already done, skipping")
            return 0

    g = load_col_graph(COL_DIR / f"{name}.col")
    n, m = g.number_of_nodes(), g.number_of_edges()
    if (n, m) != (int(row["n"]), int(row["m"])):
        log(tag, f"manifest mismatch: file has n={n} m={m}")
        return 3
    k = int(row["k"])
    adj = graph_to_adj(g)
    pignn = json.loads((ROOT / "configs" / "pignn_frozen.json").read_text())
    model = build_linearised_pignn(adj, k, pignn["embed_dim"], pignn["hidden_dim"], pignn["dropout"], seed, args.device)
    z0_a = torch.load(a_dir / "z0.pt", map_location="cpu")
    z0_diff = float((model.z0.cpu() - z0_a).abs().max())
    if z0_diff > 1e-4:
        log(tag, f"Z0 differs from A's z0.pt by {z0_diff:.2e}; refusing")
        return 3

    targets = {}
    for line in (a_dir / "trajectory.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["epoch"] <= a_tmax:
            targets[f"A@{r['epoch']}"] = r["cumulative_seconds"]
    max_steps = t_max * args.max_steps_factor
    checkpoints = sorted(set(log_checkpoint_grid(max_steps, n)) | {t_max})
    frozen = json.loads(args.config.read_text())
    hp = {key: frozen[key] for key in CONFIG_KEYS if key in frozen}
    cfg = FreeLogitConfig(k=k, n_steps=t_max, seed=seed, device=args.device, **hp)

    with (out_dir / "run_info.jsonl").open("a") as f:
        f.write(json.dumps(dict(
            time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(),
            slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
            gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            torch=torch.__version__, n_steps=t_max, max_steps=max_steps, config=str(args.config),
            z0_max_abs_diff_vs_A=z0_diff)) + "\n")
    log(tag, f"n={n} m={m} k={k} steps={t_max} max_steps={max_steps} targets={len(targets)} "
             f"z0_diff={z0_diff:.1e} cfg={hp}")
    t0 = time.time()
    result = train_linearised_checkpointed(
        adj, model, cfg, checkpoints, out_dir, time_targets=targets, max_steps=max_steps,
        graph_meta={"graph": name, "n": n, "m": m, "chi": int(row["chi"]), "arm": ARM, "a_run": str(a_dir),
                    "config_file": str(args.config), "pignn_config": pignn})
    log(tag, f"trained in {time.time() - t0:.0f}s (this attempt) final_step={result['final_step']} "
             f"final_conflicts={result['final_conflicts']} stop={result['stop_reason']} "
             f"missed_targets={result['missed_targets']}")

    eg = EndpointGraph.from_networkx(g)
    rows = []
    for ckpt_path in sorted((out_dir / "checkpoints").glob("ckpt_*.pt")):
        saved = torch.load(ckpt_path, map_location="cpu")
        if saved["epoch"] > result["final_step"]:
            continue
        res = evaluate_endpoint(eg, torch.softmax(saved["Z"], dim=1), k, seed=seed, soft_diagnostics=False)
        rows.append(dict(
            graph=name, seed=seed, k=k, arm=ARM, epoch=saved["epoch"], time_matched=saved["time_matched"],
            conflicts=res.conflicts, single_flip_optimal=res.single_flip.is_optimal,
            n_improvable=res.single_flip.n_improvable_vertices, best_gain=res.single_flip.best_gain,
            conflicts_after_descent=res.descent.conflicts_after, descent_moves=res.descent.n_moves,
            descent_seconds=res.descent.seconds))
    tmp = out_dir / "endpoint_trajectory.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, out_dir / "endpoint_trajectory.jsonl")
    (out_dir / "DONE").write_text(json.dumps(dict(
        t_max=result["final_step"], n_steps=t_max, a_t_max=a_tmax, time_match=True,
        stop_reason=result["stop_reason"], missed_targets=result["missed_targets"],
        n_endpoint_rows=len(rows), final_conflicts=result["final_conflicts"])))
    log(tag, "DONE")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--graph")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "linearised_frozen.json")
    ap.add_argument("--t-max", type=int, default=None, help="cap on the step budget (default: A's t_max)")
    ap.add_argument("--max-steps-factor", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--main", type=Path, default=ROOT / "results" / "main")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "exp3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "sweep_manifest.csv")
    ap.add_argument("--worker-seed", type=int)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, index_col="idx")
    if args.graph is not None:
        row = manifest[manifest["name"] == args.graph].iloc[0]
    elif args.task_id is not None:
        row = manifest.loc[args.task_id]
    else:
        sys.exit("give --task-id or --graph")
    if not args.config.exists():
        sys.exit(f"{args.config} does not exist (freeze the tuned rung-D config first)")
    if args.worker_seed is not None:
        sys.exit(run_one(row, args.worker_seed, args))

    log(row["name"], f"rung D: launching seeds {args.seeds} on {socket.gethostname()}")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--graph", row["name"], "--config", str(args.config),
            "--max-steps-factor", str(args.max_steps_factor), "--device", args.device, "--main", str(args.main),
            "--out", str(args.out), "--manifest", str(args.manifest)]
    if args.t_max is not None:
        base += ["--t-max", str(args.t_max)]
    procs = {s: subprocess.Popen(base + ["--worker-seed", str(s)], env=env) for s in args.seeds}
    codes = {s: p.wait() for s, p in procs.items()}
    log(row["name"], f"exit codes {codes}")
    sys.exit(1 if any(c != 0 for c in codes.values()) else 0)


if __name__ == "__main__":
    main()
