"""Experiment 3 (RQ2) main run of the free-logit arms on DIMACS at k = chi - 1. One invocation =
one graph and one arm, all seeds, run in parallel as separate processes sharing one GPU --
the same layout PI-GNN (arm A, results/main) was run with, so wall-clock times are comparable.

Arms:
  B      start from PI-GNN's exact initial logits, results/main/<graph>_seed<s>/z0.pt
  Brand  start from random logits with Z0's scale (free_logits.matched_random_logits)
Both use the frozen free-logit config (--config, chosen by tune_free_logits.py on the synthetic
tuning set; B-rand reuses B's config so that B vs B-rand differs only in the start).
--logit-noise > 0 gives the noise-control variant (written to <arm>_noise<sigma>).

Budgets, per (graph, seed):
  steps  the step budget defaults to A's budget for that run (DONE t_max: 100k, or 300k for the
         13 extended graphs), on the same log checkpoint grid, for the matched-step comparison;
  time   unless --no-time-match, training continues past the step budget until its
         cumulative seconds reach A's cumulative seconds at every A checkpoint (label
         "A@<epoch>"), capped at --max-steps-factor x the step budget; a checkpoint is saved
         where each A time is reached, for the matched-time comparison.
At every saved checkpoint the endpoint is evaluated exactly as for A (argmax, single-flip
report, strict vertex descent with descent seed = run seed) -> endpoint_trajectory.jsonl.

Output: <out>/<arm>/<graph>_seed<s>/, same files as an A run (z0.pt, meta.json,
trajectory.jsonl, checkpoints/, resume_state.pt, endpoint_trajectory.jsonl, DONE). DONE's
"t_max" is the last step actually trained (what run_soft.py reads); "n_steps" is the step budget.
Idempotent and requeue-safe like run_sweep.py.

    python scripts/run_free_logits.py --task-id 0 --arm B
    python scripts/run_free_logits.py --graph queen5_5 --arm Brand --seeds 0 --config <json>
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
from graphsml.free_logits import FreeLogitConfig, matched_random_logits, train_free_logits_checkpointed  # noqa: E402

COL_DIR = ROOT / "data" / "dimacs" / "col"
CONFIG_KEYS = ("lr", "optimizer", "weight_decay", "momentum", "logit_noise")


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def graph_to_adj(graph):
    n = graph.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in graph.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def arm_dir_name(args) -> str:
    return args.arm + (f"_noise{args.logit_noise:g}" if args.logit_noise else "")


def run_one(row: pd.Series, seed: int, args) -> int:
    name = row["name"]
    tag = f"{arm_dir_name(args)}/{name}/seed{seed}"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log(tag, "CUDA requested but not available; refusing to fall back to CPU silently.")
        return 2

    a_dir = args.main / f"{name}_seed{seed}"
    if not (a_dir / "DONE").exists():
        log(tag, f"PI-GNN run {a_dir} has no DONE marker; its z0 and timings are needed")
        return 4
    a_tmax = json.loads((a_dir / "DONE").read_text())["t_max"]
    t_max = args.t_max or a_tmax
    time_match = not args.no_time_match

    out_dir = args.out / arm_dir_name(args) / f"{name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        done = json.loads((out_dir / "DONE").read_text())
        if done["n_steps"] >= t_max and done["time_match"] == time_match and done["a_t_max"] >= a_tmax:
            log(tag, "already done, skipping")
            return 0

    g = load_col_graph(COL_DIR / f"{name}.col")
    n, m = g.number_of_nodes(), g.number_of_edges()
    if (n, m) != (int(row["n"]), int(row["m"])):
        log(tag, f"manifest mismatch: file has n={n} m={m}, manifest n={row['n']} m={row['m']}")
        return 3
    k = int(row["k"])
    z0 = torch.load(a_dir / "z0.pt", map_location="cpu")
    if tuple(z0.shape) != (n, k):
        log(tag, f"z0 shape {tuple(z0.shape)} != ({n}, {k})")
        return 3
    z_init = z0 if args.arm == "B" else matched_random_logits(z0, seed)

    targets = {}
    if time_match:
        for line in (a_dir / "trajectory.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["epoch"] <= a_tmax:
                targets[f"A@{r['epoch']}"] = r["cumulative_seconds"]
    max_steps = t_max * args.max_steps_factor if time_match else t_max
    checkpoints = sorted(set(log_checkpoint_grid(max_steps, n)) | {t_max})

    frozen = json.loads(args.config.read_text())
    hp = {key: frozen[key] for key in CONFIG_KEYS if key in frozen}
    if args.logit_noise:
        hp["logit_noise"] = args.logit_noise
    cfg = FreeLogitConfig(k=k, n_steps=t_max, seed=seed, device=args.device, **hp)

    with (out_dir / "run_info.jsonl").open("a") as f:
        f.write(json.dumps(dict(
            time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(),
            slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
            gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            torch=torch.__version__, n_steps=t_max, max_steps=max_steps, config=str(args.config))) + "\n")

    log(tag, f"n={n} m={m} k={k} steps={t_max} max_steps={max_steps} targets={len(targets)} "
             f"(last {max(targets.values()) if targets else 0:.0f}s) cfg={hp}")
    t0 = time.time()
    result = train_free_logits_checkpointed(
        graph_to_adj(g), z_init, cfg, checkpoints, out_dir, time_targets=targets, max_steps=max_steps,
        graph_meta={"graph": name, "n": n, "m": m, "chi": int(row["chi"]), "arm": args.arm,
                    "a_run": str(a_dir), "config_file": str(args.config)})
    log(tag, f"trained in {time.time() - t0:.0f}s (this attempt) final_step={result['final_step']} "
             f"final_conflicts={result['final_conflicts']} stop={result['stop_reason']} "
             f"missed_targets={result['missed_targets']}")

    eg = EndpointGraph.from_networkx(g)
    t0 = time.time()
    rows = []
    for ckpt_path in sorted((out_dir / "checkpoints").glob("ckpt_*.pt")):
        saved = torch.load(ckpt_path, map_location="cpu")
        if saved["epoch"] > result["final_step"]:
            continue
        res = evaluate_endpoint(eg, torch.softmax(saved["Z"], dim=1), k, seed=seed, soft_diagnostics=False)
        rows.append(dict(
            graph=name, seed=seed, k=k, arm=args.arm, epoch=saved["epoch"], time_matched=saved["time_matched"],
            conflicts=res.conflicts, single_flip_optimal=res.single_flip.is_optimal,
            n_improvable=res.single_flip.n_improvable_vertices, best_gain=res.single_flip.best_gain,
            conflicts_after_descent=res.descent.conflicts_after, descent_moves=res.descent.n_moves,
            descent_seconds=res.descent.seconds))
    tmp = out_dir / "endpoint_trajectory.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, out_dir / "endpoint_trajectory.jsonl")
    (out_dir / "DONE").write_text(json.dumps(dict(
        t_max=result["final_step"], n_steps=t_max, a_t_max=a_tmax, time_match=time_match,
        stop_reason=result["stop_reason"], missed_targets=result["missed_targets"],
        n_endpoint_rows=len(rows), final_conflicts=result["final_conflicts"])))
    log(tag, f"endpoints evaluated in {time.time() - t0:.0f}s; DONE")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, help="row index into the manifest (Slurm array task id)")
    ap.add_argument("--graph", help="graph name (alternative to --task-id)")
    ap.add_argument("--arm", choices=["B", "Brand"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "free_logits_frozen.json")
    ap.add_argument("--logit-noise", type=float, default=0.0, help="noise-control variant (std of logit noise)")
    ap.add_argument("--t-max", type=int, default=None, help="step budget (default: the PI-GNN run's t_max)")
    ap.add_argument("--max-steps-factor", type=int, default=10)
    ap.add_argument("--no-time-match", action="store_true", help="stop at the step budget, no A-time checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--main", type=Path, default=ROOT / "results" / "main", help="PI-GNN (arm A) runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "exp3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "sweep_manifest.csv")
    ap.add_argument("--worker-seed", type=int, help="internal: run a single seed in this process")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, index_col="idx")
    if args.graph is not None:
        row = manifest[manifest["name"] == args.graph].iloc[0]
    elif args.task_id is not None:
        row = manifest.loc[args.task_id]
    else:
        sys.exit("give --task-id or --graph")
    if not args.config.exists():
        sys.exit(f"{args.config} does not exist (freeze the tuned free-logit config first)")

    if args.worker_seed is not None:
        sys.exit(run_one(row, args.worker_seed, args))

    log(row["name"], f"arm {arm_dir_name(args)}: launching seeds {args.seeds} in parallel on {socket.gethostname()} "
                      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--graph", row["name"], "--arm", args.arm,
            "--config", str(args.config), "--logit-noise", str(args.logit_noise),
            "--max-steps-factor", str(args.max_steps_factor), "--device", args.device,
            "--main", str(args.main), "--out", str(args.out), "--manifest", str(args.manifest)]
    if args.t_max is not None:
        base += ["--t-max", str(args.t_max)]
    if args.no_time_match:
        base += ["--no-time-match"]
    procs = {s: subprocess.Popen(base + ["--worker-seed", str(s)], env=env) for s in args.seeds}
    codes = {s: p.wait() for s, p in procs.items()}
    failed = {s: c for s, c in codes.items() if c != 0}
    log(row["name"], f"exit codes {codes}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
