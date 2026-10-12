"""Step H: main PI-GNN sweep. One invocation = one graph, all seeds, run in parallel
as separate processes sharing one GPU (the pilot and a 5-way benchmark showed a run is
bound by kernel launches, ~4-5 ms/epoch, so co-located seeds do not slow each other).

For each (graph, seed): train the frozen config at k = chi - 1 for --t-max epochs with the
shared log checkpoint grid, then evaluate the endpoint (single-flip report + strict vertex
descent) at every saved checkpoint. Output lives in <out>/<graph>_seed<seed>/ -- the
directory name does NOT contain the budget, so a finished run can later be extended to a
larger --t-max in place (train_pignn_checkpointed resumes from the last checkpoint).

Idempotent and requeue-safe (the Slurm partition is killable): a run that has a DONE file
is skipped; an interrupted one resumes from its last completed checkpoint.

    python scripts/run_sweep.py --task-id 0                 # graph = row 0 of the manifest
    python scripts/run_sweep.py --graph queen5_5 --seeds 0 1
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
from graphsml.train import TrainConfig, train_pignn_checkpointed  # noqa: E402

COL_DIR = ROOT / "data" / "dimacs" / "col"


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
    tag = f"{name}/seed{seed}"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log(tag, "CUDA requested but not available; refusing to fall back to CPU silently.")
        return 2

    out_dir = args.out / f"{name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists() and json.loads((out_dir / "DONE").read_text())["t_max"] >= args.t_max:
        log(tag, "already done, skipping")
        return 0

    g = load_col_graph(COL_DIR / f"{name}.col")
    n, m = g.number_of_nodes(), g.number_of_edges()
    if (n, m) != (int(row["n"]), int(row["m"])):
        log(tag, f"manifest mismatch: file has n={n} m={m}, manifest n={row['n']} m={row['m']}")
        return 3
    k = int(row["k"])
    frozen = json.loads(args.config.read_text())
    cfg = TrainConfig(k=k, embed_dim=frozen["embed_dim"], hidden_dim=frozen["hidden_dim"],
                      dropout=frozen["dropout"], lr=frozen["lr"], n_epochs=args.t_max,
                      arch=frozen.get("arch", "sage"),
                      seed=seed, device=args.device)
    checkpoints = log_checkpoint_grid(args.t_max, n)

    with (out_dir / "run_info.jsonl").open("a") as f:
        f.write(json.dumps(dict(
            time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(),
            slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
            gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            torch=torch.__version__, t_max=args.t_max)) + "\n")

    log(tag, f"n={n} m={m} chi={int(row['chi'])} k={k} t_max={args.t_max} checkpoints={len(checkpoints)}")
    t0 = time.time()
    result = train_pignn_checkpointed(
        graph_to_adj(g), cfg, checkpoints, out_dir,
        graph_meta={"graph": name, "n": n, "m": m, "chi": int(row["chi"])})
    log(tag, f"trained in {time.time() - t0:.0f}s (this attempt) final_conflicts={result['final_conflicts']} "
             f"best_conflicts={result['best_conflicts']} paper_stop_epoch={result['paper_stop_epoch']}")

    eg = EndpointGraph.from_networkx(g)
    t0 = time.time()
    rows = []
    for ckpt_path in sorted((out_dir / "checkpoints").glob("ckpt_*.pt")):
        saved = torch.load(ckpt_path, map_location="cpu")
        if saved["epoch"] > args.t_max:
            continue
        res = evaluate_endpoint(eg, torch.softmax(saved["Z"], dim=1), k, seed=seed, soft_diagnostics=False)
        rows.append(dict(
            graph=name, seed=seed, k=k, epoch=saved["epoch"], conflicts=res.conflicts,
            single_flip_optimal=res.single_flip.is_optimal,
            n_improvable=res.single_flip.n_improvable_vertices, best_gain=res.single_flip.best_gain,
            conflicts_after_descent=res.descent.conflicts_after, descent_moves=res.descent.n_moves,
            descent_seconds=res.descent.seconds))
    tmp = out_dir / "endpoint_trajectory.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, out_dir / "endpoint_trajectory.jsonl")
    (out_dir / "DONE").write_text(json.dumps(dict(
        t_max=args.t_max, n_endpoint_rows=len(rows), final_conflicts=result["final_conflicts"])))
    log(tag, f"endpoints evaluated in {time.time() - t0:.0f}s; DONE")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, help="row index into the manifest (Slurm array task id)")
    ap.add_argument("--graph", help="graph name (alternative to --task-id)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--t-max", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pignn_frozen.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "main")
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

    if args.worker_seed is not None:
        sys.exit(run_one(row, args.worker_seed, args))

    log(row["name"], f"launching seeds {args.seeds} in parallel on {socket.gethostname()} "
                      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--graph", row["name"],
            "--t-max", str(args.t_max), "--device", args.device, "--config", str(args.config), "--out", str(args.out),
            "--manifest", str(args.manifest)]
    procs = {s: subprocess.Popen(base + ["--worker-seed", str(s)], env=env) for s in args.seeds}
    codes = {s: p.wait() for s, p in procs.items()}
    failed = {s: c for s, c in codes.items() if c != 0}
    log(row["name"], f"exit codes {codes}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
