"""TabuCol quality reference for Experiment 1 (docs/Experiment1_Steps.md sections 2.6, 9, 15).
One invocation = one graph (row of configs/sweep_manifest.csv), all seeds, run in parallel as
separate single-threaded processes (one CPU each). CPU only.

k = chi - 1, random start (the same random_colouring(n, k, seed) as the 'random' control).
Time budget per graph (decided 2026-10-09): the wall-clock time PI-GNN spent training that
graph to 100,000 epochs = mean over its 5 seeds of `cumulative_seconds` at epoch 100,000 in
results/main/<graph>_seed<s>/trajectory.jsonl (one GPU process per seed, so this is a per-run
time, ~300-700 s). The time counts only the search, not Python start-up or the start
colouring. The iterations reached are recorded, because a wall-clock budget depends on the
node's speed. --time-budget overrides it (used for the timing test).

Per seed it writes <out>/<graph>/seed<s>.json: budget, iterations, stop reason, start and best
conflicts, when the best was found, the improvement history, and the best colouring's
single-flip report and strict-vertex-descent result (descent RNG = default_rng(seed), as for
every other method); plus seed<s>_best.npy (the best colouring, for the warm-start hook of
section 15). Idempotent and requeue-safe: a seed whose json exists is skipped (a killed seed
restarts from scratch).

    python scripts/run_tabucol.py --task-id 2
    python scripts/run_tabucol.py --graph queen5_5 --seeds 0 1 --time-budget 10 --out results/tabucol_test
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import (  # noqa: E402
    EndpointGraph,
    conflict_count,
    neighbour_colour_counts,
    single_flip_report,
    strict_vertex_descent,
)
from graphsml.tabucol import tabucol  # noqa: E402

COL_DIR = ROOT / "data" / "dimacs" / "col"
PIGNN_EPOCH = 100_000


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def pignn_seconds(main: Path, name: str) -> tuple[float, list[float]]:
    per_seed = []
    for s in range(5):
        for line in (main / f"{name}_seed{s}" / "trajectory.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["epoch"] == PIGNN_EPOCH:
                per_seed.append(r["cumulative_seconds"])
    if len(per_seed) != 5:
        raise RuntimeError(f"{name}: expected 5 PI-GNN runs with epoch {PIGNN_EPOCH}, found {len(per_seed)}")
    return float(np.mean(per_seed)), per_seed


def run_one(row: pd.Series, seed: int, args) -> int:
    name, k = row["name"], int(row["k"])
    tag = f"{name}/seed{seed}"
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"seed{seed}.json"
    if out_json.exists():
        log(tag, "already done, skipping")
        return 0

    g = load_col_graph(COL_DIR / f"{name}.col")
    n, m = g.number_of_nodes(), g.number_of_edges()
    if (n, m) != (int(row["n"]), int(row["m"])):
        log(tag, f"manifest mismatch: file has n={n} m={m}, manifest n={row['n']} m={row['m']}")
        return 3
    eg = EndpointGraph.from_networkx(g)
    pignn_mean, pignn_per_seed = pignn_seconds(args.main, name)
    budget = args.time_budget if args.time_budget is not None else pignn_mean
    log(tag, f"n={n} m={m} k={k} time budget {budget:.0f}s (PI-GNN mean {pignn_mean:.0f}s)")

    res = tabucol(eg, k, seed=seed, time_budget=budget)
    assert res.best_conflicts == conflict_count(eg, res.best_colours), "M bookkeeping drifted"
    M = neighbour_colour_counts(eg, res.best_colours, k)
    flip = single_flip_report(res.best_colours, M)
    desc = strict_vertex_descent(eg, res.best_colours, k, seed=seed, M=M)

    np.save(out_dir / f"seed{seed}_best.npy", res.best_colours)
    out = dict(
        graph=name, method="tabucol", seed=seed, k=k, n=n, m=m,
        time_budget=budget, pignn_seconds_mean=pignn_mean, pignn_seconds_per_seed=pignn_per_seed,
        start="random_colouring(n, k, seed)", alpha=0.6, l_max=9,
        start_conflicts=res.start_conflicts, best_conflicts=res.best_conflicts,
        iterations=res.iterations, iter_at_best=res.iter_at_best, seconds=res.seconds,
        seconds_at_best=res.seconds_at_best, iters_per_second=res.iterations / max(res.seconds, 1e-9),
        stopped_by=res.stopped_by,
        single_flip_optimal=flip.is_optimal, n_improvable=flip.n_improvable_vertices, best_gain=flip.best_gain,
        conflicts_after_descent=desc.conflicts_after, descent_moves=desc.n_moves,
        host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
        history=res.history)
    tmp = out_dir / f"seed{seed}.json.tmp"
    tmp.write_text(json.dumps(out))
    os.replace(tmp, out_json)
    log(tag, f"start {res.start_conflicts} -> best {res.best_conflicts} (found at iter {res.iter_at_best:,}, "
             f"{res.seconds_at_best:.0f}s) in {res.iterations:,} iters / {res.seconds:.0f}s "
             f"({out['iters_per_second']:.0f} it/s), stop={res.stopped_by}; single-flip optimal={flip.is_optimal}, "
             f"after descent {desc.conflicts_after}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, help="row index into the manifest (Slurm array task id)")
    ap.add_argument("--graph", help="graph name (alternative to --task-id)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--time-budget", type=float, help="seconds; default = PI-GNN's mean training time")
    ap.add_argument("--main", type=Path, default=ROOT / "results" / "main")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tabucol")
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

    log(row["name"], f"launching seeds {args.seeds} in parallel on {socket.gethostname()}")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--graph", row["name"],
            "--main", str(args.main), "--out", str(args.out), "--manifest", str(args.manifest)]
    if args.time_budget is not None:
        base += ["--time-budget", str(args.time_budget)]
    procs = {s: subprocess.Popen(base + ["--worker-seed", str(s)], env=env) for s in args.seeds}
    codes = {s: p.wait() for s, p in procs.items()}
    log(row["name"], f"exit codes {codes}")
    sys.exit(1 if any(c != 0 for c in codes.values()) else 0)


if __name__ == "__main__":
    main()
