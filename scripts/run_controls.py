"""Step G: classical controls for Experiment 1. One invocation = one graph (a row of the
sweep manifest), all methods, all seeds. CPU only.

For each method in {random, dsatur, dsatur_rand} and each seed, build a hard k-colouring at
k = chi - 1 (the same k as the PI-GNN sweep), record its conflicts and single-flip report,
then run the same strict_vertex_descent as for PI-GNN's argmax, with the same descent seed
convention (descent RNG = default_rng(seed)). 'dsatur' is deterministic, so its 5 seeds
differ only in the descent tie-breaks.

Output: <out>/<graph>/controls.jsonl (one row per method x seed) plus a DONE marker.
Idempotent and requeue-safe: a graph with a DONE marker is skipped.

    python scripts/run_controls.py --task-id 0
    python scripts/run_controls.py --graph queen5_5 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.controls import dsatur_fixed_k, random_colouring  # noqa: E402
from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import (  # noqa: E402
    EndpointGraph,
    conflicts_from_counts,
    neighbour_colour_counts,
    single_flip_report,
    strict_vertex_descent,
)

COL_DIR = ROOT / "data" / "dimacs" / "col"
METHODS = ("random", "dsatur", "dsatur_rand")


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def start_colouring(method: str, eg: EndpointGraph, k: int, seed: int) -> np.ndarray:
    if method == "random":
        return random_colouring(eg.n, k, seed)
    if method == "dsatur":
        return dsatur_fixed_k(eg, k)
    if method == "dsatur_rand":
        return dsatur_fixed_k(eg, k, seed=seed)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, help="row index into the manifest (Slurm array task id)")
    ap.add_argument("--graph", help="graph name (alternative to --task-id)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "controls")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "sweep_manifest.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, index_col="idx")
    if args.graph is not None:
        row = manifest[manifest["name"] == args.graph].iloc[0]
    elif args.task_id is not None:
        row = manifest.loc[args.task_id]
    else:
        sys.exit("give --task-id or --graph")
    name, k = row["name"], int(row["k"])
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        log(name, "already done, skipping")
        return

    g = load_col_graph(COL_DIR / f"{name}.col")
    n, m = g.number_of_nodes(), g.number_of_edges()
    if (n, m) != (int(row["n"]), int(row["m"])):
        sys.exit(f"{name}: manifest mismatch: file has n={n} m={m}, manifest n={row['n']} m={row['m']}")
    log(name, f"host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID')} "
              f"task={os.environ.get('SLURM_ARRAY_TASK_ID')} n={n} m={m} chi={int(row['chi'])} k={k} seeds={args.seeds}")

    eg = EndpointGraph.from_networkx(g)
    rows = []
    for method in METHODS:
        for seed in args.seeds:
            t0 = time.perf_counter()
            colours = start_colouring(method, eg, k, seed)
            start_seconds = time.perf_counter() - t0
            M = neighbour_colour_counts(eg, colours, k)
            flip = single_flip_report(colours, M)
            res = strict_vertex_descent(eg, colours, k, seed=seed, M=M)
            rows.append(dict(
                graph=name, method=method, seed=seed, k=k, n=n,
                conflicts=conflicts_from_counts(colours, M), colours_used=int(len(np.unique(colours))),
                single_flip_optimal=flip.is_optimal, n_improvable=flip.n_improvable_vertices,
                best_gain=flip.best_gain, conflicts_after_descent=res.conflicts_after,
                descent_moves=res.n_moves, descent_seconds=res.seconds, start_seconds=start_seconds))
            log(name, f"{method}/seed{seed}: raw={rows[-1]['conflicts']} -> after descent="
                      f"{res.conflicts_after} ({res.n_moves} moves, {res.seconds:.1f}s)")

    tmp = out_dir / "controls.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, out_dir / "controls.jsonl")
    (out_dir / "DONE").write_text(json.dumps(dict(n_rows=len(rows), time=time.strftime("%Y-%m-%dT%H:%M:%S"))))
    log(name, "DONE")


if __name__ == "__main__":
    main()
