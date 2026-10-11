"""Experiment 3 always-logged diagnostics (plan section 6: 4C smoothness, 4E movement from
initialisation), computed after the fact from saved checkpoints, identically for every arm:
PI-GNN (arm A, results/main) and the free-logit arms (results/exp3/<arm>). CPU only, float64.

Per run directory with a DONE marker, for every checkpoint with epoch <= DONE's t_max:
eval-mode Potts loss, conflicts, row entropy, colours used, Dirichlet energy of P, distance of
Z from Z0 (absolute, relative, and relative on row-centred logits), median logit margin,
fraction of rows with p_max >= 0.99 (graphsml.diagnostics.logit_diagnostics); for PI-GNN also
the parameter distance from init_weights.pt (total and per tensor). Free logits have no
parameters besides Z, so their parameter distance is the logit distance.

Writes <run>/movement.jsonl atomically; a run whose file already covers its DONE t_max is skipped.

    python scripts/run_movement.py --task-id 2                          # arm A
    python scripts/run_movement.py --task-id 2 --runs results/exp3/B    # arm B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.diagnostics import logit_diagnostics, param_distance  # noqa: E402
from graphsml.dimacs import load_col_graph  # noqa: E402

COL_DIR = ROOT / "data" / "dimacs" / "col"


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def run_one(name: str, adj: torch.Tensor, seed: int, args) -> int:
    tag = f"{args.runs.name}/{name}/seed{seed}"
    run_dir = args.runs / f"{name}_seed{seed}"
    if not (run_dir / "DONE").exists():
        log(tag, "no DONE marker (run missing or unfinished); skipping")
        return 4
    t_max = json.loads((run_dir / "DONE").read_text())["t_max"]
    out = run_dir / "movement.jsonl"
    if out.exists() and not args.force:
        rows = [json.loads(line) for line in out.read_text().splitlines() if line]
        if rows and rows[0].get("t_max") == t_max:
            log(tag, f"already done for t_max={t_max}, skipping")
            return 0

    z0 = torch.load(run_dir / "z0.pt", map_location="cpu").double()
    w0_path = run_dir / "init_weights.pt"
    w0 = torch.load(w0_path, map_location="cpu") if w0_path.exists() else None
    t0 = time.time()
    rows = []
    for ckpt_path in sorted((run_dir / "checkpoints").glob("ckpt_*.pt")):
        epoch = int(ckpt_path.stem.split("_")[1])
        if epoch > t_max:
            continue
        saved = torch.load(ckpt_path, map_location="cpu")
        assert saved["epoch"] == epoch, (ckpt_path, saved["epoch"])
        row = dict(graph=name, seed=seed, epoch=epoch, t_max=t_max, time_matched=saved.get("time_matched", []))
        row.update(logit_diagnostics(saved["Z"].double(), z0, adj))
        if w0 is not None and "weights" in saved:
            row.update(param_distance(saved["weights"], w0))
        else:
            row.update(param_dist=row["logit_dist"], param_dist_rel=row["logit_dist_rel"])
        rows.append(row)
    tmp = run_dir / "movement.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, out)
    last = rows[-1]
    log(tag, f"{len(rows)} checkpoints in {time.time() - t0:.0f}s; at {last['epoch']}: conflicts={last['eval_conflicts']} "
             f"dirichlet={last['dirichlet']:.1f} logit_dist_rel={last['logit_dist_rel']:.2f} "
             f"param_dist_rel={last['param_dist_rel']:.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, help="row index into the manifest (Slurm array task id)")
    ap.add_argument("--graph", help="graph name (alternative to --task-id)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "main", help="directory of <graph>_seed<s> runs")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "sweep_manifest.csv")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, index_col="idx")
    if args.graph is not None:
        row = manifest[manifest["name"] == args.graph].iloc[0]
    elif args.task_id is not None:
        row = manifest.loc[args.task_id]
    else:
        sys.exit("give --task-id or --graph")
    name = row["name"]
    g = load_col_graph(COL_DIR / f"{name}.col")
    assert (g.number_of_nodes(), g.number_of_edges()) == (int(row["n"]), int(row["m"])), name
    adj = torch.zeros(g.number_of_nodes(), g.number_of_nodes(), dtype=torch.float64)
    for u, v in g.edges():
        adj[u, v] = adj[v, u] = 1.0
    codes = {s: run_one(name, adj, s, args) for s in args.seeds}
    log(name, f"exit codes {codes}")
    sys.exit(1 if any(c != 0 for c in codes.values()) else 0)


if __name__ == "__main__":
    main()
