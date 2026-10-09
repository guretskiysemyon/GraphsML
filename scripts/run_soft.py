"""Step I, panel C input: soft-space diagnostics (g_tilde, H_tilde) of Experiment1_Steps.md
section 4, computed after the fact from the eval-mode logits Z saved at every PI-GNN
checkpoint (the Step H sweep ran evaluate_endpoint with soft_diagnostics=False).
CPU only, no training.

One invocation = one graph (row of configs/sweep_manifest.csv), all seeds. For each run
<main>/<graph>_seed<s>/ that has a DONE marker, every checkpoint with epoch <= DONE's
t_max is processed (checkpoint files are not written atomically, so a run that is being
extended in place is read only up to its last finished budget). Writes, atomically:

  soft_diagnostics.jsonl  one row per checkpoint: summary statistics over the
                          non-isolated vertices (g_tilde is 0 by convention on isolated
                          ones), plus `entropy_check_diff` = |mean_v H_tilde(v) log k -
                          eval_entropy logged at training time| as a sanity check that
                          the right logits were loaded
  soft_pervertex.npz      epochs (E,), degree (n,), g_tilde (E, n), H_tilde (E, n),
                          float32, so panel C can be re-thresholded or re-binned later

Idempotent: a run whose soft_diagnostics.jsonl already covers its DONE t_max is skipped,
so after extending runs in place, resubmitting picks up only the extended ones.

Category thresholds used in the summary (descriptive choice, recorded here; the npz keeps
the raw values): stationary = g_tilde <= 0.01; near-hard = H_tilde <= 0.1;
near-uniform = H_tilde >= 0.9.

    python scripts/run_soft.py --task-id 2
    python scripts/run_soft.py --graph queen5_5 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import EndpointGraph, soft_stationarity  # noqa: E402

COL_DIR = ROOT / "data" / "dimacs" / "col"
G_STAT, H_HARD, H_UNIF = 0.01, 0.1, 0.9


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def summarise(epoch: int, g: np.ndarray, H: np.ndarray, mask: np.ndarray, k: int,
              logged_entropy: float | None) -> dict:
    gm, Hm = g[mask], H[mask]
    stat = gm <= G_STAT
    row = dict(
        epoch=epoch, n_used=int(mask.sum()),
        g_mean=float(gm.mean()), g_median=float(np.median(gm)), g_p90=float(np.quantile(gm, 0.9)),
        g_max=float(gm.max()), frac_stationary=float(stat.mean()),
        H_mean=float(Hm.mean()), H_median=float(np.median(Hm)),
        frac_near_hard=float((Hm <= H_HARD).mean()), frac_near_uniform=float((Hm >= H_UNIF).mean()),
        frac_stationary_near_hard=float((stat & (Hm <= H_HARD)).mean()),
        frac_stationary_near_uniform=float((stat & (Hm >= H_UNIF)).mean()),
        frac_not_stationary=float((~stat).mean()),
    )
    # the training-time log is the mean natural-log entropy over ALL vertices
    row["entropy_check_diff"] = (None if logged_entropy is None
                                 else abs(float(H.mean()) * np.log(k) - logged_entropy))
    return row


def run_one(name: str, k: int, eg: EndpointGraph, seed: int, args) -> int:
    tag = f"{name}/seed{seed}"
    run_dir = args.main / f"{name}_seed{seed}"
    done = run_dir / "DONE"
    if not done.exists():
        log(tag, "no DONE marker (run missing or unfinished); skipping")
        return 4
    t_max = json.loads(done.read_text())["t_max"]
    out_jsonl = run_dir / "soft_diagnostics.jsonl"
    if out_jsonl.exists() and not args.force:
        rows = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line]
        if rows and rows[0].get("t_max") == t_max:
            log(tag, f"already done for t_max={t_max}, skipping")
            return 0

    logged = {}
    for line in (run_dir / "trajectory.jsonl").read_text().splitlines():
        r = json.loads(line)
        logged[r["epoch"]] = r["eval_entropy"]

    t0 = time.time()
    mask = eg.degree > 0
    epochs, gs, Hs, rows = [], [], [], []
    for ckpt_path in sorted((run_dir / "checkpoints").glob("ckpt_*.pt")):
        epoch = int(ckpt_path.stem.split("_")[1])
        if epoch > t_max:
            continue
        saved = torch.load(ckpt_path, map_location="cpu")
        assert saved["epoch"] == epoch, (ckpt_path, saved["epoch"])
        P = torch.softmax(saved["Z"].double(), dim=1)
        g, H = soft_stationarity(eg, P, k)
        row = summarise(epoch, g, H, mask, k, logged.get(epoch))
        row.update(graph=name, seed=seed, k=k, t_max=t_max)
        rows.append(row)
        epochs.append(epoch); gs.append(g.astype(np.float32)); Hs.append(H.astype(np.float32))

    worst = max((r["entropy_check_diff"] or 0.0) for r in rows)
    if worst > 1e-3:
        log(tag, f"WARNING entropy check: max |mean H_tilde*log k - logged entropy| = {worst:.2e}")

    tmp = run_dir / "soft_pervertex.npz.tmp.npz"
    np.savez_compressed(tmp, epochs=np.array(epochs), degree=eg.degree,
                        g_tilde=np.stack(gs), H_tilde=np.stack(Hs))
    os.replace(tmp, run_dir / "soft_pervertex.npz")
    tmp = run_dir / "soft_diagnostics.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, out_jsonl)
    last = rows[-1]
    log(tag, f"{len(rows)} checkpoints up to epoch {epochs[-1]} in {time.time() - t0:.0f}s; "
             f"max entropy-check diff {worst:.1e}; at epoch {epochs[-1]}: g_mean={last['g_mean']:.4f} "
             f"H_mean={last['H_mean']:.3f} stationary={last['frac_stationary']:.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, help="row index into the manifest (Slurm array task id)")
    ap.add_argument("--graph", help="graph name (alternative to --task-id)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--main", type=Path, default=ROOT / "results" / "main")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "sweep_manifest.csv")
    ap.add_argument("--force", action="store_true", help="recompute even if already done")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, index_col="idx")
    if args.graph is not None:
        row = manifest[manifest["name"] == args.graph].iloc[0]
    elif args.task_id is not None:
        row = manifest.loc[args.task_id]
    else:
        sys.exit("give --task-id or --graph")

    name, k = row["name"], int(row["k"])
    g = load_col_graph(COL_DIR / f"{name}.col")
    assert (g.number_of_nodes(), g.number_of_edges()) == (int(row["n"]), int(row["m"])), name
    eg = EndpointGraph.from_networkx(g)
    eg.adj = eg.adj.double()  # soft_stationarity computes Q = adj @ P in P's dtype (float64 here)
    log(name, f"n={eg.n} k={k} seeds={args.seeds}")
    codes = {s: run_one(name, k, eg, s, args) for s in args.seeds}
    log(name, f"exit codes {codes}")
    sys.exit(1 if any(c not in (0,) for c in codes.values()) else 0)


if __name__ == "__main__":
    main()
