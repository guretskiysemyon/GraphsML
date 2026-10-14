"""Experiment 2 (RQ3): decoders on frozen P, with null-P controls. One invocation = one run
(graph x seed) of one arm. CPU only. See docs/Experiment2_Steps.md.

For each P variant (trained, uniform, untrained, rowperm, colperm) built from the run's saved
checkpoint at --epoch, decode with argmax, D-CE (saturation order), D-CE (random order, null
control e) and K independent samples, then apply the same strict_vertex_descent as Experiment 1
(descent RNG = default_rng(seed)) to every decoded colouring. Nothing is retrained.

Every row records the decode time and the descent time separately, and the DONE marker
records the run's total wall time, so the experiment's cost can be read off directly.

Output: <out>/<arm>/<graph>_seed<seed>/decoders.jsonl (one row per variant x decoder) + DONE.
Idempotent and requeue-safe: a run with a DONE marker is skipped.

    python scripts/run_decoders.py --task-id 0                       # Experiment 1 outputs
    python scripts/run_decoders.py --graph queen5_5 --seed 0 --samples 4
    python scripts/run_decoders.py --task-id 0 --runs results/exp3/B --arm B --variants trained colperm
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.decoders import (  # noqa: E402
    dce,
    permute_colours_within_rows,
    permute_rows,
    potts_loss_np,
    sample_colourings,
    softmax_np,
    uniform_noise_P,
)
from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import (  # noqa: E402
    EndpointGraph,
    conflicts_from_counts,
    neighbour_colour_counts,
    single_flip_report,
    strict_vertex_descent,
)

COL_DIR = ROOT / "data" / "dimacs" / "col"
VARIANTS = ("trained", "uniform", "untrained", "rowperm", "colperm")
SEEDS = (0, 1, 2, 3, 4)


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def load_logits(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    Z = obj["Z"] if isinstance(obj, dict) else obj
    return Z.detach().double().numpy()


def build_P(variant: str, trained_P: np.ndarray, run_dir: Path, k: int, seed: int) -> np.ndarray:
    n = trained_P.shape[0]
    if variant == "trained":
        return trained_P
    if variant == "uniform":
        return uniform_noise_P(n, k, seed)
    if variant == "untrained":
        return softmax_np(load_logits(run_dir / "z0.pt"))
    if variant == "rowperm":
        return permute_rows(trained_P, seed)
    if variant == "colperm":
        return permute_colours_within_rows(trained_P, seed)
    raise ValueError(variant)


def hard_row(eg: EndpointGraph, colours: np.ndarray, k: int, seed: int) -> dict:
    """Raw conflicts, single-flip report and descent for one hard colouring."""
    M = neighbour_colour_counts(eg, colours, k)
    flip = single_flip_report(colours, M)
    res = strict_vertex_descent(eg, colours, k, seed=seed, M=M)
    return dict(conflicts=conflicts_from_counts(colours, M), single_flip_optimal=flip.is_optimal,
                conflicts_after_descent=res.conflicts_after, descent_moves=res.n_moves,
                descent_seconds=res.seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, help="Slurm array id: graph row = id // 5, seed = id % 5")
    ap.add_argument("--graph", help="graph name (alternative to --task-id, with --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "main",
                    help="arm directory holding <graph>_seed<s>/checkpoints/")
    ap.add_argument("--arm", default="A", help="name of the arm, used for the output directory")
    ap.add_argument("--epoch", type=int, default=100000, help="which checkpoint's P to decode")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--samples", type=int, default=32, help="K, colourings sampled from each P")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "exp2")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "sweep_manifest.csv")
    args = ap.parse_args()
    t_start = time.perf_counter()

    manifest = pd.read_csv(args.manifest, index_col="idx")
    if args.graph is not None:
        row, seed = manifest[manifest["name"] == args.graph].iloc[0], args.seed
    elif args.task_id is not None:
        row, seed = manifest.loc[args.task_id // len(SEEDS)], SEEDS[args.task_id % len(SEEDS)]
    else:
        sys.exit("give --task-id or --graph")
    name, k = row["name"], int(row["k"])
    run_name = f"{name}_seed{seed}"
    run_dir = args.runs / run_name
    out_dir = args.out / args.arm / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        log(run_name, "already done, skipping")
        return

    g = load_col_graph(COL_DIR / f"{name}.col")
    n, m = g.number_of_nodes(), g.number_of_edges()
    if (n, m) != (int(row["n"]), int(row["m"])):
        sys.exit(f"{name}: manifest mismatch: file has n={n} m={m}, manifest n={row['n']} m={row['m']}")
    log(run_name, f"host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID')} "
                  f"task={os.environ.get('SLURM_ARRAY_TASK_ID')} arm={args.arm} n={n} m={m} k={k} "
                  f"epoch={args.epoch} variants={args.variants} K={args.samples}")
    eg = EndpointGraph.from_networkx(g)
    trained_P = softmax_np(load_logits(run_dir / "checkpoints" / f"ckpt_{args.epoch:08d}.pt"))
    if trained_P.shape != (n, k):
        sys.exit(f"{run_name}: P has shape {trained_P.shape}, expected {(n, k)}")

    base = dict(graph=name, arm=args.arm, seed=seed, epoch=args.epoch, k=k, n=n, m=m)
    rows = []
    for variant in args.variants:
        t0 = time.perf_counter()
        P = build_P(variant, trained_P, run_dir, k, seed)
        build_seconds = time.perf_counter() - t0
        vb = dict(base, variant=variant, potts_loss=potts_loss_np(eg, P), build_seconds=build_seconds)

        t0 = time.perf_counter()
        colours = P.argmax(axis=1)
        rows.append(dict(vb, decoder="argmax", decode_seconds=time.perf_counter() - t0,
                         **hard_row(eg, colours, k, seed)))

        for decoder, order in (("dce", "saturation"), ("dce_rand", "random")):
            t0 = time.perf_counter()
            colours = dce(eg, P, k, order=order, seed=seed)
            rows.append(dict(vb, decoder=decoder, decode_seconds=time.perf_counter() - t0,
                             **hard_row(eg, colours, k, seed)))

        t0 = time.perf_counter()
        samples = sample_colourings(P, args.samples, seed)
        decode_seconds = time.perf_counter() - t0
        per = [hard_row(eg, s, k, seed) for s in samples]
        raw = np.array([r["conflicts"] for r in per])
        after = np.array([r["conflicts_after_descent"] for r in per])
        rows.append(dict(vb, decoder="sample", n_samples=args.samples, decode_seconds=decode_seconds,
                         conflicts=int(raw.min()), conflicts_mean=float(raw.mean()),
                         single_flip_optimal_frac=float(np.mean([r["single_flip_optimal"] for r in per])),
                         conflicts_after_descent=int(after.min()), conflicts_after_descent_mean=float(after.mean()),
                         descent_moves=int(sum(r["descent_moves"] for r in per)),
                         descent_seconds=float(sum(r["descent_seconds"] for r in per))))

        vrows = rows[-4:]
        log(run_name, f"{variant}: L(P)={vb['potts_loss']:.1f} " + " ".join(
            f"{r['decoder']}={r['conflicts']}->{r['conflicts_after_descent']}" for r in vrows)
            + f" ({sum(r['decode_seconds'] + r['descent_seconds'] for r in vrows) + build_seconds:.1f}s)")

    tmp = out_dir / "decoders.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, out_dir / "decoders.jsonl")
    total = time.perf_counter() - t_start
    (out_dir / "DONE").write_text(json.dumps(dict(
        n_rows=len(rows), total_seconds=total, host=socket.gethostname(),
        slurm_job=os.environ.get("SLURM_JOB_ID"), time=time.strftime("%Y-%m-%dT%H:%M:%S"))))
    log(run_name, f"DONE in {total:.1f}s")


if __name__ == "__main__":
    main()
