"""Experiment 3 tuning of the free-logit arm (rung B) on the synthetic tuning set, with the same
protocol and effort as PI-GNN's (tune_stage1.py; progress_exp1.md decision 7):

  stage 1: 3 reference configs + 40 random draws x 13 graphs x 2 regimes (k_hi, k_hi - 1)
           x 2 seeds x 10,000 steps;
  stage 2: top 6 of stage 1 + 4 draws near them x 3 seeds x 50,000 steps.

Score = mean rank of final eval-mode conflicts (scripts/summarize_tuning.py, unchanged).
Every unit starts from PI-GNN's initial eval-mode logits Z0 for that graph, k and seed, built
with the frozen PI-GNN config (configs/pignn_frozen.json), i.e. exactly B's start on DIMACS.
No logit noise during tuning (rung B is the pure free-logit arm).

Units are split across tasks as in tune_stage1.py (task t runs units i with i % T == t);
one JSON line per finished unit, resumable after preemption.

    python scripts/tune_free_logits.py --task 0 --n-tasks 24                      # stage 1
    python scripts/tune_free_logits.py --write-stage2 results/tuning/free_stage1  # after summarize_tuning
    python scripts/tune_free_logits.py --task 0 --n-tasks 24 --configs results/tuning/free_stage2_input_configs.json \
        --n-epochs 50000 --eval-every 1000 --seeds 3 --out results/tuning/free_stage2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.free_logits import FreeLogitConfig, pignn_initial_logits, train_free_logits_lean  # noqa: E402
from graphsml.tune import free_logit_stage2_configs, sample_free_logit_configs  # noqa: E402

TUNING_DIR = ROOT / "data" / "synthetic" / "tuning"
CONFIG_KEYS = ("lr", "optimizer", "weight_decay")


def graph_to_adj(graph):
    n = graph.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in graph.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def write_stage2(stage1_dir: Path) -> None:
    summary = pd.read_csv(stage1_dir.parent / f"{stage1_dir.name}_summary.csv")  # sorted by mean_rank
    cfgs = free_logit_stage2_configs(summary.to_dict("records"))
    out = stage1_dir.parent / "free_stage2_input_configs.json"
    out.write_text(json.dumps(cfgs, indent=1))
    print(f"wrote {len(cfgs)} configs to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--n-epochs", type=int, default=10_000, help="optimisation steps per unit")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tuning" / "free_stage1")
    ap.add_argument("--configs", type=Path, default=None, help="JSON list of configs (default: sample stage-1 configs)")
    ap.add_argument("--graphs", nargs="*", help="debug: restrict to these tuning graphs")
    ap.add_argument("--max-units", type=int, default=None, help="debug: stop after this many units")
    ap.add_argument("--write-stage2", type=Path, metavar="STAGE1_DIR",
                    help="write free_stage2_input_configs.json from STAGE1_DIR's summary and exit")
    args = ap.parse_args()

    if args.write_stage2 is not None:
        write_stage2(args.write_stage2)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available; refusing to fall back to CPU silently (use --device cpu).")

    if args.configs is not None:
        cfgs = [{k: c[k] for k in CONFIG_KEYS} for c in json.loads(args.configs.read_text())]
    else:
        cfgs = [c.__dict__ for c in sample_free_logit_configs(args.n_random, seed=0)]
    frozen = json.loads((ROOT / "configs" / "pignn_frozen.json").read_text())
    manifest = pd.read_csv(TUNING_DIR / "manifest.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "configs.json").write_text(json.dumps(cfgs, indent=1))

    graph_ids = [gi for gi in range(len(manifest)) if not args.graphs or manifest.iloc[gi]["name"] in args.graphs]
    units = [(ci, gi, regime, seed)
             for gi in graph_ids for ci in range(len(cfgs))
             for regime in ("hi", "lo") for seed in range(args.seeds)]
    mine = [u for i, u in enumerate(units) if i % args.n_tasks == args.task]

    out_file = args.out / f"task_{args.task:03d}.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["graph_idx"], r["regime"], r["seed"]))
    print(f"task {args.task}/{args.n_tasks}: {len(mine)} units, {len(done)} already done, device={args.device}", flush=True)

    adj_cache, z0_cache = {}, {}
    n_run = 0
    with out_file.open("a") as f:
        for ci, gi, regime, seed in mine:
            if (ci, gi, regime, seed) in done:
                continue
            row = manifest.iloc[gi]
            if gi not in adj_cache:
                adj_cache[gi] = graph_to_adj(load_col_graph(TUNING_DIR / f"{row['name']}.col"))
            k = int(row["k_hi"]) - (1 if regime == "lo" else 0)
            if (gi, k, seed) not in z0_cache:
                z0_cache[(gi, k, seed)] = pignn_initial_logits(
                    adj_cache[gi], k, frozen["embed_dim"], frozen["hidden_dim"], frozen["dropout"], seed, args.device)
            cfg = FreeLogitConfig(k=k, n_steps=args.n_epochs, seed=seed, device=args.device, **cfgs[ci])
            res = train_free_logits_lean(adj_cache[gi], z0_cache[(gi, k, seed)], cfg, args.eval_every)
            f.write(json.dumps(dict(config=ci, graph_idx=gi, graph=row["name"], regime=regime, seed=seed,
                                    k=k, m=int(row["m"]), n=int(row["n"]), **res)) + "\n")
            f.flush()
            n_run += 1
            if n_run % 10 == 0:
                print(f"  {n_run} units done", flush=True)
            if args.max_units and n_run >= args.max_units:
                break
    print("finished", flush=True)


if __name__ == "__main__":
    main()
