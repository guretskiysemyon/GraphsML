"""Stage 1 hyperparameter search: random search on the synthetic tuning set.

Units = config x graph x regime (k_hi, k_hi-1) x seed. Task t of T runs units i with i % T == t,
appending one JSON line per finished unit (resumable after preemption).
Requires a GPU unless --device cpu is passed explicitly.
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
from graphsml.tune import TuneConfig, pignn_stage2_configs, sample_configs, train_lean  # noqa: E402

TUNING_DIR = ROOT / "data" / "synthetic" / "tuning"


def graph_to_adj(graph):
    n = graph.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in graph.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--n-epochs", type=int, default=10_000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tuning" / "stage1")
    ap.add_argument("--configs", type=Path, default=None, help="JSON list of configs (default: sample stage-1 configs)")
    ap.add_argument("--max-units", type=int, default=None, help="debug: stop after this many units")
    ap.add_argument("--arch", default="sage", choices=["sage", "mlp"],
                    help="sage = PI-GNN; mlp = Experiment 3 rung C (no message passing)")
    ap.add_argument("--write-stage2", type=Path, metavar="STAGE1_DIR",
                    help="write <parent>/<name>_stage2_input_configs.json from STAGE1_DIR's summary and exit")
    args = ap.parse_args()

    if args.write_stage2 is not None:
        d = args.write_stage2
        ranked = pd.read_csv(d.parent / f"{d.name}_summary.csv").to_dict("records")  # sorted by mean_rank
        out = d.parent / f"{d.name.replace('stage1', 'stage2')}_input_configs.json"
        out.write_text(json.dumps(pignn_stage2_configs(ranked), indent=1))
        print(f"wrote {out}")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available; refusing to fall back to CPU silently (use --device cpu).")

    if args.configs is not None:
        cfgs = [TuneConfig(**{k: c[k] for k in ('lr', 'hidden_dim', 'embed_dim', 'dropout')}) for c in json.loads(args.configs.read_text())]
    else:
        cfgs = sample_configs(args.n_random, seed=0)
    manifest = pd.read_csv(TUNING_DIR / "manifest.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "configs.json").write_text(json.dumps([c.__dict__ for c in cfgs], indent=1))

    units = [(ci, gi, regime, seed)
             for gi in range(len(manifest)) for ci in range(len(cfgs))
             for regime in ("hi", "lo") for seed in range(args.seeds)]
    mine = [u for i, u in enumerate(units) if i % args.n_tasks == args.task]

    out_file = args.out / f"task_{args.task:03d}.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["graph_idx"], r["regime"], r["seed"]))
    print(f"task {args.task}/{args.n_tasks}: {len(mine)} units, {len(done)} already done, device={args.device} "
          f"arch={args.arch}", flush=True)

    adj_cache = {}
    n_run = 0
    with out_file.open("a") as f:
        for ci, gi, regime, seed in mine:
            if (ci, gi, regime, seed) in done:
                continue
            row = manifest.iloc[gi]
            if gi not in adj_cache:
                adj_cache[gi] = graph_to_adj(load_col_graph(TUNING_DIR / f"{row['name']}.col"))
            k = int(row["k_hi"]) - (1 if regime == "lo" else 0)
            res = train_lean(adj_cache[gi], k, cfgs[ci], args.n_epochs, seed, args.device, args.eval_every, args.arch)
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
