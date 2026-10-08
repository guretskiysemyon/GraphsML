"""Rank stage-1 configs. Primary score: mean rank over (graph, regime) of the seed-averaged
final eval-mode conflicts (rank 1 = fewest). Also reports mean normalised conflicts (conflicts / m)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results" / "tuning" / "stage1")
    args = ap.parse_args()
    cfgs = json.loads((args.dir / "configs.json").read_text())
    rows = []
    for f in sorted(args.dir.glob("task_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows.append(dict(config=r["config"], graph=r["graph"], regime=r["regime"], seed=r["seed"],
                             m=r["m"], final=r["eval_conflicts"][-1], best=min(r["eval_conflicts"]),
                             entropy=r["eval_entropy"][-1], seconds=r["seconds"]))
    df = pd.DataFrame(rows)
    n_expected = df.groupby(["graph", "regime", "seed"]).config.nunique()
    print(f"{len(df)} units loaded; configs per (graph,regime,seed): min={n_expected.min()} max={n_expected.max()}")
    cell = df.groupby(["config", "graph", "regime"]).agg(final=("final", "mean"), m=("m", "first")).reset_index()
    # only compare configs on cells where every config has a result
    full = cell.groupby(["graph", "regime"]).config.transform("nunique") == cell.config.nunique()
    cell = cell[full].copy()
    cell["rank"] = cell.groupby(["graph", "regime"]).final.rank(method="average")
    cell["norm"] = cell.final / cell.m
    out = cell.groupby("config").agg(mean_rank=("rank", "mean"), cells=("rank", "size")).join(
        cell.pivot_table(index="config", columns="regime", values="norm", aggfunc="mean").add_prefix("norm_conf_"))
    out = out.join(pd.DataFrame(cfgs).rename_axis("config"))
    out["seconds_per_unit"] = df.groupby("config").seconds.mean()
    out = out.sort_values("mean_rank")
    out.to_csv(args.dir.parent / "stage1_summary.csv")
    print(out.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
