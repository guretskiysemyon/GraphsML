"""Freeze the Experiment 3 free-logit config: the stage-2 winner (lowest mean rank in
results/tuning/free_stage2_summary.csv, written by summarize_tuning.py), the same selection rule
that froze PI-GNN's config (configs/pignn_frozen.json). Writes configs/free_logits_frozen.json.
Refuses to overwrite an existing frozen config unless --force.

Also used for Experiment 3 rung C (the no-message-passing MLP), with its own paths, the PI-GNN
hyperparameter keys and arch "mlp":
    python scripts/freeze_free_logits.py --stage1 results/tuning/C_stage1_summary.csv \
        --stage2 results/tuning/C_stage2_summary.csv --stage2-input results/tuning/C_stage2_input_configs.json \
        --out configs/mlp_frozen.json --keys lr hidden_dim embed_dim dropout --set '{"arch": "mlp"}' \
        --tuning-start "the model's own random initialisation"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, default=ROOT / "results" / "tuning" / "free_stage1_summary.csv")
    ap.add_argument("--stage2", type=Path, default=ROOT / "results" / "tuning" / "free_stage2_summary.csv")
    ap.add_argument("--stage2-input", type=Path, default=ROOT / "results" / "tuning" / "free_stage2_input_configs.json")
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "free_logits_frozen.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keys", nargs="+", default=["lr", "optimizer", "weight_decay"])
    ap.add_argument("--set", default="{}", help="JSON of extra fields to store, e.g. {\"arch\": \"mlp\"}")
    ap.add_argument("--tuning-start", default="PI-GNN's Z0 (frozen PI-GNN config) per graph/k/seed")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} exists; not overwriting (use --force)")

    s2 = pd.read_csv(args.stage2)  # sorted by mean_rank
    best = s2.iloc[0]
    s2_inputs = json.loads(args.stage2_input.read_text())
    stage1_id = s2_inputs[int(best["config"])].get("stage1_id")
    s1 = pd.read_csv(args.stage1)
    s1_rank = float(s1.loc[s1.config == stage1_id, "mean_rank"].iloc[0]) if stage1_id is not None else None
    cast = {"lr": float, "weight_decay": float, "dropout": float, "optimizer": str, "hidden_dim": int, "embed_dim": int}
    frozen = dict(
        **{key: cast[key](best[key]) for key in args.keys}, **json.loads(args.set),
        _provenance=dict(
            selected_from=str(args.stage2.resolve().relative_to(ROOT)), stage2_config=int(best["config"]),
            stage1_id=stage1_id, stage1_mean_rank=s1_rank, stage2_mean_rank=float(best["mean_rank"]),
            runner_up_stage2_mean_rank=float(s2.iloc[1]["mean_rank"]) if len(s2) > 1 else None,
            tuning_set=f"data/synthetic/tuning (13 graphs), start = {args.tuning_start}",
            selection_criterion="mean rank of seed-averaged final eval-mode conflicts across (graph, regime) cells, "
                                "regime in {k=chi_hi, k=chi_hi-1} (same as PI-GNN)",
            stage1_budget_steps=10000, stage2_budget_steps=50000,
            selected_from_stage1=str(args.stage1.resolve().relative_to(ROOT)),
            frozen_on=time.strftime("%Y-%m-%d")))
    args.out.write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
