"""Experiment 3, item 1 of the follow-ups (descriptive, no tests): how each arm gets to where it
stops, from the always-logged diagnostics 4C/4E (scripts/run_movement.py -> movement.jsonl).

Per arm (A = results/main, every arm directory under results/exp3), per checkpoint: median over
graphs of the seed-mean of
  margin_median          median over vertices of the top-1 minus top-2 logit (saturation)
  frac_pmax_099          fraction of vertices with p_max >= 0.99 (one-hotness)
  eval_entropy           mean row entropy (nats)
  logit_dist_centred_rel ||Z - Z0|| / ||Z0|| on row-centred logits (movement in the softmax-relevant part)
  param_dist_rel         ||theta - theta0|| / ||theta0|| (= logit movement for free logits)
  dirichlet / 2m         Dirichlet energy per edge, scaled so a proper hard colouring would give 1
  softness_gap           (2 (m - conflicts) - Dirichlet) / 2m: 0 at a hard colouring, > 0 while P is soft
plus the same at the 100k checkpoint per family. Graph = unit, each graph weight 1.

Outputs: <out>/exp3_movement_tidy.csv, exp3_movement_summary.csv, exp3_movement_by_family.csv,
         exp3_movement.png
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_MATCHED = 100_000
COLS = ["margin_median", "frac_pmax_099", "eval_entropy", "logit_dist_centred_rel", "param_dist_rel",
        "dirichlet_per_2m", "softness_gap", "conflicts_per_m"]
TITLES = {"margin_median": "median logit margin (top1 - top2)", "frac_pmax_099": "fraction of vertices with p_max >= 0.99",
          "eval_entropy": "mean row entropy (nats)", "logit_dist_centred_rel": "||Z - Z0|| / ||Z0|| (row-centred)",
          "param_dist_rel": "||theta - theta0|| / ||theta0||", "dirichlet_per_2m": "Dirichlet energy / 2m",
          "softness_gap": "softness gap (2(m - conflicts) - E) / 2m", "conflicts_per_m": "argmax conflicts / m"}
LOG_Y = {"margin_median", "logit_dist_centred_rel", "param_dist_rel", "softness_gap", "conflicts_per_m", "eval_entropy"}


def load(root, arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*_seed*", "movement.jsonl"))):
        rows += [dict(json.loads(l), arm=arm) for l in open(f)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--exp3", default="results/exp3")
    ap.add_argument("--manifest", default="configs/sweep_manifest.csv")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    man = pd.read_csv(a.manifest).rename(columns={"name": "graph"})

    rows = load(a.main, "A")
    for d in sorted(glob.glob(os.path.join(a.exp3, "*"))):
        if glob.glob(os.path.join(d, "*_seed*", "movement.jsonl")):
            rows += load(d, os.path.basename(d))
    df = pd.DataFrame(rows).merge(man[["graph", "family", "m"]], on="graph")
    # step-grid checkpoints only: drop the extra time-matched ones, but keep a grid checkpoint that
    # also happened to be time-matched (A's grid per graph is a subset of every arm's grid)
    a_grid = set(zip(df[df.arm == "A"].graph, df[df.arm == "A"].epoch))
    df = df[[(not tm) or ((g, e) in a_grid) for tm, g, e in zip(df.time_matched, df.graph, df.epoch)]]
    df["dirichlet_per_2m"] = df.dirichlet / (2 * df.m)
    df["softness_gap"] = (2 * (df.m - df.eval_conflicts) - df.dirichlet) / (2 * df.m)
    df["conflicts_per_m"] = df.eval_conflicts / df.m
    df.drop(columns=["time_matched"]).to_csv(f"{a.out}/exp3_movement_tidy.csv", index=False)
    arms = sorted(df.arm.unique(), key=lambda s: (s != "A", s))
    print({arm: df[df.arm == arm].groupby(["graph", "seed"]).ngroups for arm in arms})

    per_graph = df.groupby(["arm", "graph", "family", "epoch"])[COLS].mean().reset_index()
    n_graphs = per_graph.groupby(["arm", "epoch"]).graph.nunique().rename("graphs")
    summary = per_graph.groupby(["arm", "epoch"])[COLS].median().join(n_graphs).reset_index()
    summary.to_csv(f"{a.out}/exp3_movement_summary.csv", index=False)
    fam = (per_graph[per_graph.epoch == T_MATCHED].groupby(["arm", "family"])[COLS].median()
           .join(per_graph[per_graph.epoch == T_MATCHED].groupby(["arm", "family"]).graph.nunique().rename("graphs"))
           .reset_index())
    fam.to_csv(f"{a.out}/exp3_movement_by_family.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[summary.epoch.isin([1, 100, 1000, 10000, T_MATCHED])].round(4).to_string(index=False))

    # figure: one panel per diagnostic, median over the graphs that have each epoch (all 68 up to 100k)
    full = n_graphs[n_graphs == n_graphs.max()].reset_index()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    cmap = plt.get_cmap("tab10")
    for i, arm in enumerate(arms):
        ok = full[full.arm == arm].epoch
        s = summary[(summary.arm == arm) & summary.epoch.isin(ok)]
        for ax, col in zip(axes.ravel(), COLS):
            ax.plot(s.epoch, s[col], marker="o", ms=3, label=arm, color=cmap(i))
    for ax, col in zip(axes.ravel(), COLS):
        ax.set_xscale("log")
        if col in LOG_Y:
            ax.set_yscale("log")
        ax.set_title(TITLES[col], fontsize=10)
        ax.set_xlabel("optimisation step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Experiment 3 diagnostics (4C/4E): median over {int(n_graphs.max())} graphs of seed means, "
                 "k = chi - 1", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{a.out}/exp3_movement.png", dpi=150)
    print("wrote", f"{a.out}/exp3_movement.png")


if __name__ == "__main__":
    main()
