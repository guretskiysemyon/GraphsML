"""Experiment 3 (RQ2) analysis: PI-GNN (A) vs free logits from A's Z0 (B) vs free logits from a
random start (B-rand), plus any optional arms present (noise controls). CPU only.

Protocol (docs/Experiment3_Steps.md section 6, to be fixed before any result is looked at):
  * graph = statistical unit; per graph, mean over the 5 seeds (same seed = same Z0 for A and B);
  * primary metric = raw argmax conflicts (plan: "report argmax first as the measure of raw
    optimiser quality"); secondary = conflicts after strict vertex descent, single-flip-optimal
    fraction, eval Potts loss;
  * two views: matched steps (budget T = 100k, the step budget shared by all 68 graphs; and each
    graph's final A budget) and matched wall-clock time (B's checkpoint where its cumulative
    seconds first reached A's seconds at epoch T, labelled "A@T");
  * per graph: d = (X - Y) / max(Y, 1); tie if |X - Y| < 1 or |d| <= 0.05 (the Experiment 1 rule);
    two-sided Wilcoxon signed-rank on d over graphs, descriptive (DIMACS is not an IID sample).

Inputs : <main>/<g>_seed<s>/{endpoint_trajectory,trajectory}.jsonl (arm A),
         <exp3>/<arm>/<g>_seed<s>/{endpoint_trajectory,trajectory}.jsonl (free-logit arms),
         optional movement.jsonl in each run (scripts/run_movement.py) for Potts loss / Dirichlet energy.
Outputs: <out>/exp3_tidy.csv, exp3_per_graph.csv, exp3_summary.csv, exp3_figure3.png
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
from scipy.stats import wilcoxon

TIE_REL, TIE_ABS = 0.05, 1.0
T_MATCHED = 100_000
METRICS = {"conflicts": "argmax conflicts", "conflicts_after_descent": "conflicts after descent",
           "single_flip_optimal": "single-flip-optimal fraction", "eval_loss": "Potts loss (eval)"}
# (X, Y, view, role): X better than Y = X has fewer conflicts
COMPARISONS = [("B", "A", "steps", "primary"), ("B", "A", "time", "primary"),
               ("B", "Brand", "steps", "primary"), ("Brand", "A", "steps", "secondary"),
               ("Brand", "A", "time", "secondary")]
COLOURS = {"A": "#2a78d6", "B": "#e34948", "Brand": "#f0a020"}


def load_arm(root, arm):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*_seed*"))):
        g, s = os.path.basename(d).rsplit("_seed", 1)
        f_ep, f_tr = os.path.join(d, "endpoint_trajectory.jsonl"), os.path.join(d, "trajectory.jsonl")
        if not (os.path.exists(os.path.join(d, "DONE")) and os.path.exists(f_ep)):
            continue
        ep = pd.DataFrame([json.loads(l) for l in open(f_ep)])
        tr = pd.DataFrame([json.loads(l) for l in open(f_tr)])[["epoch", "cumulative_seconds", "eval_entropy"]]
        df = ep.merge(tr, on="epoch", how="left")
        f_mv = os.path.join(d, "movement.jsonl")
        if os.path.exists(f_mv):
            mv = pd.DataFrame([json.loads(l) for l in open(f_mv)])
            df = df.merge(mv[["epoch", "eval_loss", "dirichlet", "logit_dist_rel", "param_dist_rel"]], on="epoch", how="left")
        df["graph"], df["seed"], df["arm"] = g, int(s), arm
        if "time_matched" not in df:
            df["time_matched"] = [[] for _ in range(len(df))]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def view(df, arm, t, mode):
    """One row per (graph, seed): arm at step t (mode 'steps') or at the checkpoint matched to
    A's time at epoch t (mode 'time'; arm A is its own time match). t='final' = A's last epoch."""
    d = df[df.arm == arm]
    if d.empty:
        return d
    a_final = df[df.arm == "A"].groupby(["graph", "seed"]).epoch.max().rename("a_final").reset_index()
    d = d.merge(a_final, on=["graph", "seed"])
    target = d.a_final if t == "final" else t
    if mode == "steps" or arm == "A":
        return d[d.epoch == target]
    lab = "A@" + target.astype(str) if t == "final" else pd.Series(f"A@{t}", index=d.index)
    hit = [l in tm for l, tm in zip(lab, d.time_matched)]
    return d[hit].sort_values("epoch").groupby(["graph", "seed"]).head(1)


def compare(px, py, metric):
    j = pd.concat([px.rename("X"), py.rename("Y")], axis=1).dropna()
    d = (j.X - j.Y) / np.maximum(j.Y, 1.0)
    tie = ((j.X - j.Y).abs() < TIE_ABS) | (d.abs() <= TIE_REL)
    if metric == "single_flip_optimal":  # higher is better; a fraction, so compare absolutely
        d, tie = j.X - j.Y, (j.X - j.Y).abs() < 0.2
        better = j.X > j.Y
    else:
        better = j.X < j.Y
    res = np.where(tie, "tie", np.where(better, "X better", "Y better"))
    try:
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    except ValueError:
        p = np.nan
    return dict(graphs=len(j), X_better=int((res == "X better").sum()), ties=int((res == "tie").sum()),
                Y_better=int((res == "Y better").sum()), median_d=float(d.median()) if len(d) else np.nan,
                mean_X=float(j.X.mean()) if len(j) else np.nan, mean_Y=float(j.Y.mean()) if len(j) else np.nan,
                wilcoxon_p=float(p)), pd.Series(res, index=j.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--exp3", default="results/exp3")
    ap.add_argument("--manifest", default="configs/sweep_manifest.csv")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = pd.read_csv(a.manifest).rename(columns={"name": "graph"})

    arms = {"A": load_arm(a.main, "A")}
    for d in sorted(glob.glob(os.path.join(a.exp3, "*"))):
        if not glob.glob(os.path.join(d, "*_seed*", "DONE")):
            continue  # not an arm directory (tuning output, analysis, files)
        arm = os.path.basename(d)
        arms[arm] = load_arm(d, arm)
        print(f"arm {arm}: {arms[arm].graph.nunique() if len(arms[arm]) else 0} graphs, "
              f"{arms[arm].groupby(['graph', 'seed']).ngroups if len(arms[arm]) else 0} runs")
    df = pd.concat([v for v in arms.values() if len(v)], ignore_index=True)
    df.drop(columns=["time_matched"]).assign(time_matched=df.time_matched.map(json.dumps)).to_csv(
        f"{a.out}/exp3_tidy.csv", index=False)
    # restrict to graphs that every free-logit arm has, so all comparisons use one graph set
    free = [k for k in arms if k != "A" and len(arms[k])]
    graphs = set(arms["A"].graph)
    for k in free:
        graphs &= set(arms[k].graph)
    df = df[df.graph.isin(graphs)]
    print(f"{len(graphs)} graphs common to A and {free}")

    per_graph, summary = [], []
    for t in (T_MATCHED, "final"):
        for mode in ("steps", "time"):
            for arm in ["A"] + free:
                v = view(df, arm, t, mode)
                if v.empty:
                    continue
                cols = [c for c in METRICS if c in v]
                agg = v.groupby("graph")[cols + ["epoch", "cumulative_seconds"]].mean()
                agg["n_seeds"] = v.groupby("graph").seed.nunique()
                per_graph.append(agg.reset_index().assign(arm=arm, budget=str(t), view=mode))
    pg = pd.concat(per_graph, ignore_index=True).merge(man[["graph", "family", "n", "m", "chi", "k"]], on="graph")
    pg.to_csv(f"{a.out}/exp3_per_graph.csv", index=False)

    for x, y, mode, role in COMPARISONS + [(f, "A", m, "secondary") for f in free if f not in ("B", "Brand")
                                           for m in ("steps", "time")]:
        for t in (str(T_MATCHED), "final"):
            # the B-vs-Brand comparison is between two free-logit arms: both use their step (or time) view
            px_all = pg[(pg.arm == x) & (pg.budget == t) & (pg.view == mode)].set_index("graph")
            py_all = pg[(pg.arm == y) & (pg.budget == t) & (pg.view == mode)].set_index("graph")
            for metric in METRICS:
                if metric not in px_all or metric not in py_all:
                    continue
                stats, _ = compare(px_all[metric], py_all[metric], metric)
                summary.append(dict(X=x, Y=y, view=mode, budget=t, metric=metric, role=role, **stats))
    sm = pd.DataFrame(summary)
    sm.to_csv(f"{a.out}/exp3_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sm[(sm.metric == "conflicts") & (sm.budget == str(T_MATCHED))].round(4).to_string(index=False))

    # Figure 3: per budget, median over graphs of conflicts / m (matched graph set per budget)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    tidy = df.merge(man[["graph", "m"]], on="graph")
    tidy["conf_per_edge"] = tidy.conflicts / tidy.m
    a_epochs = tidy[tidy.arm == "A"].groupby("epoch").graph.nunique()
    shared = a_epochs[a_epochs == len(graphs)].index
    for arm in ["A"] + free:
        d = tidy[(tidy.arm == arm) & tidy.epoch.isin(shared)]
        s = d.groupby(["graph", "epoch"]).conf_per_edge.mean().groupby("epoch").median()
        axes[0].plot(s.index, s.values, marker="o", ms=3, label=arm, color=COLOURS.get(arm))
        for ax, col, scale in ((axes[2], "eval_loss", "m"), (axes[3], "dirichlet", "m")):
            if col in d and d[col].notna().any():
                s2 = (d[col] / d[scale]).groupby([d.graph, d.epoch]).mean().groupby("epoch").median()
                ax.plot(s2.index, s2.values, marker="o", ms=3, label=arm, color=COLOURS.get(arm))
        # time view: B's checkpoint matched to A@e, x = median A seconds at e
        xs, ys = [], []
        for e in shared:
            v = view(tidy, arm, int(e), "time")
            if len(v) and v.graph.nunique() == len(graphs):
                ta = tidy[(tidy.arm == "A") & (tidy.epoch == e)].cumulative_seconds.median()
                xs.append(ta)
                ys.append(v.groupby("graph").conf_per_edge.mean().median())
        axes[1].plot(xs, ys, marker="o", ms=3, label=arm, color=COLOURS.get(arm))
    titles = ["argmax conflicts / m vs steps", "argmax conflicts / m vs A's wall-clock time (s)",
              "Potts loss / m vs steps", "Dirichlet energy / m vs steps"]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Experiment 3: median over {len(graphs)} graphs (seed means), k = chi - 1", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{a.out}/exp3_figure3.png", dpi=150)
    print("wrote", f"{a.out}/exp3_figure3.png")


if __name__ == "__main__":
    main()
