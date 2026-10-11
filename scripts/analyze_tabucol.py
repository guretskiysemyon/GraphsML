"""TabuCol quality reference vs PI-GNN (docs/progress_exp1.md section 11 B). Descriptive only:
no tests. CPU only.

Inputs : results/tabucol/<graph>/seed<s>.json         (scripts/run_tabucol.py)
         results/analysis/tidy_seed_level.csv          (scripts/analyze_exp1.py; PI-GNN per seed)
         configs/sweep_manifest.csv
Outputs: results/analysis/tabucol_per_graph.csv, tabucol_by_family.csv, tabucol_vs_pignn.png

Per graph: TabuCol mean best conflicts over its seeds (primary) and best of them (secondary);
PI-GNN conflicts after strict descent at epoch 100,000, mean over 5 seeds (primary) and best
of 5 (secondary, decision 12). Same tie rule as section 9 (tie if |diff| < 1 conflict or
|d| <= 0.05, d = (PI-GNN - TabuCol) / max(TabuCol, 1)). The difference is never a gap to optimum.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIE_REL, TIE_ABS = 0.05, 1.0
PIGNN_EPOCH = 100_000
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
WIN, TIE, LOSS = "#2a78d6", "#898781", "#e34948"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabucol", default="results/tabucol")
    ap.add_argument("--pignn", default="results/analysis/tidy_seed_level.csv")
    ap.add_argument("--manifest", default="configs/sweep_manifest.csv")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--n-seeds", type=int, default=10)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest).rename(columns={"name": "graph"})
    tc = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.tabucol, "*", "seed*.json")))])
    tc = tc.drop(columns=["history"])
    per = tc.groupby("graph").size()
    incomplete = sorted(set(man.graph) - set(per[per == a.n_seeds].index))
    print(f"TabuCol rows: {len(tc)}; graphs with all {a.n_seeds} seeds: {(per == a.n_seeds).sum()} / {len(man)}")
    if incomplete:
        print(f"INCOMPLETE graphs (left out): {incomplete}")
    print("stop reasons:", tc.stopped_by.value_counts().to_dict())
    zero = tc[tc.best_conflicts == 0]
    if len(zero):
        print(f"WARNING: {len(zero)} runs reached 0 conflicts at k = chi - 1 (chi or data error?):",
              sorted(zero.graph.unique()))
    print(f"best colouring single-flip optimal: {tc.single_flip_optimal.mean():.3f} of runs; "
          f"descent changed the best colouring in {(tc.conflicts_after_descent < tc.best_conflicts).sum()} runs")
    print("iterations per run: median {:,.0f}, min {:,.0f}, max {:,.0f}; budget used: median {:.0f}s".format(
        tc.iterations.median(), tc.iterations.min(), tc.iterations.max(), tc.seconds.median()))

    g = tc.groupby("graph").agg(
        tabu_mean=("best_conflicts", "mean"), tabu_best=("best_conflicts", "min"), tabu_max=("best_conflicts", "max"),
        tabu_seeds=("seed", "nunique"), budget_s=("time_budget", "first"), iterations_median=("iterations", "median"),
        seconds_at_best_median=("seconds_at_best", "median"), tabu_sfo_frac=("single_flip_optimal", "mean"))
    g = g[g.tabu_seeds == a.n_seeds]

    pg = pd.read_csv(a.pignn)
    pg = pg[pg.epoch == PIGNN_EPOCH].groupby("graph").conflicts_after_descent.agg(
        pignn_mean="mean", pignn_best="min", pignn_seeds="count")
    tab = man.set_index("graph")[["family", "n", "m", "chi", "k"]].join(g, how="inner").join(pg, how="inner")
    tab["ratio_mean"] = tab.pignn_mean / tab.tabu_mean.clip(lower=1)
    for kind in ("mean", "best"):
        p, c = tab[f"pignn_{kind}"].to_numpy(float), tab[f"tabu_{kind}"].to_numpy(float)
        d = (p - c) / np.maximum(c, 1.0)
        tie = (np.abs(p - c) < TIE_ABS) | (np.abs(d) <= TIE_REL)
        tab[f"d_{kind}"] = d
        tab[f"verdict_{kind}"] = np.where(tie, "tie", np.where(p < c, "pignn_fewer", "tabucol_fewer"))
    tab = tab.sort_values("n")
    os.makedirs(a.out, exist_ok=True)
    tab.to_csv(f"{a.out}/tabucol_per_graph.csv")

    fam = tab.groupby("family").agg(
        graphs=("n", "size"), tabu_mean=("tabu_mean", "mean"), pignn_mean=("pignn_mean", "mean"),
        median_ratio=("ratio_mean", "median"), median_d=("d_mean", "median"),
        pignn_fewer=("verdict_mean", lambda v: int((v == "pignn_fewer").sum())),
        ties=("verdict_mean", lambda v: int((v == "tie").sum())),
        tabucol_fewer=("verdict_mean", lambda v: int((v == "tabucol_fewer").sum()))).reset_index()
    fam.to_csv(f"{a.out}/tabucol_by_family.csv", index=False)

    pd.set_option("display.width", 250)
    for kind in ("mean", "best"):
        print(f"\n[{kind}] PI-GNN (100k, after descent) vs TabuCol over {len(tab)} graphs:",
              tab[f"verdict_{kind}"].value_counts().to_dict(), f"median d = {tab[f'd_{kind}'].median():.3f}")
    print(f"\nmean conflicts: TabuCol {tab.tabu_mean.mean():.1f}, PI-GNN after descent {tab.pignn_mean.mean():.1f}")
    print(fam.round(3).to_string(index=False))
    print(tab[["family", "n", "k", "tabu_mean", "tabu_best", "pignn_mean", "pignn_best", "ratio_mean",
               "verdict_mean", "iterations_median"]].round(2).to_string())

    # figure: PI-GNN after descent vs TabuCol, per graph (log-log), coloured by verdict
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.3})
    ax = axes[0]
    x, y = tab.tabu_mean.clip(lower=0.5), tab.pignn_mean.clip(lower=0.5)
    lo, hi = min(x.min(), y.min()) * 0.7, max(x.max(), y.max()) * 1.4
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, label="equal conflicts")
    for name, col, mk, lab in (("pignn_fewer", WIN, "o", "PI-GNN fewer"), ("tie", TIE, "s", "tie"),
                               ("tabucol_fewer", LOSS, "^", "TabuCol fewer")):
        s = tab.verdict_mean == name
        ax.scatter(x[s], y[s], s=46, c=col, marker=mk, edgecolors=SURFACE, linewidths=1.2, zorder=3,
                   label=f"{lab} (n={int(s.sum())})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"TabuCol best conflicts (mean of {a.n_seeds} seeds)")
    ax.set_ylabel("PI-GNN (100k) -> descent: conflicts (mean of 5)")
    ax.grid(color=GRID, lw=0.6); ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("per graph", loc="left", fontsize=11)
    ax = axes[1]
    f = fam.sort_values("median_ratio")
    ax.barh(range(len(f)), f.median_ratio, color=[WIN if r < 1 - TIE_REL else LOSS if r > 1 + TIE_REL else TIE
                                                    for r in f.median_ratio], height=0.6)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(f))); ax.set_yticklabels([f"{r.family} ({r.graphs})" for r in f.itertuples()], fontsize=9)
    ax.set_xlabel("median over graphs of PI-GNN / TabuCol conflicts (log; 1 = equal)")
    ax.set_title("by family", loc="left", fontsize=11)
    for s in ("top", "right"):
        for axx in axes:
            axx.spines[s].set_visible(False)
    fig.suptitle(f"Experiment 1: PI-GNN + descent vs TabuCol (same wall-clock time as PI-GNN's training), k = chi-1, "
                 f"{len(tab)} graphs. TabuCol is a reference, not the optimum.", x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{a.out}/tabucol_vs_pignn.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {a.out}/tabucol_vs_pignn.png")


if __name__ == "__main__":
    main()
