"""Step I, panel C: aggregate the soft-space diagnostics written by scripts/run_soft.py and
draw the (H_tilde, g_tilde) companion figure. CPU only.

Reads results/main/<graph>_seed<s>/{soft_diagnostics.jsonl, soft_pervertex.npz} and
configs/sweep_manifest.csv. Graph is the statistical unit (decision 5): seed-level summaries
are averaged within a graph, and in the joint histograms each graph has total weight 1
(each vertex is weighted 1 / (non-isolated vertices x seeds)), so large graphs do not
dominate. Matched budgets only: a budget is shown in the matched panels only if every
graph in the manifest has all 5 seeds at that checkpoint.

Writes to --out:
  soft_tidy_seed_level.csv    one row per (graph, seed, checkpoint) from soft_diagnostics.jsonl
  soft_per_graph_epoch.csv    seed means per (graph, epoch), with family / n / chi / k
  soft_summary.csv            matched budgets: means and medians over graphs
  soft_by_family.csv          per family at the largest matched budget
  figure1_panelC_soft.png     joint (H_tilde, g_tilde) histograms at selected budgets +
                              category fractions vs budget
Thresholds (same as run_soft.py): stationary g_tilde <= 0.01, near-hard H_tilde <= 0.1,
near-uniform H_tilde >= 0.9.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G_STAT, H_HARD, H_UNIF = 0.01, 0.1, 0.9
G_FLOOR = 1e-6
PANEL_BUDGETS = [1, 100, 1000, 10000, 100000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--manifest", default="configs/sweep_manifest.csv")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--n-seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = pd.read_csv(a.manifest).rename(columns={"name": "graph"})

    rows = []
    for f in sorted(glob.glob(os.path.join(a.main, "*_seed*", "soft_diagnostics.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/soft_tidy_seed_level.csv", index=False)

    # completeness and sanity report
    runs = df.groupby(["graph", "seed"]).size()
    expected = {(g, s) for g in man.graph for s in range(a.n_seeds)}
    missing = sorted(expected - set(runs.index))
    print(f"runs with soft diagnostics: {len(runs)} / {len(expected)}")
    if missing:
        print(f"MISSING {len(missing)} runs: {missing}")
    bad = df[df.entropy_check_diff.fillna(0) > 1e-3]
    print(f"checkpoints with entropy-check diff > 1e-3: {len(bad)} "
          f"(max diff overall {df.entropy_check_diff.max():.2e})")
    if len(bad):
        print(bad[["graph", "seed", "epoch", "entropy_check_diff"]].to_string(index=False))

    cols = ["g_mean", "g_median", "g_p90", "g_max", "frac_stationary", "H_mean", "H_median",
            "frac_near_hard", "frac_near_uniform", "frac_stationary_near_hard",
            "frac_stationary_near_uniform", "frac_not_stationary"]
    pg = (df.groupby(["graph", "epoch"])
            .agg(**{c: (c, "mean") for c in cols}, n_seeds=("seed", "nunique"))
            .reset_index()
            .merge(man[["graph", "family", "n", "chi", "k"]], on="graph"))
    pg["frac_stationary_intermediate"] = (pg.frac_stationary - pg.frac_stationary_near_hard
                                          - pg.frac_stationary_near_uniform)
    pg.to_csv(f"{a.out}/soft_per_graph_epoch.csv", index=False)

    full = pg[pg.n_seeds == a.n_seeds].groupby("epoch").graph.nunique()
    matched = sorted(full[full == len(man)].index)
    print(f"matched budgets (all {len(man)} graphs x {a.n_seeds} seeds): {matched}")
    m = pg[pg.epoch.isin(matched)]
    summ = m.groupby("epoch").agg(
        graphs=("graph", "nunique"),
        **{f"{c}_mean": (c, "mean") for c in cols + ["frac_stationary_intermediate"]},
        g_mean_median=("g_mean", "median"), H_mean_median=("H_mean", "median")).reset_index()
    summ.to_csv(f"{a.out}/soft_summary.csv", index=False)
    print(summ[["epoch", "graphs", "g_mean_mean", "g_mean_median", "H_mean_mean", "H_mean_median",
                "frac_stationary_near_hard_mean", "frac_stationary_near_uniform_mean",
                "frac_not_stationary_mean"]].to_string(index=False))

    last = max(matched)
    fam = (m[m.epoch == last].groupby("family")
           .agg(graphs=("graph", "nunique"), g_mean=("g_mean", "mean"), H_mean=("H_mean", "mean"),
                frac_stationary_near_hard=("frac_stationary_near_hard", "mean"),
                frac_stationary_near_uniform=("frac_stationary_near_uniform", "mean"),
                frac_not_stationary=("frac_not_stationary", "mean")).reset_index())
    fam.insert(1, "epoch", last)
    fam.to_csv(f"{a.out}/soft_by_family.csv", index=False)
    print(f"\nby family at epoch {last}:\n{fam.to_string(index=False)}")

    # joint histograms, each graph weight 1
    budgets = [b for b in PANEL_BUDGETS if b in matched]
    hbins = np.linspace(0, 1, 41)
    gbins = np.linspace(np.log10(G_FLOOR), 0, 41)  # g_tilde <= 1 (it is a gap per unit degree)
    hists = {b: np.zeros((len(hbins) - 1, len(gbins) - 1)) for b in budgets}
    for f in sorted(glob.glob(os.path.join(a.main, "*_seed*", "soft_pervertex.npz"))):
        z = np.load(f)
        mask = z["degree"] > 0
        w = 1.0 / (mask.sum() * a.n_seeds)
        epochs = list(z["epochs"])
        for b in budgets:
            if b not in epochs:
                continue
            i = epochs.index(b)
            H = z["H_tilde"][i][mask]
            lg = np.log10(np.clip(z["g_tilde"][i][mask], G_FLOOR, 1.0))
            h, _, _ = np.histogram2d(H, lg, bins=[hbins, gbins])
            hists[b] += h * w

    fig = plt.figure(figsize=(4 * len(budgets), 7.5))
    gs = fig.add_gridspec(2, len(budgets), height_ratios=[1.15, 1])
    vmax = max(h.max() for h in hists.values())
    for j, b in enumerate(budgets):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(hists[b].T, origin="lower", aspect="auto", cmap="viridis",
                       extent=[0, 1, np.log10(G_FLOOR), 0],
                       norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=vmax))
        ax.axhline(np.log10(G_STAT), color="w", lw=0.8, ls="--")
        ax.axvline(H_HARD, color="w", lw=0.8, ls=":")
        ax.axvline(H_UNIF, color="w", lw=0.8, ls=":")
        ax.set_title(f"epoch {b:,}")
        ax.set_xlabel("H̃ (normalised row entropy)")
        if j == 0:
            ax.set_ylabel(f"log10 g̃ (floored at {G_FLOOR:g})")
    fig.colorbar(im, ax=fig.axes, shrink=0.5, location="right", pad=0.01,
                 label="graphs' worth of vertices (each graph sums to 1)")
    ax = fig.add_subplot(gs[1, :])
    for c, lab in [("frac_stationary_near_hard", f"stationary (g̃≤{G_STAT}) & near-hard (H̃≤{H_HARD})"),
                   ("frac_stationary_intermediate", "stationary & intermediate H̃"),
                   ("frac_stationary_near_uniform", f"stationary & near-uniform (H̃≥{H_UNIF})"),
                   ("frac_not_stationary", f"not stationary (g̃>{G_STAT})")]:
        ax.plot(summ.epoch, summ[f"{c}_mean"], "o-", label=lab)
    ax.set_xscale("log"); ax.set_xlabel("epochs"); ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of vertices\n(mean over graphs)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5)); ax.grid(alpha=0.3)
    fig.suptitle(f"Experiment 1 panel C, k=chi-1, {len(man)} graphs x {a.n_seeds} seeds "
                 f"(non-isolated vertices; matched budgets)")
    fig.savefig(f"{a.out}/figure1_panelC_soft.png", dpi=110, bbox_inches="tight")
    print(f"wrote {a.out}/figure1_panelC_soft.png")


if __name__ == "__main__":
    main()
