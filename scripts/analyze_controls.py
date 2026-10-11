"""Step G analysis: PI-GNN vs classical controls, compared after strict vertex descent.

Implements exactly the protocol and the interpretation rules written in
docs/progress_exp1.md section 9 *before* any control result existed. CPU only.

Post hoc additions (docs/progress_exp1.md section 11, rules written before the results): the
'dsatur_rand_v' control from controls_posthoc.jsonl is reported next to the pre-registered rows
with role "post hoc", and the order-sensitivity check (deterministic DSATUR vs dsatur_rand_v)
is written to controls_posthoc_order.csv. Both are skipped if the post hoc file is absent.

Inputs : results/controls/<graph>/controls.jsonl   (scripts/run_controls.py)
         results/controls/<graph>/controls_posthoc.jsonl (optional, post hoc)
         results/analysis/per_graph_epoch.csv       (scripts/analyze_exp1.py; PI-GNN seed means)
         configs/sweep_manifest.csv
Outputs: results/analysis/controls_per_graph.csv, controls_summary.csv, controls_by_family.csv,
         controls_vs_pignn.png
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

# --- fixed in advance (docs/progress_exp1.md section 9) ---
TIE_REL = 0.05  # |d| <= 5% is a tie ...
TIE_ABS = 1.0  # ... and so is an absolute difference below one conflict
ALPHA = 0.025  # two primary controls, Bonferroni
PIGNN_EPOCH = 100_000
PRIMARY = ["random", "dsatur"]
SECONDARY = ["dsatur_rand", "pignn_epoch1"]
POSTHOC = ["dsatur_rand_v"]  # added after the step G results (section 11)
LABEL = {"dsatur_rand_v": "DSATUR (random vertex ties)", "random": "random start", "dsatur": "DSATUR (deterministic)",
         "dsatur_rand": "DSATUR (randomised)", "pignn_epoch1": "PI-GNN at epoch 1"}

# palette.md: diverging blue <-> red with a gray midpoint, light surface
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
WIN, TIE, LOSS = "#2a78d6", "#898781", "#e34948"


def load_controls(path):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "*", "controls*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def role(c):
    return "primary" if c in PRIMARY else "post hoc" if c in POSTHOC else "exploratory"


def verdict(p, c):
    d = (p - c) / np.maximum(c, 1.0)
    tie = (np.abs(p - c) < TIE_ABS) | (np.abs(d) <= TIE_REL)
    return d, np.where(tie, "tie", np.where(p < c, "win", "loss"))


def reading(med, pval, wins, losses):
    if med < -TIE_REL and pval < ALPHA and wins > losses:
        return "PI-GNN is a better start for strict descent than this control"
    if med > TIE_REL and pval < ALPHA and losses > wins:
        return "PI-GNN is a worse start for strict descent than this control"
    if pval >= ALPHA or abs(med) <= TIE_REL:
        return "not distinguishable from this control as a start for strict descent"
    return "unclassified (significant but sign conditions disagree): report numbers only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", default="results/controls")
    ap.add_argument("--pignn", default="results/analysis/per_graph_epoch.csv")
    ap.add_argument("--manifest", default="configs/sweep_manifest.csv")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()

    ctl = load_controls(a.controls)
    man = pd.read_csv(a.manifest).rename(columns={"name": "graph"})
    pg = pd.read_csv(a.pignn)
    print(f"control rows: {len(ctl)}  graphs: {ctl.graph.nunique()}  "
          f"rows per (graph, method): {sorted(ctl.groupby(['graph', 'method']).size().unique())}")

    after = ctl.groupby(["graph", "method"]).conflicts_after_descent.mean().unstack()
    raw = ctl.groupby(["graph", "method"]).conflicts.mean().unstack().add_prefix("raw_")
    tab = man.set_index("graph")[["family", "n", "chi", "k"]].join(after, how="inner").join(raw)
    p100 = pg[pg.epoch == PIGNN_EPOCH].set_index("graph")
    p1 = pg[pg.epoch == 1].set_index("graph")
    tab["pignn_100k"] = p100.after
    tab["pignn_epoch1"] = p1.after
    tab["raw_pignn_100k"] = p100.conflicts
    tab = tab.dropna(subset=["pignn_100k"] + list(PRIMARY)).sort_values("n")
    print(f"graphs compared: {len(tab)}")
    posthoc = [c for c in POSTHOC if c in tab.columns and tab[c].notna().all()]
    if POSTHOC and not posthoc:
        print("post hoc controls absent or incomplete, skipped:", POSTHOC)

    summary = []
    for c in PRIMARY + SECONDARY + posthoc:
        d, v = verdict(tab.pignn_100k.to_numpy(float), tab[c].to_numpy(float))
        tab[f"d_{c}"], tab[f"verdict_{c}"] = d, v
        nz = d[d != 0]
        try:
            pval = float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue) if nz.size else 1.0
        except ValueError:
            pval = 1.0
        wins, ties, losses = (int((v == s).sum()) for s in ("win", "tie", "loss"))
        med = float(np.median(d))
        absdiff = np.abs(tab.pignn_100k.to_numpy(float) - tab[c].to_numpy(float))
        summary.append(dict(
            control=c, role=role(c), graphs=len(tab),
            wins=wins, ties=ties, losses=losses, median_d=med,
            # descriptive only: how many of the wins / losses are by at most 5 conflicts in absolute terms
            wins_abs_le5=int(((v == "win") & (absdiff <= 5)).sum()), losses_abs_le5=int(((v == "loss") & (absdiff <= 5)).sum()),
            mean_ctrl_after=float(tab[c].mean()),
            mean_pignn_after=float(tab.pignn_100k.mean()), wilcoxon_p=pval,
            reading=reading(med, pval, wins, losses) if c in PRIMARY else f"{role(c)}, no claim"))
    summ = pd.DataFrame(summary)

    fam = []
    for c in PRIMARY + SECONDARY + posthoc:
        for f, g in tab.groupby("family"):
            v = g[f"verdict_{c}"]
            fam.append(dict(control=c, family=f, graphs=len(g), wins=int((v == "win").sum()),
                            ties=int((v == "tie").sum()), losses=int((v == "loss").sum()),
                            median_d=float(g[f"d_{c}"].median())))
    fam = pd.DataFrame(fam)

    # post hoc order-sensitivity check (section 11 A): deterministic DSATUR vs dsatur_rand_v
    if "dsatur_rand_v" in posthoc:
        rv = ctl[ctl.method == "dsatur_rand_v"].groupby("graph").conflicts_after_descent
        order_tab = tab[["family", "n", "dsatur", "dsatur_rand_v", "pignn_100k", "verdict_dsatur"]].copy()
        order_tab["rand_v_min"], order_tab["rand_v_max"] = rv.min(), rv.max()
        order_tab["d_det_vs_rand_v"], order_tab["det_vs_rand_v"] = verdict(
            order_tab.dsatur.to_numpy(float), order_tab.dsatur_rand_v.to_numpy(float))
        order_tab = order_tab.rename(columns={"verdict_dsatur": "pignn_vs_dsatur_stepG"})
        order_tab.to_csv(f"{a.out}/controls_posthoc_order.csv")
        cnt = lambda v: {k: int((v == k).sum()) for k in ("win", "tie", "loss")}
        print("\n[post hoc] deterministic DSATUR vs dsatur_rand_v ('win' = deterministic has fewer conflicts):",
              cnt(order_tab.det_vs_rand_v))
        sub = order_tab[order_tab.pignn_vs_dsatur_stepG == "loss"]
        print(f"  on the {len(sub)} graphs where DSATUR beat PI-GNN in step G:", cnt(sub.det_vs_rand_v))
        print(order_tab.groupby("family").det_vs_rand_v.value_counts().unstack(fill_value=0).to_string())

    os.makedirs(a.out, exist_ok=True)
    tab.to_csv(f"{a.out}/controls_per_graph.csv")
    summ.to_csv(f"{a.out}/controls_summary.csv", index=False)
    fam.to_csv(f"{a.out}/controls_by_family.csv", index=False)

    # ---- figure ----
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "font.size": 10})
    fig = plt.figure(figsize=(15.5, 5.2 + 0.5 * len(posthoc)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.28)
    allv = np.concatenate([tab[["pignn_100k"] + PRIMARY].to_numpy(float).ravel()])
    lo, hi = max(allv.min() * 0.7, 0.5), allv.max() * 1.4
    for i, c in enumerate(PRIMARY):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=1, label="equal conflicts")
        for name, col, mk in (("win", WIN, "o"), ("tie", TIE, "s"), ("loss", LOSS, "^")):
            s = tab[tab[f"verdict_{c}"] == name]
            ax.scatter(s[c], s.pignn_100k, s=46, c=col, marker=mk, edgecolors=SURFACE, linewidths=1.2,
                       zorder=3, label=f"PI-GNN {'fewer' if name == 'win' else 'same' if name == 'tie' else 'more'}"
                                       f" conflicts (n={len(s)})")
        # direct labels for the 3 points that are far from the diagonal in both ratio and magnitude
        far = tab.assign(r=(tab.pignn_100k - tab[c]).abs() * np.abs(np.log(tab.pignn_100k / tab[c]))).nlargest(3, "r")
        for rank, (g, r) in enumerate(far.iterrows()):
            above = r.pignn_100k > r[c]
            ax.annotate(g, (r[c], r.pignn_100k), xytext=(-6, 7) if above else (6, -11 if rank % 2 == 0 else 5),
                        textcoords="offset points", ha="right" if above else "left", fontsize=7.5, color=INK2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"{LABEL[c]} -> descent: conflicts (mean of 5 seeds)")
        if i == 0:
            ax.set_ylabel("PI-GNN (100k epochs) -> descent: conflicts")
        ax.set_title(LABEL[c], loc="left", fontsize=11, color=INK)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax = fig.add_subplot(gs[0, 2])
    order = PRIMARY + SECONDARY + posthoc
    for j, c in enumerate(order[::-1]):
        r = summ[summ.control == c].iloc[0]
        left = 0
        for cnt, col, txt in ((r.wins, WIN, "white"), (r.ties, TIE, INK), (r.losses, LOSS, "white")):
            ax.barh(j, cnt, left=left, height=0.58, color=col, edgecolor=SURFACE, linewidth=2)
            if cnt >= 3:
                ax.text(left + cnt / 2, j, str(int(cnt)), ha="center", va="center", color=txt, fontsize=9, fontweight="bold")
            elif cnt > 0:  # too narrow to hold its count: put it beside (last segment) or above the segment
                last = col == LOSS
                ax.text(left + cnt + 0.6 if last else left + cnt / 2, j if last else j + 0.36, str(int(cnt)),
                        ha="left" if last else "center", va="center" if last else "bottom", color=INK2, fontsize=8)
            left += cnt
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{LABEL[c]}\n({role(c)})" for c in order[::-1]], fontsize=9, color=INK2)
    ax.set_xlim(0, len(tab) + 3); ax.set_xticks(range(0, len(tab) + 1, 10)); ax.set_xlabel("graphs")
    ax.set_title("PI-GNN vs control, per graph", loc="left", fontsize=11, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=WIN, label="PI-GNN fewer conflicts"), Patch(color=TIE, label="tie"),
                       Patch(color=LOSS, label="PI-GNN more conflicts")], loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Experiment 1, step G: conflicts after strict vertex descent, PI-GNN vs classical starts "
                 f"(k = chi-1, {len(tab)} graphs; tie = within 5% or under 1 conflict)", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.savefig(f"{a.out}/controls_vs_pignn.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(summ.round(4).to_string(index=False))
    print(fam[fam.control.isin(PRIMARY)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
