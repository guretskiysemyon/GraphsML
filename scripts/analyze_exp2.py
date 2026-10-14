"""Experiment 2 (RQ3) analysis: decoders on frozen P, null-P controls, family breakdown, and
the Experiment 3 arms through the same decoders. CPU only. See docs/Experiment2_Steps.md.

Paired comparisons use Experiment 1's rules unchanged (scripts/analyze_controls.py): per graph,
average the seeds; d = (x - y) / max(y, 1); a tie is |d| <= 5 % or |x - y| < 1 conflict; Wilcoxon
on the nonzero d; descriptive p-values. The primary metric is conflicts after strict descent.

Inputs : results/exp2/<arm>/<graph>_seed<s>/decoders.jsonl and DONE   (scripts/run_decoders.py)
         configs/sweep_manifest.csv
Outputs: results/analysis/exp2_summary.csv   comparisons on PI-GNN's own P (arm A)
         results/analysis/exp2_by_family.csv the three primary comparisons per graph family
         results/analysis/exp2_arms.csv      every arm vs A, after argmax+descent and after D-CE+descent
         results/analysis/exp2_arm_spread.csv per graph, the spread across arms under each decoder
         results/analysis/exp2_cost.csv      measured CPU time per arm
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TIE_REL = 0.05
TIE_ABS = 1.0
ARMS = ["A", "A_nodrop", "B", "Brand", "C", "D", "B_noise0.5", "B_noise1", "B_noise2"]

# (x, y) as (variant, decoder); x is the reference "better if lower" side
PRIMARY = [
    (("trained", "dce"), ("trained", "argmax"), "D-CE vs argmax, both on trained P"),
    (("trained", "dce"), ("uniform", "dce"), "trained P vs no information (D-CE; uniform = DSATUR)"),
    (("trained", "dce"), ("colperm", "dce"), "trained P vs colours scrambled within rows (D-CE)"),
]
SECONDARY = [
    (("trained", "dce"), ("rowperm", "dce"), "trained P vs rows shuffled across vertices (D-CE)"),
    (("trained", "dce"), ("untrained", "dce"), "trained P vs untrained GNN (D-CE)"),
    (("trained", "dce"), ("trained", "dce_rand"), "saturation vs random order (D-CE, trained P)"),
    (("trained", "sample"), ("trained", "argmax"), "best of 32 samples vs argmax (trained P)"),
    (("trained", "argmax"), ("uniform", "argmax"), "trained P vs no information (argmax)"),
    (("trained", "argmax"), ("colperm", "argmax"), "trained P vs colours scrambled (argmax)"),
    (("uniform", "dce"), ("colperm", "dce"), "no information vs scrambled colours (D-CE)"),
]


def compare(x: pd.Series, y: pd.Series, conflicts: bool = True) -> dict:
    """conflicts=False for values that are not conflict counts (e.g. relative spreads): then d is
    the plain relative difference and only the 5 % band counts as a tie."""
    x, y = x.align(y, join="inner")
    xv, yv = x.to_numpy(float), y.to_numpy(float)
    if conflicts:
        d = (xv - yv) / np.maximum(yv, 1.0)
        tie = (np.abs(xv - yv) < TIE_ABS) | (np.abs(d) <= TIE_REL)
    else:
        d = (xv - yv) / np.where(yv > 0, yv, 1.0)
        tie = np.abs(d) <= TIE_REL
    nz = d[d != 0]
    try:
        p = float(wilcoxon(nz).pvalue) if nz.size else 1.0
    except ValueError:
        p = 1.0
    return dict(graphs=len(d), better=int((~tie & (xv < yv)).sum()), ties=int(tie.sum()),
                worse=int((~tie & (xv > yv)).sum()), median_d=float(np.median(d)) if len(d) else np.nan,
                mean_x=float(xv.mean()), mean_y=float(yv.mean()), wilcoxon_p=p)


def reading(r: dict) -> str:
    if r["wilcoxon_p"] < 0.05 and r["median_d"] < -TIE_REL and r["better"] > r["worse"]:
        return "first is better"
    if r["wilcoxon_p"] < 0.05 and r["median_d"] > TIE_REL and r["worse"] > r["better"]:
        return "second is better"
    if r["wilcoxon_p"] >= 0.05 or abs(r["median_d"]) <= TIE_REL:
        return "not distinguishable"
    return "mixed (significant, but the median is within the tie band or signs disagree)"


def load(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, done = [], []
    for arm_dir in sorted(glob.glob(os.path.join(root, "*"))):
        arm = os.path.basename(arm_dir)
        for f in glob.glob(os.path.join(arm_dir, "*", "decoders.jsonl")):
            rows += [json.loads(l) for l in open(f)]
            dfile = os.path.join(os.path.dirname(f), "DONE")
            if os.path.exists(dfile):
                done.append(dict(arm=arm, **json.load(open(dfile))))
    return pd.DataFrame(rows), pd.DataFrame(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/exp2")
    ap.add_argument("--manifest", default="configs/sweep_manifest.csv")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)

    df, done = load(args.root)
    man = pd.read_csv(args.manifest)
    family = man.set_index("name").family
    n_runs = df.groupby("arm")[["graph", "seed"]].apply(lambda t: len(t.drop_duplicates()))
    print("runs per arm:", n_runs.to_dict())

    collapsed = df[(df.arm == "A") & (df.variant == "trained") & (df.decoder == "argmax") & (df.conflicts == df.m)]
    print(f"arm A runs whose trained argmax puts every edge in conflict (one colour): {len(collapsed)} "
          f"on {collapsed.graph.nunique()} graphs {sorted(collapsed.graph.unique())}")

    # seed means per graph
    g = df.groupby(["arm", "graph", "variant", "decoder"])[["conflicts", "conflicts_after_descent"]].mean()

    def series(arm, variant, decoder, col="conflicts_after_descent"):
        try:
            return g.loc[(arm, slice(None), variant, decoder), col].droplevel([0, 2, 3])
        except KeyError:
            return pd.Series(dtype=float)

    # 1. comparisons on PI-GNN's own P
    out = []
    for role, comps in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for (xv, xd), (yv, yd), label in comps:
            for col, metric in (("conflicts_after_descent", "after descent"), ("conflicts", "raw")):
                r = compare(series("A", xv, xd, col), series("A", yv, yd, col))
                out.append(dict(role=role, comparison=label, metric=metric, **r, reading=reading(r)))
    summary = pd.DataFrame(out)
    summary.to_csv(os.path.join(args.out, "exp2_summary.csv"), index=False)
    print("\n== PI-GNN's own P (arm A), per-graph seed means ==")
    print(summary[summary.metric == "after descent"].drop(columns=["metric", "role"]).round(4).to_string(index=False))

    # 2. family breakdown of the primary comparisons
    fam_rows = []
    for (xv, xd), (yv, yd), label in PRIMARY:
        x, y = series("A", xv, xd), series("A", yv, yd)
        for fam, graphs in family.groupby(family).groups.items():
            r = compare(x[x.index.isin(graphs)], y[y.index.isin(graphs)])
            fam_rows.append(dict(comparison=label, family=fam, **r))
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(os.path.join(args.out, "exp2_by_family.csv"), index=False)
    print("\n== primary comparisons by family (after descent; better/ties/worse, median d) ==")
    piv = fam.assign(cell=fam.apply(lambda r: f"{r.better}/{r.ties}/{r.worse} {r.median_d:+.2f}", axis=1))
    print(piv.pivot(index="family", columns="comparison", values="cell").to_string())

    # 3. arms through the same decoders
    arms = [a for a in ARMS if a in n_runs.index]
    arm_rows = []
    for arm in arms:
        if arm == "A":
            continue
        for dec in ("argmax", "dce"):
            r = compare(series(arm, "trained", dec), series("A", "trained", dec))
            arm_rows.append(dict(arm=arm, comparison=f"arm vs A, both {dec}+descent", **r, reading=reading(r)))
        r = compare(series(arm, "trained", "dce"), series("A", "uniform", "dce"))
        arm_rows.append(dict(arm=arm, comparison="arm D-CE vs no information (DSATUR)", **r, reading=reading(r)))
        r = compare(series(arm, "trained", "dce"), series(arm, "colperm", "dce"))
        arm_rows.append(dict(arm=arm, comparison="arm trained vs its colperm (D-CE)", **r, reading=reading(r)))
        r = compare(series(arm, "trained", "dce"), series(arm, "trained", "argmax"))
        arm_rows.append(dict(arm=arm, comparison="arm D-CE vs argmax", **r, reading=reading(r)))
    arms_df = pd.DataFrame(arm_rows)
    arms_df.to_csv(os.path.join(args.out, "exp2_arms.csv"), index=False)
    if len(arms_df):
        print("\n== Experiment 3 arms vs PI-GNN (A), after descent ==")
        print(arms_df.round(4).to_string(index=False))

        # spread across arms per graph: does D-CE shrink the differences between optimisers?
        sp = []
        for dec in ("argmax", "dce"):
            tab = pd.concat({a: series(a, "trained", dec) for a in arms}, axis=1).dropna()
            rel = (tab.max(axis=1) - tab.min(axis=1)) / tab.mean(axis=1).clip(lower=1)
            sp.append(pd.DataFrame(dict(decoder=dec, graph=tab.index, rel_range=rel.values,
                                        cv=(tab.std(axis=1) / tab.mean(axis=1).clip(lower=1)).values)))
        spread = pd.concat(sp)
        spread.to_csv(os.path.join(args.out, "exp2_arm_spread.csv"), index=False)
        w = spread.pivot(index="graph", columns="decoder", values="rel_range")
        r = compare(w.dce, w.argmax, conflicts=False)
        print(f"\n== spread across the {len(arms)} arms per graph, (max - min) / mean after descent ==")
        print(spread.groupby("decoder")[["rel_range", "cv"]].median().round(4).to_string())
        print(f"D-CE spread smaller / tie / larger than argmax spread on {r['better']}/{r['ties']}/{r['worse']} "
              f"graphs (tie = within 5 % relative), median relative change {r['median_d']:+.3f}, "
              f"Wilcoxon p = {r['wilcoxon_p']:.2g}")
        means = pd.DataFrame({dec: [series(a, "trained", dec).mean() for a in arms] for dec in ("argmax", "dce")},
                             index=arms)
        means["rank_argmax"], means["rank_dce"] = means.argmax.rank(), means.dce.rank()
        print("\nmean conflicts after descent per arm (graph means), and rank under each decoder:")
        print(means.round(2).to_string())

    # 4. cost
    if len(done):
        cost = done.groupby("arm").total_seconds.agg(runs="count", cpu_hours=lambda s: s.sum() / 3600,
                                                      median_s="median", max_s="max")
        cost.to_csv(os.path.join(args.out, "exp2_cost.csv"))
        print("\n== measured cost ==")
        print(cost.round(3).to_string())


if __name__ == "__main__":
    main()
