"""Experiment 1 analysis: aggregate the sweep and draw Figure 1 panels.

Reads results/main/<graph>_seed<s>/{endpoint_trajectory,trajectory}.jsonl and
configs/sweep_manifest.csv. CPU only, no GPU needed. All values are raw at each
checkpoint (decision 11); graph is the statistical unit, seeds are averaged (decisions 5, 12).
Panels use a matched graph set per budget: only graphs that have a checkpoint at epoch T.
"""
import argparse, json, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(main):
    ep, tr = [], []
    for d in sorted(glob.glob(os.path.join(main, "*_seed*"))):
        for name, acc in (("endpoint_trajectory.jsonl", ep), ("trajectory.jsonl", tr)):
            f = os.path.join(d, name)
            if os.path.exists(f):
                acc += [json.loads(l) for l in open(f)]
    ep = pd.DataFrame(ep)
    tr = pd.DataFrame(tr)
    # trajectory.jsonl has no graph/seed columns; recover them from the directory name
    return ep, tr


def load_traj(main):
    rows = []
    for d in sorted(glob.glob(os.path.join(main, "*_seed*"))):
        g, s = os.path.basename(d).rsplit("_seed", 1)
        f = os.path.join(d, "trajectory.jsonl")
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l); r["graph"] = g; r["seed"] = int(s); rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--manifest", default="configs/sweep_manifest.csv")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ep, _ = load(a.main)
    tr = load_traj(a.main)
    man = pd.read_csv(a.manifest).rename(columns={"name": "graph"})
    df = ep.merge(tr[["graph", "seed", "epoch", "eval_entropy", "n_colours_used", "best_conflicts_so_far"]],
                  on=["graph", "seed", "epoch"], how="left")
    df["gain"] = df.conflicts - df.conflicts_after_descent
    df.to_csv(f"{a.out}/tidy_seed_level.csv", index=False)

    # per graph x epoch (mean over seeds)
    pg = df.groupby(["graph", "epoch"]).agg(
        sfo=("single_flip_optimal", "mean"), conflicts=("conflicts", "mean"),
        after=("conflicts_after_descent", "mean"), gain=("gain", "mean"),
        improvable=("n_improvable", "mean"), entropy=("eval_entropy", "mean"),
        colours=("n_colours_used", "mean"), best_so_far=("best_conflicts_so_far", "mean"),
        n_seeds=("seed", "nunique")).reset_index().merge(man[["graph", "family", "n", "chi", "k"]], on="graph")
    pg.to_csv(f"{a.out}/per_graph_epoch.csv", index=False)

    # matched set per budget: graphs and seeds present at that epoch
    cnt = pg.groupby("epoch").graph.nunique()
    full = cnt[cnt == pg.graph.nunique()].index  # epochs shared by all graphs
    agg = pg[pg.epoch.isin(full)].groupby("epoch").agg(
        sfo=("sfo", "mean"), conflicts=("conflicts", "mean"), after=("after", "mean"),
        entropy=("entropy", "mean"), graphs=("graph", "nunique")).reset_index()
    # median of per-graph relative gain is less dominated by the huge DSJR graphs
    pg["rel_gain"] = pg.gain / pg.conflicts.clip(lower=1)
    agg["median_rel_gain"] = pg[pg.epoch.isin(full)].groupby("epoch").rel_gain.median().values
    agg.to_csv(f"{a.out}/figure1_summary.csv", index=False)

    # extension candidates: >10% conflict drop 50k->100k and not single-flip-optimal at 100k
    t = pg.pivot(index="graph", columns="epoch", values="conflicts")
    s = pg.pivot(index="graph", columns="epoch", values="sfo")
    last = t.columns.max()
    c = pd.DataFrame({"conf_50k": t[50000], "conf_last": t[last], "sfo_last": s[last]})
    c["rel_drop"] = (c.conf_50k - c.conf_last) / c.conf_50k.clip(lower=1)
    c["abs_drop"] = c.conf_50k - c.conf_last
    c["extend"] = (c.rel_drop > 0.10) & (c.abs_drop >= 3) & (c.sfo_last < 0.5)
    c.sort_values("rel_drop", ascending=False).to_csv(f"{a.out}/extension_candidates.csv")

    # paper early-stop vs T_max
    ps = []
    for d in glob.glob(os.path.join(a.main, "*_seed*")):
        f = os.path.join(d, "run_info.jsonl")
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if "paper_stop_epoch" in r:
                    ps.append({"dir": os.path.basename(d), "paper_stop_epoch": r["paper_stop_epoch"]})
    if ps:
        pd.DataFrame(ps).to_csv(f"{a.out}/paper_stop_epochs.csv", index=False)

    # ---- Figure 1 ----
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    ax[0].plot(agg.epoch, agg.sfo, "o-"); ax[0].set_title("A: single-flip-optimal fraction")
    ax[1].plot(agg.epoch, agg.conflicts, "o-", label="argmax"); ax[1].plot(agg.epoch, agg.after, "s-", label="after descent")
    ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("B: mean conflicts (raw)")
    ax[2].plot(agg.epoch, agg.median_rel_gain, "o-"); ax[2].set_title("B': median relative descent gain")
    ax[3].plot(agg.epoch, agg.entropy, "o-"); ax[3].set_title("C: mean row entropy (eval)")
    for x in ax: x.set_xscale("log"); x.set_xlabel("epochs"); x.grid(alpha=.3)
    fig.suptitle(f"Experiment 1, k=chi-1, {pg.graph.nunique()} graphs x 5 seeds (matched epochs)")
    fig.tight_layout(); fig.savefig(f"{a.out}/figure1.png", dpi=130)

    # per-graph curves, grouped by family
    fams = sorted(pg.family.unique())
    fig, axs = plt.subplots(2, (len(fams) + 1) // 2, figsize=(5 * ((len(fams) + 1) // 2), 8), squeeze=False)
    for x, fam in zip(axs.ravel(), fams):
        for g, d in pg[pg.family == fam].groupby("graph"):
            x.plot(d.epoch, d.sfo, alpha=.6, label=g)
        x.set_xscale("log"); x.set_title(fam); x.set_ylim(-.05, 1.05)
        if pg[pg.family == fam].graph.nunique() <= 8: x.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{a.out}/sfo_by_family.png", dpi=110)
    print(agg.to_string()); print("extend:", list(c.index[c.extend]))


if __name__ == "__main__":
    main()
