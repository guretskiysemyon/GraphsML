"""RQ1 follow-up (post hoc, descriptive): at PI-GNN's endpoint, how close is P to one-hot
per vertex, and are the vertices that still have an improving single flip the soft ones
or fully one-hot ones? CPU only, read-only on results/main; reads the eval-mode logits Z
of the checkpoints at 100k and (for runs extended in place) at their DONE t_max.

Per vertex (non-isolated only): p_max, 1 - p_max, logit margin z_(1) - z_(2), hard gain
M[v, own] - min_c M[v, c] of the argmax colouring, and the norm of the Potts-loss gradient
w.r.t. that vertex's logits, dL/dz_vc = p_vc (q_vc - sum_c' p_vc' q_vc'), q = A P.
The hard counts are checked against endpoint_trajectory.jsonl (conflicts, n_improvable).

Writes results/analysis/onehot_runs.csv (one row per run x epoch) and
onehot_improvable_vertices.csv (one row per improvable vertex), then prints a summary.

    python scripts/onehot_vs_sfo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import (EndpointGraph, conflicts_from_counts,  # noqa: E402
                               neighbour_colour_counts, single_flip_report)

MAIN = ROOT / "results" / "main"
OUT = ROOT / "results" / "analysis"
THR = (0.99, 0.999999)  # p_max thresholds: "near one-hot", "numerically one-hot"


def per_vertex(eg: EndpointGraph, Z: torch.Tensor, k: int):
    Z = Z.double()
    P = torch.softmax(Z, dim=1)
    Q = eg.adj @ P
    s = (P * Q).sum(dim=1, keepdim=True)
    grad = (P * (Q - s)).norm(dim=1).numpy()
    top2 = Z.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).numpy()
    pmax = P.max(dim=1).values.numpy()
    colours = Z.argmax(dim=1).numpy()
    M = neighbour_colour_counts(eg, colours, k)
    own = M[np.arange(eg.n), colours]
    gain = own - M.min(axis=1)
    return colours, M, own, gain, pmax, margin, grad


def q(x, p):
    return float(np.quantile(x, p)) if x.size else np.nan


def main():
    manifest = pd.read_csv(ROOT / "configs" / "sweep_manifest.csv", index_col="idx")
    runs, verts, mismatches = [], [], 0
    t0 = time.time()
    for _, row in manifest.iterrows():
        name, k, fam = row["name"], int(row["k"]), row["family"]
        eg = EndpointGraph.from_networkx(load_col_graph(ROOT / "data" / "dimacs" / "col" / f"{name}.col"))
        eg.adj = eg.adj.double()
        mask = eg.degree > 0
        for seed in range(5):
            rd = MAIN / f"{name}_seed{seed}"
            t_max = json.loads((rd / "DONE").read_text())["t_max"]
            logged = {}
            for line in (rd / "endpoint_trajectory.jsonl").read_text().splitlines():
                r = json.loads(line)
                logged[r["epoch"]] = r
            for epoch in sorted({100_000, t_max}):
                saved = torch.load(rd / "checkpoints" / f"ckpt_{epoch:08d}.pt", map_location="cpu")
                assert saved["epoch"] == epoch
                colours, M, own, gain, pmax, margin, grad = per_vertex(eg, saved["Z"], k)
                rep = single_flip_report(colours, M)
                conf = conflicts_from_counts(colours, M)
                lg = logged[epoch]
                if (conf, rep.n_improvable_vertices) != (lg["conflicts"], lg["n_improvable"]):
                    mismatches += 1
                    print(f"MISMATCH {name} s{seed} e{epoch}: recomputed ({conf}, {rep.n_improvable_vertices}) "
                          f"vs logged ({lg['conflicts']}, {lg['n_improvable']})", flush=True)
                imp = rep.improvable_mask & mask
                cnf = (own > 0) & mask & ~imp
                ok = (own == 0) & mask
                r = dict(graph=name, family=fam, seed=seed, epoch=epoch, t_max=t_max, n=int(mask.sum()), k=k,
                         conflicts=conf, sfo=rep.is_optimal, n_improvable=int(imp.sum()),
                         n_conflicted_not_improvable=int(cnf.sum()),
                         pmax_mean=float(pmax[mask].mean()), one_minus_pmax_median=q(1 - pmax[mask], 0.5),
                         one_minus_pmax_p90=q(1 - pmax[mask], 0.9), margin_median=q(margin[mask], 0.5),
                         margin_p10=q(margin[mask], 0.1), grad_median=q(grad[mask], 0.5))
                for name_, sel in (("imp", imp), ("cnf", cnf), ("ok", ok)):
                    r[f"{name_}_pmax_median"] = q(pmax[sel], 0.5)
                    r[f"{name_}_margin_median"] = q(margin[sel], 0.5)
                    r[f"{name_}_grad_median"] = q(grad[sel], 0.5)
                    for t in THR:
                        r[f"{name_}_frac_pmax_ge_{t}"] = float((pmax[sel] >= t).mean()) if sel.any() else np.nan
                for t in THR:
                    r[f"all_frac_pmax_ge_{t}"] = float((pmax[mask] >= t).mean())
                runs.append(r)
                for v in np.flatnonzero(imp):
                    verts.append(dict(graph=name, family=fam, seed=seed, epoch=epoch, v=int(v),
                                      degree=int(eg.degree[v]), own=int(own[v]), gain=int(gain[v]),
                                      pmax=float(pmax[v]), margin=float(margin[v]), grad=float(grad[v])))
        print(f"[{time.time() - t0:.0f}s] {name} done", flush=True)

    R, V = pd.DataFrame(runs), pd.DataFrame(verts)
    R.to_csv(OUT / "onehot_runs.csv", index=False)
    V.to_csv(OUT / "onehot_improvable_vertices.csv", index=False)
    print(f"\nwrote {len(R)} run rows, {len(V)} improvable-vertex rows; count mismatches vs logs: {mismatches}")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for label, sub in (("100k, 68 graphs", R[R.epoch == 100_000]),
                       ("300k, 13 extended graphs", R[R.epoch == 300_000]),
                       ("100k, same 13 graphs", R[(R.epoch == 100_000) & (R.t_max == 300_000)])):
        print(f"\n=== {label}: {len(sub)} runs, single-flip optimal {int(sub.sfo.sum())}")
        cols = ["all_frac_pmax_ge_0.99", "all_frac_pmax_ge_0.999999", "imp_frac_pmax_ge_0.99",
                "imp_frac_pmax_ge_0.999999", "imp_margin_median", "ok_margin_median",
                "imp_grad_median", "ok_grad_median"]
        g = sub.groupby("graph").agg(family=("family", "first"), sfo=("sfo", "mean"),
                                     n_imp=("n_improvable", "mean"), **{c: (c, "mean") for c in cols})
        print("per graph, then median over graphs by family:")
        print(g.groupby("family").agg(graphs=("sfo", "size"), sfo=("sfo", "mean"), n_imp=("n_imp", "median"),
                                      **{c: (c, "median") for c in cols}).round(4))
        print("all graphs (median over graphs):", g[cols].median().round(5).to_dict())
    if len(V):
        for ep in sorted(V.epoch.unique()):
            v = V[V.epoch == ep]
            print(f"\nimprovable vertices at epoch {ep}: {len(v)}; p_max quantiles "
                  f"{np.quantile(v.pmax, [0.1, 0.5, 0.9]).round(6).tolist()}; "
                  f">=0.99 {np.mean(v.pmax >= 0.99):.3f}; >=1-1e-6 {np.mean(v.pmax >= 0.999999):.3f}; "
                  f"margin median {v.margin.median():.1f}; gain distribution "
                  f"{v.gain.value_counts().sort_index().head(6).to_dict()}")


if __name__ == "__main__":
    main()
