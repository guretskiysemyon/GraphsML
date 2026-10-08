"""Build the synthetic tuning set (Step E): graphs disjoint from the DIMACS test set.

Writes DIMACS .col files plus manifest.csv to data/synthetic/tuning/.
For every graph we store k_hi, a number of colours known to be sufficient
(planted k, or the DSATUR colour count, whichever is smaller). Tuning evaluates two
regimes: k = k_hi (colourable, zero conflicts attainable) and k = k_hi - 1
(hard: chi may equal k_hi, so conflicts are usually unavoidable).
Note chi is not proven for these graphs, so k_hi - 1 is not guaranteed infeasible.
"""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "tuning"
SEED = 20261007


def planted(n, k, p, rng):
    classes = np.arange(n) % k
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            if classes[u] != classes[v] and rng.random() < p:
                g.add_edge(u, v)
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    specs = [
        ("planted_n60_k5", "planted", lambda: planted(60, 5, 0.5, rng), 5),
        ("planted_n100_k12", "planted", lambda: planted(100, 12, 0.6, rng), 12),
        ("planted_n120_k8", "planted", lambda: planted(120, 8, 0.4, rng), 8),
        ("planted_n200_k40", "planted", lambda: planted(200, 40, 0.5, rng), 40),
        ("planted_n250_k15", "planted", lambda: planted(250, 15, 0.3, rng), 15),
        ("planted_n450_k25", "planted", lambda: planted(450, 25, 0.2, rng), 25),
        ("er_n100_p0.3", "erdos_renyi", lambda: nx.gnp_random_graph(100, 0.3, seed=int(rng.integers(1 << 30))), None),
        ("er_n250_p0.1", "erdos_renyi", lambda: nx.gnp_random_graph(250, 0.1, seed=int(rng.integers(1 << 30))), None),
        ("er_n450_p0.5", "erdos_renyi", lambda: nx.gnp_random_graph(450, 0.5, seed=int(rng.integers(1 << 30))), None),
        ("regular_n200_d20", "random_regular", lambda: nx.random_regular_graph(20, 200, seed=int(rng.integers(1 << 30))), None),
        ("regular_n400_d10", "random_regular", lambda: nx.random_regular_graph(10, 400, seed=int(rng.integers(1 << 30))), None),
        ("ws_n200_k20", "watts_strogatz", lambda: nx.watts_strogatz_graph(200, 20, 0.1, seed=int(rng.integers(1 << 30))), None),
        ("ws_n450_k10", "watts_strogatz", lambda: nx.watts_strogatz_graph(450, 10, 0.1, seed=int(rng.integers(1 << 30))), None),
    ]
    rows = []
    for name, family, build, k_planted in specs:
        g = build()
        dsatur = max(nx.greedy_color(g, strategy="DSATUR").values()) + 1
        k_hi = dsatur if k_planted is None else min(k_planted, dsatur)
        with (OUT / f"{name}.col").open("w") as f:
            f.write(f"c synthetic tuning graph {name}\n")
            f.write(f"p edge {g.number_of_nodes()} {g.number_of_edges()}\n")
            for u, v in g.edges():
                f.write(f"e {u + 1} {v + 1}\n")
        rows.append(dict(name=name, family=family, n=g.number_of_nodes(), m=g.number_of_edges(),
                         k_planted=k_planted or "", k_dsatur=dsatur, k_hi=k_hi))
        print(rows[-1])
    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
