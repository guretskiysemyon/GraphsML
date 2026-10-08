"""Step F: budget pilot. Run the frozen config on a small diagnostic subset of
DIMACS graphs (spanning n and density) at k = chi - 1, with a generous exploratory
checkpoint grid, and report per-epoch time, loss/conflict trajectories, and
endpoint-metric evolution (single-flip-optimal fraction, descent gains) -- to
decide T_max per graph and confirm the checkpoint grid is reasonable before the
main DIMACS sweep (Step H). Single seed per graph: exploratory, not final numbers.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsml.checkpoints import log_checkpoint_grid  # noqa: E402
from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import EndpointGraph, evaluate_endpoint  # noqa: E402
from graphsml.train import TrainConfig, train_pignn_checkpointed  # noqa: E402

PILOT_GRAPHS = ["queen5_5", "jean", "queen11_11", "le450_15a", "flat1000_50_0"]
COL_DIR = ROOT / "data" / "dimacs" / "col"
OUT_ROOT = ROOT / "results" / "pilot"


def graph_to_adj(graph):
    n = graph.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in graph.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-max", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--graphs", nargs="*", default=PILOT_GRAPHS)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available; refusing to fall back to CPU silently (use --device cpu).")

    frozen = json.loads((ROOT / "configs" / "pignn_frozen.json").read_text())
    chi = pd.read_csv(ROOT / "data" / "dimacs" / "chi_bounds.csv").set_index("name")["chi"]

    for name in args.graphs:
        g = load_col_graph(COL_DIR / f"{name}.col")
        n = g.number_of_nodes()
        adj = graph_to_adj(g)
        k = int(chi[name]) - 1
        checkpoints = log_checkpoint_grid(args.t_max, n)

        cfg = TrainConfig(k=k, embed_dim=frozen["embed_dim"], hidden_dim=frozen["hidden_dim"],
                           dropout=frozen["dropout"], lr=frozen["lr"], n_epochs=args.t_max,
                           seed=args.seed, device=args.device)
        out_dir = OUT_ROOT / f"{name}_seed{args.seed}_t{args.t_max}"

        print(f"=== {name}: n={n} m={g.number_of_edges()} chi={chi[name]} k={k} "
              f"checkpoints={len(checkpoints)} ===", flush=True)
        t0 = time.time()
        result = train_pignn_checkpointed(adj, cfg, checkpoints, out_dir,
                                           graph_meta={"graph": name, "n": n, "m": g.number_of_edges(), "chi": int(chi[name])})
        wall = time.time() - t0
        print(f"  wall={wall:.1f}s ({wall / args.t_max * 1000:.3f} ms/epoch) "
              f"final_conflicts={result['final_conflicts']} best_conflicts={result['best_conflicts']} "
              f"paper_stop_epoch={result['paper_stop_epoch']}", flush=True)

        eg = EndpointGraph.from_networkx(g)
        endpoint_rows = []
        for ckpt_path in sorted((out_dir / "checkpoints").glob("ckpt_*.pt")):
            saved = torch.load(ckpt_path, map_location="cpu")
            P = torch.softmax(saved["Z"], dim=1)
            res = evaluate_endpoint(eg, P, k, seed=args.seed, soft_diagnostics=False)
            endpoint_rows.append(dict(
                epoch=saved["epoch"], conflicts=res.conflicts,
                single_flip_optimal=res.single_flip.is_optimal,
                n_improvable=res.single_flip.n_improvable_vertices,
                best_gain=res.single_flip.best_gain,
                conflicts_after_descent=res.descent.conflicts_after,
                descent_moves=res.descent.n_moves,
                descent_seconds=res.descent.seconds,
            ))
        (out_dir / "endpoint_trajectory.jsonl").write_text(
            "\n".join(json.dumps(r) for r in endpoint_rows) + "\n")

        last5 = endpoint_rows[-5:]
        print("  last checkpoints (epoch, conflicts, single_flip_optimal, conflicts_after_descent):")
        for r in last5:
            print(f"    {r['epoch']:>8} {r['conflicts']:>6} {str(r['single_flip_optimal']):>5} "
                  f"{r['conflicts_after_descent']:>6}")


if __name__ == "__main__":
    main()
