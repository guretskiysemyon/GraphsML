"""Step A validation: reproduce the paper's zero-conflict results at k = chi.

Uses the Table III (PI-SAGE) hyperparameters from progress_exp1.md on
anna, myciel5, queen5_5. Early stopping is off (decision 2); we still log the
epoch at which the paper's own early-stopping rule would have fired.
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.train import TrainConfig, train_pignn  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dimacs" / "col"

# Table III (supplement), PI-SAGE, k = chi
TABLE_III = {
    "anna": dict(k=11, embed_dim=43, hidden_dim=22, lr=0.03507, dropout=0.3298),
    "jean": dict(k=10, embed_dim=50, hidden_dim=62, lr=0.01663, dropout=0.3185),
    "myciel5": dict(k=6, embed_dim=16, hidden_dim=18, lr=0.01333, dropout=0.3964),
    "myciel6": dict(k=7, embed_dim=8, hidden_dim=22, lr=0.01779, dropout=0.2225),
    "queen5_5": dict(k=5, embed_dim=77, hidden_dim=32, lr=0.02988, dropout=0.3784),
    "queen6_6": dict(k=7, embed_dim=20, hidden_dim=12, lr=0.05105, dropout=0.3425),
    "queen7_7": dict(k=7, embed_dim=67, hidden_dim=12, lr=0.02175, dropout=0.2339),
    "queen8_8": dict(k=9, embed_dim=32, hidden_dim=10, lr=0.02728, dropout=0.2878),
    "queen8_12": dict(k=12, embed_dim=107, hidden_dim=23, lr=0.01730, dropout=0.1796),
    "queen9_9": dict(k=10, embed_dim=109, hidden_dim=16, lr=0.02636, dropout=0.3257),
    "queen11_11": dict(k=11, embed_dim=75, hidden_dim=25, lr=0.04600, dropout=0.2974),
    "queen13_13": dict(k=13, embed_dim=112, hidden_dim=199, lr=0.14426, dropout=0.1571),
}

# PI-SAGE best-coloring conflicts reported in Table I of the paper (one entry per graph)
PAPER_PI_SAGE = {
    "anna": 0, "jean": 0, "myciel5": 0, "myciel6": 0, "queen5_5": 0, "queen6_6": 0,
    "queen7_7": 0, "queen8_8": 1, "queen9_9": 1, "queen8_12": 0, "queen11_11": 17, "queen13_13": 26,
}


def graph_to_adj(graph, device):
    n = graph.number_of_nodes()
    adj = torch.zeros(n, n, device=device)
    for u, v in graph.edges():
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-init", choices=["normal", "xavier"], default="normal")
    parser.add_argument("--save-dir", type=Path, default=None, help="save per-seed trajectories as .npz")
    parser.add_argument("--graphs", nargs="+", default=list(TABLE_III.keys()))
    args = parser.parse_args()

    for name in args.graphs:
        cfg_dict = TABLE_III[name]
        graph = load_col_graph(DATA_DIR / f"{name}.col")
        adj = graph_to_adj(graph, args.device)
        print(f"=== {name}: n={graph.number_of_nodes()} m={graph.number_of_edges()} k={cfg_dict['k']} ===")

        print(f"  paper PI-SAGE best conflicts (Table I): {PAPER_PI_SAGE[name]}")
        for seed in range(args.seeds):
            cfg = TrainConfig(
                k=cfg_dict["k"],
                embed_dim=cfg_dict["embed_dim"],
                hidden_dim=cfg_dict["hidden_dim"],
                dropout=cfg_dict["dropout"],
                lr=cfg_dict["lr"],
                n_epochs=args.n_epochs,
                seed=seed,
                embed_init=args.embed_init,
                device=args.device,
            )
            t0 = time.time()
            result = train_pignn(adj, cfg)
            elapsed = time.time() - t0
            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.save_dir / f"{name}_seed{seed}.npz",
                    loss=np.array(result["loss_history"], dtype=np.float32),
                    eval_conflicts=np.array(result["eval_conflicts_history"], dtype=np.int32),
                    train_conflicts=np.array(result["train_conflicts_history"], dtype=np.int32),
                    eval_entropy=np.array(result["eval_entropy_history"], dtype=np.float32),
                )
            print(
                f"  seed={seed} eval(best={result['best_conflicts']} final={result['final_conflicts']}) "
                f"train_dropout_on(best={result['best_train_conflicts']} final={result['final_train_conflicts']}) "
                f"paper_stop_epoch={result['paper_stop_epoch']} "
                f"time={elapsed:.1f}s"
            )


if __name__ == "__main__":
    main()
