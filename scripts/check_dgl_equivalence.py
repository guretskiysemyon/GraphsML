"""Optional equivalence check (open item 5): our PIGNN vs the paper's DGL GNNSage.

Run with a Python that has DGL (not the main venv), e.g. the throwaway env in the scratchpad.
Usage: python scripts/check_dgl_equivalence.py --minimal-utils /path/to/gcp-with-gnns-example
Loads identical weights into both models, then compares eval-mode logits, Potts loss,
parameter gradients, and the parameters after a few AdamW steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.losses import potts_loss  # noqa: E402
from graphsml.model import PIGNN, build_mean_adjacency  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dimacs" / "col"

# Table III, PI-SAGE
CASES = {
    "anna": dict(k=11, embed_dim=43, hidden_dim=22, lr=0.03507, dropout=0.3298),
    "queen5_5": dict(k=5, embed_dim=77, hidden_dim=32, lr=0.02988, dropout=0.3784),
}


def max_abs(a, b):
    return (a - b).abs().max().item()


def run_case(name, cfg, dtype, minimal_utils, n_steps):
    graph = load_col_graph(DATA_DIR / f"{name}.col")
    n = graph.number_of_nodes()
    src, dst = zip(*graph.edges())
    g = dgl.graph((list(src) + list(dst), list(dst) + list(src)), num_nodes=n)
    adj = torch.zeros(n, n, dtype=dtype)
    for u, v in graph.edges():
        adj[u, v] = adj[v, u] = 1.0

    torch.manual_seed(0)
    ours = PIGNN(n, cfg["embed_dim"], cfg["hidden_dim"], cfg["k"], cfg["dropout"]).to(dtype)
    theirs = minimal_utils.GNNSage(g, cfg["embed_dim"], cfg["hidden_dim"], cfg["k"], cfg["dropout"], "mean").to(dtype)
    embed = torch.nn.Embedding(n, cfg["embed_dim"]).to(dtype)

    # copy our weights into theirs
    with torch.no_grad():
        embed.weight.copy_(ours.embedding.weight)
        for mine, ref in ((ours.conv1, theirs.layers[0]), (ours.conv2, theirs.layers[1])):
            ref.fc_self.weight.copy_(mine.fc_self.weight)
            ref.fc_neigh.weight.copy_(mine.fc_neigh.weight)
            ref.fc_self.bias.copy_(mine.bias)

    mean_adj = build_mean_adjacency(adj)
    ours.eval()
    theirs.eval()
    z_ours = ours(mean_adj)
    z_theirs = theirs(embed.weight)
    loss_ours = potts_loss(F.softmax(z_ours, 1), adj)
    loss_theirs = minimal_utils.loss_func_mod(F.softmax(z_theirs, 1), adj)
    print(f"[{name} {dtype}] logits max|diff|={max_abs(z_ours, z_theirs):.2e} "
          f"loss ours={loss_ours.item():.6f} theirs={loss_theirs.item():.6f}")

    loss_ours.backward()
    loss_theirs.backward()
    pairs = [
        ("embedding", ours.embedding.weight, embed.weight),
        ("conv1.fc_self", ours.conv1.fc_self.weight, theirs.layers[0].fc_self.weight),
        ("conv1.fc_neigh", ours.conv1.fc_neigh.weight, theirs.layers[0].fc_neigh.weight),
        ("conv1.bias", ours.conv1.bias, theirs.layers[0].fc_self.bias),
        ("conv2.fc_self", ours.conv2.fc_self.weight, theirs.layers[1].fc_self.weight),
        ("conv2.fc_neigh", ours.conv2.fc_neigh.weight, theirs.layers[1].fc_neigh.weight),
        ("conv2.bias", ours.conv2.bias, theirs.layers[1].fc_self.bias),
    ]
    worst_grad = max(max_abs(a.grad, b.grad) for _, a, b in pairs)
    print(f"    grads: max|diff| over all params={worst_grad:.2e}")

    # a few identical optimisation steps (dropout off so both are deterministic)
    opt_o = torch.optim.AdamW(ours.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    opt_t = torch.optim.AdamW(list(theirs.parameters()) + list(embed.parameters()), lr=cfg["lr"], weight_decay=1e-2)
    ours.zero_grad(); theirs.zero_grad(); embed.zero_grad()
    for _ in range(n_steps):
        opt_o.zero_grad(); opt_t.zero_grad()
        potts_loss(F.softmax(ours(mean_adj), 1), adj).backward()
        minimal_utils.loss_func_mod(F.softmax(theirs(embed.weight), 1), adj).backward()
        opt_o.step(); opt_t.step()
    drift = max(max_abs(a, b) for _, a, b in pairs)
    z_drift = max_abs(ours(mean_adj), theirs(embed.weight))
    print(f"    after {n_steps} AdamW steps: params max|diff|={drift:.2e}, logits max|diff|={z_drift:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal-utils", required=True, help="dir containing minimal_utils.py")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    sys.path.insert(0, args.minimal_utils)
    import minimal_utils

    for name, cfg in CASES.items():
        for dtype in (torch.float64, torch.float32):
            run_case(name, cfg, dtype, minimal_utils, args.steps)


if __name__ == "__main__":
    main()
