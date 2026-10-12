"""Plain-PyTorch PI-GNN: trainable embedding -> 2 GraphSAGE (mean) layers -> logits.

Implemented from the method description in Schuetz et al., arXiv:2202.01606, and
the public gcp-with-gnns-example repo's use of DGL's SAGEConv(aggregator_type="mean"),
without importing DGL. The mean-aggregator SAGEConv computes, for each node v:

    h_self  = feat[v]
    h_neigh = mean_{u in N(v)} feat[u]            (0 if v has no neighbours)
    out[v]  = W_self @ h_self + W_neigh @ h_neigh + b

followed by an optional activation. This module uses a dense adjacency matrix,
which is fine at the node counts used here (DIMACS graphs considered go up to a
few thousand nodes).
"""

from __future__ import annotations

import torch
from torch import nn


class SAGEConvMean(nn.Module):
    """One mean-aggregator GraphSAGE layer, matching dgl.nn.SAGEConv(aggregator_type='mean')."""

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, mean_adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """mean_adj: (n, n) row-normalised adjacency (mean_adj[v, u] = 1/deg(v) if u in N(v) else 0).
        h: (n, in_feats).
        """
        h_neigh = mean_adj @ h
        return self.fc_self(h) + self.fc_neigh(h_neigh) + self.bias


def build_mean_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Row-normalise a 0/1 adjacency matrix so that each row of degree>0 sums to 1.
    Isolated nodes (degree 0) get an all-zero row, i.e. mean over an empty neighbourhood is 0.
    """
    deg = adj.sum(dim=1, keepdim=True)
    safe_deg = deg.clamp(min=1.0)
    mean_adj = adj / safe_deg
    mean_adj = torch.where(deg > 0, mean_adj, torch.zeros_like(mean_adj))
    return mean_adj


class PIGNN(nn.Module):
    """Trainable node embedding -> SAGEConvMean -> ReLU -> dropout -> SAGEConvMean -> logits."""

    def __init__(
        self,
        n_nodes: int,
        embed_dim: int,
        hidden_dim: int,
        k: int,
        dropout: float,
        embed_init: str = "normal",
    ):
        super().__init__()
        # "normal" = nn.Embedding default N(0, 1), as in the paper's code; "xavier" kept for comparison.
        self.embedding = nn.Embedding(n_nodes, embed_dim)
        if embed_init == "xavier":
            nn.init.xavier_uniform_(self.embedding.weight)
        elif embed_init != "normal":
            raise ValueError(f"unknown embed_init {embed_init!r}")
        self.conv1 = SAGEConvMean(embed_dim, hidden_dim)
        self.conv2 = SAGEConvMean(hidden_dim, k)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, mean_adj: torch.Tensor) -> torch.Tensor:
        """Returns logits Z of shape (n_nodes, k)."""
        h0 = self.embedding.weight
        h1 = self.relu(self.conv1(mean_adj, h0))
        h1 = self.dropout(h1)
        z = self.conv2(mean_adj, h1)
        return z


class SelfLinear(nn.Module):
    """SAGEConvMean with the neighbour term removed: out[v] = W_self @ h[v] + b, same
    initialisation as SAGEConvMean's self weight."""

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        nn.init.xavier_uniform_(self.fc_self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc_self(h) + self.bias


class NodeMLP(nn.Module):
    """Experiment 3 rung C, the no-message-passing control: PIGNN with both SAGE layers'
    neighbour terms deleted. Trainable embedding -> linear -> ReLU -> dropout -> linear -> logits,
    same widths, dropout and initialisation as PIGNN; each vertex's logits depend on its own
    embedding only, so the graph enters only through the loss."""

    def __init__(self, n_nodes: int, embed_dim: int, hidden_dim: int, k: int, dropout: float,
                 embed_init: str = "normal"):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, embed_dim)
        if embed_init == "xavier":
            nn.init.xavier_uniform_(self.embedding.weight)
        elif embed_init != "normal":
            raise ValueError(f"unknown embed_init {embed_init!r}")
        self.lin1 = SelfLinear(embed_dim, hidden_dim)
        self.lin2 = SelfLinear(hidden_dim, k)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, mean_adj: torch.Tensor | None = None) -> torch.Tensor:
        """mean_adj is accepted (and ignored) so the training loops can call every arch alike."""
        h1 = self.dropout(self.relu(self.lin1(self.embedding.weight)))
        return self.lin2(h1)


ARCHS = {"sage": PIGNN, "mlp": NodeMLP}


def build_model(arch: str, n_nodes: int, embed_dim: int, hidden_dim: int, k: int, dropout: float,
                embed_init: str = "normal") -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}")
    return ARCHS[arch](n_nodes, embed_dim, hidden_dim, k, dropout, embed_init)
