import torch

from graphsml.model import PIGNN, build_mean_adjacency


def _adj_from_edges(n, edges):
    adj = torch.zeros(n, n)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def test_mean_adjacency_rows_sum_to_one_or_zero():
    # Node 3 is isolated.
    adj = _adj_from_edges(4, [(0, 1), (1, 2)])
    mean_adj = build_mean_adjacency(adj)
    row_sums = mean_adj.sum(dim=1)
    assert torch.allclose(row_sums, torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_mean_adjacency_averages_neighbours():
    adj = _adj_from_edges(3, [(0, 1), (0, 2)])
    mean_adj = build_mean_adjacency(adj)
    h = torch.tensor([[1.0, 0.0], [0.0, 2.0], [4.0, 0.0]])
    h_neigh = mean_adj @ h
    assert torch.allclose(h_neigh[0], torch.tensor([2.0, 1.0]))  # mean of nodes 1,2


def test_pignn_forward_shape():
    n, k = 10, 4
    edges = [(i, (i + 1) % n) for i in range(n)]
    adj = _adj_from_edges(n, edges)
    mean_adj = build_mean_adjacency(adj)
    model = PIGNN(n_nodes=n, embed_dim=8, hidden_dim=6, k=k, dropout=0.1)
    z = model(mean_adj)
    assert z.shape == (n, k)


def test_pignn_isolated_node_gets_finite_logits():
    n, k = 5, 3
    adj = _adj_from_edges(n, [(0, 1), (1, 2)])  # node 3, 4 isolated
    mean_adj = build_mean_adjacency(adj)
    model = PIGNN(n_nodes=n, embed_dim=4, hidden_dim=4, k=k, dropout=0.0)
    z = model(mean_adj)
    assert torch.isfinite(z).all()
