import torch

from graphsml.losses import argmax_colouring, conflict_count, potts_loss


def _adj_from_edges(n, edges):
    adj = torch.zeros(n, n)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def test_potts_loss_equals_conflict_count_for_one_hot():
    # Triangle, 3 colours but forced collision: colours = [0, 0, 1]
    adj = _adj_from_edges(3, [(0, 1), (1, 2), (0, 2)])
    colours = torch.tensor([0, 0, 1])
    p = torch.nn.functional.one_hot(colours, num_classes=2).float()
    loss = potts_loss(p, adj)
    n_conf = conflict_count(colours, adj)
    assert n_conf == 1  # only edge (0,1) is monochromatic
    assert loss.item() == n_conf


def test_potts_loss_zero_for_proper_colouring():
    # Odd cycle C5, k=3 is 2-colourable-impossible but 3-colourable properly
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    adj = _adj_from_edges(5, edges)
    colours = torch.tensor([0, 1, 0, 1, 2])
    p = torch.nn.functional.one_hot(colours, num_classes=3).float()
    assert conflict_count(colours, adj) == 0
    assert potts_loss(p, adj).item() == 0.0


def test_argmax_colouring_matches_max_component():
    p = torch.tensor([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]])
    c = argmax_colouring(p)
    assert c.tolist() == [1, 0]


def test_conflict_count_counts_each_edge_once():
    adj = _adj_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    colours = torch.tensor([0, 0, 0, 0])
    assert conflict_count(colours, adj) == 4
