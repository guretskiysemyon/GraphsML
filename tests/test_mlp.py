import json

import networkx as nx
import pytest
import torch

from graphsml.model import PIGNN, NodeMLP, build_mean_adjacency, build_model
from graphsml.train import TrainConfig, train_pignn_checkpointed
from graphsml.tune import TuneConfig, pignn_stage2_configs, sample_configs, train_lean


def _adj(g):
    n = g.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in g.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def test_mlp_ignores_the_graph():
    torch.manual_seed(0)
    model = NodeMLP(10, 6, 5, 3, dropout=0.3).eval()
    a1 = build_mean_adjacency(_adj(nx.petersen_graph()))
    a2 = build_mean_adjacency(_adj(nx.cycle_graph(10)))
    z1 = model(a1)
    assert z1.shape == (10, 3)
    assert torch.equal(z1, model(a2))
    assert torch.equal(z1, model())


def test_sage_does_use_the_graph_and_build_model_dispatches():
    torch.manual_seed(0)
    sage = build_model("sage", 10, 6, 5, 3, 0.0)
    assert isinstance(sage, PIGNN) and isinstance(build_model("mlp", 10, 6, 5, 3, 0.0), NodeMLP)
    sage.eval()
    a1 = build_mean_adjacency(_adj(nx.petersen_graph()))
    a2 = build_mean_adjacency(_adj(nx.cycle_graph(10)))
    assert not torch.equal(sage(a1), sage(a2))
    with pytest.raises(ValueError):
        build_model("gat", 10, 6, 5, 3, 0.0)


def test_mlp_has_the_self_half_of_pignn_parameters():
    sage, mlp = build_model("sage", 10, 6, 5, 3, 0.0), build_model("mlp", 10, 6, 5, 3, 0.0)
    n_sage = sum(p.numel() for p in sage.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    emb = 10 * 6
    assert n_sage - emb == 2 * (6 * 5 + 5 * 3) + 5 + 3
    assert n_mlp - emb == (6 * 5 + 5 * 3) + 5 + 3


def test_checkpointed_mlp_training_records_arch_and_resumes(tmp_path):
    adj = _adj(nx.petersen_graph())

    def cfg(n):
        return TrainConfig(k=3, embed_dim=4, hidden_dim=4, dropout=0.2, lr=0.05, n_epochs=n, seed=1,
                           device="cpu", arch="mlp")

    train_pignn_checkpointed(adj, cfg(6), [3, 6], tmp_path / "ref")
    assert json.loads((tmp_path / "ref" / "meta.json").read_text())["arch"] == "mlp"
    train_pignn_checkpointed(adj, cfg(3), [3], tmp_path / "ext")
    train_pignn_checkpointed(adj, cfg(6), [3, 6], tmp_path / "ext")
    a = torch.load(tmp_path / "ref" / "checkpoints" / "ckpt_00000006.pt")["Z"]
    b = torch.load(tmp_path / "ext" / "checkpoints" / "ckpt_00000006.pt")["Z"]
    assert torch.equal(a, b)


def test_lean_training_with_mlp_runs():
    adj = _adj(nx.petersen_graph())
    res = train_lean(adj, 3, TuneConfig(lr=0.05, hidden_dim=4, embed_dim=4, dropout=0.1), 20, 0, "cpu",
                     eval_every=10, arch="mlp")
    assert res["eval_epochs"] == [10, 20]


def test_pignn_stage2_configs_stay_in_the_top_box():
    cfgs = [dict(c.__dict__, config=i) for i, c in enumerate(sample_configs(40, seed=0))]
    s2 = pignn_stage2_configs(cfgs)
    assert len(s2) == 10 and [c["stage1_id"] for c in s2[:6]] == list(range(6))
    for key in ("lr", "hidden_dim", "embed_dim", "dropout"):
        vals = [c[key] for c in s2[:6]]
        assert all(min(vals) - 1 <= c[key] <= max(vals) + 1 for c in s2[6:])
