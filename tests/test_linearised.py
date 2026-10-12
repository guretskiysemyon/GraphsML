import json

import networkx as nx
import torch

from graphsml.free_logits import FreeLogitConfig, pignn_initial_logits
from graphsml.linearised import build_linearised_pignn, train_linearised_checkpointed, train_linearised_lean
from graphsml.losses import potts_loss


def _adj(g):
    n = g.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in g.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def _model(seed=0):
    return build_linearised_pignn(_adj(nx.petersen_graph()), 3, 4, 5, 0.3, seed=seed)


def _flat_jacobian(model):
    """Explicit J0 (n*k x total params) by brute force, for checking the matrix-free version."""
    params = {n: p.detach().clone() for n, p in model.net.named_parameters()}
    flat0 = torch.cat([p.reshape(-1) for p in params.values()])
    shapes = [(n, p.shape, p.numel()) for n, p in params.items()]

    def f(flat):
        d, i = {}, 0
        for n, s, c in shapes:
            d[n] = flat[i:i + c].reshape(s)
            i += c
        return model._f(d).reshape(-1)

    return torch.autograd.functional.jacobian(f, flat0)


def test_start_is_pignn_z0():
    model = _model(seed=3)
    assert torch.equal(model(), model.z0)
    assert torch.equal(model.z0, pignn_initial_logits(_adj(nx.petersen_graph()), 3, 4, 5, 0.3, seed=3))


def test_logits_are_linear_in_delta_and_match_explicit_jacobian():
    torch.manual_seed(0)
    model = _model()
    J = _flat_jacobian(model)
    with torch.no_grad():
        for d in model.delta:
            d.normal_()
    flat_delta = torch.cat([d.detach().reshape(-1) for d in model.delta])
    expected = model.z0.reshape(-1) + J @ flat_delta
    assert torch.allclose(model().reshape(-1), expected, atol=1e-5)


def test_one_sgd_step_is_the_tangent_kernel_preconditioned_update():
    """Plain GD on delta: Delta Z = -lr * J0 J0^T grad_Z L, the plan's rung D update."""
    model = _model()
    adj = _adj(nx.petersen_graph())
    J = _flat_jacobian(model)
    z = model.z0.clone().requires_grad_(True)
    g = torch.autograd.grad(potts_loss(torch.softmax(z, 1), adj), z)[0].reshape(-1)
    lr = 0.05
    opt = torch.optim.SGD(list(model.delta), lr=lr)
    opt.zero_grad()
    potts_loss(torch.softmax(model(), 1), adj).backward()
    opt.step()
    with torch.no_grad():
        dz = (model() - model.z0).reshape(-1)
    assert torch.allclose(dz, -lr * (J @ (J.T @ g)), atol=1e-5)


def test_training_lowers_the_loss():
    adj = _adj(nx.petersen_graph())
    model = _model()
    cfg = FreeLogitConfig(k=3, lr=0.05, n_steps=200, optimizer="adam", device="cpu")
    p0 = torch.softmax(model.z0, 1)
    res = train_linearised_lean(adj, model, cfg, eval_every=100)
    assert res["final_loss"] < potts_loss(p0, adj).item()


def test_checkpointed_resume_is_exact(tmp_path):
    adj = _adj(nx.petersen_graph())
    cfg = lambda n: FreeLogitConfig(k=3, lr=0.05, n_steps=n, optimizer="adam", device="cpu", patience=1000)
    train_linearised_checkpointed(adj, _model(), cfg(6), [3, 6], tmp_path / "ref", time_targets={"A@1": 0.0})
    train_linearised_checkpointed(adj, _model(), cfg(3), [3], tmp_path / "ext")
    train_linearised_checkpointed(adj, _model(), cfg(6), [3, 6], tmp_path / "ext")
    a = torch.load(tmp_path / "ref" / "checkpoints" / "ckpt_00000006.pt")
    b = torch.load(tmp_path / "ext" / "checkpoints" / "ckpt_00000006.pt")
    assert torch.equal(a["Z"], b["Z"])
    rows = [json.loads(l) for l in (tmp_path / "ref" / "trajectory.jsonl").read_text().splitlines()]
    assert rows[0]["time_matched"] == ["A@1"] and rows[-1]["epoch"] == 6
    assert torch.equal(torch.load(tmp_path / "ref" / "z0.pt"), _model().z0)
