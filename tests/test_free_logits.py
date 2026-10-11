import json

import networkx as nx
import numpy as np
import pytest
import torch

import graphsml.free_logits as fl_mod
from graphsml.diagnostics import dirichlet_energy, logit_diagnostics, param_distance
from graphsml.free_logits import (FreeLogitConfig, matched_random_logits, pignn_initial_logits,
                                  train_free_logits_checkpointed, train_free_logits_lean)
from graphsml.losses import conflict_count
from graphsml.train import TrainConfig, train_pignn_checkpointed
from graphsml.tune import free_logit_stage2_configs, sample_free_logit_configs


def _adj(g):
    n = g.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in g.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def _cfg(n_steps, seed=0, **kw):
    return FreeLogitConfig(**dict(dict(k=3, lr=0.05, n_steps=n_steps, seed=seed, device="cpu", patience=1000), **kw))


def test_pignn_initial_logits_match_the_saved_z0(tmp_path):
    """B must start from exactly the Z0 that the PI-GNN run saved."""
    adj = _adj(nx.petersen_graph())
    cfg = TrainConfig(k=3, embed_dim=7, hidden_dim=5, dropout=0.3, lr=0.01, n_epochs=2, seed=4, device="cpu")
    train_pignn_checkpointed(adj, cfg, [1, 2], tmp_path / "a")
    saved = torch.load(tmp_path / "a" / "z0.pt")
    rebuilt = pignn_initial_logits(adj, 3, 7, 5, 0.3, seed=4)
    assert torch.equal(saved, rebuilt)
    # dropout has no parameters, so the no-dropout noise control starts from the same Z0
    assert torch.equal(rebuilt, pignn_initial_logits(adj, 3, 7, 5, 0.0, seed=4))


def test_matched_random_logits_scale_and_seeding():
    z0 = torch.randn(50, 4) * 3.0 + 7.0
    r = matched_random_logits(z0, seed=1)
    assert r.shape == z0.shape and r.dtype == z0.dtype
    z0c = z0.double() - z0.double().mean(1, keepdim=True)
    assert torch.allclose(r.double().mean(1), torch.zeros(50, dtype=torch.float64), atol=1e-5)
    assert abs(r.double().std(unbiased=False).item() - z0c.std(unbiased=False).item()) < 1e-4
    assert torch.equal(r, matched_random_logits(z0, seed=1))
    assert not torch.equal(r, matched_random_logits(z0, seed=2))


def test_dirichlet_energy_matches_edge_sum_and_one_hot_identity():
    g = nx.gnp_random_graph(12, 0.4, seed=0)
    adj = _adj(g).double()
    p = torch.softmax(torch.randn(12, 3, dtype=torch.float64), dim=1)
    brute = sum(float(((p[u] - p[v]) ** 2).sum()) for u, v in g.edges())
    assert dirichlet_energy(p, adj) == pytest.approx(brute, rel=1e-12)
    c = torch.randint(0, 3, (12,))
    onehot = torch.nn.functional.one_hot(c, 3).double()
    assert dirichlet_energy(onehot, adj) == pytest.approx(2 * (g.number_of_edges() - conflict_count(c, adj)))


def test_logit_and_param_distance():
    adj = _adj(nx.cycle_graph(5)).double()
    z0 = torch.randn(5, 3, dtype=torch.float64)
    d = logit_diagnostics(z0 + 1.0, z0, adj)  # a per-row shift: softmax unchanged
    assert d["logit_dist"] == pytest.approx(np.sqrt(15))
    assert d["logit_dist_centred_rel"] == pytest.approx(0.0, abs=1e-12)
    assert d["eval_conflicts"] == logit_diagnostics(z0, z0, adj)["eval_conflicts"]
    s0 = {"a": torch.ones(2, 2), "b": torch.zeros(3)}
    s1 = {"a": torch.ones(2, 2) * 2, "b": torch.ones(3)}
    pd_ = param_distance(s1, s0)
    assert pd_["param_dist"] == pytest.approx(np.sqrt(4 + 3))
    assert pd_["param_dist_rel"] == pytest.approx(np.sqrt(7) / 2)
    assert pd_["param_dist_rel/b"] is None


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd"])
def test_free_logits_lower_the_potts_loss_and_colour_a_star(optimizer):
    # On a star every single-flip-optimal 2-colouring is proper, so a local minimum must reach 0
    # conflicts (an even cycle would not do: two "domain walls" are a single-flip-optimal trap).
    adj = _adj(nx.star_graph(9))
    z_init = 0.01 * torch.randn(10, 2, generator=torch.Generator().manual_seed(0))
    lr = 1.0 if optimizer == "sgd" else 0.1
    res = train_free_logits_lean(adj, z_init, _cfg(1000, k=2, lr=lr, optimizer=optimizer, weight_decay=1e-4),
                                 eval_every=100)
    assert res["eval_epochs"][-1] == 1000
    assert res["eval_conflicts"][-1] == 0
    p0 = torch.softmax(z_init, 1)
    assert res["final_loss"] < 0.5 * (adj * (p0 @ p0.T)).sum().item()


def test_checkpointed_run_writes_expected_artifacts(tmp_path):
    adj = _adj(nx.petersen_graph())
    z_init = torch.randn(10, 3)
    out = tmp_path / "run"
    res = train_free_logits_checkpointed(adj, z_init, _cfg(20), [1, 5, 10, 20], out, graph_meta={"graph": "p"})
    assert torch.equal(torch.load(out / "z0.pt"), z_init)
    assert json.loads((out / "meta.json").read_text())["optimizer"] == "adam"
    rows = [json.loads(line) for line in (out / "trajectory.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 5, 10, 20]
    assert res["final_step"] == 20 and res["stop_reason"] == "complete"
    last = torch.load(out / "checkpoints" / "ckpt_00000020.pt")
    assert rows[-1]["eval_conflicts"] == conflict_count(last["Z"].argmax(1), adj)


def test_resume_after_crash_matches_uninterrupted_run_with_noise(tmp_path, monkeypatch):
    adj = _adj(nx.petersen_graph())
    z_init = torch.randn(10, 3, generator=torch.Generator().manual_seed(1))
    cfg = _cfg(8, seed=3, logit_noise=0.5)
    ckpts = list(range(1, 9))
    train_free_logits_checkpointed(adj, z_init, cfg, ckpts, tmp_path / "ref")

    real = fl_mod.potts_loss
    calls = {"n": 0}

    def flaky(p, a):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("simulated crash")
        return real(p, a)

    monkeypatch.setattr(fl_mod, "potts_loss", flaky)
    with pytest.raises(RuntimeError):
        train_free_logits_checkpointed(adj, z_init, cfg, ckpts, tmp_path / "crashed")
    monkeypatch.setattr(fl_mod, "potts_loss", real)
    train_free_logits_checkpointed(adj, z_init, cfg, ckpts, tmp_path / "crashed")

    for e in ckpts:
        a = torch.load(tmp_path / "ref" / "checkpoints" / f"ckpt_{e:08d}.pt")["Z"]
        b = torch.load(tmp_path / "crashed" / "checkpoints" / f"ckpt_{e:08d}.pt")["Z"]
        assert torch.equal(a, b)
    rows = (tmp_path / "crashed" / "trajectory.jsonl").read_text().splitlines()
    assert [json.loads(r)["epoch"] for r in rows] == ckpts


def test_time_targets_extend_training_and_tag_checkpoints(tmp_path):
    adj = _adj(nx.petersen_graph())
    z_init = torch.randn(10, 3)
    # reached at once: tagged at step 1, run stops at the step budget
    res = train_free_logits_checkpointed(adj, z_init, _cfg(5), [1, 5, 10], tmp_path / "a",
                                         time_targets={"A@1": 0.0}, max_steps=10)
    rows = [json.loads(line) for line in (tmp_path / "a" / "trajectory.jsonl").read_text().splitlines()]
    assert rows[0]["epoch"] == 1 and rows[0]["time_matched"] == ["A@1"]
    assert res["final_step"] == 5 and res["missed_targets"] == [] and res["stop_reason"] == "complete"
    # never reached: runs on to max_steps and reports the miss
    res = train_free_logits_checkpointed(adj, z_init, _cfg(5), [1, 5, 10, 20], tmp_path / "b",
                                         time_targets={"A@far": 1e9}, max_steps=20)
    assert res["final_step"] == 20 and res["missed_targets"] == ["A@far"]
    assert res["stop_reason"] == "max_steps_targets_missed"


def test_same_trajectory_with_and_without_time_targets(tmp_path):
    adj = _adj(nx.petersen_graph())
    z_init = torch.randn(10, 3)
    train_free_logits_checkpointed(adj, z_init, _cfg(6), [3, 6], tmp_path / "plain")
    train_free_logits_checkpointed(adj, z_init, _cfg(6), [3, 6], tmp_path / "timed",
                                   time_targets={"x": 0.0}, max_steps=6)
    a = torch.load(tmp_path / "plain" / "checkpoints" / "ckpt_00000006.pt")["Z"]
    b = torch.load(tmp_path / "timed" / "checkpoints" / "ckpt_00000006.pt")["Z"]
    assert torch.equal(a, b)


def test_calling_again_is_a_noop_and_extension_resumes(tmp_path):
    adj = _adj(nx.petersen_graph())
    z_init = torch.randn(10, 3)
    out = tmp_path / "run"
    r1 = train_free_logits_checkpointed(adj, z_init, _cfg(5), [1, 5], out)
    r2 = train_free_logits_checkpointed(adj, z_init, _cfg(5), [1, 5], out)
    assert r2["final_step"] == r1["final_step"] == 5 and r2["stop_reason"] == "already_complete"
    assert len((out / "trajectory.jsonl").read_text().splitlines()) == 2
    train_free_logits_checkpointed(adj, z_init, _cfg(10), [1, 5, 10], out)
    train_free_logits_checkpointed(adj, z_init, _cfg(10), [1, 5, 10], tmp_path / "ref")
    a = torch.load(out / "checkpoints" / "ckpt_00000010.pt")["Z"]
    b = torch.load(tmp_path / "ref" / "checkpoints" / "ckpt_00000010.pt")["Z"]
    assert torch.equal(a, b)


def test_free_logit_config_sampling_and_stage2():
    cfgs = sample_free_logit_configs(40, seed=0)
    assert len(cfgs) == 43
    assert cfgs == sample_free_logit_configs(40, seed=0)
    assert all(c.weight_decay == 0.0 for c in cfgs if c.optimizer == "adam")
    assert all(1e-3 <= c.lr <= 1.0 for c in cfgs[3:])
    ranked = [dict(c.__dict__, config=i) for i, c in enumerate(cfgs)]
    s2 = free_logit_stage2_configs(ranked)
    assert len(s2) == 10 and [c["stage1_id"] for c in s2[:6]] == list(range(6))
    lrs = [c["lr"] for c in s2[:6]]
    assert all(min(lrs) - 1e-12 <= c["lr"] <= max(lrs) + 1e-12 for c in s2[6:])
