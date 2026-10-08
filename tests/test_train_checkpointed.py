import json

import networkx as nx
import pytest
import torch

import graphsml.train as train_mod
from graphsml.train import TrainConfig, train_pignn_checkpointed


def _graph_adj():
    g = nx.cycle_graph(6)
    n = g.number_of_nodes()
    adj = torch.zeros(n, n)
    for u, v in g.edges():
        adj[u, v] = adj[v, u] = 1.0
    return adj


def _cfg(n_epochs, seed=0):
    return TrainConfig(k=3, embed_dim=4, hidden_dim=4, dropout=0.1, lr=0.05,
                        n_epochs=n_epochs, seed=seed, device="cpu", patience=1000)


def test_fresh_run_writes_expected_artifacts(tmp_path):
    adj = _graph_adj()
    checkpoints = [1, 5, 10, 20]
    out_dir = tmp_path / "run"
    result = train_pignn_checkpointed(adj, _cfg(20), checkpoints, out_dir, graph_meta={"graph": "c6"})

    assert (out_dir / "z0.pt").exists()
    assert (out_dir / "init_weights.pt").exists()
    assert (out_dir / "meta.json").exists()
    traj_lines = (out_dir / "trajectory.jsonl").read_text().splitlines()
    assert len(traj_lines) == len(checkpoints)
    epochs_logged = [json.loads(line)["epoch"] for line in traj_lines]
    assert epochs_logged == checkpoints
    for e in checkpoints:
        assert (out_dir / "checkpoints" / f"ckpt_{e:08d}.pt").exists()
    assert result["final_conflicts"] == json.loads(traj_lines[-1])["eval_conflicts"]


def test_resume_after_simulated_crash_matches_uninterrupted_run(tmp_path, monkeypatch):
    adj = _graph_adj()
    checkpoints = [1, 2, 3, 4, 5, 6, 7, 8]
    cfg = _cfg(8, seed=1)

    ref_dir = tmp_path / "ref"
    ref_result = train_pignn_checkpointed(adj, cfg, checkpoints, ref_dir)

    crashed_dir = tmp_path / "crashed"
    call_count = {"n": 0}
    real_potts_loss = train_mod.potts_loss

    def flaky_loss(p, a):
        call_count["n"] += 1
        if call_count["n"] == 4:
            raise RuntimeError("simulated crash")
        return real_potts_loss(p, a)

    monkeypatch.setattr(train_mod, "potts_loss", flaky_loss)
    with pytest.raises(RuntimeError):
        train_pignn_checkpointed(adj, cfg, checkpoints, crashed_dir)

    monkeypatch.setattr(train_mod, "potts_loss", real_potts_loss)
    resumed_result = train_pignn_checkpointed(adj, cfg, checkpoints, crashed_dir, resume=True)

    assert resumed_result["final_conflicts"] == ref_result["final_conflicts"]
    assert resumed_result["best_conflicts"] == ref_result["best_conflicts"]

    ref_last = torch.load(ref_dir / "checkpoints" / f"ckpt_{checkpoints[-1]:08d}.pt")
    resumed_last = torch.load(crashed_dir / "checkpoints" / f"ckpt_{checkpoints[-1]:08d}.pt")
    assert torch.allclose(ref_last["Z"], resumed_last["Z"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_resume_on_cuda_does_not_move_rng_state_off_cpu(tmp_path):
    """Regression test: torch.load(resume_path, map_location=device) used to move the
    saved RNG-state ByteTensors onto the GPU, and torch.set_rng_state/torch.cuda.set_rng_state
    both require them on CPU -- this crashed every GPU resume. A device="cpu" test alone
    can't catch this, since map_location="cpu" matches trivially there."""
    adj = _graph_adj()
    out_dir = tmp_path / "run"

    def gpu_cfg(n_epochs):
        return TrainConfig(k=3, embed_dim=4, hidden_dim=4, dropout=0.1, lr=0.05,
                            n_epochs=n_epochs, seed=0, device="cuda", patience=1000)

    train_pignn_checkpointed(adj, gpu_cfg(1), [1], out_dir)  # partial: stop after checkpoint 1
    resumed = train_pignn_checkpointed(adj, gpu_cfg(3), [1, 2, 3], out_dir, resume=True)
    assert resumed["checkpoints"] == [1, 2, 3]


def test_calling_again_after_completion_is_a_noop(tmp_path):
    adj = _graph_adj()
    checkpoints = [1, 3, 5]
    out_dir = tmp_path / "run"
    cfg = _cfg(5, seed=2)
    r1 = train_pignn_checkpointed(adj, cfg, checkpoints, out_dir)
    r2 = train_pignn_checkpointed(adj, cfg, checkpoints, out_dir, resume=True)
    assert r1 == r2
    assert len((out_dir / "trajectory.jsonl").read_text().splitlines()) == len(checkpoints)


def test_checkpoints_must_include_t_max(tmp_path):
    adj = _graph_adj()
    with pytest.raises(ValueError):
        train_pignn_checkpointed(adj, _cfg(10), [1, 5], tmp_path / "bad")
