"""Continuation contracts: replay, optimizer, atomic publication, and history."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from Algorithms.alpha_zero import checkpointing as cp
from Algorithms.alpha_zero.runtime import ROOT, use_repository_open_spiel

use_repository_open_spiel()
import jax
import jax.numpy as jnp
from open_spiel.python.algorithms.alpha_zero import replay_buffer, utils


def example(value=0):
    return utils.TrainInput(
        observation=jnp.full((15,), value / 48, dtype=jnp.float32),
        legals_mask=jnp.ones(6, dtype=jnp.bool_),
        policy=jnp.full((6,), 1 / 6, dtype=jnp.float32),
        value=jnp.asarray(1, dtype=jnp.float32))


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_replay_roundtrip_preserves_ring_and_next_sample(tmp_path):
    original = replay_buffer.Buffer(8, seed=42)
    for i in range(13):
        original.append(example(i))
    original.sample(3)
    cp.save_buffer(tmp_path / "replay.npz", original)
    restored = replay_buffer.Buffer(8)
    cp.load_buffer(tmp_path / "replay.npz", restored, example())
    assert len(restored) == 8
    assert restored.total_seen == 13
    assert_tree_equal(original.buffer_state, restored.buffer_state)
    assert_tree_equal(original.sample(4), restored.sample(4))
    original.append(example(99))
    restored.append(example(99))
    assert_tree_equal(original.buffer_state, restored.buffer_state)
    with pytest.raises(ValueError, match="capacity"):
        cp.load_buffer(tmp_path / "replay.npz", replay_buffer.Buffer(9), example())


def test_empty_and_scalar_evaluation_buffers(tmp_path):
    for size in (0, 5):
        original = replay_buffer.Buffer(3)
        for i in range(size):
            original.append(jnp.asarray(i, dtype=jnp.float32))
        cp.save_buffer(tmp_path / "evaluation.npz", original)
        restored = replay_buffer.Buffer(3)
        cp.load_buffer(tmp_path / "evaluation.npz", restored, jnp.asarray(0, dtype=jnp.float32))
        assert restored.total_seen == size
        assert_tree_equal(original.buffer_state, restored.buffer_state)


def snapshot_config(tmp_path):
    model_path = tmp_path / "checkpoint-1"
    model_path.mkdir()
    (model_path / "fake-model").write_text("model bytes")
    cp.write_json(tmp_path / "manifest.json", {"settings": {}})
    return SimpleNamespace(path=str(tmp_path), session_id="test")


def test_interrupted_snapshot_keeps_previous_and_corruption_rejected(tmp_path, monkeypatch):
    config = snapshot_config(tmp_path)
    replay = replay_buffer.Buffer(8)
    replay.append(example())
    first = cp.save_snapshot(config, 1, replay, [], 1, 1)
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError("interrupted write")
        patch.setattr(cp, "save_buffer", fail)
        with pytest.raises(OSError, match="interrupted"):
            cp.save_snapshot(config, 1, replay, [], 2, 2)
    assert cp.find_snapshot(tmp_path) == first
    assert json.loads((first.parent / "latest.json").read_text())["completed_step"] == 1
    with pytest.raises(ValueError, match="Model-only"):
        cp.validate_snapshot(tmp_path / "checkpoint-1")
    (first / "replay.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        cp.validate_snapshot(first)


def test_history_preserves_abandoned_segment(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    history = tmp_path / "learner.jsonl"
    original = ''.join(json.dumps({"step": i}) + '\n' for i in (1, 2, 3))
    history.write_text(original)
    (tmp_path / "evaluation.jsonl").write_text(''.join(
        json.dumps({"training_step": i}) + '\n' for i in (1, 2, 3)))
    exports = tmp_path / "inference-checkpoints"
    exports.mkdir()
    for i in (1, 2, 3):
        (exports / f"step-{i:06d}.npz").write_bytes(b"export")
    for name in ("checkpoint-0", "checkpoint-2", "checkpoint-3", "checkpoint--1"):
        (tmp_path / name).mkdir()
    cp.prepare_history(tmp_path, session, 2)
    assert [json.loads(line)["step"] for line in history.read_text().splitlines()] == [1, 2]
    assert (session / "previous-learner.jsonl").read_text() == original
    assert (tmp_path / "checkpoint-2").exists()
    assert (session / "checkpoint-3").exists()
    assert (session / "checkpoint--1").exists()
    assert [json.loads(line)["training_step"] for line in
            (tmp_path / "evaluation.jsonl").read_text().splitlines()] == [1, 2]
    assert (exports / "step-000002.npz").exists()
    assert (session / "inference-checkpoints/step-000003.npz").exists()


def test_model_optimizer_and_replay_continue_identical_next_update(tmp_path):
    from open_spiel.python.algorithms.alpha_zero import model_linen
    model = model_linen.Model.build_model("ayo_mlp", [15], 6, 256, 3,
                                         0.0001, 0.001, str(tmp_path))
    replay = replay_buffer.Buffer(8, seed=27)
    for i in range(8):
        replay.append(example(i))
    model.update(replay.sample(4))
    model.save_checkpoint(1)
    cp.save_buffer(tmp_path / "replay.npz", replay)
    restored = model_linen.Model.build_model("ayo_mlp", [15], 6, 256, 3,
                                            0.0001, 0.001, str(tmp_path))
    restored.load_checkpoint(1)
    restored_replay = replay_buffer.Buffer(8)
    cp.load_buffer(tmp_path / "replay.npz", restored_replay, example())
    assert_tree_equal(model._state.opt_state, restored._state.opt_state)
    assert int(restored._state.step) == 1
    model.update(replay.sample(4))
    restored.update(restored_replay.sample(4))
    assert int(restored._state.step) == 2
    assert_tree_equal(model._state.params, restored._state.params)
    assert_tree_equal(model._state.opt_state, restored._state.opt_state)


def test_process_restart_continues_learner_and_preserves_history(tmp_path):
    run = tmp_path / "run"
    command = [sys.executable, "-B", "-m", "experiments.alpha_zero.train"]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    def launch(*args):
        return subprocess.run(command + list(args), cwd=ROOT, env=environment,
                              capture_output=True, text=True, timeout=180)
    result = launch("--config", str(ROOT / "experiments/alpha_zero/configs/sanity.json"),
                    "--output", str(run))
    assert result.returncode == 0, result.stdout + result.stderr
    snapshot = cp.find_snapshot(run)
    before = cp.validate_snapshot(snapshot)
    checkpoint_hashes = {str(p.relative_to(run)): cp.digest(p)
                         for p in (run / "checkpoint-2").rglob("*") if p.is_file()}
    result = launch("--resume", str(run), "--max-steps", "4")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "next step 3" in result.stdout
    assert cp.validate_snapshot(snapshot) == before
    for name, digest in checkpoint_hashes.items():
        assert cp.digest(run / name) == digest
    rows = [json.loads(line) for line in (run / "learner.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert [row["total_states"] for row in rows] == [8, 16, 24, 32]
    assert [row["total_trajectories"] for row in rows] == [2, 4, 6, 8]
    assert [row["gradient_updates"] for row in rows] == [2, 6, 12, 20]
    evaluations = [json.loads(line) for line in (run / "evaluation.jsonl").read_text().splitlines()]
    assert [row["training_step"] for row in evaluations] == [1, 2, 3, 4]
    assert all(row["status"] == "completed" and row["games"] == 2 for row in evaluations)
    assert len(list((run / "inference-checkpoints").glob("step-*.npz"))) == 4
    curve = tmp_path / "greedy-curve.jsonl"
    offline = subprocess.run(
        [sys.executable, "-B", "-m", "experiments.alpha_zero.offline_evaluate",
         "--run", str(run), "--checkpoints", "1:4", "--games", "2",
         "--opponent", "GREEDY_HSTAR", "--output", str(curve)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert offline.returncode == 0, offline.stdout + offline.stderr
    assert [json.loads(line)["training_step"] for line in curve.read_text().splitlines()] == [1, 2, 3, 4]
    latest, learner = cp.validate_snapshot(cp.find_snapshot(run))
    assert learner["completed_step"] == 4
    assert latest["manifest"]["settings"]["max_steps"] == 4
    actor_logs = list((run / "sessions").glob("*/log-actor-0.txt"))
    assert len(actor_logs) == 2
    for log in actor_logs:
        text = log.read_text()
        assert text.index("Initial checkpoint loaded") < text.index("Game 1:")
    assert launch("--resume", str(run), "--max-steps", "4").returncode != 0
    assert launch("--resume", str(run), "--resume-step", "2", "--max-steps", "5").returncode != 0
    assert launch("--resume", str(run), "--max-steps", "5", "--config",
                  str(ROOT / "experiments/alpha_zero/configs/starter.json")).returncode != 0
    result = launch("--resume", str(run), "--max-steps", "5", "--evaluation-games", "0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "next step 5" in result.stdout
    evaluations = [json.loads(line) for line in (run / "evaluation.jsonl").read_text().splitlines()]
    assert [row["training_step"] for row in evaluations] == [1, 2, 3, 4, 5]
    assert evaluations[-1]["status"] == "skipped" and evaluations[-1]["games"] == 0
    assert (run / "inference-checkpoints/step-000005.npz").is_file()
    metadata, learner = cp.validate_snapshot(cp.find_snapshot(run))
    assert learner["completed_step"] == 5
    assert metadata["manifest"]["settings"]["evaluation_games"] == 0
    assert launch("--resume", str(run), "--max-steps", "6", "--actors", "0",
                  "--dry-run").returncode != 0
    result = launch("--resume", str(run), "--max-steps", "6", "--actors", "3",
                    "--evaluation-games", "0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "next step 6" in result.stdout
    rows = [json.loads(line) for line in (run / "learner.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4, 5, 6]
    session = max((run / "sessions").iterdir())
    assert json.loads((session / "session.json").read_text())["settings"]["actors"] == 3
    assert len(list(session.glob("log-actor-*.txt"))) == 3
    metadata, learner = cp.validate_snapshot(cp.find_snapshot(run))
    assert learner["completed_step"] == 6
    assert metadata["manifest"]["settings"]["actors"] == 3
