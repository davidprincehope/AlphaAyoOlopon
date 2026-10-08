"""Exercise the Modal worker's spawn policy and real learner timing on CPU."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_modal_worker_spawn_learning_and_resume(tmp_path):
    # Exercise the same worker entrypoint plumbing without its GPU preflight.
    launcher = tmp_path / "cpu_modal_probe.py"
    launcher.write_text(
        "from experiments.alpha_zero import modal_worker\n"
        "if __name__ == '__main__':\n"
        "    modal_worker.install_round_timing()\n"
        "    modal_worker.train.main()\n", encoding="utf-8")
    run = tmp_path / "run"
    env = dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu")
    command = [sys.executable, str(launcher)]
    for args in (
        ["--config", str(ROOT / "experiments/alpha_zero/configs/sanity.json"),
         "--output", str(run), "--actors", "2", "--evaluation-games", "0"],
        ["--resume", str(run), "--max-steps", "3", "--actors", "2",
         "--evaluation-games", "0"],
    ):
        result = subprocess.run(command + args, cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (run / "learner.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[-1]["gradient_updates"] > rows[-2]["gradient_updates"]
    assert rows[-1]["session_id"] != rows[-2]["session_id"]
    for row in rows:
        timing = row["timing"]
        assert timing["collection_seconds"] > 0
        assert timing["learning_and_model_checkpoint_seconds"] > 0
        assert timing["export_and_evaluation_seconds"] >= 0
        assert abs(timing["round_seconds"] - sum(timing[key] for key in (
            "collection_seconds", "learning_and_model_checkpoint_seconds",
            "export_and_evaluation_seconds"))) < 1e-6
    assert (run / "training-checkpoints/step-000003/metadata.json").exists()


def test_compiled_actor_inference_matches_and_refreshes_parameters(tmp_path):
    probe = tmp_path / "inference_probe.py"
    probe.write_text('''
import numpy as np
from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
from open_spiel.python.algorithms.alpha_zero.model_linen import Model
from open_spiel.python.algorithms.alpha_zero.utils import TrainInput

model = Model.build_model("ayo_mlp", [15], 6, 256, 3, 0.0001, 0.001, None)
observation = np.linspace(0, 1, 15, dtype=np.float32)
mask = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
original = Model.inference
expected = original(model, observation, mask)
from experiments.alpha_zero import modal_worker
actual = model.inference(observation, mask)
for a, b in zip(actual, expected):
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
assert isinstance(actual[1], np.ndarray)
assert np.all(actual[1][~mask] == 0)
old = model._state.params
model.update(TrainInput(observation=np.stack([observation] * 4),
                       legals_mask=np.stack([mask] * 4),
                       policy=np.stack([mask / mask.sum()] * 4).astype(np.float32),
                       value=np.ones(4, dtype=np.float32)))
assert model._state.params is not old
expected = original(model, observation, mask)
actual = model.inference(observation, mask)
assert model._modal_inference_view.params is model._state.params
for a, b in zip(actual, expected):
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
''', encoding="utf-8")
    result = subprocess.run([sys.executable, str(probe)], cwd=ROOT,
                            env=dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
