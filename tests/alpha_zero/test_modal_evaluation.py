"""Modal offline wrapper persists completion/failure and validates launch options."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from experiments.alpha_zero import modal_evaluate as remote
from experiments.alpha_zero import offline_evaluate as offline


@pytest.mark.parametrize("fail", [False, True])
def test_modal_commits_at_checkpoint_and_exit_without_training(tmp_path, monkeypatch, fail):
    import jax
    commits, calls = [], []
    monkeypatch.setattr(remote, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr(remote, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(remote, "results_volume", SimpleNamespace(commit=lambda: commits.append(True)))
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu")])
    def evaluate(run, config, output, *, on_checkpoint):
        calls.append((run, config, output))
        on_checkpoint({"status": "completed"})
        if fail:
            raise RuntimeError("match failed")
        return {"status": "completed"}
    monkeypatch.setattr(offline, "run_evaluation", evaluate)
    config = asdict(offline.EvaluationConfig(checkpoints=[1], games=2, simulations=64))
    function = remote.evaluate_remote.local
    if fail:
        with pytest.raises(RuntimeError, match="match failed"):
            function("source", "evaluation", config)
    else:
        assert function("source", "evaluation", config)["status"] == "completed"
    assert len(commits) == 2
    assert calls[0][0] == tmp_path / "checkpoints/source"
    assert calls[0][2] == tmp_path / "results/evaluation/results.jsonl"


def test_modal_launch_overrides_and_background_submission(monkeypatch):
    received = []
    monkeypatch.setattr(remote, "evaluate_remote", SimpleNamespace(
        spawn=lambda *args: (received.append(args) or SimpleNamespace(object_id="fc-test"))))
    remote.main(run_name="source", evaluation_name="future", checkpoints="1,25:75:25",
                games=4, simulations=8, opponent="MCTS", seed=9, background=True)
    run, name, config = received[0]
    assert run == "source" and name == "future"
    assert config["checkpoints"] == [1, 25, 50, 75]
    assert config["games"] == 4 and config["simulations"] == 8 and config["seed"] == 9
    assert config["opponent"]["name"] == "MCTS"
    with pytest.raises(ValueError, match="single path component"):
        remote.main(run_name="../unsafe")
