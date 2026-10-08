"""The repeat suite changes the frozen checkpoint, preserving all four protocols."""
from dataclasses import asdict
import json
import subprocess

import pytest

from experiments.checkpoint_strength import modal_neural_vs_mcts as wrapper


@pytest.mark.parametrize("checkpoint", [100, 1000])
def test_suite_changes_only_checkpoint(checkpoint):
    configurations = wrapper.suite_configurations(checkpoint)
    assert len(configurations) == 4
    for name, filename in wrapper.SUITE_CASES:
        original = json.loads((wrapper.ROOT / "experiments/checkpoint_strength/configs" / filename).read_text())
        actual = asdict(configurations[name])
        for key, value in original.items():
            assert actual[key] == (checkpoint if key == "checkpoint" else value)
        assert actual["games"] == 400 and actual["workers"] == 8
    assert [(c.neural_mode, c.simulations, c.mcts_budget) for c in configurations.values()] == [
        ("puct", 64, 64), ("puct", 128, 128), ("puct", 64, 128), ("policy", 0, 64)]


def test_cleanup_timeout_does_not_fail_completed_games(monkeypatch, capsys):
    def busy(*args, **kwargs):
        assert kwargs["timeout"] == 2
        raise subprocess.TimeoutExpired(args[0], 2)
    monkeypatch.setattr(wrapper.subprocess, "run", busy)
    wrapper.stop_mps("nvidia-cuda-mps-control", {})
    assert "mps_cleanup_at_container_exit" in capsys.readouterr().out
