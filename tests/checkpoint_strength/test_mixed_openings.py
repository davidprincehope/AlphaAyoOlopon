"""Mixed-depth benchmarks are frozen, globally unique, and replayable."""
from collections import Counter

import pyspiel
import pytest
from Model.ayo_olopon import ayo_olopon  # noqa: F401
from experiments.checkpoint_strength.generate_openings import freeze_mixed_benchmark
from experiments.checkpoint_strength.opening_dataset import ROOT, load_dataset


SOURCE = ROOT / "experiments/checkpoint_strength/dataset/investigation_v1"


def test_mixed_selection_reproducible_unique_and_complete(tmp_path):
    counts = {2: 20, 4: 45, 6: 45, 8: 45, 10: 45}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    payload = freeze_mixed_benchmark(SOURCE, a, counts)
    freeze_mixed_benchmark(SOURCE, b, counts)
    assert a.read_bytes() == b.read_bytes()
    game = pyspiel.load_game("ayo_olopon")
    dataset = load_dataset(a, game)
    assert len(dataset.openings) == 200
    assert Counter(len(o.actions) for o in dataset.openings) == counts
    assert len({o.position_sha256 for o in dataset.openings}) == 200
    assert [len(o.actions) for o in dataset.openings[:5]] == [2, 4, 6, 8, 10]
    for opening in dataset.openings:
        state = opening.new_state(game)
        assert state.move_number() == len(opening.actions) and not state.is_terminal()
    assert all(s["terminal_trials"] == 0 for s in payload["statistics"].values())
    with pytest.raises(FileExistsError):
        freeze_mixed_benchmark(SOURCE, a, counts)


def test_insufficient_depth_does_not_write_an_artifact(tmp_path):
    output = tmp_path / "impossible.json"
    with pytest.raises(ValueError, match="Depth 2 has fewer than 40"):
        freeze_mixed_benchmark(SOURCE, output, {2: 40, 4: 40, 6: 40, 8: 40, 10: 40})
    assert not output.exists()


@pytest.mark.parametrize("counts", [{}, {3: 20}, {2: 0}, {2: True}, {"2": 20}])
def test_invalid_allocation_rejected(tmp_path, counts):
    with pytest.raises(ValueError, match="depth_counts"):
        freeze_mixed_benchmark(SOURCE, tmp_path / "bad.json", counts)


def test_saved_v2_dataset_matches_128_simulation_configuration():
    from experiments.alpha_zero.offline_evaluate import load_config
    config = load_config(ROOT / "experiments/checkpoint_strength/configs/progression_mixed_v2_128.json")
    game = pyspiel.load_game("ayo_olopon")
    dataset = load_dataset(config.openings, game)
    assert Counter(len(o.actions) for o in dataset.openings) == {2: 32, 4: 42, 6: 42, 8: 42, 10: 42}
    assert len(dataset.openings) == 200 and config.games == 400
    assert config.simulations == 128 and config.checkpoints == [1, *range(50, 601, 50)]
