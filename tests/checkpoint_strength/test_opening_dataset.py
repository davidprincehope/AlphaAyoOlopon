"""Validate the published book, replay memory and rejection of corrupt inputs."""
from collections import Counter
import json

import pytest
from experiments.alpha_zero import offline_evaluate as offline
from experiments.checkpoint_strength.opening_dataset import (
    DEFAULT_OPENINGS, LEGACY_OPENINGS, ROOT, digest, load_dataset, snapshot,
)


def test_le4capture_book_preserves_complete_state_and_cross_agent_rules():
    neural = offline.pyspiel.load_game("ayo_olopon_alpha_zero")
    plain = offline.pyspiel.load_game("ayo_olopon")
    book = load_dataset(DEFAULT_OPENINGS, neural)
    assert book.content_sha256 == "681ac741957726f17d3e29241330f76d4d0da80f7dbfd2e05605736ac68d0a9b"
    assert len(book.openings) == len({o.position_sha256 for o in book.openings}) == 200
    assert Counter(len(o.actions) for o in book.openings) == {2: 32, 4: 42, 6: 42, 8: 42, 10: 42}
    for opening in book.openings:
        state = opening.new_state(neural)
        assert state.move_number() == len(opening.actions)
        assert abs(state.captured[0] - state.captured[1]) <= 4
        assert snapshot(state) == snapshot(opening.new_state(plain))
        assert state._positions_since_capture
        assert state.clone()._positions_since_capture == state._positions_since_capture
    with pytest.raises(ValueError, match="rules/horizon"):
        load_dataset(DEFAULT_OPENINGS, offline.pyspiel.load_game("ayo_olopon_alpha_zero(max_moves=4)"))


def test_historical_dataset_path_loads_the_same_frozen_book():
    game = offline.pyspiel.load_game("ayo_olopon")
    assert not (ROOT / LEGACY_OPENINGS).exists()
    assert load_dataset(LEGACY_OPENINGS, game) == load_dataset(DEFAULT_OPENINGS, game)


@pytest.mark.parametrize("corruption", ["hash", "action", "snapshot", "duplicate", "terminal"])
def test_loader_rejects_tampered_or_invalid_openings(tmp_path, corruption):
    data = json.loads((ROOT / DEFAULT_OPENINGS).read_text())
    if corruption in ("hash", "action"):
        data["openings"][0]["actions"][0] = 99
    elif corruption == "snapshot":
        data["openings"][0]["state"]["repetition_positions"] = []
    elif corruption == "duplicate":
        data["openings"][1] = dict(data["openings"][0], opening_id="different-id")
    else:
        data["openings"][0]["state"]["is_terminal"] = True
    if corruption != "hash":
        data["content_sha256"] = digest({k: v for k, v in data.items() if k != "content_sha256"})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_dataset(path, offline.pyspiel.load_game("ayo_olopon"))
