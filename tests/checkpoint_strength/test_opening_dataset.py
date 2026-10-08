"""The isolated opening dataset is reproducible and preserves complete game state."""
from collections import Counter
import gzip
import hashlib
import json
import random

import pytest

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
import pyspiel
from Algorithms.alpha_zero import game as adapter  # noqa: F401
from experiments.checkpoint_strength.generate_openings import (
    characterize, freeze_benchmark, generate_candidates, save_investigation,
)
from experiments.checkpoint_strength.opening_dataset import (
    DEFAULT_OPENINGS, ROOT, digest, load_dataset, replay, snapshot,
)


def test_generation_is_reproducible_counts_actual_decisions_and_isolates_rng(tmp_path):
    game = pyspiel.load_game("ayo_olopon")
    before = random.getstate()
    a = generate_candidates(game, count=25, seed=7)
    b = generate_candidates(game, count=25, seed=7)
    assert a == b and random.getstate() == before
    assert a != generate_candidates(game, count=25, seed=8)
    assert Counter(r["target_plies"] for r in a) == {2: 25, 4: 25, 6: 25, 8: 25, 10: 25}
    for record in a:
        state = replay(game, record["actions"])
        assert snapshot(state) == record["state"]
        assert state.move_number() == len(record["actions"]) == record["plies"]
        assert state.is_terminal() or record["plies"] == record["target_plies"]
    report = characterize(a)
    assert all(s["terminal_trials"] + s["eligible_nonterminal"] == 25 for s in report.values())
    first = save_investigation(tmp_path / "a", game, 25, 7)
    second = save_investigation(tmp_path / "b", game, 25, 7)
    assert first == second
    assert (tmp_path / "a/candidates.jsonl.gz").read_bytes() == (tmp_path / "b/candidates.jsonl.gz").read_bytes()
    with pytest.raises(FileExistsError):
        save_investigation(tmp_path / "a", game, 25, 7)
    output = tmp_path / "frozen.json"
    freeze_benchmark(tmp_path / "a", output, 4, 10)
    assert len(load_dataset(output, game).openings) == 10
    # Moving the saved report through a CRLF checkout must not change provenance.
    report_path = tmp_path / "b/depth_analysis.json"
    report_path.write_bytes(report_path.read_bytes().replace(b"\n", b"\r\n"))
    other = tmp_path / "other.json"
    freeze_benchmark(tmp_path / "b", other, 4, 10)
    assert output.read_bytes() == other.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_benchmark(tmp_path / "a", output, 4, 10)
    with pytest.raises(ValueError, match="fewer"):
        freeze_benchmark(tmp_path / "a", tmp_path / "impossible.json", 2, 100)


def test_committed_benchmark_and_investigation_are_consistent():
    game = pyspiel.load_game("ayo_olopon_alpha_zero")
    benchmark = load_dataset(DEFAULT_OPENINGS, game)
    assert len(benchmark.openings) == 100
    assert {len(o.actions) for o in benchmark.openings} == {6}
    assert len({o.position_sha256 for o in benchmark.openings}) == 100
    directory = ROOT / "experiments/checkpoint_strength/dataset/investigation_v1"
    analysis = json.loads((directory / "depth_analysis.json").read_text())
    candidates = (directory / "candidates.jsonl.gz").read_bytes()
    assert hashlib.sha256(candidates).hexdigest() == analysis["candidates_sha256"]
    raw = gzip.decompress(candidates)
    assert hashlib.sha256(raw).hexdigest() == analysis["candidates_content_sha256"]
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 25000 and characterize(records) == analysis["by_depth"]
    payload = json.loads((ROOT / DEFAULT_OPENINGS).read_text())
    assert payload["investigation_sha256"] == digest(analysis)
    # Confirm the exact documented selection rule, independently of the freeze writer.
    seen, expected = set(), []
    for record in records:
        if record["target_plies"] == 6 and not record["state"]["is_terminal"] and record["position_sha256"] not in seen:
            seen.add(record["position_sha256"])
            expected.append(record["actions"])
    assert [list(o.actions) for o in benchmark.openings] == expected[:100]


def test_replay_preserves_repetition_memory_horizon_and_cross_agent_rules():
    neural_game = pyspiel.load_game("ayo_olopon_alpha_zero")
    plain_game = pyspiel.load_game("ayo_olopon")
    benchmark = load_dataset(DEFAULT_OPENINGS, neural_game)
    assert load_dataset(DEFAULT_OPENINGS, plain_game).content_sha256 == benchmark.content_sha256
    for opening in benchmark.openings:
        state = opening.new_state(neural_game)
        assert state.ply == state.move_number() == 6
        assert snapshot(state) == snapshot(opening.new_state(plain_game))
        # Manual board assignment loses repetition memory; replay does not.
        assert state._positions_since_capture
        assert state.clone()._positions_since_capture == state._positions_since_capture
    with pytest.raises(ValueError, match="rules/horizon"):
        load_dataset(DEFAULT_OPENINGS, pyspiel.load_game("ayo_olopon_alpha_zero(max_moves=4)"))


@pytest.mark.parametrize("corruption", ["hash", "action", "snapshot", "duplicate", "terminal"])
def test_loader_rejects_tampered_or_invalid_openings(tmp_path, corruption):
    data = json.loads((ROOT / DEFAULT_OPENINGS).read_text())
    if corruption == "hash":
        data["openings"][0]["actions"][0] = 99
    elif corruption == "action":
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
        load_dataset(path, pyspiel.load_game("ayo_olopon"))
