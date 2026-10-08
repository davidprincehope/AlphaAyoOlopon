"""Direct network actions must be legal, deterministic, value-independent, and search-free."""
import json

import numpy as np
import pytest

from experiments.checkpoint_strength import neural_vs_mcts as comparison
from tests.checkpoint_strength.test_neural_vs_mcts import source  # noqa: F401


def test_direct_policy_masks_illegal_moves_breaks_ties_and_ignores_value():
    calls, traces = [], []
    observation = list(range(15))
    mask = [0, 1, 0, 1, 1, 0]

    class State:
        def legal_actions(self):
            return [4, 1, 3]
        def legal_actions_mask(self):
            return mask
        def observation_tensor(self):
            return observation
        def clone(self):
            raise AssertionError("Direct policy must not evaluate successors")
        def apply_action(self, action):
            raise AssertionError("Direct policy must not apply actions during prediction")

    class Model:
        def inference(self, observed, legals):
            calls.append((observed, legals))
            # Illegal actions deliberately have the highest scores. The value changes
            # sign but must have no effect; tied legal actions 3/4 choose action 3.
            return (-100 if len(calls) == 1 else 100), np.asarray([1, .1, 1, .4, .4, 1])

    agent = comparison.make_policy_agent(Model(), on_prediction=traces.append)
    assert [agent.step(State()), agent.step(State())] == [3, 3]
    assert calls == [(observation, mask), (observation, mask)]
    assert len(traces) == 2 and all(t["chosen_action"] == 3 for t in traces)


def test_policy_matches_use_no_neural_search_and_one_prediction_per_decision(source, tmp_path, monkeypatch):
    run, game, item, _ = source
    calls = []
    original_inference = item["model"].inference
    def traced_inference(observation, mask):
        calls.append(True)
        return original_inference(observation, mask)
    monkeypatch.setattr(item["model"], "inference", traced_inference)
    def forbidden(*args, **kwargs):
        raise AssertionError("C600 policy must not construct a neural MCTS agent")
    monkeypatch.setattr(comparison, "make_neural_agent", forbidden)
    mcts_budgets = []
    original_bot = comparison.make_bot
    def bot(game, simulations, **kwargs):
        mcts_budgets.append(simulations)
        return original_bot(game, simulations, **kwargs)
    monkeypatch.setattr(comparison, "make_bot", bot)
    config = comparison.NeuralVsMCTSConfig(games=4, neural_mode="policy", simulations=0,
                                          mcts_simulations=4, workers=1)
    output = tmp_path / "policy"
    report = comparison.run_neural_vs_mcts(run, config, output)
    records = [json.loads(line) for line in (output / "results.games.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "results.manifest.json").read_text())
    assert mcts_budgets == [4] * 4
    assert len(calls) == sum(r["policy_decisions"].get("C600", 0) for r in records)
    assert manifest["evaluation_policy"]["C600"]["search"] == "none"
    assert manifest["evaluation_policy"]["C600"]["simulations"] == 0
    assert not manifest["evaluation_policy"]["C600"]["value_head_used"]
    assert manifest["evaluation_policy"]["MCTS"]["simulations"] == 4
    assert report["games"] == 4
    for record in records:
        state = comparison.offline.pyspiel.load_game(str(game)).new_initial_state()
        for action in record["opening_actions"]:
            state.apply_action(action)
        traces = iter(record["neural_policy_predictions"])
        for action in record["continuation_actions"]:
            if record[f"player_{state.current_player()}_policy"] == "C600":
                trace = next(traces)
                assert trace["legal_actions"] == sorted(state.legal_actions())
                assert trace["chosen_action"] == action
                probabilities = trace["policy_probabilities"]
                assert action == max(sorted(state.legal_actions()), key=lambda a: probabilities[a])
            assert action in state.legal_actions()
            state.apply_action(action)
        assert next(traces, None) is None and state.is_terminal()
        assert list(state.returns()) == record["returns_by_player"]


@pytest.mark.parametrize("changes", [{"neural_mode": "unknown"},
    {"neural_mode": "policy"}, {"neural_mode": "policy", "simulations": 0},
    {"neural_mode": "policy", "simulations": 2, "mcts_simulations": 64},
    {"neural_mode": "policy", "simulations": False, "mcts_simulations": 64}])
def test_invalid_policy_configuration(changes):
    with pytest.raises(ValueError):
        comparison.NeuralVsMCTSConfig(**changes)


def test_saved_policy_configuration_has_no_search_and_mcts_64():
    from pathlib import Path
    config = comparison.NeuralVsMCTSConfig(**json.loads(Path(
        "experiments/checkpoint_strength/configs/c600_policy_vs_mcts_64.json").read_text()))
    assert config.neural_mode == "policy" and config.simulations == 0 and config.mcts_budget == 64
