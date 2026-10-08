"""Adapters for existing Ayo agents; specifications are JSON-serializable."""
import importlib
import random
from types import SimpleNamespace


def make_agent(game, seed, specification):
    name = specification.get("name", "RAND").upper()
    params = dict(specification.get("params", {}))
    if name in ("RAND", "RANDOM", "GREEDY_HSTAR"):
        if params:
            raise ValueError(f"{name} accepts no parameters")
        from experiments.agent_benchmark.random_vs_greedy_hstar import _policy_action
        rng = random.Random(seed)
        return SimpleNamespace(step=lambda state: _policy_action(
            state, "RAND" if name == "RANDOM" else name, rng))
    if name in ("GREEDY", "HEURISTIC"):
        from experiments.heuristic.run_experiment import _policy_action, PolicyStats
        policy = params.pop("heuristic", "H_CTM")
        weights = params.pop("weights", None)
        if params:
            raise ValueError(f"Unknown GREEDY parameters: {sorted(params)}")
        rng, stats = random.Random(seed), PolicyStats()
        return SimpleNamespace(step=lambda state: _policy_action(
            policy, state, state.current_player(), rng, stats, weights))
    if name == "MINIMAX":
        from Algorithms.ayo_minimax import make_bot
        return make_bot(game, **params)
    if name == "MCTS":
        from Algorithms.ayo_mcts import make_bot
        params.setdefault("seed", seed)
        return make_bot(game, **params)
    if "factory" in specification:
        module, function = specification["factory"].split(":", 1)
        return getattr(importlib.import_module(module), function)(game=game, seed=seed, **params)
    raise ValueError(f"Unknown agent {name!r}; use RAND, GREEDY_HSTAR, GREEDY, MINIMAX, MCTS, or a module:function factory")
