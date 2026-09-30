"""AlphaZero adapters for the existing Ayo agent benchmark match runner."""
import json
from pathlib import Path
import random
import warnings

import numpy as np


def make_neural_agent(game, model, simulations, uct_c, seed):
    from open_spiel.python.algorithms import mcts
    from open_spiel.python.algorithms.alpha_zero.evaluator import AlphaZeroEvaluator
    from types import SimpleNamespace
    bot = mcts.MCTSBot(game, uct_c, simulations, AlphaZeroEvaluator(game, model),
                       solve=False, random_state=np.random.RandomState(seed),
                       dirichlet_noise=None, child_selection_fn=mcts.SearchNode.puct_value)
    def step(state):
        root = bot.mcts_search(state)
        return min(root.children, key=lambda child: (-child.explore_count, child.action)).action
    return SimpleNamespace(step=step)


def evaluate_model(game, model, settings, step, *, opponent=None, games=None,
                   simulations=None, seed=None):
    from experiments.agent_benchmark.agents import make_agent
    from experiments.agent_benchmark.random_vs_greedy_hstar import run_matches
    opponent = settings.evaluation_opponent if opponent is None else opponent
    games = settings.evaluation_games if games is None else games
    seed = settings.evaluation_seed if seed is None else seed
    simulations = settings.max_simulations if simulations is None else simulations
    if simulations < 2:
        raise ValueError("Neural search requires at least 2 simulations")
    # Isolate even custom baseline agents that use module-level RNG functions.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        factories = {
            "ALPHAZERO": lambda game, seed: make_neural_agent(game, model, simulations, settings.uct_c, seed),
            "OPPONENT": lambda game, seed: make_agent(game, seed, opponent)}
        summary, records = run_matches(game, factories, games, seed)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    stats = summary["policies"]["ALPHAZERO"]
    return {"checkpoint_id": step, "training_step": step, "opponent": opponent,
            "mcts_simulations": simulations, "evaluation_seed": seed,
            **{key: stats[key] for key in ("games", "wins", "draws", "losses", "win_rate",
                                            "score_rate", "average_game_length")},
            "player0_score_rate": summary["by_seat"]["0"]["policies"]["ALPHAZERO"]["score_rate"],
            "player1_score_rate": summary["by_seat"]["1"]["policies"]["ALPHAZERO"]["score_rate"],
            "termination_reasons": summary["termination_reasons"],
            "summary": summary, "matches": records, "status": "completed"}


def evaluate_round(config, game, model, step):
    """Measurement only: no learner/replay/optimizer writes or acceptance logic."""
    from Algorithms.alpha_zero.inference import InferenceModel, export_round
    from Algorithms.alpha_zero.config import Settings
    run = Path(config.path)
    try:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        values = {key: getattr(config, key, value) for key, value in manifest["settings"].items()}
        # Older resume manifests lack the evaluation fields.
        for key in ("evaluation_games", "evaluation_opponent", "evaluation_seed"):
            values[key] = getattr(config, key)
        settings = Settings(**values)
        manifest["settings"] = values
        export_round(run, step, model._state.params, manifest)
        report = evaluate_model(game, InferenceModel(model._state.params), settings, step)
    except Exception as exc:
        report = {"training_step": step, "checkpoint_id": step, "status": "error", "error": repr(exc)}
        warnings.warn(f"Round {step} evaluation failed (training continues): {exc}")
    report["session_id"] = config.session_id
    try:
        with (run / "evaluation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report) + "\n")
    except OSError as exc:
        warnings.warn(f"Could not record round {step} evaluation: {exc}")
    return report
