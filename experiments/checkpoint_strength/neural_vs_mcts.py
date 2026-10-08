"""Frozen neural checkpoint versus vanilla MCTS, with parallel opening pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import threading
import time

from experiments.alpha_zero import offline_evaluate as offline
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import make_neural_agent
from Algorithms.ayo_mcts import make_bot
from experiments.agent_benchmark.random_vs_greedy_hstar import play_game, summarize
from experiments.checkpoint_strength.head_to_head import paired_score_interval
from experiments.checkpoint_strength.metrics import first_player_advantage
from experiments.checkpoint_strength.opening_dataset import ROOT, digest, load_dataset


@dataclass(frozen=True)
class NeuralVsMCTSConfig:
    checkpoint: int = 600
    games: int = 400
    simulations: int = 64  # Neural budget; also the legacy default for MCTS.
    seed: int = 20261003
    openings: str = "experiments/checkpoint_strength/dataset/ayo_mixed_v2_200.json"
    workers: int = 8
    mcts_simulations: int | None = None
    neural_mode: str = "puct"

    def __post_init__(self):
        if self.neural_mode not in ("puct", "policy"):
            raise ValueError("neural_mode must be puct or policy")
        if self.neural_mode == "policy":
            if type(self.simulations) is not int or self.simulations != 0:
                raise ValueError("Direct policy mode requires zero neural simulations")
        elif type(self.simulations) is not int or self.simulations < 2:
            raise ValueError("Neural PUCT simulations must be an integer >= 2")
        offline.EvaluationConfig(checkpoints=[self.checkpoint], games=self.games,
                                 simulations=None if self.neural_mode == "policy" else self.simulations,
                                 seed=self.seed, openings=self.openings)
        if self.mcts_simulations is not None and (
                type(self.mcts_simulations) is not int or self.mcts_simulations < 2):
            raise ValueError("MCTS simulations must be an integer >= 2, or null to match the neural budget")
        if self.mcts_budget < 2:
            raise ValueError("Direct policy mode requires an explicit MCTS budget >= 2")
        if type(self.workers) is not int or not 1 <= self.workers <= 28:
            raise ValueError("workers must be an integer between 1 and 28")
        if self.seed + self.games // 2 - 1 >= 2**32:
            raise ValueError("Paired seeds must fit NumPy RandomState")

    @property
    def mcts_budget(self):
        return self.simulations if self.mcts_simulations is None else self.mcts_simulations


def make_policy_agent(model, *, on_prediction=None):
    """One current-state prediction per decision; use only the policy head."""
    import numpy as np
    from types import SimpleNamespace

    def step(state):
        legal = sorted(state.legal_actions())
        if not legal:
            raise ValueError("Direct policy requires a nonterminal state with legal actions")
        mask = state.legal_actions_mask()
        _, policy = model.inference(state.observation_tensor(), mask)
        policy = np.asarray(policy)
        if (policy.shape != (len(mask),) or not np.isfinite(policy).all()
                or (policy < 0).any()):
            raise ValueError("Invalid neural policy probabilities")
        action = int(max(legal, key=lambda action: policy[action]))
        if on_prediction is not None:
            on_prediction({"chosen_action": action, "legal_actions": legal,
                           "policy_probabilities": policy.tolist()})
        return action

    return SimpleNamespace(step=step)


def neural_policy_settings(config):
    if config.neural_mode == "policy":
        return {"search": "none", "simulations": 0, "neural_policy": True,
                "neural_value": False, "policy_evaluations_per_move": 1,
                "successor_evaluations": 0, "value_head_used": False,
                "dirichlet_noise": None, "temperature_sampling": False,
                "action_selection": "maximum predicted legal-action probability, lowest action ID breaks ties"}
    return {"search": "PUCT", "simulations": config.simulations,
            "leaf_value": "frozen neural value", "prior": "frozen neural policy", "solve": False,
            "dirichlet_noise": None, "temperature_sampling": False,
            "action_selection": "maximum visit count, lowest action ID breaks ties"}


def play_pair(game, model, config, uct_c, index, opening, dataset_hash):
    """Use original global pair IDs/seeds regardless of execution order."""
    name = f"C{config.checkpoint}"
    predictions = []
    neural_factory = (lambda game, seed: make_policy_agent(model, on_prediction=predictions.append)) if config.neural_mode == "policy" else (
        lambda game, seed: make_neural_agent(game, model, config.simulations, uct_c, seed))
    factories = {
        name: neural_factory,
        "MCTS": lambda game, seed: make_bot(game, config.mcts_budget, rollouts_per_leaf=1,
                                           uct_c=uct_c, seed=seed)}
    records = []
    for seat, assignment in enumerate(((name, "MCTS"), ("MCTS", name))):
        predictions.clear()
        record = play_game(2 * index + seat, assignment, config.seed + index,
                           game.max_game_length(), game=game,
                           agent_factories=factories, opening=opening)
        record.update(opening_pair_id=index, opening_dataset_sha256=dataset_hash)
        if config.neural_mode == "policy":
            record["neural_policy_predictions"] = list(predictions)
        records.append(record)
    return records


_WORKER = None


def initialize_worker(game_string, checkpoint_source, checkpoint_hash, configuration, uct_c):
    global _WORKER
    if offline.source_hash(checkpoint_source) != checkpoint_hash:
        raise RuntimeError("Worker checkpoint hash differs from preflight")
    game = offline.pyspiel.load_game(game_string)
    model = offline.InferenceModel(offline.load_parameters(checkpoint_source))
    state = game.new_initial_state()
    model.inference(state.observation_tensor(), state.legal_actions_mask())
    _WORKER = (game, model, NeuralVsMCTSConfig(**configuration), uct_c)


def worker_pair(index, opening, dataset_hash):
    return play_pair(*_WORKER, index, opening, dataset_hash)


def make_report(config, provenance, records):
    names = (f"C{config.checkpoint}", "MCTS")
    summary = summarize(records)
    return {
        "status": "completed", "config": asdict(config), "opening_dataset": provenance,
        **summary,
        "by_seat": {name: {f"P{seat}": summarize([
            r for r in records if r[f"player_{seat}_policy"] == name])["policies"][name]
            for seat in (0, 1)} for name in names},
        "by_opening_depth": {str(depth): summarize([
            r for r in records if r["opening_plies"] == depth])["policies"]
            for depth in sorted({r["opening_plies"] for r in records})},
        "first_player_advantage": first_player_advantage(records),
        "score_intervals": {name: paired_score_interval(records, name, config.seed) for name in names}}


def run_neural_vs_mcts(run, config, output, *, on_progress=None, hardware=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run):
        raise ValueError("Results must be outside the source training run")
    if output.exists():
        raise FileExistsError(output)
    training = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (training.get("architecture_version") != "ayo_mlp_185863_v1"
            or training.get("observation_version") != offline.adapter.OBSERVATION_VERSION
            or training.get("value_perspective") != "player_to_move"):
        raise ValueError("Incompatible checkpoint architecture/observation/value contract")
    settings = Settings(**training["settings"])
    if training.get("game") != settings.game_string:
        raise ValueError("Saved game and settings disagree")
    game = offline.pyspiel.load_game(settings.game_string)
    dataset = load_dataset(config.openings, game)
    if config.games > 2 * len(dataset.openings):
        raise ValueError("Requested games exceed twice the opening count; repeats are forbidden")
    started = time.perf_counter()
    items, _ = offline.prepare_checkpoints(
        run, offline.EvaluationConfig(checkpoints=[config.checkpoint]), game, training,
        lambda event, **values: print(json.dumps({"event": event, **values}), flush=True))
    item, = items
    selected = dataset.openings[:config.games // 2]
    provenance = {
        "benchmark_id": dataset.benchmark_id, "path": dataset.path,
        "file_sha256": dataset.file_sha256, "content_sha256": dataset.content_sha256,
        "opening_count": len(selected), "available_openings": len(dataset.openings),
        "games_per_opening": 2,
        "selection": "entire_dataset" if len(selected) == len(dataset.openings) else "prefix_smoke_test",
        "selected_openings_sha256": digest([{"opening_id": o.opening_id, "actions": o.actions,
                                             "state_sha256": o.state_sha256} for o in selected])}
    sources = [
        "experiments/checkpoint_strength/neural_vs_mcts.py",
        "experiments/checkpoint_strength/modal_neural_vs_mcts.py",
        "experiments/checkpoint_strength/head_to_head.py",
        "experiments/checkpoint_strength/opening_dataset.py",
        "experiments/checkpoint_strength/metrics.py",
        "experiments/alpha_zero/offline_evaluate.py",
        "experiments/alpha_zero/modal_common.py",
        "experiments/alpha_zero/modal_evaluate.py",
        "experiments/agent_benchmark/random_vs_greedy_hstar.py",
        "Algorithms/ayo_mcts.py", "Algorithms/alpha_zero/evaluation.py",
        "Algorithms/alpha_zero/inference.py", "Algorithms/alpha_zero/config.py",
        "Algorithms/alpha_zero/game.py", "Algorithms/alpha_zero/runtime.py",
        "Model/ayo_olopon/ayo_olopon.py", "open_spiel/open_spiel/python/algorithms/mcts.py",
        "open_spiel/open_spiel/python/algorithms/alpha_zero/evaluator.py",
        "open_spiel/open_spiel/python/algorithms/alpha_zero/model_linen.py"]
    manifest = {
        "experiment": "neural_checkpoint_vs_vanilla_mcts", "schema_version": 1,
        "status": "running", "run": str(run), "config": asdict(config),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "game": settings.game_string, "uct_c": settings.uct_c,
        "run_manifest_sha256": offline.source_hash(run / "manifest.json"),
        "model_contract": {key: training[key] for key in (
            "architecture_version", "observation_version", "value_perspective")},
        "checkpoints": [{key: value for key, value in item.items() if key != "model"}],
        "opening_dataset": provenance,
        "seed_schedule": "seed + global opening index; same seed for each swapped-seat pair",
        "evaluation_policy": {
            f"C{config.checkpoint}": neural_policy_settings(config),
            "MCTS": {"search": "UCT", "simulations": config.mcts_budget, "rollouts_per_leaf": 1,
                "leaf_value": "uniform random rollout to terminal", "prior": "uniform legal actions",
                "neural_policy": False, "neural_value": False, "solve": False,
                "action_selection": "OpenSpiel MCTSBot.step (best_child sort_key)"}},
        "comparison_note": (f"Neural action mode={config.neural_mode}. "
                            f"Simulations per move: C{config.checkpoint}={config.simulations}, "
                            f"MCTS={config.mcts_budget}. Simulation budgets do not equate runtime/compute; "
                            "direct policy mode uses only current-state policy probabilities, without search or value-based action selection."),
        "execution": {"workers": min(config.workers, len(selected)),
                      "unit": "complete swapped-seat opening pair", "start_method": "spawn",
                      "final_record_order": "global game_id", "hardware": hardware},
        "score_definition": "win=1, draw=0.5, loss=0",
        "code_sha256": {name: offline.source_hash(ROOT / name) for name in sources},
        "packages": {name: version(name) for name in ("open_spiel", "numpy", "jax", "flax")},
        "completed_games": 0, "total_games": config.games,
        "artifacts": {"manifest": "results.manifest.json", "games": "results.games.jsonl",
                      "summary": "results.json"}}
    output.mkdir(parents=True, exist_ok=False)

    def commit():
        temporary = output / "results.manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "results.manifest.json")
        if on_progress is not None:
            on_progress(dict(manifest))

    commit()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print(json.dumps({"event": "neural_vs_mcts_heartbeat",
                              "completed_games": manifest["completed_games"],
                              "total_games": config.games}), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    records = []
    try:
        with (output / "results.games.jsonl").open("x", encoding="utf-8") as stream:
            def record_pair(pair):
                for record in pair:
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
                records.extend(pair)
                manifest["completed_games"] = len(records)
                print(json.dumps({"event": "neural_vs_mcts_pair_completed",
                                  "completed_games": len(records), "total_games": config.games,
                                  "opening_id": pair[0]["opening_id"]}), flush=True)
                if len(records) % 20 == 0:
                    commit()

            if config.workers == 1:
                for index, opening in enumerate(selected):
                    record_pair(play_pair(game, item["model"], config, settings.uct_c,
                                          index, opening, dataset.content_sha256))
            else:
                with ProcessPoolExecutor(
                        max_workers=min(config.workers, len(selected)),
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=initialize_worker,
                        initargs=(settings.game_string, item["checkpoint_source"],
                                  item["checkpoint_sha256"], asdict(config), settings.uct_c)) as pool:
                    futures = [pool.submit(worker_pair, index, opening, dataset.content_sha256)
                               for index, opening in enumerate(selected)]
                    for future in as_completed(futures):
                        record_pair(future.result())
        records.sort(key=lambda r: r["game_id"])
        if [r["game_id"] for r in records] != list(range(config.games)):
            raise RuntimeError("Missing or duplicate game IDs")
        if offline.source_hash(item["checkpoint_source"]) != item["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint changed during evaluation")
        if offline.source_hash(run / "manifest.json") != manifest["run_manifest_sha256"]:
            raise RuntimeError("Training manifest changed during evaluation")
        temporary = output / "results.games.jsonl.tmp"
        temporary.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        temporary.replace(output / "results.games.jsonl")
        report = make_report(config, provenance, records)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started,
                        games_sha256=offline.source_hash(output / "results.games.jsonl"),
                        summary_sha256=offline.source_hash(output / "results.json"))
        commit()
        print(json.dumps({"event": "neural_vs_mcts_completed", "policies": report["policies"],
                          "score_intervals": report["score_intervals"]}), flush=True)
        return report
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        commit()
        raise
    finally:
        stop.set()
        thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = NeuralVsMCTSConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    run_neural_vs_mcts(args.run, config, args.output)


if __name__ == "__main__":
    main()
