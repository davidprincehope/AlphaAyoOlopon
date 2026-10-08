"""Parallel frozen-checkpoint matches against the unchanged GREEDY_HSTAR agent."""
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
from experiments.agent_benchmark.agents import make_agent
from experiments.agent_benchmark.random_vs_greedy_hstar import play_game, summarize
from experiments.checkpoint_strength.head_to_head import paired_score_interval
from experiments.checkpoint_strength.metrics import first_player_advantage
from experiments.checkpoint_strength.opening_dataset import ROOT, digest, load_dataset, replay


OPPONENT = {"name": "GREEDY_HSTAR", "params": {}}
SOURCES = (
    "experiments/checkpoint_strength/greedy_progression.py",
    "experiments/checkpoint_strength/opening_dataset.py",
    "experiments/checkpoint_strength/head_to_head.py",
    "experiments/checkpoint_strength/metrics.py",
    "experiments/alpha_zero/offline_evaluate.py",
    "experiments/alpha_zero/modal_common.py",
    "experiments/alpha_zero/modal_evaluate.py",
    "experiments/agent_benchmark/agents.py",
    "experiments/agent_benchmark/random_vs_greedy_hstar.py",
    "Algorithms/H_star_ayo.py", "Algorithms/eoh_ayo.py",
    "Algorithms/alpha_zero/evaluation.py", "Algorithms/alpha_zero/inference.py",
    "Algorithms/alpha_zero/config.py", "Algorithms/alpha_zero/game.py",
    "Algorithms/alpha_zero/runtime.py",
    "Model/ayo_olopon/ayo_olopon.py",
    "open_spiel/open_spiel/python/algorithms/mcts.py",
    "open_spiel/open_spiel/python/algorithms/alpha_zero/evaluator.py",
    "open_spiel/open_spiel/python/algorithms/alpha_zero/model_linen.py",
)


@dataclass(frozen=True)
class GreedyCheckpointConfig:
    checkpoint: int
    simulations: int
    games: int = 400
    seed: int = 20261003
    workers: int = 8
    openings: str = "experiments/checkpoint_strength/dataset/ayo_mixed_v2_200.json"

    def __post_init__(self):
        offline.EvaluationConfig(checkpoints=[self.checkpoint], simulations=self.simulations,
                                 games=self.games, seed=self.seed, openings=self.openings)
        if type(self.workers) is not int or not 1 <= self.workers <= 28:
            raise ValueError("workers must be an integer between 1 and 28")
        if self.seed + self.games // 2 - 1 >= 2**32:
            raise ValueError("Paired seeds must fit NumPy RandomState")


def opening_provenance(dataset, games):
    selected = dataset.openings[:games // 2]
    return {
        "protocol": "fixed_openings_v1", "benchmark_id": dataset.benchmark_id,
        "path": dataset.path, "file_sha256": dataset.file_sha256,
        "content_sha256": dataset.content_sha256, "opening_count": len(selected),
        "available_openings": len(dataset.openings), "games_per_opening": 2,
        "selection": "entire_dataset" if len(selected) == len(dataset.openings) else "prefix_smoke_test",
        "selected_openings_sha256": digest([{"opening_id": o.opening_id, "actions": o.actions,
                                             "state_sha256": o.state_sha256} for o in selected]),
    }


def play_pair(game, model, config, uct_c, index, opening, dataset_hash):
    factories = {
        "ALPHAZERO": lambda game, seed: make_neural_agent(game, model, config.simulations, uct_c, seed),
        "OPPONENT": lambda game, seed: make_agent(game, seed, OPPONENT),
    }
    rows = []
    for seat, assignment in enumerate((("ALPHAZERO", "OPPONENT"), ("OPPONENT", "ALPHAZERO"))):
        row = play_game(2 * index + seat, assignment, config.seed + index,
                        game.max_game_length(), game=game, agent_factories=factories, opening=opening)
        row.update(opening_pair_id=index, opening_dataset_sha256=dataset_hash)
        rows.append(row)
    return rows


_WORKER = None


def initialize_worker(game_string, checkpoint_path, checkpoint_hash, configuration, uct_c):
    global _WORKER
    if offline.source_hash(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("Worker checkpoint hash differs from preflight")
    game = offline.pyspiel.load_game(game_string)
    model = offline.InferenceModel(offline.load_parameters(checkpoint_path))
    state = game.new_initial_state()
    model.inference(state.observation_tensor(), state.legal_actions_mask())
    _WORKER = game, model, GreedyCheckpointConfig(**configuration), uct_c


def worker_pair(index, opening, dataset_hash):
    return play_pair(*_WORKER, index, opening, dataset_hash)


def report_from_records(config, provenance, records):
    summary = summarize(records)
    stats = summary["policies"]["ALPHAZERO"]
    summary["by_seat"] = {
        str(seat): summarize([row for row in records if row[f"player_{seat}_policy"] == "ALPHAZERO"])
        for seat in (0, 1)
    }
    by_seat = {f"P{seat}": summary["by_seat"][str(seat)]["policies"]["ALPHAZERO"] for seat in (0, 1)}
    return {
        "status": "completed", "checkpoint_id": config.checkpoint,
        "training_step": config.checkpoint, "learner_round": config.checkpoint,
        "opponent": OPPONENT, "mcts_simulations": config.simulations, "evaluation_seed": config.seed,
        **{key: stats[key] for key in ("games", "wins", "draws", "losses", "win_rate", "score_rate",
                                     "average_game_length")},
        "player0_score_rate": by_seat["P0"]["score_rate"],
        "player1_score_rate": by_seat["P1"]["score_rate"], "by_seat": by_seat,
        "termination_reasons": summary["termination_reasons"], "summary": summary, "matches": records,
        "opening_dataset": provenance, "first_player_advantage": first_player_advantage(records),
        "by_opening_depth": {str(depth): summarize([r for r in records if r["opening_plies"] == depth])["policies"]
                             for depth in sorted({r["opening_plies"] for r in records})},
        "score_interval": paired_score_interval(records, "ALPHAZERO", config.seed),
    }


def validate_reports(rows, run, dataset, budget, expected_steps, *, replay_games=False):
    """Accept complete, compatible checkpoint records and recompute their statistics."""
    if [row["learner_round"] for row in rows] != list(expected_steps):
        raise ValueError("Reports must contain exactly the expected ordered checkpoints")
    game = offline.pyspiel.load_game(json.loads((Path(run) / "manifest.json").read_text())["game"])
    for row in rows:
        step = row["learner_round"]
        config = GreedyCheckpointConfig(checkpoint=step, simulations=budget)
        if (row["status"] != "completed" or row["games"] != 400 or len(row["matches"]) != 400
                or row["opponent"] != OPPONENT or row["mcts_simulations"] != budget
                or row["evaluation_seed"] != config.seed
                or row["opening_dataset"]["content_sha256"] != dataset.content_sha256
                or offline.source_hash(Path(run) / "inference-checkpoints" / f"step-{step:06d}.npz")
                    != row["checkpoint_sha256"]):
            raise ValueError(f"Incompatible saved C{step} report at {budget} simulations")
        matches = row["matches"]
        if [match["game_id"] for match in matches] != list(range(400)):
            raise ValueError("Missing or duplicate game IDs in saved results")
        for match in matches:
            index, seat = divmod(match["game_id"], 2)
            opening = dataset.openings[index]
            assignment = ("ALPHAZERO", "OPPONENT") if seat == 0 else ("OPPONENT", "ALPHAZERO")
            if (match["opening_id"] != opening.opening_id or match["opening_pair_id"] != index
                    or match["seed"] != config.seed + index
                    or match["opening_dataset_sha256"] != dataset.content_sha256
                    or match["opening_state_sha256"] != opening.state_sha256
                    or match["opening_actions"] != list(opening.actions)
                    or (match["player_0_policy"], match["player_1_policy"]) != assignment):
                raise ValueError("Saved opening/seed/seat protocol differs")
            if replay_games:
                state = replay(game, match["opening_actions"] + match["continuation_actions"])
                if (not state.is_terminal() or list(state.returns()) != match["returns_by_player"]
                        or list(state.captured) != match["final_captured"]
                        or state.move_number() != match["game_length"]):
                    raise ValueError("Saved game does not reproduce its terminal outcome")
        recalculated = report_from_records(config, row["opening_dataset"], matches)
        for key in ("wins", "draws", "losses", "score_rate", "win_rate", "by_seat"):
            if recalculated[key] != row[key]:
                raise ValueError(f"Saved statistics mismatch for C{step}: {key}")
        row["score_interval"] = recalculated["score_interval"]
    return rows


def run_checkpoint(run, config, output, *, on_progress=None, hardware=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run):
        raise ValueError("Results must be outside the source training run")
    if output.exists():
        raise FileExistsError(output)
    training = json.loads((run / "manifest.json").read_text())
    settings = Settings(**training["settings"])
    if (training["game"] != settings.game_string
            or training["architecture_version"] != "ayo_mlp_185863_v1"
            or training["observation_version"] != offline.adapter.OBSERVATION_VERSION
            or training["value_perspective"] != "player_to_move"):
        raise ValueError("Incompatible saved model/game contract")
    game = offline.pyspiel.load_game(settings.game_string)
    dataset = load_dataset(config.openings, game)
    if config.games > 2 * len(dataset.openings):
        raise ValueError("Requested games exceed twice the opening count")
    selected = dataset.openings[:config.games // 2]
    items, _ = offline.prepare_checkpoints(
        run, offline.EvaluationConfig(checkpoints=[config.checkpoint]), game, training,
        lambda event, **values: print(json.dumps({"event": event, **values}), flush=True))
    item, = items
    provenance = opening_provenance(dataset, config.games)
    manifest = {
        "experiment": "checkpoint_progression_vs_greedy_hstar", "schema_version": 1,
        "status": "running", "config": asdict(config), "run": str(run),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "game": settings.game_string, "uct_c": settings.uct_c,
        "checkpoint": {key: value for key, value in item.items() if key != "model"},
        "run_manifest_sha256": offline.source_hash(run / "manifest.json"),
        "opening_dataset": provenance,
        "evaluation_policy": {"opponent": OPPONENT, "simulations": config.simulations,
                              "dirichlet_noise": None, "temperature_sampling": False,
                              "action_selection": "maximum visit count, lowest action ID breaks ties"},
        "seed_schedule": "seed + global opening index; same seed for both seat assignments",
        "execution": {"workers": config.workers, "start_method": "spawn", "unit": "complete opening pair",
                      "hardware": hardware},
        "code_sha256": {name: offline.source_hash(ROOT / name) for name in SOURCES},
        "packages": {name: version(name) for name in ("open_spiel", "numpy", "jax", "flax")},
        "completed_games": 0, "total_games": config.games,
    }
    output.mkdir(parents=True)
    started = time.perf_counter()

    def commit():
        temporary = output / "results.manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "results.manifest.json")
        if on_progress is not None:
            on_progress(dict(manifest))

    commit()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print(json.dumps({"event": "greedy_checkpoint_heartbeat", "checkpoint": config.checkpoint,
                              "simulations": config.simulations, "completed_games": manifest["completed_games"],
                              "total_games": config.games}), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    records = []
    try:
        with (output / "results.games.jsonl").open("x") as stream:
            def record_pair(pair):
                for row in pair:
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                records.extend(pair)
                manifest["completed_games"] = len(records)
                if len(records) % 20 == 0 or len(records) == config.games:
                    commit()
                    print(json.dumps({"event": "greedy_checkpoint_progress", "checkpoint": config.checkpoint,
                                      "simulations": config.simulations, "completed_games": len(records),
                                      "total_games": config.games}), flush=True)

            if config.workers == 1:
                for index, opening in enumerate(selected):
                    record_pair(play_pair(game, item["model"], config, settings.uct_c,
                                          index, opening, dataset.content_sha256))
            else:
                with ProcessPoolExecutor(max_workers=min(config.workers, len(selected)),
                                         mp_context=multiprocessing.get_context("spawn"),
                                         initializer=initialize_worker,
                                         initargs=(settings.game_string, item["checkpoint_source"],
                                                   item["checkpoint_sha256"], asdict(config), settings.uct_c)) as pool:
                    futures = [pool.submit(worker_pair, index, opening, dataset.content_sha256)
                               for index, opening in enumerate(selected)]
                    for future in as_completed(futures):
                        record_pair(future.result())
        records.sort(key=lambda row: row["game_id"])
        if [row["game_id"] for row in records] != list(range(config.games)):
            raise RuntimeError("Missing or duplicate game IDs")
        if offline.source_hash(item["checkpoint_source"]) != item["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint changed during evaluation")
        if offline.source_hash(run / "manifest.json") != manifest["run_manifest_sha256"]:
            raise RuntimeError("Source training manifest changed during evaluation")
        temporary = output / "results.games.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in records))
        temporary.replace(output / "results.games.jsonl")
        report = report_from_records(config, provenance, records)
        report.update({key: value for key, value in item.items() if key != "model"})
        report["checkpoint_elapsed_seconds"] = time.perf_counter() - started
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started,
                        games_sha256=offline.source_hash(output / "results.games.jsonl"),
                        summary_sha256=offline.source_hash(output / "results.json"))
        commit()
        print(json.dumps({"event": "greedy_checkpoint_completed", "checkpoint": config.checkpoint,
                          "simulations": config.simulations, "score_rate": report["score_rate"]}), flush=True)
        return report
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        commit()
        raise
    finally:
        stop.set()
        thread.join(timeout=1)
