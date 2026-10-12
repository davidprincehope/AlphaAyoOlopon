"""Paired C600 versus minimax evaluation, with resumable opening-pair artifacts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from experiments.alpha_zero import offline_evaluate as offline
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import make_neural_agent
from Algorithms.ayo_minimax import make_bot
from experiments.agent_benchmark.random_vs_greedy_hstar import play_game, summarize
from experiments.checkpoint_strength.head_to_head import paired_score_interval
from experiments.checkpoint_strength.opening_dataset import ROOT, load_dataset, replay

WORKER = None


def initialize(game_string, checkpoint_source, checkpoint_hash, config, uct_c):
    global WORKER
    if offline.source_hash(checkpoint_source) != checkpoint_hash:
        raise ValueError("Worker checkpoint hash mismatch")
    game = offline.pyspiel.load_game(game_string)
    model = offline.InferenceModel(offline.load_parameters(checkpoint_source))
    state = game.new_initial_state()
    model.inference(state.observation_tensor(), state.legal_actions_mask())
    WORKER = game, model, config, uct_c


def play_opening(index, opening, dataset_hash):
    game, model, config, uct_c = WORKER
    factories = {
        "C600": lambda game, seed: make_neural_agent(game, model, config.simulations, uct_c, seed),
        "MINIMAX": lambda game, seed: make_bot(game, **config.opponent["params"]),
    }
    records = []
    for seat, assignment in enumerate((("C600", "MINIMAX"), ("MINIMAX", "C600"))):
        record = play_game(2 * index + seat, assignment, config.seed + index,
                           game.max_game_length(), game=game, agent_factories=factories, opening=opening)
        record.update(opening_pair_id=index, opening_dataset_sha256=dataset_hash)
        records.append(record)
    return records


def verify_pair(game, pair, index, opening, config, dataset_hash):
    if len(pair) != 2:
        raise ValueError("Incomplete opening pair")
    for seat, record in enumerate(pair):
        assignment = ("C600", "MINIMAX") if seat == 0 else ("MINIMAX", "C600")
        if (record["game_id"] != 2 * index + seat or record["opening_pair_id"] != index
                or record["seed"] != config.seed + index or record["opening_id"] != opening.opening_id
                or record["opening_state_sha256"] != opening.state_sha256
                or record["opening_actions"] != list(opening.actions)
                or record["opening_dataset_sha256"] != dataset_hash
                or (record["player_0_policy"], record["player_1_policy"]) != assignment):
            raise ValueError("Opening, seed or seat protocol mismatch")
        state = replay(game, record["opening_actions"] + record["continuation_actions"])
        if (not state.is_terminal() or record["truncated"]
                or list(state.returns()) != record["returns_by_player"]
                or list(state.captured) != record["final_captured"]
                or state.move_number() != record["game_length"]):
            raise ValueError("Game replay does not match the recorded terminal result")
        for player, name in enumerate(assignment):
            value = state.returns()[player]
            if (record["policy_returns"][name] != value or record["policy_outcome"][name]
                    != ("win" if value > 0 else "loss" if value < 0 else "draw")):
                raise ValueError("Recorded policy outcome mismatch")


def run(run_directory, config, output, workers):
    run_directory, output = Path(run_directory).resolve(), Path(output).resolve()
    if output.is_relative_to(run_directory) or not 1 <= workers <= 28:
        raise ValueError("Use an external results directory and 1-28 workers")
    if config.checkpoints != [600] or config.opponent["name"] != "MINIMAX":
        raise ValueError("This runner evaluates C600 against MINIMAX")
    training = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    settings = Settings(**training["settings"])
    if (training["architecture_version"] != "ayo_mlp_185863_v1"
            or training["observation_version"] != offline.adapter.OBSERVATION_VERSION
            or training["value_perspective"] != "player_to_move" or training["game"] != settings.game_string):
        raise ValueError("Incompatible training contract")
    game = offline.pyspiel.load_game(settings.game_string)
    make_bot(game, **config.opponent["params"])
    dataset = load_dataset(config.openings, game)
    if config.games > 2 * len(dataset.openings):
        raise ValueError("Opening repeats are forbidden")
    selected = dataset.openings[:config.games // 2]
    items, _ = offline.prepare_checkpoints(run_directory, config, game, training,
                                          lambda event, **values: None)
    item, = items
    sources = [
        "Algorithms/H_star_ayo.py", "Algorithms/ayo_minimax.py", "Algorithms/alpha_zero/evaluation.py",
        "Algorithms/alpha_zero/game.py", "Algorithms/alpha_zero/inference.py",
        "Algorithms/alpha_zero/config.py", "Algorithms/alpha_zero/runtime.py",
        "Model/ayo_olopon/ayo_olopon.py", "experiments/alpha_zero/offline_evaluate.py",
        "experiments/agent_benchmark/random_vs_greedy_hstar.py",
        "experiments/checkpoint_strength/neural_vs_minimax.py",
        "experiments/checkpoint_strength/opening_dataset.py", "experiments/checkpoint_strength/head_to_head.py",
        "open_spiel/open_spiel/python/algorithms/minimax.py",
        "open_spiel/open_spiel/python/algorithms/mcts.py",
        "open_spiel/open_spiel/python/algorithms/alpha_zero/evaluator.py",
        "open_spiel/open_spiel/python/algorithms/alpha_zero/model_linen.py",
    ]
    contract = {
        "config": asdict(config), "game": settings.game_string, "uct_c": settings.uct_c,
        "checkpoint_sha256": item["checkpoint_sha256"],
        "run_manifest_sha256": offline.source_hash(run_directory / "manifest.json"),
        "opening_dataset_sha256": dataset.content_sha256,
        "opening_dataset_file_sha256": dataset.file_sha256,
        "code_sha256": {name: offline.source_hash(ROOT / name) for name in sources},
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "evaluation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Resume requires unchanged code, configuration, model and opening book")
    else:
        if any(output.iterdir()):
            raise FileExistsError("Output directory contains unrelated files")
        manifest = {"experiment": "c600_puct64_vs_minimax_depth6", "contract": contract,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "minimax": {"search": "alpha-beta", "depth_unit": "complete player actions",
                                "leaf_evaluator": "clipped H* formula divided by 11",
                                "terminal_utilities": [-1, 0, 1], "tie_break": "first legal action"},
                    "neural": {"search": "PUCT", "dirichlet_noise": None, "temperature_sampling": False,
                               "action_selection": "maximum visits, lowest action ID ties"},
                    "seed_schedule": "20261003 + opening index; both seats share the seed",
                    "workers": workers, "completed_games": 0, "status": "running"}
    pair_directory = output / "opening_pairs"
    pair_directory.mkdir(exist_ok=True)
    records, pending = [], []
    def save_progress():
        manifest["completed_games"] = len(records)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    for index, opening in enumerate(selected):
        path = pair_directory / f"pair-{index:03d}.json"
        if path.exists():
            pair = json.loads(path.read_text(encoding="utf-8"))
            verify_pair(game, pair, index, opening, config, dataset.content_sha256)
            records.extend(pair)
        else:
            pending.append((index, opening))
    manifest["status"] = "running"
    save_progress()
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                 mp_context=multiprocessing.get_context("spawn"), initializer=initialize,
                                 initargs=(settings.game_string, item["checkpoint_source"],
                                           item["checkpoint_sha256"], config, settings.uct_c)) as pool:
            futures = {pool.submit(play_opening, index, opening, dataset.content_sha256): (index, opening)
                       for index, opening in pending}
            for future in as_completed(futures):
                index, opening = futures[future]
                pair = future.result()
                verify_pair(game, pair, index, opening, config, dataset.content_sha256)
                path = pair_directory / f"pair-{index:03d}.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(pair) + "\n", encoding="utf-8")
                temporary.replace(path)
                records.extend(pair)
                save_progress()
                print(json.dumps({"completed_games": len(records), "total_games": config.games,
                                  "policies": summarize(records)["policies"]}), flush=True)
    records.sort(key=lambda record: record["game_id"])
    if [record["game_id"] for record in records] != list(range(config.games)):
        raise ValueError("Missing or duplicate game IDs")
    for name, expected in contract["code_sha256"].items():
        if offline.source_hash(ROOT / name) != expected:
            raise ValueError("Evaluation source changed during the run")
    if (offline.source_hash(item["checkpoint_source"]) != contract["checkpoint_sha256"]
            or offline.source_hash(run_directory / "manifest.json") != contract["run_manifest_sha256"]):
        raise ValueError("Training artifacts changed during evaluation")
    report = {"status": "completed", "checkpoint": 600, "neural_simulations": config.simulations,
              "opponent": config.opponent, "opening_pairs": len(selected),
              "opening_dataset": {"path": dataset.path, "content_sha256": dataset.content_sha256,
                                  "file_sha256": dataset.file_sha256},
              **summarize(records),
              "by_seat": {f"P{seat}": summarize([r for r in records if r[f"player_{seat}_policy"] == "C600"])["policies"]["C600"] for seat in (0, 1)},
              "score_intervals": {name: paired_score_interval(records, name, config.seed) for name in ("C600", "MINIMAX")},
              "verification": {"games_replayed": len(records), "complete_swapped_seat_pairs": True},
              "checkpoint_sha256": item["checkpoint_sha256"]}
    (output / "results.games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    games_sha256=offline.source_hash(output / "results.games.jsonl"),
                    results_sha256=offline.source_hash(output / "results.json"))
    save_progress()
    print(json.dumps({"status": "completed", "policies": report["policies"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.run, offline.load_config(args.config), args.output, args.workers)
