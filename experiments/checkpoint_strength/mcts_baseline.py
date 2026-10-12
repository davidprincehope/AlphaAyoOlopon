"""Seeded vanilla UCT/random-rollout baseline on the saved opening benchmark."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
from experiments.agent_benchmark.agents import make_agent
from experiments.agent_benchmark.random_vs_greedy_hstar import run_matches
from experiments.checkpoint_strength.metrics import first_player_advantage


def run_baseline(game, dataset, output, *, seed=20261003, uct_c=1.5,
                 budgets=(64, 128), on_progress=None):
    if not budgets or any(type(b) is not int or b < 2 for b in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Simulation budgets must be distinct integers >= 2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    games = 2 * len(dataset.openings)
    started = time.perf_counter()
    provenance = {"benchmark_id": dataset.benchmark_id, "content_sha256": dataset.content_sha256,
                  "file_sha256": dataset.file_sha256, "opening_count": len(dataset.openings),
                  "opening_order": [opening.opening_id for opening in dataset.openings]}
    status = {"status": "running", "experiment": "vanilla_mcts_vs_greedy_hstar",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "budgets": list(budgets),
              "completed_budgets": [], "completed_games": 0, "total_games": games * len(budgets),
              "games_per_budget": games, "opening_dataset": provenance, "seed": seed,
              "seed_schedule": "seed + opening index; same seed in both seats and both budgets",
              "game": str(game), "search": "UCT", "uct_c": uct_c, "rollouts_per_leaf": 1,
              "prior": "uniform legal actions", "leaf_value": "uniform random rollout to terminal",
              "neural_policy": False, "neural_value": False, "solve": False,
              "action_selection": "OpenSpiel MCTSBot.step (best_child sort_key)",
              "confidence_intervals": None,
              "uncertainty_note": "Paired seat games share an opening; no confidence intervals computed."}
    root = Path(__file__).resolve().parents[2]
    files = ["Algorithms/ayo_mcts.py", "Algorithms/H_star_ayo.py", "Algorithms/eoh_ayo.py",
             "Algorithms/alpha_zero/game.py", "Model/ayo_olopon/ayo_olopon.py",
             "experiments/agent_benchmark/agents.py", "experiments/agent_benchmark/random_vs_greedy_hstar.py",
             "experiments/checkpoint_strength/opening_dataset.py", "experiments/checkpoint_strength/metrics.py",
             "experiments/checkpoint_strength/mcts_baseline.py",
             "open_spiel/open_spiel/python/algorithms/mcts.py"]
    status["code_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    def save():
        temporary = output / "manifest.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "manifest.json")
    def notify(event):
        print(json.dumps(event), flush=True)
        if on_progress:
            on_progress(event)
    save()
    try:
        with (output / "results.jsonl").open("x", encoding="utf-8") as results:
            for budget in budgets:
                status.update(current_simulations=budget, budget_completed_games=0)
                directory = output / str(budget)
                directory.mkdir()
                save()
                notify({"event": "budget_started", "simulations": budget})
                with (directory / "games.jsonl").open("x", encoding="utf-8") as stream:
                    def record_game(record):
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        status["completed_games"] += 1
                        status["budget_completed_games"] += 1
                        save()
                        notify({"event": "game_completed", "simulations": budget,
                                "completed_games": status["completed_games"], "total_games": status["total_games"],
                                "budget_completed_games": status["budget_completed_games"],
                                "opening_id": record["opening_id"], "opening_plies": record["opening_plies"],
                                "seat": "P0" if record["player_0_policy"] == "MCTS" else "P1",
                                "outcome": record["policy_outcome"]["MCTS"]})
                    mcts_spec = {"name": "MCTS", "params": {"simulations": budget,
                                 "rollouts_per_leaf": 1, "uct_c": uct_c}}
                    factories = {"MCTS": lambda game, seed: make_agent(game, seed, mcts_spec),
                                 "GREEDY_HSTAR": lambda game, seed: make_agent(game, seed, {"name": "GREEDY_HSTAR"})}
                    summary, records = run_matches(game, factories, games, seed,
                                                   openings=dataset, on_game=record_game)
                report = {"status": "completed", "simulations": budget, "seed": seed,
                          "agent": "MCTS", "opponent": "GREEDY_HSTAR", "opening_dataset": provenance,
                          **summary["policies"]["MCTS"], "summary": summary,
                          "by_seat": {f"P{seat}": summary["by_seat"][str(seat)]["policies"]["MCTS"]
                                      for seat in (0, 1)},
                          "first_player_advantage": first_player_advantage(records)}
                (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                results.write(json.dumps(report) + "\n")
                results.flush()
                status["completed_budgets"].append(budget)
                save()
                notify({"event": "budget_completed", "simulations": budget,
                        "wins": report["wins"], "draws": report["draws"], "losses": report["losses"],
                        "win_rate": report["win_rate"], "score_rate": report["score_rate"],
                        "first_player_advantage": report["first_player_advantage"]})
        status.update(status="completed", current_simulations=None,
                      elapsed_seconds=time.perf_counter() - started,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        notify({"event": "experiment_completed", "completed_games": status["completed_games"]})
        return status
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        save()
        raise
