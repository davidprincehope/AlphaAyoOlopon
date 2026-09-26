"""Compare H*-evaluated Ayo minimax at depths 2-5 against random play.

Each depth writes a game-by-game JSONL log and checkpoint summary, so completed
depths remain available if a later depth is interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import open_spiel

# This checkout keeps the Python OpenSpiel sources beneath open_spiel/open_spiel.
OPEN_SPIEL_PYTHON = str(ROOT / "open_spiel" / "open_spiel")
if OPEN_SPIEL_PYTHON not in open_spiel.__path__:
  open_spiel.__path__.append(OPEN_SPIEL_PYTHON)

import pyspiel

from Algorithms import H_star_ayo
from Algorithms import ayo_minimax
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers the game


DEFAULT_GAMES = 10_000
DEFAULT_DEPTHS = (1, 2, 3, 4, 5)
DEFAULT_SEED = 20260924 
DEFAULT_MAX_ACTIONS = 300
POLICY_MINIMAX = "MINIMAX_HSTAR"
POLICY_RANDOM = "RAND"
DEPTH_1_BASELINE = {
    "source": "experiments/agent_benchmark/results/random_vs_greedy_hstar_10000.json",
    "games": 10_000,
    "master_seed": 20260923,
    "score_rate": 0.80065,
    "win_rate": 0.7566,
}


def play_game(game, bot, depth: int, game_id: int, assignment: tuple[str, str],
              seed: int, max_actions: int) -> dict:
  state = game.new_initial_state()
  rng = random.Random(seed)
  started = time.perf_counter()
  minimax_seconds = 0.0
  actions = 0

  while not state.is_terminal() and actions < max_actions:
    player = int(state.current_player())
    if assignment[player] == POLICY_MINIMAX:
      decision_started = time.perf_counter()
      action = int(bot.step(state))
      minimax_seconds += time.perf_counter() - decision_started
    else:
      action = int(rng.choice(list(state.legal_actions())))
    if action not in state.legal_actions():
      raise RuntimeError(f"{assignment[player]} selected illegal action {action}")
    state.apply_action(action)
    actions += 1

  if state.is_terminal():
    returns = [float(value) for value in state.returns()]
    report = state.last_relay_report
    if report and report.get("reason") in {
        "repeated_relay_state", "relay_lap_limit_exceeded"
    }:
      termination = "repetition"
    elif returns == [0.0, 0.0]:
      termination = "terminal_draw"
    else:
      termination = "terminal_win"
    truncated = False
  else:
    difference = int(state.captured[0] - state.captured[1])
    returns = ([1.0, -1.0] if difference > 0 else
               [-1.0, 1.0] if difference < 0 else [0.0, 0.0])
    termination = "action_limit"
    truncated = True

  minimax_seat = assignment.index(POLICY_MINIMAX)
  outcome_value = returns[minimax_seat]
  outcome = ("win" if outcome_value > 0 else
             "loss" if outcome_value < 0 else "draw")
  return {
      "depth": depth,
      "game_id": game_id,
      "seed": seed,
      "player_0_policy": assignment[0],
      "player_1_policy": assignment[1],
      "minimax_seat": minimax_seat,
      "minimax_outcome": outcome,
      "returns_by_player": returns,
      "termination_reason": termination,
      "truncated": truncated,
      "game_length": actions,
      "final_captured": [int(value) for value in state.captured],
      "minimax_decision_seconds": minimax_seconds,
      "elapsed_seconds": time.perf_counter() - started,
  }


def summarize(records: list[dict]) -> dict:
  outcomes = Counter(record["minimax_outcome"] for record in records)
  terminations = Counter(record["termination_reason"] for record in records)
  games = len(records)
  wins, draws, losses = (outcomes[name] for name in ("win", "draw", "loss"))
  return {
      "games": games,
      "wins": wins,
      "draws": draws,
      "losses": losses,
      "win_rate": wins / games if games else None,
      "score_rate": (wins + 0.5 * draws) / games if games else None,
      "seat_counts": {
          "0": sum(record["minimax_seat"] == 0 for record in records),
          "1": sum(record["minimax_seat"] == 1 for record in records),
      },
      "termination_reasons": dict(terminations),
      "average_game_length": (
          sum(record["game_length"] for record in records) / games
          if games else None
      ),
      "total_minimax_decision_seconds": sum(
          record["minimax_decision_seconds"] for record in records
      ),
      "average_minimax_decision_seconds_per_game": (
          sum(record["minimax_decision_seconds"] for record in records) / games
          if games else None
      ),
      "total_elapsed_seconds": sum(record["elapsed_seconds"] for record in records),
  }


def run_depth(game, depth: int, games: int, seed: int, max_actions: int,
              output_dir: Path, resume: bool = False) -> dict:
  bot = ayo_minimax.make_bot(
      game, maximum_depth=depth, value_function=H_star_ayo.evaluate_state
  )
  log_path = output_dir / f"depth_{depth}_games_{games}.jsonl"
  summary_path = output_dir / f"depth_{depth}_summary.json"
  assignments = ((POLICY_MINIMAX, POLICY_RANDOM),
                 (POLICY_RANDOM, POLICY_MINIMAX))
  records = []

  if resume and log_path.exists():
    with log_path.open("r", encoding="utf-8") as stream:
      for line_number, line in enumerate(stream, start=1):
        if not line.strip():
          continue
        record = json.loads(line)
        expected_game_id = len(records)
        if record.get("depth") != depth or record.get("game_id") != expected_game_id:
          raise ValueError(
              f"Cannot resume {log_path}: line {line_number} is not the "
              f"expected depth-{depth} record for game {expected_game_id}."
          )
        records.append(record)
    if len(records) > games:
      raise ValueError(
          f"Cannot resume {log_path}: it already contains {len(records)} "
          f"games, but target is {games}."
      )
    file_mode = "a"
    start_game_id = len(records)
  else:
    file_mode = "w"
    start_game_id = 0

  with log_path.open(file_mode, encoding="utf-8", newline="\n") as stream:
    for game_id in range(start_game_id, games):
      pair_id = game_id // 2
      assignment = assignments[game_id % 2]
      game_seed = seed + pair_id
      record = play_game(
          game, bot, depth, game_id, assignment, game_seed, max_actions
      )
      records.append(record)
      stream.write(json.dumps(record, sort_keys=True) + "\n")
      if (game_id + 1) % 100 == 0:
        stream.flush()
        print(f"depth {depth}: {game_id + 1}/{games} games", flush=True)

  result = {
      "experiment": "ayo_minimax_depth_vs_random",
      "status": "completed",
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "depth": depth,
      "evaluator": "Algorithms/H_star_ayo.py:evaluate_state",
      "games": games,
      "master_seed": seed,
      "max_actions": max_actions,
      "starting_position": "standard_initial_state",
      "seat_balance": {POLICY_MINIMAX: games // 2, POLICY_RANDOM: games // 2},
      "random_seed_protocol": "same seed for paired seat assignments; pair seeds increment from master seed",
      "tie_break": "first action in OpenSpiel legal-action order",
      "action_limit_score": "captured-seed differential; reported as action_limit, not natural_draw",
      "log_file": log_path.name,
      "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
      "summary": summarize(records),
  }
  summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                      help="games per depth; must be even for seat balance")
  parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
  parser.add_argument(
      "--resume", action="store_true",
      help="Resume existing per-depth JSONL logs and append missing games.",
  )
  parser.add_argument(
      "--output-dir", type=Path,
      default=ROOT / "experiments" / "minimax_depth_vs_random" / "results",
  )
  args = parser.parse_args()
  if args.games < 1 or args.games % 2:
    parser.error("--games must be a positive even number")
  if not args.depths or any(depth < 1 for depth in args.depths):
    parser.error("--depths must contain positive depths")
  if len(set(args.depths)) != len(args.depths):
    parser.error("--depths must not contain duplicates")
  if args.max_actions < 1:
    parser.error("--max-actions must be positive")

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  results = []
  for depth in args.depths:
    print(f"Starting depth {depth}: {args.games} games; seed {args.seed}",
          flush=True)
    result = run_depth(
        game, depth, args.games, args.seed, args.max_actions, output_dir,
        resume=args.resume,
    )
    results.append(result)
    print(json.dumps({"depth": depth, **result["summary"]}, sort_keys=True),
          flush=True)

  combined = {
      "experiment": "ayo_minimax_depth_vs_random",
      "status": "completed",
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "games_per_depth": args.games,
      "depths": args.depths,
      "evaluator": "Algorithms/H_star_ayo.py:evaluate_state",
      "master_seed": args.seed,
      "max_actions": args.max_actions,
      "existing_depth_1_greedy_baseline": DEPTH_1_BASELINE,
      "results": [
          {
              "depth": result["depth"],
              **result["summary"],
              "score_rate_change_vs_depth_1": (
                  result["summary"]["score_rate"]
                  - DEPTH_1_BASELINE["score_rate"]
              ),
          }
          for result in results
      ],
  }
  combined_path = output_dir / "experiment_summary.json"
  combined_path.write_text(
      json.dumps(combined, indent=2) + "\n", encoding="utf-8"
  )
  print(f"Complete. Summary: {combined_path}", flush=True)


if __name__ == "__main__":
  main()
