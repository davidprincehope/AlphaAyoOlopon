"""Reproducible greedy-policy experiments for the Ayo heuristic study."""

from __future__ import annotations

import json
import argparse
import hashlib
import random
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import pyspiel

from Model.ayo_olopon import ayo_olopon  # noqa: F401
from Algorithms import ayo_heuristic


OPENING_SEED = 20260921
GAME_SEED = 20260922
OPENING_COUNT = 100
DEVELOPMENT_COUNT = 70
GAME_LENGTH_LIMIT = 300
FROZEN_OPENINGS_PATH = Path(__file__).resolve().parent / "openings" / "opening_positions.json"


@dataclass
class PolicyStats:
  action_count: int = 0
  tied_action_selections: int = 0
  evaluation_seconds: float = 0.0


def _snapshot(state, opening_seed: int, plies: int) -> dict:
  return {
      "opening_seed": opening_seed,
      "plies_from_initial": plies,
      "board": list(state.board),
      "captured": list(state.captured),
      "current_player": int(state.current_player()),
      "is_terminal": bool(state.is_terminal()),
  }


def _state_hash(snapshot: dict) -> str:
  payload = {
      "board": snapshot["board"],
      "captured": snapshot["captured"],
      "current_player": snapshot["current_player"],
  }
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).hexdigest()


def generate_openings(seed=OPENING_SEED) -> list[dict]:
  """Generates 100 unique reachable early/middle-game snapshots."""
  rng = random.Random(seed)
  openings = []
  seen = set()
  game = pyspiel.load_game("ayo_olopon")
  attempts = 0
  while len(openings) < OPENING_COUNT:
    attempts += 1
    opening_seed = rng.randrange(1, 2**31)
    local = random.Random(opening_seed)
    state = game.new_initial_state()
    target_plies = local.randint(4, 32)
    for plies in range(target_plies):
      if state.is_terminal() or not state.legal_actions():
        break
      state.apply_action(local.choice(state.legal_actions()))
    if state.is_terminal() or not state.legal_actions():
      continue
    # 4--12 plies is treated as early game; 13--32 as middle game.
    key = (tuple(state.board), tuple(state.captured), int(state.current_player()))
    if key in seen:
      continue
    seen.add(key)
    openings.append(_snapshot(state, opening_seed, plies + 1))
  openings.sort(key=lambda item: item["opening_seed"])
  for index, opening in enumerate(openings):
    opening["opening_id"] = index
    opening["phase"] = "early" if opening["plies_from_initial"] <= 12 else "middle"
    opening["split"] = "development" if index < DEVELOPMENT_COUNT else "verification"
  return openings


def load_frozen_openings(path=FROZEN_OPENINGS_PATH, manifest_path=None) -> list[dict]:
  """Loads and validates the immutable opening suite."""
  payload = json.loads(Path(path).read_text())
  positions = payload["positions"]
  if len(positions) != OPENING_COUNT:
    raise ValueError(f"expected {OPENING_COUNT} openings, found {len(positions)}")
  ids = [item["opening_id"] for item in positions]
  if ids != list(range(OPENING_COUNT)):
    raise ValueError("opening ids are not a complete ordered range")
  splits = [item["split"] for item in positions]
  if splits.count("development") != DEVELOPMENT_COUNT or splits.count("verification") != OPENING_COUNT - DEVELOPMENT_COUNT:
    raise ValueError("opening split counts do not match the frozen protocol")
  states = set()
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  for item in positions:
    key = (tuple(item["board"]), tuple(item["captured"]), int(item["current_player"]))
    if key in states:
      raise ValueError("duplicate opening state detected")
    states.add(key)
    snapshot_hash = _state_hash(item)
    if "state_hash" in item and item["state_hash"] != snapshot_hash:
      raise ValueError(f"opening {item['opening_id']} state hash mismatch")
    state = state_from_snapshot(game, item)
    if state.is_terminal() or not state.legal_actions():
      raise ValueError(f"opening {item['opening_id']} is not playable and non-terminal")
    item["state_hash"] = snapshot_hash
  if manifest_path and Path(manifest_path).exists():
    manifest = json.loads(Path(manifest_path).read_text())
    expected = manifest.get("opening_state_hashes")
    if expected and expected != [item["state_hash"] for item in positions]:
      raise ValueError("frozen opening hashes do not match the corrected manifest")
  return positions


def state_from_snapshot(game, snapshot):
  state = game.new_initial_state()
  state.board = list(snapshot["board"])
  state.captured = list(snapshot["captured"])
  state._current_player = int(snapshot["current_player"])
  state._positions_since_capture = {state._position_key()}
  return state


def _policy_action(policy_name, state, player, rng, stats: PolicyStats, weights=None):
  legal = list(state.legal_actions())
  if policy_name == "RAND":
    return rng.choice(legal)

  evaluator = ayo_heuristic.candidate_heuristic(policy_name) if weights is None else (
      lambda child, fixed_player: ayo_heuristic.evaluate_state(
          child,
          fixed_player,
          weights=weights,
          terminal_value=max(10.0, sum(abs(float(value)) for value in weights.values()) + 1e-9),
      )
  )
  values = []
  start = time.perf_counter()
  for action in legal:
    values.append((evaluator(state.child(action), player), action))
  stats.evaluation_seconds += time.perf_counter() - start
  best_value = max(value for value, _ in values)
  best_actions = [action for value, action in values if value == best_value]
  stats.tied_action_selections += int(len(best_actions) > 1)
  return min(best_actions)


def play_game(snapshot, player_policies, game_seed, weights=None):
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = state_from_snapshot(game, snapshot)
  rng = random.Random(game_seed)
  stats = {0: PolicyStats(), 1: PolicyStats()}
  actions = []
  error_message = None
  while not state.is_terminal() and len(actions) < GAME_LENGTH_LIMIT:
    player = int(state.current_player())
    policy = player_policies[player]
    stats[player].action_count += 1
    try:
      action = _policy_action(policy, state, player, rng, stats[player], weights)
      state.apply_action(action)
    except Exception as error:  # pragma: no cover - defensive experiment guard
      error_message = repr(error)
      break
    actions.append({"player": player, "policy": policy, "action": int(action)})

  length_limited = not state.is_terminal() and error_message is None
  if error_message is not None:
    returns = [0.0, 0.0]
    termination_reason = "error"
    terminal = False
    natural_draw = False
    truncated = False
  elif length_limited:
    score_difference = state.captured[0] - state.captured[1]
    scoring_returns = [1.0, -1.0] if score_difference > 0 else [-1.0, 1.0] if score_difference < 0 else [0.0, 0.0]
    returns = scoring_returns
    termination_reason = "action_limit"
    terminal = False
    natural_draw = False
    truncated = True
  else:
    returns = state.returns()
    report = state.last_relay_report
    if report and report.get("reason") in {"repeated_relay_state", "relay_lap_limit_exceeded"}:
      termination_reason = "repetition"
    elif returns == [0.0, 0.0]:
      termination_reason = "terminal_draw"
    else:
      termination_reason = "terminal_win"
    terminal = True
    natural_draw = termination_reason == "terminal_draw"
    truncated = False
  winner = next((p for p, value in enumerate(returns) if value > 0), None) if terminal else None
  return {
      "returns": list(returns),
      "terminal": terminal,
      "termination_reason": termination_reason,
      "winner": winner,
      "natural_draw": natural_draw,
      "truncated": truncated,
      "error": error_message,
      "draw": natural_draw,
      "action_count": len(actions),
      "game_length": len(actions),
      "length_limited": length_limited,
      "final_captured": list(state.captured),
      "final_seed_differential": state.captured[0] - state.captured[1],
      "actions": actions,
      "policy_stats": {str(p): asdict(stats[p]) for p in (0, 1)},
  }


def _result_for_policy(game_result, policy, player_assignment):
  player = player_assignment[policy]
  if game_result["truncated"]:
    differential = game_result["final_seed_differential"] * (1 if player == 0 else -1)
    value = 1.0 if differential > 0 else -1.0 if differential < 0 else 0.0
  else:
    value = game_result["returns"][player]
  differential = game_result["final_seed_differential"] * (1 if player == 0 else -1)
  return {
      "win": int(value > 0),
      "draw": int(value == 0),
      "natural_draw": int(game_result["natural_draw"]),
      "truncation": int(game_result["truncated"]),
      "repetition": int(game_result["termination_reason"] == "repetition"),
      "loss": int(value < 0),
      "score": 1.0 if value > 0 else 0.5 if value == 0 else 0.0,
      "seed_differential": differential,
      "game_length": game_result["game_length"],
      "tied_action_selections": game_result["policy_stats"][str(player)]["tied_action_selections"],
      "evaluation_seconds": game_result["policy_stats"][str(player)]["evaluation_seconds"],
  }


def run_matchups(openings, policies, weights=None, output_path=None):
  raw_games = []
  aggregates = {}
  matchup_index = 0
  for left_index, left in enumerate(policies):
    for right in policies[left_index + 1:]:
      matchup_key = f"{left}_vs_{right}"
      totals = {policy: {"wins": 0, "natural_draws": 0, "truncations": 0, "repetitions": 0, "draws": 0, "losses": 0, "scores": [], "seed_differentials": [], "game_lengths": [], "ties": [], "evaluation_seconds": []} for policy in (left, right)}
      for opening in openings:
        for assignment in ((left, right), (right, left)):
          player_policies = {0: assignment[0], 1: assignment[1]}
          seed = GAME_SEED + matchup_index * 1000 + opening["opening_id"] * 2 + (assignment[0] == right)
          result = play_game(opening, player_policies, seed, weights=weights)
          assignment_map = {assignment[0]: 0, assignment[1]: 1}
          for policy in (left, right):
            metric = _result_for_policy(result, policy, assignment_map)
            totals[policy]["wins"] += metric["win"]
            totals[policy]["natural_draws"] += metric["natural_draw"]
            totals[policy]["truncations"] += metric["truncation"]
            totals[policy]["repetitions"] += metric["repetition"]
            totals[policy]["draws"] += metric["draw"]
            totals[policy]["losses"] += metric["loss"]
            for field, key in (("score", "scores"), ("seed_differential", "seed_differentials"), ("game_length", "game_lengths"), ("tied_action_selections", "ties"), ("evaluation_seconds", "evaluation_seconds")):
              totals[policy][key].append(metric[field])
          raw_games.append({
              "matchup": matchup_key,
              "opening": opening,
              "assignment": player_policies,
              "terminal": result["terminal"],
              "termination_reason": result["termination_reason"],
              "winner": result["winner"],
              "natural_draw": result["natural_draw"],
              "truncated": result["truncated"],
              "action_count": result["action_count"],
              "result": result,
          })
        matchup_index += 1
      for policy in (left, right):
        data = totals[policy]
        n = len(data["scores"])
        aggregates[f"{matchup_key}:{policy}"] = {
            "matchup": matchup_key,
            "policy": policy,
            "games": n,
            "wins": data["wins"],
            "natural_draws": data["natural_draws"],
            "truncations": data["truncations"],
            "repetitions": data["repetitions"],
            "draws": data["draws"],
            "losses": data["losses"],
            "score_rate": sum(data["scores"]) / n,
            "average_final_seed_differential": sum(data["seed_differentials"]) / n,
            "average_game_length": sum(data["game_lengths"]) / n,
            "average_tied_action_selections": sum(data["ties"]) / n,
            "average_evaluation_time_ms": 1000 * sum(data["evaluation_seconds"]) / n,
        }
  if output_path:
    Path(output_path).write_text(json.dumps({"games": raw_games, "aggregates": aggregates}, indent=2))
  return raw_games, aggregates


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--use-frozen-openings", action="store_true", default=False,
                      help="use the checked-in frozen opening suite (default)")
  parser.add_argument("--regenerate-openings", action="store_true",
                      help="explicitly generate a separate opening suite")
  parser.add_argument("--seed", type=int, default=None,
                      help="seed required with --regenerate-openings")
  args = parser.parse_args()
  if args.regenerate_openings and args.seed is None:
    parser.error("--regenerate-openings requires --seed")
  if args.regenerate_openings and args.use_frozen_openings:
    parser.error("choose only one opening mode")

  base = Path(__file__).resolve().parent
  corrected_dir = base / "raw_results_corrected"
  corrected_dir.mkdir(exist_ok=True)
  corrected_manifest = corrected_dir / "corrected_experiment_manifest.json"
  if args.regenerate_openings:
    openings = generate_openings(args.seed)
    opening_path = base / "openings" / f"opening_positions_seed_{args.seed}.json"
    opening_path.write_text(json.dumps({
        "opening_seed": args.seed,
        "development_count": DEVELOPMENT_COUNT,
        "verification_count": OPENING_COUNT - DEVELOPMENT_COUNT,
        "positions": openings,
    }, indent=2))
  else:
    opening_path = FROZEN_OPENINGS_PATH
    openings = load_frozen_openings(opening_path, corrected_manifest if corrected_manifest.exists() else None)

  development = [item for item in openings if item["split"] == "development"]
  verification = [item for item in openings if item["split"] == "verification"]
  candidates = ["RAND", "H_C", "H_CT", "H_CTM", "H_CTMS"]
  _, candidate_development = run_matchups(development, candidates, output_path=corrected_dir / "candidate_development.json")
  run_matchups(verification, candidates, output_path=corrected_dir / "candidate_verification.json")

  candidate_rates = {
      candidate: [item["score_rate"] for item in candidate_development.values() if item["policy"] == candidate]
      for candidate in candidates[1:]
  }
  selected_candidate = max(candidates[1:], key=lambda candidate: sum(candidate_rates[candidate]) / len(candidate_rates[candidate]))

  # Selection is made from development results only. The predefined vectors
  # are restricted to the features in the selected candidate.
  weight_vectors = [
      [1, 1, 1], [2, 1, 1], [4, 2, 1],
      [4, 2, 0.5], [6, 3, 1], [8, 3, 1],
  ]
  if any(ayo_heuristic.equivalent_weight_vectors(first, second)
         for index, first in enumerate(weight_vectors)
         for second in weight_vectors[index + 1:]):
    raise ValueError("corrected weight candidates contain equivalent ratios")
  feature_order = ("capture", "tactical", "mobility", "seed")
  selected_features = {
      "H_C": ("capture",),
      "H_CT": ("capture", "tactical"),
      "H_CTM": ("capture", "tactical", "mobility"),
      "H_CTMS": feature_order,
  }[selected_candidate]
  vector_results = {}
  for vector in weight_vectors:
    compact_vector = vector[:len(selected_features)]
    weights = {feature: weight for feature, weight in zip(selected_features, compact_vector)}
    _, aggregate = run_matchups(development, [selected_candidate, "RAND"], weights=weights, output_path=corrected_dir / f"weights_{'_'.join(map(str, compact_vector))}_development.json")
    vector_results["_".join(map(str, compact_vector))] = aggregate
  selected_vector = max(
      [vector[:len(selected_features)] for vector in weight_vectors],
      key=lambda vector: vector_results["_".join(map(str, vector))][f"{selected_candidate}_vs_RAND:{selected_candidate}"]["score_rate"],
  )
  selected_weights = {feature: weight for feature, weight in zip(selected_features, selected_vector)}
  _, verification_aggregate = run_matchups(verification, [selected_candidate, "RAND"], weights=selected_weights, output_path=corrected_dir / "selected_verification.json")
  opening_bytes = Path(opening_path).read_bytes()
  (corrected_dir / "corrected_experiment_manifest.json").write_text(json.dumps({
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "opening_generation_seed": OPENING_SEED if not args.regenerate_openings else args.seed,
      "opening_file": str(Path(opening_path).relative_to(base)),
      "opening_file_sha256": hashlib.sha256(opening_bytes).hexdigest(),
      "opening_state_hashes": [_state_hash(item) for item in openings],
      "game_seed": GAME_SEED,
      "game_length_limit": GAME_LENGTH_LIMIT,
      "development_openings": DEVELOPMENT_COUNT,
      "verification_openings": OPENING_COUNT - DEVELOPMENT_COUNT,
      "candidates": candidates,
      "selected_candidate": selected_candidate,
      "weight_vectors": weight_vectors,
      "selected_weights": list(selected_vector),
      "selected_weight_ratio": ayo_heuristic.normalize_weight_vector(selected_vector),
      "terminal_value": 10.0,
      "opening_mode": "regenerated" if args.regenerate_openings else "frozen",
      "verification_aggregate": verification_aggregate,
  }, indent=2))


if __name__ == "__main__":
  main()
