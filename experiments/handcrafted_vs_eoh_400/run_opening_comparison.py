"""Generate and compare H* with the handcrafted heuristic on 400 openings.

For each opening and each depth (1, 2, 3), both policy seat assignments are
played. Search depth counts actions with max backup for the policy being
evaluated and min backup for its opponent. Results are deterministic once the
opening file is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import pyspiel

from experiments.handcrafted_vs_eoh import compare as shared
from experiments.handcrafted_vs_eoh import compare_initial
from experiments.heuristic import run_experiment as frozen_protocol
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers game


OUTPUT_DIR = Path(__file__).resolve().parent
OPENINGS_PATH = OUTPUT_DIR / "opening_set_400.json"
OPENING_GENERATION_SEED = 20260924
TARGET_PLIES = (4, 32)
DEVELOPMENT_COUNT = 400
DEPTHS = (1, 2, 3)
MAX_ACTIONS = 300


def state_hash(snapshot: dict) -> str:
  payload = {
      "board": snapshot["board"],
      "captured": snapshot["captured"],
      "current_player": snapshot["current_player"],
  }
  return hashlib.sha256(
      json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def generate_positions(count: int = DEVELOPMENT_COUNT,
                       seed: int = OPENING_GENERATION_SEED,
                       existing: list[dict] | None = None) -> list[dict]:
  """Sample unique nonterminal reachable positions after 4-32 random plies."""
  rng = random.Random(seed)
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  seen = {
      (tuple(item["board"]), tuple(item["captured"]), int(item["current_player"]))
      for item in (existing or [])
  }
  generated = []
  attempts = 0
  while len(generated) < count:
    attempts += 1
    if attempts > max(100_000, count * 2_000):
      raise RuntimeError("could not generate the requested number of distinct openings")
    opening_seed = rng.randrange(1, 2**31)
    local_rng = random.Random(opening_seed)
    target = local_rng.randint(*TARGET_PLIES)
    state = game.new_initial_state()
    plies = 0
    while plies < target and not state.is_terminal() and state.legal_actions():
      state.apply_action(local_rng.choice(state.legal_actions()))
      plies += 1
    if state.is_terminal() or not state.legal_actions():
      continue
    key = (tuple(state.board), tuple(state.captured), int(state.current_player()))
    if key in seen:
      continue
    seen.add(key)
    generated.append({
        "opening_seed": opening_seed,
        "plies_from_initial": plies,
        "board": [int(value) for value in state.board],
        "captured": [int(value) for value in state.captured],
        "current_player": int(state.current_player()),
        "is_terminal": False,
        "phase": "early" if plies <= 12 else "middle",
        "state_hash": state_hash({
            "board": list(state.board),
            "captured": list(state.captured),
            "current_player": int(state.current_player()),
        }),
    })
  generated.sort(key=lambda item: item["opening_seed"])
  return generated


def _existing_openings(count: int) -> list[dict]:
  """Load the requested prefix of the prior frozen suite, if selected."""
  if count == 0:
    return []
  source = ROOT / "experiments" / "heuristic" / "openings" / "opening_positions.json"
  payload = json.loads(source.read_text(encoding="utf-8"))
  positions = payload["positions"]
  if count != 100 or len(positions) < count:
    raise ValueError("supported reuse count is 0 or all 100 existing frozen openings")
  return [dict(item, source_suite="frozen_heuristic_100") for item in positions[:count]]


def prepare_openings(path: Path = OPENINGS_PATH, reuse_existing: int = 0,
                     seed: int = OPENING_GENERATION_SEED) -> dict:
  if reuse_existing not in (0, 100):
    raise ValueError("reuse_existing must be 0 or 100")
  reused = _existing_openings(reuse_existing)
  new_positions = generate_positions(DEVELOPMENT_COUNT - reuse_existing, seed, reused)
  for index, item in enumerate(reused + new_positions):
    item["opening_id"] = index
    item["split"] = "comparison"
    item.setdefault("source_suite", "new_random_reachable")
  hashes = [item["state_hash"] for item in reused + new_positions]
  if len(hashes) != DEVELOPMENT_COUNT or len(set(hashes)) != DEVELOPMENT_COUNT:
    raise ValueError("opening set must contain exactly 400 distinct positions")
  metadata = {
      "opening_count": DEVELOPMENT_COUNT,
      "reused_existing_count": reuse_existing,
      "newly_generated_count": DEVELOPMENT_COUNT - reuse_existing,
      "generation_seed": seed,
      "plies_sampled_inclusive": list(TARGET_PLIES),
      "sampling": "uniform target length 4-32; random legal actions from actual OpenSpiel Ayo state",
      "positions": reused + new_positions,
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
  metadata["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
  return metadata


def load_openings(path: Path = OPENINGS_PATH) -> dict:
  raw = path.read_bytes()
  payload = json.loads(raw)
  positions = payload["positions"]
  if len(positions) != DEVELOPMENT_COUNT:
    raise ValueError(f"expected 400 positions, got {len(positions)}")
  hashes = []
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  for index, item in enumerate(positions):
    if item["opening_id"] != index:
      raise ValueError("opening ids must be contiguous and ordered")
    if item["is_terminal"]:
      raise ValueError(f"opening {index} must be nonterminal")
    expected = state_hash(item)
    if item["state_hash"] != expected:
      raise ValueError(f"opening {index} state hash mismatch")
    state = frozen_protocol.state_from_snapshot(game, item)
    if state.is_terminal() or not state.legal_actions():
      raise ValueError(f"opening {index} is not a playable Ayo state")
    hashes.append(expected)
  if len(set(hashes)) != DEVELOPMENT_COUNT:
    raise ValueError("opening positions are not distinct")
  return {"payload": payload, "file_sha256": hashlib.sha256(raw).hexdigest()}


def _play(opening: dict, depth: int, assignment: tuple[str, str], match_id: str) -> dict:
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = frozen_protocol.state_from_snapshot(game, opening)
  policy_for_player = {0: assignment[0], 1: assignment[1]}
  policy_time = {name: 0.0 for name in shared.PLAYERS}
  decisions = {name: 0 for name in shared.PLAYERS}
  actions = []
  while not state.is_terminal() and len(actions) < MAX_ACTIONS:
    player = int(state.current_player())
    policy = policy_for_player[player]
    start = time.perf_counter()
    action, search = compare_initial.choose_action(state, policy, depth)
    policy_time[policy] += time.perf_counter() - start
    decisions[policy] += 1
    actions.append({"player": player, "policy": policy, "action": action, "search": search})
    state.apply_action(action)

  truncated = not state.is_terminal()
  if truncated:
    difference = int(state.captured[0] - state.captured[1])
    returns = [1.0, -1.0] if difference > 0 else [-1.0, 1.0] if difference < 0 else [0.0, 0.0]
    reason = "action_limit"
  else:
    returns = [float(value) for value in state.returns()]
    report = state.last_relay_report
    if report and report.get("reason") in {"repeated_relay_state", "relay_lap_limit_exceeded"}:
      reason = "repetition"
    elif returns == [0.0, 0.0]:
      reason = "terminal_draw"
    else:
      reason = "terminal_win"
  per_policy_return = {name: returns[assignment.index(name)] for name in shared.PLAYERS}
  return {
      "match_id": match_id,
      "opening_id": int(opening["opening_id"]),
      "opening_hash": opening["state_hash"],
      "depth_plies": depth,
      "player_0_policy": assignment[0],
      "player_1_policy": assignment[1],
      "policy_returns": per_policy_return,
      "policy_outcome": {name: "win" if value > 0 else "loss" if value < 0 else "draw"
                          for name, value in per_policy_return.items()},
      "termination_reason": reason,
      "truncated": truncated,
      "game_length": len(actions),
      "final_captured": [int(value) for value in state.captured],
      "decisions": decisions,
      "policy_seconds": policy_time,
      "actions": actions,
  }


def _summary(matches: list[dict]) -> dict:
  output = {}
  for depth in DEPTHS:
    games = [match for match in matches if match["depth_plies"] == depth]
    policies = {}
    by_seat = {}
    for policy in shared.PLAYERS:
      results = [game["policy_outcome"][policy] for game in games]
      policies[policy] = {
          "games": len(games),
          "wins": results.count("win"),
          "draws": results.count("draw"),
          "losses": results.count("loss"),
          "score_rate": (sum(game["policy_returns"][policy] * 0.5 + 0.5 for game in games) / len(games)
                         if games else None),
          "decision_count": sum(game["decisions"][policy] for game in games),
          "policy_seconds": sum(game["policy_seconds"][policy] for game in games),
      }
      by_seat[policy] = {}
      for seat in (0, 1):
        subset = [game for game in games if (game["player_0_policy"] == policy) == (seat == 0)]
        outcomes = [game["policy_outcome"][policy] for game in subset]
        by_seat[policy][str(seat)] = {
            "games": len(subset), "wins": outcomes.count("win"),
            "draws": outcomes.count("draw"), "losses": outcomes.count("loss"),
        }
    output[str(depth)] = {
        "games": len(games),
        "termination_reasons": {reason: sum(game["termination_reason"] == reason for game in games)
                                for reason in ("terminal_win", "terminal_draw", "repetition", "action_limit")},
        "policies": policies,
        "by_policy_seat": by_seat,
    }
  return output


def run(openings_path: Path = OPENINGS_PATH, output_dir: Path = OUTPUT_DIR,
        match_limit: int | None = None, depths: tuple[int, ...] = DEPTHS) -> dict:
  opening_data = load_openings(openings_path)
  positions = opening_data["payload"]["positions"]
  output_dir.mkdir(parents=True, exist_ok=True)
  log_path = output_dir / "matches_400.jsonl"
  matches = []
  total = len(positions) * len(depths) * 2
  with log_path.open("w", encoding="utf-8", newline="\n") as stream:
    for depth in depths:
      for opening in positions:
        for seat, assignment in enumerate((shared.PLAYERS, tuple(reversed(shared.PLAYERS)))):
          match_id = f"d{depth}-o{opening['opening_id']:03d}-s{seat}"
          match = _play(opening, depth, assignment, match_id)
          matches.append(match)
          stream.write(json.dumps(match, sort_keys=True, separators=(",", ":")) + "\n")
          if len(matches) % 100 == 0:
            print(f"completed {len(matches)}/{total} games", flush=True)
          if match_limit is not None and len(matches) >= match_limit:
            break
        if match_limit is not None and len(matches) >= match_limit:
          break
      if match_limit is not None and len(matches) >= match_limit:
        break
  result = {
      "experiment": "handcrafted_vs_eoh_400_openings",
      "status": "pilot" if match_limit is not None else "completed",
      "opening_file": str(openings_path),
      "opening_file_sha256": opening_data["file_sha256"],
      "opening_count": len(positions),
      "reused_existing_count": opening_data["payload"]["reused_existing_count"],
      "newly_generated_count": opening_data["payload"]["newly_generated_count"],
      "generation_seed": opening_data["payload"]["generation_seed"],
      "plies_sampled_inclusive": list(TARGET_PLIES),
      "opening_profile": {
          "early_4_to_12_plies": sum(item["plies_from_initial"] <= 12 for item in positions),
          "middle_13_to_32_plies": sum(item["plies_from_initial"] > 12 for item in positions),
      },
      "handcrafted_policy": "H_CTM weights capture=4, tactical=2, mobility=0.5; terminal=+/-10",
      "eoh_policy": "H_star_ayo.py; candidate g2-m2-05",
      "depth_definition": "individual actions; maximize on root-policy turns, minimize on opponent turns, evaluate heuristic at depth cutoff",
      "depths_plies": list(depths),
      "seat_swaps_per_opening_depth": 1,
      "games": len(matches),
      "game_length_limit": MAX_ACTIONS,
      "action_limit_score": "captured-seed differential; equal differential is draw",
      "randomness_during_games": "none; policies and legal-action-order tie breaks are deterministic",
      "match_log": log_path.name,
      "summary": _summary(matches),
  }
  (output_dir / "results_400.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reuse-existing", type=int, choices=(0, 100), default=0,
                      help="include all 100 positions in the previous frozen opening suite")
  parser.add_argument("--regenerate-openings", action="store_true",
                      help="explicitly replace the frozen 400-position set")
  parser.add_argument("--prepare-only", action="store_true",
                      help="freeze the opening set without running games")
  parser.add_argument("--run-only", action="store_true",
                      help="run an already frozen opening_set_400.json")
  parser.add_argument("--pilot-matches", type=int,
                      help="smoke-run only the first N games; writes the normal match/result filenames")
  parser.add_argument("--pilot-depth", type=int, choices=DEPTHS,
                      help="restrict a pilot to one search depth (requires --pilot-matches)")
  parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                      help="directory for match and summary output files")
  args = parser.parse_args()
  if args.pilot_depth is not None and args.pilot_matches is None:
    parser.error("--pilot-depth requires --pilot-matches")
  if args.run_only:
    if not OPENINGS_PATH.exists():
      parser.error("--run-only requires an existing opening_set_400.json")
  elif args.regenerate_openings or not OPENINGS_PATH.exists():
    payload = prepare_openings(reuse_existing=args.reuse_existing)
    print(f"froze {payload['opening_count']} positions ({payload['newly_generated_count']} new); sha256={payload['file_sha256']}")
  if args.prepare_only:
    return
  chosen_depths = (args.pilot_depth,) if args.pilot_depth is not None else DEPTHS
  output = run(output_dir=args.output_dir, match_limit=args.pilot_matches,
               depths=chosen_depths)
  print(json.dumps({"status": output["status"], "games": output["games"], "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
  main()
