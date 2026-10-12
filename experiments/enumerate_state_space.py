"""Exhaustively enumerate reachable positional states of Ayo Olopon.

This experiment intentionally uses positional transposition keys only:
history, including ``_positions_since_capture``, is not part of the key.
The game implementation remains the sole source of legal actions and state
transitions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

import pyspiel

# Make the repository root importable when this file is run directly from the
# experiments directory (``python experiments/enumerate_state_space.py``).
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Importing the package registers the game with OpenSpiel.
from Model.ayo_olopon import ayo_olopon  # noqa: F401


TOTAL_SEEDS = 48
EXPECTED_HOUSES = 12
DEFAULT_OUTPUT = Path("experiments/ayo_olopon_state_space.json")
DEFAULT_CHECKPOINT = Path("experiments/ayo_olopon_state_space.checkpoint.pkl")


class EnumerationInvariantError(RuntimeError):
    """Raised when a generated state violates an experiment invariant."""


def canonical_state(state: pyspiel.State) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return exactly (board, captured, player) as an immutable key."""
    return (
        tuple(int(value) for value in state.board),
        tuple(int(value) for value in state.captured),
        int(state.current_player()),
    )


def validate_state(state: pyspiel.State, key: tuple[Any, ...], context: str) -> None:
    """Fail loudly if the model produces an invalid positional state."""
    board, captured, player = key
    if len(board) != EXPECTED_HOUSES:
        raise EnumerationInvariantError(
            f"{context}: expected 12 houses, got {len(board)}; state={key}"
        )
    if any(not isinstance(value, int) or value < 0 for value in board):
        raise EnumerationInvariantError(f"{context}: invalid board; state={key}")
    if any(not isinstance(value, int) or value < 0 for value in captured):
        raise EnumerationInvariantError(f"{context}: invalid captured; state={key}")
    if not state.is_terminal() and player not in (0, 1):
        raise EnumerationInvariantError(f"{context}: invalid current player; state={key}")
    if sum(board) + sum(captured) != TOTAL_SEEDS:
        raise EnumerationInvariantError(
            f"{context}: seed invariant failed; state={key}; "
            f"total={sum(board) + sum(captured)}"
        )


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically so an interrupted write is not resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return payload


def enumerate_positions(
    max_states: int | None = None,
    progress: bool = True,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run frontier BFS and return reproducible summary statistics."""
    game = pyspiel.load_game("ayo_olopon")
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = _read_checkpoint(checkpoint_path)
        frontier = checkpoint["frontier"]
        visited = checkpoint["visited"]
        states_by_depth = Counter(checkpoint["states_by_depth"])
        terminal_states = checkpoint["terminal_states"]
        duplicate_attempts = checkpoint["duplicate_attempts"]
        transitions_generated = checkpoint["transitions_generated"]
        maximum_frontier = checkpoint["maximum_frontier"]
        depth = checkpoint["depth"]
        initial_state = checkpoint["initial_state"]
        if progress:
            print(
                f"Resuming from depth {depth}: frontier={len(frontier):,}, "
                f"visited={len(visited):,}",
                flush=True,
            )
    else:
        initial = game.new_initial_state()
        initial_key = canonical_state(initial)
        validate_state(initial, initial_key, "initial state")
        initial_state = {
            "board": list(initial_key[0]),
            "captured": list(initial_key[1]),
            "current_player": initial_key[2],
        }

        visited = {initial_key}
        frontier = [initial]
        states_by_depth = Counter({0: 1})
        terminal_states = 1 if initial.is_terminal() else 0
        duplicate_attempts = 0
        transitions_generated = 0
        maximum_frontier = 1
        depth = 0
    enumeration_complete = True

    tracemalloc.start()
    started = time.perf_counter()
    while frontier:
        next_frontier = []
        for state_index, state in enumerate(frontier):
            if state.is_terminal():
                continue

            actions = list(state.legal_actions())
            for action in actions:
                transitions_generated += 1
                child = state.child(action)
                child_key = canonical_state(child)
                validate_state(
                    child,
                    child_key,
                    f"depth={depth + 1}, state_index={state_index}, action={action}",
                )
                if child_key in visited:
                    duplicate_attempts += 1
                    continue
                visited.add(child_key)
                next_frontier.append(child)
                states_by_depth[depth + 1] += 1
                if child.is_terminal():
                    terminal_states += 1
                if max_states is not None and len(visited) >= max_states:
                    enumeration_complete = False
                    frontier = []
                    break

            if not enumeration_complete:
                break

        depth += 1
        maximum_frontier = max(maximum_frontier, len(next_frontier))
        frontier = next_frontier if enumeration_complete else []
        if progress:
            print(
                f"BFS depth {depth}: frontier={len(frontier):,}, "
                f"visited={len(visited):,}",
                flush=True,
            )
        if checkpoint_path is not None:
            _write_checkpoint(
                checkpoint_path,
                {
                    "format_version": 1,
                    "frontier": frontier,
                    "visited": visited,
                    "states_by_depth": dict(states_by_depth),
                    "terminal_states": terminal_states,
                    "duplicate_attempts": duplicate_attempts,
                    "transitions_generated": transitions_generated,
                    "maximum_frontier": maximum_frontier,
                    "depth": depth,
                    "initial_state": initial_state,
                },
            )

    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "search_method": "Frontier-Based BFS",
        "state_representation": "(board, captured, current_player)",
        "initial_state": initial_state,
        "unique_reachable_states": len(visited),
        "terminal_states": terminal_states,
        "states_by_bfs_depth": {str(k): v for k, v in sorted(states_by_depth.items())},
        "maximum_bfs_depth": max(states_by_depth),
        "maximum_frontier_size": maximum_frontier,
        "peak_visited_set_size": len(visited),
        "duplicate_transposition_attempts": duplicate_attempts,
        "total_transitions_generated": transitions_generated,
        "runtime_seconds": elapsed,
        "peak_tracemalloc_bytes": peak_bytes,
        "enumeration_complete": enumeration_complete,
    }


def print_report(summary: dict[str, Any]) -> None:
    """Print a concise human-readable report."""
    print("=== Ayo Olopon Positional State-Space Enumeration ===")
    print(f"Search method: {summary['search_method']}")
    print(f"State representation: {summary['state_representation']}")
    print(f"Initial state: {summary['initial_state']}")
    print(f"Unique reachable states: {summary['unique_reachable_states']:,}")
    print(f"Terminal states: {summary['terminal_states']:,}")
    print(f"Maximum BFS depth: {summary['maximum_bfs_depth']:,}")
    print(f"Maximum frontier size: {summary['maximum_frontier_size']:,}")
    print(f"Duplicate/transposition detections: {summary['duplicate_transposition_attempts']:,}")
    print(f"Total transitions generated: {summary['total_transitions_generated']:,}")
    print(f"Runtime: {summary['runtime_seconds']:.3f} seconds")
    print(f"Peak tracemalloc memory: {summary['peak_tracemalloc_bytes']:,} bytes")
    print(f"Enumeration complete: {'YES' if summary['enumeration_complete'] else 'NO'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Optional safety bound; a bounded run is reported as incomplete.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-depth progress output.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Write a resumable pickle checkpoint after each BFS depth.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint supplied with --checkpoint.",
    )
    args = parser.parse_args()

    try:
        summary = enumerate_positions(
            max_states=args.max_states,
            progress=not args.quiet,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
        )
    except EnumerationInvariantError as error:
        print(f"INVARIANT FAILURE: {error}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_report(summary)
    print(f"JSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
