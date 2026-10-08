"""Read and validate the isolated checkpoint-strength opening dataset.

Action histories are canonical: replay restores the horizon and repetition
memory as well as the visible board.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENINGS = "experiments/checkpoint_strength/dataset/ayo_fixed_v1.json"
SCHEMA_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rules_contract(game):
    if game.get_type().short_name not in ("ayo_olopon", "ayo_olopon_alpha_zero"):
        raise ValueError("Opening benchmarks require the Python Ayo game")
    if game.num_houses_per_player != 6 or game.num_seeds_per_house != 4:
        raise ValueError("Opening benchmarks require the standard 6 x 4 board")
    # Normalize newlines so Windows-generated files also validate on Linux.
    source = (ROOT / "Model/ayo_olopon/ayo_olopon.py").read_text(encoding="utf-8")
    return {"rules": "python_ayo_olopon", "rules_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "houses_per_player": 6, "seeds_per_house": 4,
            "max_actions": game.max_game_length(), "cutoff": "collect"}


def snapshot(state):
    return {"board": [int(x) for x in state.board],
            "captured": [int(x) for x in state.captured],
            "current_player": int(state.current_player()), "ply": state.move_number(),
            "legal_actions": [int(a) for a in state.legal_actions()],
            "is_terminal": state.is_terminal(), "truncated": bool(state.truncated),
            "repetition_positions": [[int(player), list(captured), list(board)]
                                     for player, captured, board in sorted(state._positions_since_capture)]}


def position_digest(state_snapshot):
    """Visible-position identity, deliberately stricter deduplication than histories."""
    return digest({key: state_snapshot[key] for key in ("board", "captured", "current_player")})


def replay(game, actions):
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("An opening must contain a nonempty action sequence")
    state = game.new_initial_state()
    for ply, action in enumerate(actions):
        if type(action) is not int or state.is_terminal() or action not in state.legal_actions():
            raise ValueError(f"Illegal opening action {action!r} at ply {ply + 1}")
        state.apply_action(action)
    return state


@dataclass(frozen=True)
class Opening:
    opening_id: str
    actions: tuple[int, ...]
    position_sha256: str
    state_sha256: str

    def new_state(self, game):
        state = replay(game, self.actions)
        if state.is_terminal() or digest(snapshot(state)) != self.state_sha256:
            raise ValueError(f"Opening {self.opening_id} no longer reproduces its saved state")
        return state


@dataclass(frozen=True)
class OpeningDataset:
    path: str
    file_sha256: str
    benchmark_id: str
    content_sha256: str
    contract: dict
    openings: tuple[Opening, ...]


def load_dataset(path, game):
    path = Path(path)
    # Repository-relative configuration works independently of the caller's cwd.
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported opening benchmark schema")
    content_hash = data.get("content_sha256")
    if content_hash != digest({k: v for k, v in data.items() if k != "content_sha256"}):
        raise ValueError("Opening benchmark content hash mismatch")
    if data.get("rules_contract") != rules_contract(game):
        raise ValueError("Opening benchmark rules/horizon do not match the evaluation game")
    if not isinstance(data.get("benchmark_id"), str) or not data["benchmark_id"]:
        raise ValueError("Opening benchmark requires an ID")
    records = data.get("openings")
    if not isinstance(records, list) or not records or data.get("opening_count") != len(records):
        raise ValueError("Opening benchmark count is invalid")
    openings, ids, positions = [], set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Opening records must be JSON objects")
        opening_id = record.get("opening_id")
        if not isinstance(opening_id, str) or not opening_id or opening_id in ids:
            raise ValueError("Duplicate or invalid opening ID")
        state = replay(game, record.get("actions"))
        actual = snapshot(state)
        if (state.is_terminal() or actual != record.get("state") or
                record.get("plies") != len(record["actions"]) or
                digest(actual) != record.get("state_sha256") or
                position_digest(actual) != record.get("position_sha256")):
            raise ValueError(f"Opening {opening_id} state metadata mismatch or terminal position")
        if record["position_sha256"] in positions:
            raise ValueError("Duplicate visible opening position")
        ids.add(opening_id)
        positions.add(record["position_sha256"])
        openings.append(Opening(opening_id, tuple(record["actions"]),
                                record["position_sha256"], record["state_sha256"]))
    return OpeningDataset(str(path.resolve()), hashlib.sha256(raw).hexdigest(), data["benchmark_id"],
                            content_hash, data["rules_contract"], tuple(openings))
