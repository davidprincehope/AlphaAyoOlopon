"""Headless relay frames shared by the replay viewer and browser game."""
from typing import Any, Mapping


def _sowing_trace(state: Mapping[str, Any], player: int, action: int, *, max_frames: int | None = None) -> list[dict[str, Any]]:
    """Reconstruct relay-sowing frames without mutating the game model."""
    board = list(state["board"])
    captured = list(state.get("captured", (0, 0)))
    house = player * 6 + action
    original_house = house
    seen = set()
    frames = []

    for relay in range(1, 1001):
        if max_frames is not None and len(frames) >= max_frames:
            break
        key = (house, tuple(board))
        if key in seen:
            frames.append({
                "board": list(board), "captured": list(captured),
                "current_player": player, "is_terminal": False,
                "description": f"Relay cycle detected: state repeated after {relay - 1} relays",
                "source_house": original_house,
                "selected_house": original_house,
                "cycle": True,
            })
            break
        seen.add(key)
        before = list(board)
        seeds = board[house]
        board[house] = 0
        frames.append({
            "board": list(board), "captured": list(captured),
            "current_player": player, "is_terminal": False,
            "description": f"Player {player} picks up {seeds} seeds from pit {house}",
            "source_house": original_house, "selected_house": original_house,
            "pickup_house": house, "active_house": house,
            "phase": "pickup", "hand_seeds": seeds,
        })
        landing = house
        previous_house = house
        total_seeds = seeds
        seed_number = 0
        while seeds:
            landing = (landing + 1) % 12
            if landing != house:
                board[landing] += 1
                seeds -= 1
                seed_number += 1
                frames.append({
                    "board": list(board), "captured": list(captured),
                    "current_player": player, "is_terminal": False,
                    "description": f"Relay {relay}: move seed {seed_number}/{total_seeds} from pit {previous_house} to pit {landing}",
                    "source_house": original_house, "selected_house": original_house,
                    "move_from": previous_house, "active_house": landing,
                    "landing_house": landing, "phase": "sow",
                    "hand_seeds": total_seeds - seed_number,
                })
                previous_house = landing
                if board[landing] == 4:
                    capturer = player if seeds == 0 else landing // 6
                    board[landing] = 0
                    captured[capturer] += 4
                    frames.append({
                        "board": list(board), "captured": list(captured),
                        "current_player": player, "is_terminal": False,
                        "description": f"Capture: pit {landing} reached 4 seeds",
                        "source_house": original_house, "selected_house": original_house,
                        "active_house": landing, "landing_house": landing,
                        "phase": "capture",
                    })
                    if seeds == 0:
                        return frames
        landing_seeds = board[landing]
        frames.append({
            "board": list(board), "captured": list(captured),
            "current_player": player, "is_terminal": False,
            "description": f"Relay {relay} ends in pit {landing} ({landing_seeds} seeds)",
            "source_house": original_house, "selected_house": original_house,
            "active_house": landing, "landing_house": landing,
            "landing_seeds": landing_seeds, "board_before": before,
            "phase": "land",
        })
        if landing_seeds == 4:
            board[landing] = 0
            captured[player] += 4
            frames.append({
                "board": list(board), "captured": list(captured),
                "current_player": player, "is_terminal": False,
                "description": f"Capture: pit {landing} had exactly 4 seeds",
                "source_house": original_house, "selected_house": original_house,
                "active_house": landing,
                "landing_house": landing, "phase": "capture",
            })
            break
        if landing_seeds == 1:
            break
        house = landing
    return frames
