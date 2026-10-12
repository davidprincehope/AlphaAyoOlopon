"""Greedy Ayo moves using only the mover's immediate captured-seed gain."""
from types import SimpleNamespace


def capture_greedy_action(state):
    """Resolve each complete legal action; lowest action ID breaks gain ties."""
    player = state.current_player()
    legal = sorted(state.legal_actions())
    if not legal:
        raise ValueError("Capture greedy requires a state with legal actions")
    before = state.captured[player]
    return int(max(legal, key=lambda action: state.child(action).captured[player] - before))


def make_bot(game, seed=None):
    if game.get_type().short_name not in ("ayo_olopon", "ayo_olopon_alpha_zero"):
        raise ValueError("Capture greedy requires the Ayo Olopon game")
    return SimpleNamespace(step=capture_greedy_action)
