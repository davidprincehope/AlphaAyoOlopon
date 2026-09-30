"""Training observations and a finite horizon, reusing the existing Ayo rules."""

import numpy as np
import pyspiel

from Model.ayo_olopon.ayo_olopon import AyoState


GAME_NAME = "ayo_olopon_alpha_zero"
OBSERVATION_VERSION = "canonical_player_relative_horizon_v1"
GAME_TYPE = pyspiel.GameType(
    short_name=GAME_NAME,
    long_name="Ayo Olopon (AlphaZero adapter)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={"max_moves": 1000, "cutoff": "collect"},
)


class AlphaZeroAyoGame(pyspiel.Game):
    """Fixed 6x4 board; action IDs retain the original player's local pit order."""

    def __init__(self, params=None):
        params = dict(params or {})
        self.max_moves = int(params.get("max_moves", 1000))
        self.cutoff = params.get("cutoff", "collect")
        if self.max_moves < 1 or self.cutoff != "collect":
            raise ValueError("max_moves must be positive; cutoff must be collect")
        info = pyspiel.GameInfo(
            num_distinct_actions=6, max_chance_outcomes=0, num_players=2,
            min_utility=-1.0, max_utility=1.0, utility_sum=0.0,
            max_game_length=self.max_moves,
        )
        super().__init__(GAME_TYPE, info, params)
        self.num_houses_per_player = 6
        self.num_seeds_per_house = 4
        self.enable_cycle_reporting = False

    def __reduce__(self):
        # pyspiel's default pickle does not restore Python game attributes.
        return (type(self), (self.get_parameters(),))

    def new_initial_state(self):
        return AlphaZeroAyoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        if params:
            raise ValueError("Observer parameters are not supported")
        return AlphaZeroAyoObserver()


class AlphaZeroAyoState(AyoState):
    def __init__(self, game):
        super().__init__(game)
        self.ply = 0
        self.max_moves = game.max_moves
        self.cutoff = game.cutoff
        self.truncated = False

    def _apply_action(self, action):
        super()._apply_action(action)
        self.ply += 1
        if not self.is_terminal() and self.ply >= self.max_moves:
            self.truncated = True
            if self.cutoff == "collect":
                self._collect_and_terminate()
            else:
                self._game_over = True
                self._returns = [0.0, 0.0]


class AlphaZeroAyoObserver:
    """15 floats: current row, opponent row, captures, remaining horizon.

    The underlying game stays in absolute player order. At terminal states,
    the requested observer player supplies the otherwise absent perspective.
    """

    def __init__(self):
        self.tensor = np.zeros(15, dtype=np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state, player):
        perspective = player if state.is_terminal() else state.current_player()
        self.tensor[:] = canonical_observation(
            state.board, state.captured, perspective,
            state.move_number(), state.max_moves,
        )

    def string_from(self, state, player):
        del player
        return f"{state}\nPly: {state.ply}/{state.max_moves}"


def canonical_observation(board, captured, player, moves_played, max_moves):
    """Encode a six-house rotation from the specified player's perspective."""
    if len(board) != 12 or len(captured) != 2 or player not in (0, 1):
        raise ValueError("Expected 12 houses, two scores, and player 0 or 1")
    if max_moves < 1 or not 0 <= moves_played <= max_moves:
        raise ValueError("Move count must be within the configured horizon")
    first = 6 * player
    other = 6 * (1 - player)
    encoded = np.empty(15, dtype=np.float32)
    encoded[:6] = np.asarray(board[first:first + 6], dtype=np.float32) / 48
    encoded[6:12] = np.asarray(board[other:other + 6], dtype=np.float32) / 48
    encoded[12] = captured[player] / 48
    encoded[13] = captured[1 - player] / 48
    encoded[14] = (max_moves - moves_played) / max_moves
    return encoded


pyspiel.register_game(GAME_TYPE, AlphaZeroAyoGame)
