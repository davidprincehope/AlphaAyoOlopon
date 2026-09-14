"""Python implementation of the OpenSpiel Ayo Olopon game."""

import numpy as np
import pyspiel


_NUM_PLAYERS = 2
_DEFAULT_HOUSES_PER_PLAYER = 6
_DEFAULT_SEEDS_PER_HOUSE = 4
_MAX_GAME_LENGTH = 1000


_GAME_TYPE = pyspiel.GameType(
    short_name="ayo_olopon",
    long_name="Ayo Olopon",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    # The C++ Oware implementation exposes observations, not separate
    # information-state encodings. The observer below provides the RL input.
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "num_houses_per_player": _DEFAULT_HOUSES_PER_PLAYER,
        "num_seeds_per_house": _DEFAULT_SEEDS_PER_HOUSE,
    },
)


_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_DEFAULT_HOUSES_PER_PLAYER,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH,
)


class AyoGame(pyspiel.Game):
    """Game-level configuration shared by all Ayo/Oware states."""

    def __init__(self, params=None):
        """Create a game using the supplied board-size parameters.

        Returns: None.
        """
        # Example input: {"num_houses_per_player": 6, "num_seeds_per_house": 4}
        params = params or {}
        super().__init__(_GAME_TYPE, _GAME_INFO, params)
        self.num_houses_per_player = int(
            params.get("num_houses_per_player", _DEFAULT_HOUSES_PER_PLAYER)
        )
        self.num_seeds_per_house = int(
            params.get("num_seeds_per_house", _DEFAULT_SEEDS_PER_HOUSE)
        )

    def new_initial_state(self):
        """Return a new game state in the starting position.

        Returns: A new AyoState object.
        """
        # Example input: no arguments (call as game.new_initial_state())
        return AyoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Create the observer used to convert states into RL inputs.

        Returns: A new AyoObserver object.
        """
        # Example input: iig_obs_type=None, params=None
        return AyoObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params or {},
            self,
        )


class AyoState(pyspiel.State):
    """One mutable position in an Ayo Olopon game.

    ``board`` is laid out in sowing order:

      player 0: board[0:6]
      player 1: board[6:12]

    Actions are local to the current player's row. Thus action 0 means
    ``board[0]`` for player 0 and ``board[6]`` for player 1.
    """

    def __init__(self, game):
        """Initialize the board, scores, turn, and repetition tracking.

        Returns: None.
        """
        # Example input: game = AyoGame()
        super().__init__(game)
        self.num_houses_per_player = game.num_houses_per_player
        self.num_houses = _NUM_PLAYERS * self.num_houses_per_player
        self.total_seeds = (
            _NUM_PLAYERS
            * self.num_houses_per_player
            * game.num_seeds_per_house
        )
        self.board = [game.num_seeds_per_house] * self.num_houses
        self.captured = [0, 0]
        self._current_player = 0
        self._game_over = False
        self._returns = [0.0, 0.0]
        self._positions_since_capture = {self._position_key()}

    def _position_key(self):
        """Return a hashable representation of the current game position.

        Returns: A tuple containing the player, scores, and board.
        """
        # Example input: no arguments (call as state._position_key())
        return (self._current_player, tuple(self.captured), tuple(self.board))

    def current_player(self):
        """Return the current player, or TERMINAL after the game ends.

        Returns: The current player number or pyspiel.PlayerId.TERMINAL.
        """
        # Example input: no arguments (call as state.current_player())
        return pyspiel.PlayerId.TERMINAL if self._game_over else self._current_player

    def _player_lower_house(self, player):
        """Return the first board index belonging to a player.

        Returns: An integer board index.
        """
        # Example input: player=1
        return player * self.num_houses_per_player

    def _player_upper_house(self, player):
        """Return the last board index belonging to a player.

        Returns: An integer board index.
        """
        # Example input: player=1
        return self._player_lower_house(player) + self.num_houses_per_player - 1

    def _lower_house(self, house):
        """Return the first index in the row containing a house.

        Returns: An integer board index.
        """
        # Example input: house=8
        return (house // self.num_houses_per_player) * self.num_houses_per_player

    def _upper_house(self, house):
        """Return the last index in the row containing a house.

        Returns: An integer board index.
        """
        # Example input: house=8
        return self._lower_house(house) + self.num_houses_per_player - 1

    def _action_to_house(self, player, action):
        """Convert a player's local action into a board index.

        Returns: An integer board index.
        """
        # Example input: player=1, action=2
        return player * self.num_houses_per_player + action

    def _house_to_action(self, house):
        """Convert a board index into its local action number.

        Returns: An integer action number.
        """
        # Example input: house=8
        return house % self.num_houses_per_player

    def _opponent_seeds(self):
        """Return the number of seeds currently in the opponent's row.

        Returns: The opponent's seed count as an integer.
        """
        # Example input: no arguments (call as state._opponent_seeds())
        opponent = 1 - self._current_player
        lower = self._player_lower_house(opponent)
        upper = self._player_upper_house(opponent)
        return sum(self.board[lower : upper + 1])

    def _legal_actions(self, player):
        """Return the actions the specified player may currently choose.

        Returns: A list of legal action numbers.
        """
        # Example input: player=0
        if self._game_over or player not in (0, 1):
            return []

        lower = self._player_lower_house(self._current_player)
        upper = self._player_upper_house(self._current_player)

        if self._opponent_seeds() == 0:
            # The move must reach the opponent row and give it a seed.
            return [
                self._house_to_action(house)
                for house in range(lower, upper + 1)
                if self.board[house] - (upper - house) > 0
            ]

        return [
            self._house_to_action(house)
            for house in range(lower, upper + 1)
            if self.board[house] > 0
        ]

    def _distribute_seeds(self, house):
        """Sow one seed at a time, skipping the source house.

        Returns: The index of the house receiving the final seed.
        """
        # Example input: house=2
        to_distribute = self.board[house]
        if to_distribute == 0:
            raise ValueError("Cannot sow from an empty house")
        self.board[house] = 0
        index = house
        while to_distribute > 0:
            index = (index + 1) % self.num_houses
            if index != house:
                self.board[index] += 1
                to_distribute -= 1
        return index

    def _in_opponent_row(self, house):
        """Return whether a house belongs to the opponent's row.

        Returns: True if the house is in the opponent's row; otherwise False.
        """
        # Example input: house=8
        return house // self.num_houses_per_player != self._current_player

    def _sow_relay(self, house):
        """Sow successive laps until the Ayo move-ending rule is reached.

        The last seed of each lap is inspected immediately. A landing pit
        containing four seeds is captured and ends the move. A landing pit
        containing one seed was empty before that seed was dropped, so it
        ends the move without a capture. Any other landing count means the
        pit was non-empty before the last seed; all of its seeds are picked
        up and become the source for the next lap.

        Returns: True if the move captured four seeds, otherwise False.
        """
        while True:
            last_house = self._distribute_seeds(house)
            landing_seeds = self.board[last_house]

            if landing_seeds == 4:
                self.board[last_house] = 0
                self.captured[self._current_player] += 4
                return True

            if landing_seeds == 1:
                return False

            house = last_house

    def _score_terminal(self):
        """Return whether the captured-seed scores meet a terminal condition.

        Returns: True if a score-based terminal condition is met; otherwise False.
        """
        # Example input: no arguments (call as state._score_terminal())
        limit = self.total_seeds // 2
        return (
            self.captured[0] > limit
            or self.captured[1] > limit
            or (self.captured[0] == limit and self.captured[1] == limit)
        )

    def _set_score_returns_and_end(self):
        """Mark the game finished and assign returns from the final scores.

        Returns: None.
        """
        # Example input: no arguments (call as state._set_score_returns_and_end())
        self._game_over = True
        if self.captured[0] > self.captured[1]:
            self._returns = [1.0, -1.0]
        elif self.captured[0] < self.captured[1]:
            self._returns = [-1.0, 1.0]
        else:
            self._returns = [0.0, 0.0]

    def _collect_and_terminate(self):
        """Award remaining seeds to the owners of their rows.

        Returns: None.
        """
        # Example input: no arguments (call as state._collect_and_terminate())
        for house, seeds in enumerate(self.board):
            owner = house // self.num_houses_per_player
            self.captured[owner] += seeds
            self.board[house] = 0
        self._set_score_returns_and_end()

    def _apply_action(self, action):
        """Apply an action, including sowing, captures, and end checks.

        Returns: None.
        """
        # Example input: action=2
        if action not in self._legal_actions(self._current_player):
            raise ValueError(f"Illegal action: {action}")

        house = self._action_to_house(self._current_player, action)
        if self._sow_relay(house):
            # Captured seeds cannot return to the board, so earlier positions
            # cannot recur after a capture.
            self._positions_since_capture.clear()

        self._current_player = 1 - self._current_player

        if self._score_terminal():
            self._set_score_returns_and_end()
            return

        if self._position_key() in self._positions_since_capture:
            self._collect_and_terminate()
            return
        self._positions_since_capture.add(self._position_key())

        if not self._legal_actions(self._current_player):
            self._collect_and_terminate()

    def is_terminal(self):
        """Return whether the game has reached a terminal state.

        Returns: True if the game is over; otherwise False.
        """
        # Example input: no arguments (call as state.is_terminal())
        return self._game_over

    def returns(self):
        """Return final payoffs, or zero payoffs while play continues.

        Returns: A list containing the two players' payoffs.
        """
        # Example input: no arguments (call as state.returns())
        return list(self._returns) if self._game_over else [0.0, 0.0]

    def _action_to_string(self, player, action):
        """Return the display label for a player's action.

        Returns: A string action label such as ``A2`` or ``a2``.
        """
        # Example input: player=0, action=2
        return f"{'A' if player == 0 else 'a'}{action}"

    def __str__(self):
        """Return a readable summary of the board and game status.

        Returns: A string containing the board, scores, player, and status.
        """
        # Example input: no arguments (call as str(state))
        return (
            f"Board: {self.board}\n"
            f"Captured: {self.captured}\n"
            f"Current player: {self.current_player()}\n"
            f"Terminal: {self._game_over}"
        )


class AyoObserver:
    """Flat normalized observation: houses followed by both scores."""

    def __init__(self, iig_obs_type, params, game):
        """Create storage for a normalized observation tensor.

        Returns: None.
        """
        # Example input: iig_obs_type=None, params={}, game=AyoGame()
        del iig_obs_type
        if params:
            raise ValueError(f"Observation parameters are not supported: {params}")
        size = 2 * game.num_houses_per_player + _NUM_PLAYERS
        self.tensor = np.zeros(size, dtype=np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state, player):
        """Copy a game state into the observer's normalized tensor.

        Returns: None.
        """
        # Example input: state=game.new_initial_state(), player=0
        del player
        self.tensor[: state.num_houses] = np.asarray(state.board) / state.total_seeds
        self.tensor[state.num_houses :] = np.asarray(state.captured) / state.total_seeds

    def string_from(self, state, player):
        """Return a string representation of the observed state.

        Returns: A string representation of the state.
        """
        # Example input: state=game.new_initial_state(), player=0
        del player
        return str(state)


pyspiel.register_game(_GAME_TYPE, AyoGame)
