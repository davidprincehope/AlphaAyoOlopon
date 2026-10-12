"""Authoritative human/agent turns, with complete state clones for takebacks."""
from collections import deque
import secrets
import threading
import time

from Model.ayo_olopon.replay import _sowing_trace


class GameError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class GameSession:
    def __init__(self, engine, human_player=0, seed=None):
        if type(human_player) is not int or human_player not in (0, 1):
            raise GameError('Choose to play first or second')
        self.engine = engine
        self.human_player = human_player
        self.seed = secrets.randbelow(2**32) if seed is None else seed
        self.state = engine.game.new_initial_state()
        self.revision = 0
        self.moves = []
        self._frames = []
        self._rounds = deque(maxlen=24)
        self._lock = threading.RLock()
        self.last_used = time.monotonic()

    def _check_revision(self, revision):
        if type(revision) is not int or revision != self.revision:
            raise GameError('The board has changed. Refresh it before making another move.', 409)

    def _check_turn(self, human):
        if self.state.is_terminal():
            raise GameError('This game has finished. Start a new game.', 409)
        if (self.state.current_player() == self.human_player) != human:
            raise GameError('It is not your turn.' if human else 'It is your turn.', 409)

    def _apply(self, action):
        if type(action) is not int or action not in self.state.legal_actions():
            raise GameError('Choose a highlighted pit containing a legal move.')
        player = int(self.state.current_player())
        before = list(self.state.captured)
        board_before = list(self.state.board)
        next_state = self.state.clone()
        next_state.apply_action(action)
        frames = _sowing_trace({'board': board_before, 'captured': before}, player, action,
                               max_frames=600)[:600]
        # Animation is descriptive. The final frame always comes from the
        # actual engine, including repetition, horizon and row collection.
        frames.append({'board': list(next_state.board), 'captured': list(next_state.captured),
                       'current_player': int(next_state.current_player()),
                       'is_terminal': next_state.is_terminal(), 'phase': 'complete',
                       'source_house': player * 6 + action, 'hand_seeds': 0,
                       'shortened': len(frames) == 600,
                       'description': 'Move complete'})
        self.state = next_state
        self._frames = frames
        self.moves.append({'player': player, 'action': action,
                           'captured': [int(a - b) for a, b in zip(self.state.captured, before)]})
        self.revision += 1

    def human_move(self, action, revision):
        with self._lock:
            self._check_revision(revision)
            self._check_turn(True)
            if type(action) is not int or action not in self.state.legal_actions():
                raise GameError('Choose a highlighted pit containing a legal move.')
            saved = (self.state.clone(), len(self.moves))
            self._apply(action)
            self._rounds.append(saved)
            return self.snapshot()

    def agent_move(self, revision):
        with self._lock:
            self._check_revision(revision)
            self._check_turn(False)
            seed = (self.seed + self.state.move_number()) % 2**32
            action = self.engine.choose_action(self.state.clone(), seed)
            self._apply(action)
            return self.snapshot()

    def undo(self, revision):
        with self._lock:
            self._check_revision(revision)
            if not self._rounds:
                raise GameError('There is no move to take back.', 409)
            self.state, length = self._rounds.pop()
            del self.moves[length:]
            self._frames = []
            self.revision += 1
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            finished = self.state.is_terminal()
            result = None
            if finished:
                value = self.state.returns()[self.human_player]
                result = 'won' if value > 0 else 'lost' if value < 0 else 'draw'
            turn = 'finished' if finished else (
                'human' if self.state.current_player() == self.human_player else 'agent')
            return {'board': [int(n) for n in self.state.board],
                    'captured': [int(n) for n in self.state.captured],
                    'human_player': self.human_player, 'turn': turn,
                    'legal_actions': [int(a) for a in self.state.legal_actions()] if turn == 'human' else [],
                    'revision': self.revision, 'move_number': self.state.move_number(),
                    'moves': [dict(move) for move in self.moves[-12:]],
                    'last_move': dict(self.moves[-1]) if self.moves else None,
                    'can_undo': bool(self._rounds), 'result': result,
                    'animation_frames': self._frames,
                    'truncated': bool(self.state.truncated)}


class SessionStore:
    def __init__(self, engine, *, limit=32, ttl=4 * 60 * 60):
        self.engine, self.limit, self.ttl = engine, limit, ttl
        self._sessions = {}
        self._lock = threading.Lock()

    def _expire(self):
        now = time.monotonic()
        for token, game in list(self._sessions.items()):
            if now - game.last_used > self.ttl:
                del self._sessions[token]

    def create(self, human_player):
        with self._lock:
            self._expire()
            session = GameSession(self.engine, human_player)
            if len(self._sessions) >= self.limit:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].last_used)
                del self._sessions[oldest]
            token = secrets.token_urlsafe(32)
            self._sessions[token] = session
            return token, session

    def get(self, token):
        with self._lock:
            self._expire()
            game = self._sessions.get(token)
            if game is None:
                raise GameError('This game has expired. Start a new game.', 404)
            game.last_used = time.monotonic()
            return game
