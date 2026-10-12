"""Browser turns use the real adapter, including horizon and repetition history."""
from concurrent.futures import ThreadPoolExecutor

import pyspiel
import pytest

from Algorithms.alpha_zero import game as adapter
from Algorithms.alpha_zero.config import Settings
from play.session import GameError, GameSession, SessionStore


class FirstLegalEngine:
    def __init__(self, max_moves=12):
        self.game = pyspiel.load_game(Settings(max_moves=max_moves).game_string)
        self.info = {'name': 'AlphaZero', 'checkpoint': 600, 'simulations': 64,
                     'max_moves': max_moves}
        self.fail = False

    def choose_action(self, state, seed):
        if self.fail:
            # Even a faulty agent cannot mutate the authoritative state.
            state.apply_action(state.legal_actions()[0])
            raise RuntimeError('inference failed')
        return state.legal_actions()[0]


@pytest.mark.parametrize('seat', [0, 1])
def test_both_seats_and_full_round_undo_preserve_state_history(seat):
    session = GameSession(FirstLegalEngine(), seat, seed=7)
    if seat == 1:
        session.agent_move(0)
    before = session.state.clone()
    old_snapshot = session.snapshot()
    action = session.state.legal_actions()[0]
    after_human = session.human_move(action, old_snapshot['revision'])
    session.agent_move(after_human['revision'])
    assert sum(session.state.board) + sum(session.state.captured) == 48
    restored = session.undo(session.revision)
    assert restored['turn'] == 'human'
    assert restored['board'] == old_snapshot['board']
    assert restored['captured'] == old_snapshot['captured']
    assert session.state.history() == before.history()
    assert session.state._positions_since_capture == before._positions_since_capture
    assert session.state.observation_tensor() == before.observation_tensor()
    assert restored['revision'] > after_human['revision']
    assert restored['move_number'] == seat
    assert not restored['can_undo']
    # Replaying the same move after undo yields precisely the same engine state.
    before.apply_action(action)
    replayed = session.human_move(action, restored['revision'])
    assert replayed['board'] == before.board
    assert replayed['captured'] == before.captured
    assert session.state.history() == before.history()


@pytest.mark.parametrize('action', [-1, 6, True, '0', None])
def test_invalid_human_moves_do_not_change_state(action):
    session = GameSession(FirstLegalEngine())
    before = session.snapshot()
    with pytest.raises(GameError):
        session.human_move(action, 0)
    assert session.snapshot() == before


def test_wrong_turn_stale_revision_and_duplicate_requests_are_rejected():
    session = GameSession(FirstLegalEngine())
    with pytest.raises(GameError, match='your turn'):
        session.agent_move(0)
    with pytest.raises(GameError, match='board has changed'):
        session.human_move(0, True)

    def submit():
        try:
            session.human_move(0, 0)
            return 200
        except GameError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: submit(), range(2)))
    assert sorted(results) == [200, 409]
    assert session.state.move_number() == 1
    with pytest.raises(GameError, match='not your turn'):
        session.human_move(0, 1)
    with pytest.raises(GameError, match='board has changed'):
        session.agent_move(0)


def test_inference_failure_leaves_game_retryable_and_unchanged():
    engine = FirstLegalEngine()
    session = GameSession(engine, 1)
    before = session.snapshot()
    engine.fail = True
    with pytest.raises(RuntimeError):
        session.agent_move(0)
    assert session.snapshot() == before
    engine.fail = False
    assert session.agent_move(0)['turn'] == 'human'


def test_horizon_result_terminal_rejection_and_undo():
    session = GameSession(FirstLegalEngine(max_moves=2))
    after_human = session.human_move(0, 0)
    finished = session.agent_move(after_human['revision'])
    assert finished['turn'] == 'finished'
    assert finished['result'] in ('won', 'lost', 'draw')
    assert finished['truncated']
    assert sum(finished['captured']) == 48 and sum(finished['board']) == 0
    with pytest.raises(GameError, match='finished'):
        session.human_move(0, finished['revision'])
    restored = session.undo(finished['revision'])
    assert restored['turn'] == 'human' and restored['move_number'] == 0
    assert restored['board'] == [4] * 12 and not restored['truncated']


@pytest.mark.parametrize('seat', [0, 1])
def test_animation_has_pickup_sowing_and_authoritative_final_frame(seat):
    session = GameSession(FirstLegalEngine(max_moves=4), seat)
    if seat == 1:
        game = session.agent_move(0)
    else:
        game = session.human_move(0, 0)
    frames = game['animation_frames']
    assert frames[0]['phase'] == 'pickup'
    assert any(frame['phase'] == 'sow' for frame in frames)
    for frame in frames:
        # Seeds absent from the board and scores are held by the sowing player.
        held = 48 - sum(frame['board']) - sum(frame['captured'])
        assert 0 <= held <= 48
        if 'hand_seeds' in frame:
            assert held == frame['hand_seeds']
    assert frames[-1]['board'] == game['board']
    assert frames[-1]['captured'] == game['captured']
    assert frames[-1]['phase'] == 'complete'
    assert not frames[-1]['is_terminal']


def test_animation_ends_at_collected_terminal_state_and_is_cleared_by_undo():
    session = GameSession(FirstLegalEngine(max_moves=2))
    session.human_move(0, 0)
    game = session.agent_move(1)
    last = game['animation_frames'][-1]
    assert last['is_terminal'] and last['phase'] == 'complete'
    assert last['board'] == [0] * 12
    assert last['captured'] == game['captured']
    assert sum(last['captured']) == 48
    restored = session.undo(game['revision'])
    assert restored['animation_frames'] == []


def test_empty_opponent_row_only_allows_feeding_pits():
    session = GameSession(FirstLegalEngine())
    session.state.board = [1, 1, 1, 1, 1, 1] + [0] * 6
    session.state.captured = [20, 22]
    assert session.snapshot()['legal_actions'] == [5]
    before = session.snapshot()
    with pytest.raises(GameError):
        session.human_move(0, 0)
    assert session.snapshot() == before


def test_store_isolates_games_expires_and_does_not_evict_for_invalid_seat():
    store = SessionStore(FirstLegalEngine(), limit=2, ttl=100)
    token_a, a = store.create(0)
    token_b, b = store.create(1)
    a.human_move(0, 0)
    assert b.snapshot()['board'] == [4] * 12
    assert token_a != token_b
    with pytest.raises(GameError):
        store.create(True)
    assert store.get(token_a) is a
    assert store.get(token_b) is b
    a.last_used -= 101
    with pytest.raises(GameError, match='expired'):
        store.get(token_a)
    assert store.get(token_b) is b
