"""Exercise the local HTTP boundary with a deterministic engine and real rules."""
from http.client import HTTPConnection
import json
import threading
from types import SimpleNamespace

import pyspiel
import pytest

from Algorithms.alpha_zero import game as adapter
from Algorithms.alpha_zero.config import Settings
from play.server import GameHandler, GameServer
from play.session import GameError


class TestEngine:
    __test__ = False

    def __init__(self):
        self.game = pyspiel.load_game(Settings(max_moves=4).game_string)
        self.info = {'name': 'AlphaZero', 'checkpoint': 600, 'simulations': 64,
                     'max_moves': 4}

    def choose_action(self, state, seed):
        return state.legal_actions()[0]


@pytest.fixture
def send_request():
    server = GameServer(TestEngine(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def send(path, body=None, token=None, headers=None):
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        sent_headers = {'Content-Type': 'application/json'}
        if token:
            sent_headers['Authorization'] = f'Bearer {token}'
        sent_headers.update(headers or {})
        payload = None if body is None else json.dumps(body)
        connection.request('GET' if body is None else 'POST', path, payload, sent_headers)
        response = connection.getresponse()
        data = response.read()
        mime = response.getheader('Content-Type')
        result = (response.status, json.loads(data) if mime.startswith('application/json') else data,
                  dict(response.getheaders()))
        connection.close()
        return result

    yield send
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_assets_info_auth_and_no_filesystem_access(send_request):
    for path in ('/', '/style.css', '/app.js', '/favicon.svg'):
        status, data, headers = send_request(path)
        assert status == 200 and data
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert "script-src 'self'" in headers['Content-Security-Policy']
    assert send_request('/api/info')[1]['checkpoint'] == 600
    assert send_request('/api/game')[0] == 401
    assert send_request('/api/game', token='missing')[0] == 404
    for path in ('/../README.md', '/manifest.json', '/.git/config', '/static/../../README.md'):
        assert send_request(path)[0] == 404


def test_http_game_flow_independent_sessions_and_terminal_undo(send_request):
    status, data, _ = send_request('/api/new', {'human_player': 0})
    assert status == 201
    token = data['session_id']
    other = send_request('/api/new', {'human_player': 1})[1]['session_id']
    game = send_request('/api/move', {'action': 0, 'revision': 0}, token)[1]
    assert game['turn'] == 'agent'
    assert send_request('/api/move', {'action': 0, 'revision': 0}, token)[0] == 409
    assert send_request('/api/game', token=other)[1]['move_number'] == 0
    game = send_request('/api/agent', {'revision': game['revision']}, token)[1]
    assert game['turn'] == 'human' and game['move_number'] == 2
    game = send_request('/api/move', {'action': game['legal_actions'][0], 'revision': game['revision']}, token)[1]
    game = send_request('/api/agent', {'revision': game['revision']}, token)[1]
    assert game['turn'] == 'finished'
    status, restored, _ = send_request('/api/undo', {'revision': game['revision']}, token)
    assert status == 200 and restored['move_number'] == 2 and restored['turn'] == 'human'


def test_cross_origin_requests_and_bad_input_are_rejected(send_request):
    assert send_request('/api/new', {}, headers={'Origin': 'https://example.com'})[0] == 403
    assert send_request('/api/info', headers={'Host': 'example.com'})[0] == 403
    assert send_request('/api/new', {}, headers={'Content-Type': 'text/plain'})[0] == 415
    assert send_request('/api/new', [0])[0] == 400
    assert send_request('/api/new', {'human_player': True})[0] == 400
    assert send_request('/api/new', {'human_player': 2})[0] == 400
    assert send_request('/api/new', {'padding': 'x' * 5000})[0] == 413
    token = send_request('/api/new', {})[1]['session_id']
    assert send_request('/api/move', {'action': 0}, token)[0] == 409
    assert send_request('/api/move', {'action': True, 'revision': 0}, token)[0] == 400


def test_default_http_port_accepts_normalized_local_host_and_origin():
    handler = object.__new__(GameHandler)
    handler.server = SimpleNamespace(server_port=80)
    for host in ('127.0.0.1', 'localhost'):
        handler.headers = {'Host': host, 'Origin': f'http://{host}'}
        handler._check_request()
    handler.headers = {'Host': 'localhost', 'Origin': 'http://example.com'}
    with pytest.raises(GameError):
        handler._check_request()


def test_another_server_cannot_silently_share_the_same_port():
    server = GameServer(TestEngine(), port=0)
    try:
        with pytest.raises(OSError):
            GameServer(TestEngine(), port=server.server_port)
    finally:
        server.server_close()
