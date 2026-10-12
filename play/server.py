"""A loopback-only JSON service and a fixed set of browser assets."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from pathlib import Path
import traceback
from urllib.parse import urlsplit

from play.session import GameError, SessionStore

STATIC = Path(__file__).resolve().parent / 'static'
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/favicon.svg': ('favicon.svg', 'image/svg+xml')}


class GameServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = not hasattr(socket, 'SO_EXCLUSIVEADDRUSE')

    def __init__(self, engine, port=8765):
        super().__init__(('127.0.0.1', port), GameHandler)
        self.engine = engine
        self.sessions = SessionStore(engine)

    def server_bind(self):
        # Windows otherwise allows multiple local servers on the same port,
        # leaving browsers connected to an older imported version of the app.
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class GameHandler(BaseHTTPRequestHandler):
    def _send(self, status, data, content_type='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; "
                         "style-src 'self'; connect-src 'self'; img-src 'self'; "
                         "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        try:
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            # Starting another game aborts pending browser requests. The move
            # may already be committed; reconnecting reads its saved revision.
            pass

    def _check_request(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.server.server_port == 80:
            allowed.update({'127.0.0.1', 'localhost'})
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        if host not in allowed or origin is not None and origin != f'http://{host}':
            raise GameError('This app accepts requests from its local browser page only.', 403)

    def _body(self):
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            raise GameError('Send an application/json request.', 415)
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            raise GameError('Invalid request length.') from None
        if size < 0 or size > 4096:
            raise GameError('The request is too large.', 413)
        try:
            body = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeError):
            raise GameError('Invalid JSON request.') from None
        if not isinstance(body, dict):
            raise GameError('Send a JSON object.')
        return body

    def _session(self):
        authorization = self.headers.get('Authorization', '')
        if not authorization.startswith('Bearer '):
            raise GameError('Start a game first.', 401)
        return self.server.sessions.get(authorization[7:])

    def _dispatch(self, post=False):
        try:
            self._check_request()
            path = urlsplit(self.path).path
            if not post:
                if path == '/api/info':
                    self._send(200, self.server.engine.info)
                elif path == '/api/game':
                    self._send(200, self._session().snapshot())
                elif path in ASSETS:
                    filename, mime = ASSETS[path]
                    self._send(200, (STATIC / filename).read_bytes(), mime)
                else:
                    raise GameError('Page not found.', 404)
                return
            body = self._body()
            if path == '/api/new':
                token, session = self.server.sessions.create(body.get('human_player', 0))
                self._send(201, {'session_id': token, 'game': session.snapshot()})
                return
            if path not in ('/api/move', '/api/agent', '/api/undo'):
                raise GameError('Page not found.', 404)
            session = self._session()
            revision = body.get('revision')
            if path == '/api/move':
                game = session.human_move(body.get('action'), revision)
            elif path == '/api/agent':
                game = session.agent_move(revision)
            else:
                game = session.undo(revision)
            self._send(200, game)
        except GameError as exc:
            self._send(exc.status, {'error': str(exc)})
        except ConnectionError:
            pass
        except Exception:
            traceback.print_exc()
            self._send(500, {'error': 'The move could not be completed. Try again or start a new game.'})

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch(post=True)

    def do_OPTIONS(self):
        self._send(405, {'error': 'Use the local browser page.'})
