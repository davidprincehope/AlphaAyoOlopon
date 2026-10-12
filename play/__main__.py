"""Start with python -B -m play; checkpoint files remain local and read-only."""
import argparse
import os
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Play Ayo Olopon against a trained AlphaZero agent in your browser.')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/alpha_zero/modal-a100-600-20261003')
    parser.add_argument('--checkpoint', type=int, default=600)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    try:
        from play.engine import AlphaZeroEngine
        from play.server import GameServer
        print(f'Loading AlphaZero C{args.checkpoint}...', flush=True)
        engine = AlphaZeroEngine(args.run, args.checkpoint, args.simulations)
        server = GameServer(engine, args.port)
    except FileNotFoundError as exc:
        parser.error(f'Missing local model data: {exc}. Supply --run with a training manifest and retained checkpoint; see play/README.md.')
    except ImportError as exc:
        parser.error(f'Missing inference dependency: {exc}. Install requirements-alpha-zero.txt first.')
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Play at {url} (Ctrl+C to stop)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nGame server stopped.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
