"""Interactive replay viewer for recorded Ayo Olopon episodes.

The viewer uses only the Python standard library, so it can be used from a
notebook or a normal Python script without adding a plotting dependency.

Examples
--------
    import json
    from Model.ayo_olopon.visualizer import visualize_episode

    with open("episode_debug.json", encoding="utf-8") as stream:
        games = json.load(stream)
    visualize_episode(games[0])

An episode may be one of the dictionaries produced by the random-policy
tests (with a ``history`` field), a list of state snapshots, or a path to a
JSON file containing either form.
"""

from __future__ import annotations

import json
import html
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Iterable, Mapping, Sequence


_BOARD_COLOR = "#3b2415"
_PIT_COLOR = "#c89155"
_PIT_EDGE = "#f1c27d"
_P0_COLOR = "#fdffff"
_P1_COLOR = "#ff8a65"
_TEXT_COLOR = "#f8f4ec"


def _state_dict(state: Any) -> dict[str, Any]:
    """Convert a recorded snapshot or an OpenSpiel state to a plain dict."""
    if isinstance(state, Mapping):
        data = dict(state)
    else:
        data = {
            "board": list(state.board),
            "captured": list(state.captured),
            "current_player": state.current_player(),
            "is_terminal": state.is_terminal(),
        }
    if "board" not in data or len(data["board"]) != 12:
        raise ValueError("Every visualized state must contain a 12-pit 'board'.")
    data["board"] = [int(value) for value in data["board"]]
    data["captured"] = [int(value) for value in data.get("captured", (0, 0))]
    return data


def _snapshots(episode: Any, game_index: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(states, moves)`` from supported episode representations."""
    if isinstance(episode, (str, Path)):
        with Path(episode).open(encoding="utf-8") as stream:
            episode = json.load(stream)

    if isinstance(episode, list) and episode and isinstance(episode[0], Mapping):
        if "game_id" in episode[0] and "history" in episode[0]:
            episode = episode[game_index]
        else:
            return [_state_dict(state) for state in episode], []

    if isinstance(episode, Mapping):
        raw_states = episode.get("history") or episode.get("states")
        if raw_states is None and "initial_state" in episode:
            raw_states = [episode["initial_state"]]
        if raw_states is None:
            raise ValueError("Episode must contain 'history', 'states', or 'initial_state'.")
        return [_state_dict(state) for state in raw_states], list(episode.get("moves", []))

    raise TypeError("episode must be an episode dict, state list, or JSON path")


def _sowing_trace(state: Mapping[str, Any], player: int, action: int) -> list[dict[str, Any]]:
    """Reconstruct relay-sowing frames without mutating the game model."""
    board = list(state["board"])
    captured = list(state.get("captured", (0, 0)))
    house = player * 6 + action
    original_house = house
    seen = set()
    frames = []

    for relay in range(1, 1001):
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


def _expanded_frames(states: Sequence[Mapping[str, Any]], moves: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand move-level snapshots into relay-by-relay replay frames."""
    if not moves:
        return [dict(state) for state in states]
    frames = []
    for index, move in enumerate(moves):
        before = dict(states[index]) if index < len(states) else dict(move["state_before"])
        before["description"] = "Start position" if index == 0 else f"Move {index}: player {move.get('player', '?')} selects pit {move.get('action', '?')}"
        frames.append(before)
        frames.extend(_sowing_trace(before, int(move["player"]), int(move["action"])))
        if index + 1 < len(states):
            after = dict(states[index + 1])
            after["description"] = f"Move {index + 1} complete"
            frames.append(after)
    if len(states) > len(moves) + 1:
        frames.extend(dict(state) for state in states[len(moves) + 1:])
    return frames


def _notebook_html(states: Sequence[Mapping[str, Any]], moves: Sequence[Mapping[str, Any]], title: str) -> str:
    """Build a self-contained inline replay for Jupyter/IPython."""
    import uuid

    frames = _expanded_frames(states, moves)
    root_id = "ayo-replay-" + uuid.uuid4().hex
    states_json = json.dumps(frames, separators=(",", ":"))
    safe_title = html.escape(title)
    return f'''<div id="{root_id}" style="font-family:system-ui,sans-serif;max-width:980px;background:#3b2415;color:#f8f4ec;padding:16px;border-radius:12px">
<style>
#{root_id} .ayo-title{{text-align:center;font-size:20px;font-weight:700;margin-bottom:12px}}
#{root_id} .ayo-frame{{background:transparent;padding:8px 0}}
#{root_id} .ayo-side-label{{display:flex;justify-content:space-between;color:#083b50;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin:0 10px 8px}}
#{root_id} .ayo-board{{position:relative;background:#f6a313;border:8px solid #dc6900;border-radius:82px;padding:26px 30px;display:grid;grid-template-columns:repeat(6,1fr);gap:18px;box-shadow:none}}
#{root_id} .ayo-pit{{position:relative;background:#c94f03;border:0;border-radius:50%;height:92px;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#542000;box-sizing:border-box;transition:transform .12s,border-color .12s,box-shadow .12s}}
#{root_id} .ayo-pit .num{{position:absolute;font-size:11px;font-weight:700;color:#542000;left:50%;transform:translateX(-50%)}} #{root_id} .north .num{{top:-22px}} #{root_id} .south .num{{bottom:-22px}} #{root_id} .seed-bed{{display:flex;flex-wrap:wrap;align-content:center;justify-content:center;gap:3px;width:70px;max-height:56px;overflow:hidden}} #{root_id} .static-seed{{display:block;width:17px;height:11px;border-radius:50%;background:#f5f7fb;box-shadow:0 1px 1px #8b3000;transform:rotate(-22deg)}} #{root_id} .static-seed:nth-child(3n){{transform:rotate(24deg)}}
#{root_id} .selected{{outline:5px solid #fff1a8;outline-offset:3px;transform:scale(1.05);z-index:2}} #{root_id} .travel-seed{{position:absolute;width:19px;height:13px;border-radius:50%;background:#fff;box-shadow:0 1px 3px #7a2900;z-index:20;pointer-events:none;transform:translate(-50%,-50%);transition:left .25s linear,top .25s linear}}
#{root_id} .ayo-meta{{display:flex;justify-content:space-between;gap:12px;margin:10px 2px 0;font-size:14px;font-weight:700;flex-wrap:wrap;color:#07384b}} #{root_id} .ayo-action{{margin-top:12px;background:#fff1bf;color:#542000;border:2px solid #dc6900;border-radius:7px;padding:9px 12px;font-weight:700;min-height:20px}}
#{root_id} .ayo-controls{{display:flex;align-items:center;gap:8px;margin-top:14px;flex-wrap:wrap}} #{root_id} button{{font:inherit;padding:6px 11px;cursor:pointer}}
#{root_id} input[type=range]{{flex:1;min-width:180px}} #{root_id} .ayo-status{{margin-top:10px;font-size:13px;opacity:.9}}
</style>
<div class="ayo-title">{safe_title}</div><div class="ayo-frame"><div class="ayo-side-label"><span>North · Player 1</span><span>← counter-clockwise path →</span></div><div class="ayo-board" data-board></div><div class="ayo-side-label" style="margin-top:8px"><span>South · Player 0</span><span>12 pits · 48 seeds</span></div></div>
<div class="ayo-meta"><span data-p1></span><span data-turn></span><span data-p0></span></div>
<div class="ayo-action" data-action>Preparing replay...</div>
<div class="ayo-controls"><button data-play>Pause</button><button data-prev>&lt;</button><button data-next>&gt;</button><button data-start>|&lt;</button><button data-end>&gt;|</button><label>Speed <input data-speed type="range" min="80" max="1200" value="350"></label><input data-slider type="range" min="0" max="{len(frames)-1}" value="0"></div>
<div class="ayo-status" data-status></div>
<script>
(() => {{
  const root=document.getElementById({json.dumps(root_id)}), states={states_json};
  let frame=0, playing=true, timer;
  const board=root.querySelector('[data-board]'), slider=root.querySelector('[data-slider]');
  function draw() {{ const s=states[frame], order=[11,10,9,8,7,6,0,1,2,3,4,5];
    board.innerHTML=order.map((i,n)=>{{const cls=n<6?'north p1':'south p0'; const selected=i===(s.selected_house ?? s.source_house); const inFlight=s.phase==='sow' && i===s.landing_house; const visibleSeeds=Math.max(0,s.board[i]-(inFlight?1:0)); const seeds='<span class="static-seed"></span>'.repeat(Math.min(visibleSeeds,24)); return `<div class="ayo-pit ${{cls}} ${{selected?'selected':''}}" data-pit="${{i}}"><span class="num">${{i}}</span><span class="seed-bed">${{seeds}}</span></div>`}}).join('');
    root.querySelector('[data-p1]').textContent='Player 1 score: '+s.captured[1]; root.querySelector('[data-p0]').textContent='Player 0 score: '+s.captured[0];
    root.querySelector('[data-turn]').textContent=s.is_terminal?'GAME OVER':'Player '+s.current_player+' to move'; slider.value=frame;
    root.querySelector('[data-action]').textContent=s.description || ('Frame '+(frame+1));
    root.querySelector('[data-status]').textContent='Frame '+(frame+1)+' / '+states.length+(s.cycle?' · cycle detected':'');
    if (s.phase==='sow' && Number.isInteger(s.move_from) && Number.isInteger(s.landing_house)) {{
      const source=board.querySelector(`[data-pit="${{s.move_from}}"]`), landing=board.querySelector(`[data-pit="${{s.landing_house}}"]`);
      if (source && landing) {{ const br=board.getBoundingClientRect(), a=source.getBoundingClientRect(), b=landing.getBoundingClientRect(); const seed=document.createElement('div'); seed.className='travel-seed'; seed.style.left=(a.left+a.width/2-br.left)+'px'; seed.style.top=(a.top+a.height/2-br.top)+'px'; board.appendChild(seed); requestAnimationFrame(()=>{{seed.style.left=(b.left+b.width/2-br.left)+'px';seed.style.top=(b.top+b.height/2-br.top)+'px'}}); }}
    }}
  }}
  function step(n) {{ frame=Math.max(0,Math.min(states.length-1,n)); draw(); }}
  function tick() {{ if(!playing)return; if(frame>=states.length-1){{playing=false;root.querySelector('[data-play]').textContent='Play';return}} step(frame+1); timer=setTimeout(tick,+root.querySelector('[data-speed]').value); }}
  root.querySelector('[data-play]').onclick=()=>{{playing=!playing;root.querySelector('[data-play]').textContent=playing?'Pause':'Play';if(playing)tick()}};
  root.querySelector('[data-prev]').onclick=()=>step(frame-1); root.querySelector('[data-next]').onclick=()=>step(frame+1); root.querySelector('[data-start]').onclick=()=>step(0); root.querySelector('[data-end]').onclick=()=>step(states.length-1); slider.oninput=()=>step(+slider.value);
  draw(); tick();
}})();
</script></div>'''


def _display_in_notebook(states: Sequence[Mapping[str, Any]], moves: Sequence[Mapping[str, Any]], title: str) -> bool:
    """Display inline when called from IPython; return whether it did so."""
    try:
        get_ipython()  # noqa: F821 - provided by IPython
    except NameError:
        return False
    from IPython.display import HTML, display

    display(HTML(_notebook_html(states, moves, title)))
    return True


class _AyoReplay:
    def __init__(self, root: tk.Tk, states: Sequence[Mapping[str, Any]], moves: Sequence[Mapping[str, Any]], title: str, interval_ms: int, autoplay: bool):
        self.root = root
        self.states = states
        self.moves = moves
        self.index = 0
        self.interval_ms = max(100, int(interval_ms))
        self.playing = False

        root.title(title)
        root.configure(bg=_BOARD_COLOR)
        root.minsize(900, 570)

        self.canvas = tk.Canvas(root, width=1100, height=620, bg=_BOARD_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(12, 0))

        controls = tk.Frame(root, bg=_BOARD_COLOR)
        controls.pack(fill="x", padx=16, pady=10)
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="|<", width=4, command=lambda: self.show(0)).pack(side="left", padx=(8, 2))
        ttk.Button(controls, text="<", width=4, command=lambda: self.show(self.index - 1)).pack(side="left", padx=2)
        ttk.Button(controls, text=">", width=4, command=lambda: self.show(self.index + 1)).pack(side="left", padx=2)
        ttk.Button(controls, text=">|", width=4, command=lambda: self.show(len(self.states) - 1)).pack(side="left", padx=(2, 14))
        ttk.Label(controls, text="Speed", background=_BOARD_COLOR, foreground=_TEXT_COLOR).pack(side="left")
        self.speed = tk.IntVar(value=self.interval_ms)
        tk.Scale(controls, from_=100, to=2000, resolution=100, orient="horizontal", variable=self.speed,
                 showvalue=False, length=130, bg=_BOARD_COLOR, fg=_TEXT_COLOR, highlightthickness=0).pack(side="left")
        self.timeline = tk.Scale(controls, from_=0, to=max(0, len(states) - 1), orient="horizontal",
                                 variable=tk.IntVar(value=0), showvalue=False, command=self._timeline_changed,
                                 bg=_BOARD_COLOR, fg=_TEXT_COLOR, highlightthickness=0)
        self.timeline.pack(side="right", fill="x", expand=True, padx=(14, 0))

        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, bg=_BOARD_COLOR, fg=_TEXT_COLOR, anchor="w").pack(fill="x", padx=18, pady=(0, 10))
        self._draw()
        if autoplay:
            self.toggle_play()

    def pump(self) -> None:
        """Keep a non-blocking replay responsive inside a notebook."""
        try:
            if self.root.winfo_exists():
                self.root.update()
                self.root.after(30, self.pump)
        except tk.TclError:
            # The user closed the replay window.
            return

    def _timeline_changed(self, value: str) -> None:
        if int(float(value)) != self.index:
            self.show(int(float(value)))

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.index >= len(self.states) - 1:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.show(self.index + 1)
        self.root.after(max(100, int(self.speed.get())), self._tick)

    def show(self, index: int) -> None:
        self.index = max(0, min(len(self.states) - 1, int(index)))
        self.timeline.set(self.index)
        self._draw()

    def _draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(900, canvas.winfo_width())
        height = max(570, canvas.winfo_height())
        canvas.create_text(width / 2, 28, text="AYO OLOPON", fill=_TEXT_COLOR, font=("Segoe UI", 20, "bold"))
        state = self.states[self.index]
        board_left, board_top = 95, 105
        pit_w, pit_h, gap = max(92, (width - 250) // 6), 130, 14
        board_width = 6 * pit_w + 5 * gap
        canvas.create_rectangle(board_left - 28, board_top - 28, board_left + board_width + 28,
                                board_top + 2 * pit_h + gap + 28, fill="#74451f", outline=_PIT_EDGE, width=3)
        canvas.create_text(board_left, board_top - 45, text="PLAYER 1", anchor="w", fill=_P1_COLOR, font=("Segoe UI", 12, "bold"))
        canvas.create_text(board_left, board_top + 2 * pit_h + gap + 45, text="PLAYER 0", anchor="w", fill=_P0_COLOR, font=("Segoe UI", 12, "bold"))

        for local in range(6):
            x = board_left + local * (pit_w + gap)
            self._pit(canvas, x, board_top, pit_w, pit_h, state["board"][11 - local], 1, 11 - local)
            self._pit(canvas, x, board_top + pit_h + gap, pit_w, pit_h, state["board"][local], 0, local)

        cap_y = board_top + pit_h + gap / 2
        self._score(canvas, board_left - 86, cap_y, state["captured"][1], _P1_COLOR, "P1 score")
        self._score(canvas, board_left + board_width + 86, cap_y, state["captured"][0], _P0_COLOR, "P0 score")

        move_text = state.get("description", "Start position" if self.index == 0 else f"Frame {self.index}")
        player = state.get("current_player", "—")
        if state.get("is_terminal"):
            result = state.get("returns", "Game over")
            turn_text = f"GAME OVER  ·  {result}"
        else:
            turn_text = f"Player {player} to move"
        self.status.set(f"{move_text}    ·    {turn_text}    ·    Frame {self.index + 1}/{len(self.states)}")

    def _move_text(self, move_index: int) -> str:
        if move_index >= len(self.moves):
            return ""
        move = self.moves[move_index]
        return f"P{move.get('player', '?')} chose pit {move.get('action', '?')}"

    @staticmethod
    def _pit(canvas: tk.Canvas, x: int, y: int, width: int, height: int, seeds: int, player: int, label: int) -> None:
        canvas.create_oval(x, y, x + width, y + height, fill=_PIT_COLOR, outline=_PIT_EDGE, width=2)
        canvas.create_text(x + width / 2, y + 18, text=str(label), fill="#4b2b17", font=("Segoe UI", 10, "bold"))
        canvas.create_text(x + width / 2, y + height - 18, text=str(seeds), fill="#fff7ed", font=("Segoe UI", 17, "bold"))
        radius = 4
        for seed in range(min(seeds, 48)):
            col, row = seed % 8, seed // 8
            cx = x + width / 2 + (col - 3.5) * 8
            cy = y + 48 + row * 10
            canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                               fill=_P0_COLOR if player == 0 else _P1_COLOR, outline="")

    @staticmethod
    def _score(canvas: tk.Canvas, x: float, y: float, score: int, color: str, label: str) -> None:
        canvas.create_oval(x - 38, y - 54, x + 38, y + 54, fill="#26180e", outline=color, width=3)
        canvas.create_text(x, y - 10, text=str(score), fill=color, font=("Segoe UI", 24, "bold"))
        canvas.create_text(x, y + 25, text=label, fill=_TEXT_COLOR, font=("Segoe UI", 9))


def visualize_episode(episode: Any, *, game_index: int = 0, title: str = "Ayo Olopon Replay",
                      interval_ms: int = 700, autoplay: bool = True, block: bool = False) -> Any:
    """Open an interactive replay window for a fully played episode.

    Controls include play/pause, previous/next, jump-to-start/end, playback
    speed, and a timeline scrubber. The default is notebook-friendly: it
    returns immediately and keeps the window responsive. Set ``block=True``
    in a normal Python script when you want the call to own the event loop.
    """
    states, moves = _snapshots(episode, game_index=game_index)
    if not states:
        raise ValueError("The episode contains no states to visualize.")
    if not block and _display_in_notebook(states, moves, title):
        return None
    root = tk.Tk()
    replay_frames = _expanded_frames(states, moves)
    replay = _AyoReplay(root, replay_frames, [], title, interval_ms, autoplay)
    if block:
        root.mainloop()
    else:
        root.after(30, replay.pump)
    return replay


__all__ = ["visualize_episode"]
