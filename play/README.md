# Play against AlphaZero

Run a local browser game against a frozen trained checkpoint. The app uses the
same Ayo rules and PUCT agent as the project's AlphaZero evaluations. It loads
parameters for inference only.

From the repository root, with the environment activated:

```sh
python -m pip install -r requirements-alpha-zero.txt
python -B -m play
```

The server opens your browser at **http://127.0.0.1:8765**. Choose to play first
or second and select **Start match**. Click a highlighted pit in your bottom row,
or press **1–6**, selecting your pits from left to right. Pit labels use the
visualizer's global numbering, 0–11. The agent replies automatically. **Undo last turn** restores
the position before your last move, including the agent's reply. Up to 24 of
your turns can be taken back. **New match** starts again with the selected seat.
Press **Ctrl+C** in the terminal to stop the server.

## Local checkpoint required

The default opponent is **C600**, with **64 PUCT simulations per move**, from
`runs/alpha_zero/modal-a100-600-20261003`. That retained run exists in the local
development workspace. Checkpoints are excluded from Git; a fresh clone needs
a compatible trained run before it can serve the game.

For a compact run, retain these two files together:

```text
your-run/
  manifest.json
  inference-checkpoints/
    step-000600.npz
```

Choose a different retained run or checkpoint with:

```sh
python -B -m play --run path/to/your-run --checkpoint 600 --simulations 64
```

The manifest must match the canonical `ayo_mlp_185863_v1` architecture,
player-to-move value convention, and observation/game versions. Compact NPZ
exports also have their embedded checkpoint round and model contract checked.
Historical full checkpoint directories are supported by the existing inference
loader. The app does not download weights or substitute a random agent.

## Options and behavior

| Option | Default | Purpose |
| --- | --- | --- |
| `--run` | Local retained C600 run | Directory containing the manifest and checkpoint |
| `--checkpoint` | `600` | Positive learner round |
| `--simulations` | `64` | Search budget, from 2 to 4096; larger budgets take longer |
| `--port` | `8765` | Local HTTP port |
| `--no-browser` | Off | Print the URL without opening a browser |

CPU inference is selected unless `JAX_PLATFORMS` is already set. Startup loads
and warms the model before accepting games, so the first move can start
immediately. A busy port or missing model produces an actionable terminal error.

The server binds to `127.0.0.1`. Browser assets are local, with no CDN or build
step. Each tab stores its own game token in session storage; reloading resumes
that game. Games are held in memory, expire after four hours of inactivity,
and disappear when the server stops. At most 32 games are retained, with the
least recently used evicted when capacity is reached. Model inference is shared
and serialized; each move has its own search RNG.

The Python state is authoritative. Illegal pits, stale revisions, duplicate
requests, and moves outside the correct turn are rejected. Undo restores a full
state clone, including move count and repetition history. A failed or lost
response is reconciled against the server before offering an agent retry.

The match interface uses a carved wooden board, recessed pits, ivory seeds,
and a green playing surface. Player scores and progress toward 25 captures sit
above the board, with the active player and legal pits highlighted. Match setup,
undo, recent moves, and expandable rules sit beside the board on wider screens
and below it on mobile. All browser assets remain local.

Seed-by-seed relay animation follows the rules engine. Open **Move replay** for
**Pause**, previous/next step, speed, and timeline controls. **Skip animation**
remains beside the board during playback. **Replay move** reviews the last move without
changing the game. Finish or skip playback to continue making moves. Very long
traces show up to 600 intermediate frames, then the authoritative final position.

Small counts and legal-pit outlines accompany the seeds. Controls support keyboard
use, turn messages are announced to assistive technology, and automatic animation
is skipped for reduced-motion users. The final frame always uses the rules
engine's state, including collection at the trained game's move limit.

Run the gameplay and HTTP regression tests with:

```sh
python -B -m pytest tests/play -q
```
