# Ayo Olopon

A Python implementation of the Ayo Olopon board game built on the OpenSpiel framework. The project includes a custom game definition, relay-sowing logic, cycle detection, and a test suite covering gameplay rules and terminal-state handling.

## Overview

Ayo Olopon is a two-player turn-based sowing game. This repository models the game as an OpenSpiel environment so it can be used with RL tooling, game-tree search, and rule-based validation.

The core game logic is implemented in:

- `Model/ayo_olopon/ayo_olopon.py`

The project also includes:

- `Model/oware/oware.py` — a related Oware implementation
- `tests/ayo_olopon/test_ayo_olopon.py` — Ayo-specific validation tests
- `tests/oware/test_oware.py` — Oware validation tests
- `OpenSpiel/` — OpenSpiel examples and C++ components

## Features

- Two-player sequential game implementation
- Standard Ayo/Oware sowing rules
- Capture logic and relay sowing
- Repetition / cycle detection with reporting
- Observation tensor support for RL-style inputs
- Registration as an OpenSpiel game
- Pytest coverage for rule behavior and state transitions

## Project Layout

```text
AlphaAyoOlopon/
├── Algorithms/
├── Game_complexity/
├── Model/
│   ├── ayo_olopon/
│   ├── jekinje/
│   └── oware/
├── OpenSpiel/
├── Resources/
├── tests/
│   ├── ayo_olopon/
│   └── oware/
├── episode_debug.json
├── .gitignore
├── README.md
└── ...
```

## Requirements

- Python 3.10+
- NumPy
- OpenSpiel / PySpiel
- pytest

## Setup

From the project root, install the required dependencies:

```bash
pip install numpy pytest
```

If your environment already includes OpenSpiel, you can proceed directly. If not, install the library compatible with your Python environment, then confirm the game loads correctly.

## Running the Game

You can load the game through PySpiel:

```python
import pyspiel

game = pyspiel.load_game("ayo_olopon")
state = game.new_initial_state()
print(state)
print(state.legal_actions())
```

## Running Tests

Run the Ayo test suite from the project root:

```bash
python -m pytest tests/ayo_olopon/test_ayo_olopon.py
```

You can also run the full project tests:

```bash
python -m pytest
```

## Notes

- The registered game name is `ayo_olopon`.
- The state observation tensor is normalized using the total number of seeds in the position.
- The implementation includes support for relay-cycle reporting when `enable_cycle_reporting` is enabled.

## License

This project does not currently declare a license file in the repository. If you plan to share or distribute it publicly, add an appropriate license before publication.
