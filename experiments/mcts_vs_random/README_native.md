# Native OpenSpiel MCTS

`Algorithms/ayo_mcts_cpp.h` exposes `AyoMctsBotCpp`, backed by OpenSpiel's
C++ `MCTSBot` with UCT selection and `RandomRolloutEvaluator`. Defaults match
the Python adapter: 1,000 simulations per move, one random rollout per leaf,
exploration constant `sqrt(2)`, and solver backups disabled.

Build the native executable from the repository root:

```powershell
$env:PATH = "C:\ProgramData\mingw64\mingw64\bin;$env:PATH"
cmake -S Model\ayo_olopon_C++ -B build\ayo_olopon_cpp_working `
  -G "MinGW Makefiles" `
  -DOPEN_SPIEL_ROOT="$PWD\open_spiel\open_spiel" `
  -DOPEN_SPIEL_BUILD_WITH_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build\ayo_olopon_cpp_working --target ayo_mcts_native -- -j2
```

Run a benchmark against random play:

```powershell
$env:PATH = "C:\ProgramData\mingw64\mingw64\bin;$env:PATH"
build\ayo_olopon_cpp_working\ayo_mcts_native.exe `
  --games 10000 --simulations 1000 --rollouts-per-leaf 1 `
  --uct-c 1.4142135623730951 --seed 20260924 --max-actions 300
```

The runner writes a JSONL record after each game and a summary when complete.
It alternates seats in pairs, uses the same seed for each pair, and accepts
`--resume` to validate and continue an interrupted log. Change the compute
budget with `--simulations`; use `--output <path>` to choose another log.
Random sequences differ from Python's NumPy RNG, while the UCT policy and
random-rollout evaluation follow the same OpenSpiel algorithm and settings.
