# Ayo Olopon C++ OpenSpiel model

The model follows the structure of the existing `OpenSpiel/oware` game:

- `ayo_olopon_board.h/.cc` stores positional board data.
- `ayo_olopon.h/.cc` implements the OpenSpiel game and state transitions.
- `../../tests/ayo_olopon_C++/ayo_olopon_test.cc` contains parity tests.

## Build and run

The repository includes an OpenSpiel checkout under `open_spiel`.
Its CMake entry point is `open_spiel/open_spiel/CMakeLists.txt`.

From the repository root:

```powershell
cmake -S Model/ayo_olopon_C++ -B build/ayo_olopon_cpp \
  -G "MinGW Makefiles" \
  -DOPEN_SPIEL_ROOT=/c/Users/user/Documents/AlphaAyoOlopon/open_spiel/open_spiel \
  -DOPEN_SPIEL_BUILD_WITH_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/ayo_olopon_cpp --target ayo_olopon_cpp_tests
ctest --test-dir build/ayo_olopon_cpp --output-on-failure
```

The state-space experiment is built and run with:

```bash
cmake --build build/ayo_olopon_cpp --target ayo_state_space_enumerator
./build/ayo_olopon_cpp/ayo_state_space_enumerator ayo_state_space_results.json
```

For a bounded smoke test, provide a maximum depth, for example:

```bash
./build/ayo_olopon_cpp/ayo_state_space_enumerator smoke.json 3
```

Run these commands from the MSYS2 UCRT64 terminal so `g++`, `mingw32-make`,
and their runtime libraries are on `PATH`. The enumerator stops only when the
frontier is empty and reports `Enumeration complete: YES`; its output JSON
contains the per-depth counts and all requested metrics.
