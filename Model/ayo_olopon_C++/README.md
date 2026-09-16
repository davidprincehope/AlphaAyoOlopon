# Ayo Olopon C++ OpenSpiel model

The model follows the structure of the existing `OpenSpiel/oware` game:

- `ayo_olopon_board.h/.cc` stores positional board data.
- `ayo_olopon.h/.cc` implements the OpenSpiel game and state transitions.
- `../../tests/ayo_olopon_C++/ayo_olopon_test.cc` contains parity tests.

## Build and run

The repository includes a full OpenSpiel checkout under `open_spiel-master`.
Its CMake entry point is `open_spiel-master/open_spiel/CMakeLists.txt`.

From the repository root:

```powershell
cmake -S Model/ayo_olopon_C++ -B build/ayo_olopon_cpp `
  -DOPEN_SPIEL_ROOT=C:/path/to/AlphaAyoOlopon/open_spiel-master/open_spiel
cmake --build build/ayo_olopon_cpp --config Release --target ayo_olopon_cpp_tests
ctest --test-dir build/ayo_olopon_cpp -C Release --output-on-failure
```

The executable is under `build/ayo_olopon_cpp/` after compilation.
