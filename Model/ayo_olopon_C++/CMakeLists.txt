cmake_minimum_required(VERSION 3.17)
project(ayo_olopon_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT OPEN_SPIEL_ROOT)
  message(FATAL_ERROR
    "Set OPEN_SPIEL_ROOT to open_spiel-master/open_spiel")
endif()

# OpenSpiel's bundled CMake uses environment variables for these switches.
# Forward the normal -D options so this wrapper does not accidentally enable
# Python or fall back to its Testing build type.
if(DEFINED OPEN_SPIEL_BUILD_WITH_PYTHON)
  set(ENV{OPEN_SPIEL_BUILD_WITH_PYTHON} "${OPEN_SPIEL_BUILD_WITH_PYTHON}")
endif()
if(DEFINED CMAKE_BUILD_TYPE)
  set(ENV{BUILD_TYPE} "${CMAKE_BUILD_TYPE}")
endif()

# This is the CMake entry point in the bundled OpenSpiel clone.
# The enumerator embeds its own game and needs only OpenSpiel's core. Avoid
# configuring the bundled project's unrelated games, examples, and tests.
set(OPEN_SPIEL_MINIMAL_BUILD ON)
add_subdirectory(${OPEN_SPIEL_ROOT} ${CMAKE_BINARY_DIR}/open_spiel)

set(REPOSITORY_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

add_executable(ayo_olopon_cpp_tests
  ${CMAKE_CURRENT_LIST_DIR}/ayo_olopon.cc
  ${CMAKE_CURRENT_LIST_DIR}/ayo_olopon_board.cc
  ${REPOSITORY_ROOT}/tests/ayo_olopon_C++/ayo_olopon_test.cc
  $<TARGET_OBJECTS:open_spiel_core>
)

target_include_directories(ayo_olopon_cpp_tests PRIVATE
  ${REPOSITORY_ROOT}
  ${OPEN_SPIEL_ROOT}/..
  ${OPEN_SPIEL_ROOT}/abseil-cpp
  ${OPEN_SPIEL_ROOT}/json/include
)

target_link_libraries(ayo_olopon_cpp_tests PRIVATE
  absl::algorithm
  absl::flat_hash_map
  absl::optional
  absl::random_random
  absl::str_format
  absl::strings
  absl::time
)

enable_testing()
add_test(NAME ayo_olopon_cpp_tests COMMAND ayo_olopon_cpp_tests)

add_executable(ayo_minimax_native
  ${REPOSITORY_ROOT}/Algorithms/ayo_minimax_cpp.cc
  ${CMAKE_CURRENT_LIST_DIR}/ayo_olopon.cc
  ${CMAKE_CURRENT_LIST_DIR}/ayo_olopon_board.cc
  ${REPOSITORY_ROOT}/experiments/minimax_depth_vs_random/run_native.cpp
  ${OPEN_SPIEL_ROOT}/algorithms/minimax.cc
  $<TARGET_OBJECTS:open_spiel_core>
)
target_include_directories(ayo_minimax_native PRIVATE
  ${REPOSITORY_ROOT}
  ${OPEN_SPIEL_ROOT}/..
  ${OPEN_SPIEL_ROOT}/abseil-cpp
  ${OPEN_SPIEL_ROOT}/json/include
)
target_link_libraries(ayo_minimax_native PRIVATE
  absl::algorithm
  absl::flat_hash_map
  absl::optional
  absl::random_random
  absl::str_format
  absl::strings
  absl::time
)

add_executable(ayo_state_space_enumerator
  ${CMAKE_CURRENT_LIST_DIR}/ayo_olopon.cc
  ${CMAKE_CURRENT_LIST_DIR}/ayo_olopon_board.cc
  ${REPOSITORY_ROOT}/experiments/enumerate_state_space.cc
  $<TARGET_OBJECTS:open_spiel_core>
)
target_include_directories(ayo_state_space_enumerator PRIVATE
  ${REPOSITORY_ROOT}
  ${OPEN_SPIEL_ROOT}/..
  ${OPEN_SPIEL_ROOT}/abseil-cpp
  ${OPEN_SPIEL_ROOT}/json/include
)
target_link_libraries(ayo_state_space_enumerator PRIVATE
  absl::algorithm
  absl::flat_hash_map
  absl::optional
  absl::random_random
  absl::str_format
  absl::strings
  absl::time
)
if(WIN32)
  target_link_libraries(ayo_state_space_enumerator PRIVATE psapi)
endif()
