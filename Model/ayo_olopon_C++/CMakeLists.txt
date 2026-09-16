cmake_minimum_required(VERSION 3.17)
project(ayo_olopon_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT OPEN_SPIEL_ROOT)
  message(FATAL_ERROR
    "Set OPEN_SPIEL_ROOT to open_spiel-master/open_spiel")
endif()

# This is the CMake entry point in the bundled OpenSpiel clone.
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
  ${OPEN_SPIEL_ROOT}
  ${OPEN_SPIEL_ROOT}/abseil-cpp
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
