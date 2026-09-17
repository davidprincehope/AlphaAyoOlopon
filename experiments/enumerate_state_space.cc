#include "Model/ayo_olopon_C++/ayo_olopon.h"

#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::State;

struct StateKey {
  uint64_t low = 0;
  uint64_t high = 0;
  bool operator==(const StateKey& other) const {
    return low == other.low && high == other.high;
  }
};

struct StateKeyHash {
  size_t operator()(const StateKey& key) const {
    uint64_t x = key.low ^ (key.high + 0x9e3779b97f4a7c15ULL +
                            (key.low << 6) + (key.low >> 2));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<size_t>(x);
  }
};

StateKey CanonicalState(const State& state) {
  const auto* ayo = dynamic_cast<const open_spiel::ayo_olopon::AyoState*>(&state);
  if (ayo == nullptr) throw std::runtime_error("State is not Ayo Olopon");
  const auto& board = ayo->Board();
  if (board.seeds.size() != 12 || board.score.size() != 2)
    throw std::runtime_error("Unexpected Ayo board dimensions");

  StateKey key;
  int bit = 0;
  auto append = [&](int value, const char* field) {
    if (value < 0 || value > 48)
      throw std::runtime_error(std::string("Invalid ") + field + " value");
    const uint64_t v = static_cast<uint64_t>(value);
    if (bit < 64) {
      key.low |= v << bit;
      if (bit > 58) key.high |= v >> (64 - bit);
    } else {
      key.high |= v << (bit - 64);
    }
    bit += 6;
  };
  for (int value : board.seeds) append(value, "seed");
  for (int value : board.score) append(value, "captured");
  append(ayo->IsTerminal() ? 2 : ayo->CurrentPlayer(), "player");
  return key;
}

void CheckInvariant(const State& state, int depth, Action action) {
  const auto* ayo = dynamic_cast<const open_spiel::ayo_olopon::AyoState*>(&state);
  const auto& board = ayo->Board();
  int total = 0;
  for (int value : board.seeds) total += value;
  for (int value : board.score) total += value;
  if (total != 48) {
    std::cerr << "Invariant failure at depth " << depth << ", action "
              << action << ": total=" << total << "\n" << state << "\n";
    throw std::runtime_error("48-seed conservation invariant failed");
  }
}

uint64_t PeakResidentBytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters{};
  return GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))
             ? static_cast<uint64_t>(counters.PeakWorkingSetSize)
             : 0;
#else
  struct rusage usage{};
  return getrusage(RUSAGE_SELF, &usage) == 0
             ? static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL
             : 0;
#endif
}

struct Metrics {
  std::vector<uint64_t> states_by_depth;
  uint64_t terminal_states = 0;
  uint64_t duplicate_attempts = 0;
  uint64_t transitions = 0;
  uint64_t peak_frontier = 0;
  bool complete = false;
};

void WriteJson(const std::string& path, const Metrics& m, uint64_t elapsed_ms,
               uint64_t peak_rss) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot open output: " + path);
  uint64_t total = 0;
  for (uint64_t n : m.states_by_depth) total += n;
  out << "{\n  \"complete\": " << (m.complete ? "true" : "false")
      << ",\n  \"total_unique_states\": " << total
      << ",\n  \"states_by_depth\": [\n";
  for (size_t i = 0; i < m.states_by_depth.size(); ++i) {
    out << "    {\"depth\": " << i << ", \"states\": "
        << m.states_by_depth[i] << "}"
        << (i + 1 == m.states_by_depth.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"max_depth\": "
      << (m.states_by_depth.empty() ? 0 : m.states_by_depth.size() - 1)
      << ",\n  \"terminal_states\": " << m.terminal_states
      << ",\n  \"duplicate_attempts\": " << m.duplicate_attempts
      << ",\n  \"legal_transitions_generated\": " << m.transitions
      << ",\n  \"peak_frontier\": " << m.peak_frontier
      << ",\n  \"elapsed_seconds\": " << std::fixed << std::setprecision(3)
      << elapsed_ms / 1000.0 << ",\n  \"peak_rss_bytes\": " << peak_rss
      << "\n}\n";
}

int Run(int argc, char** argv) {
  if (argc > 2) throw std::runtime_error("Usage: enumerate_state_space [output.json]");
  const std::string output = argc == 2 ? argv[1] : "ayo_state_space_results.json";
  const auto start = std::chrono::steady_clock::now();
  const std::shared_ptr<const Game> game = open_spiel::LoadGame("ayo_olopon");
  std::vector<std::unique_ptr<State>> frontier;
  frontier.push_back(game->NewInitialState());

  absl::flat_hash_set<StateKey, StateKeyHash> visited;
  Metrics m;
  CheckInvariant(*frontier.front(), 0, -1);
  visited.insert(CanonicalState(*frontier.front()));
  m.states_by_depth.push_back(1);
  m.peak_frontier = 1;

  for (int depth = 0; !frontier.empty(); ++depth) {
    std::vector<std::unique_ptr<State>> next;
    for (const auto& state : frontier) {
      if (state->IsTerminal()) {
        ++m.terminal_states;
        continue;
      }
      for (Action action : state->LegalActions()) {
        ++m.transitions;
        std::unique_ptr<State> child = state->Child(action);
        CheckInvariant(*child, depth + 1, action);
        if (!visited.insert(CanonicalState(*child)).second) {
          ++m.duplicate_attempts;
        } else {
          next.push_back(std::move(child));
        }
      }
    }
    if (next.empty()) break;
    m.states_by_depth.push_back(next.size());
    m.peak_frontier = std::max<uint64_t>(m.peak_frontier, next.size());
    std::cout << "BFS depth " << depth + 1 << ": frontier=" << next.size()
              << ", visited=" << visited.size() << "\n" << std::flush;
    frontier = std::move(next);
  }

  m.complete = true;
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  const uint64_t peak_rss = PeakResidentBytes();
  WriteJson(output, m, elapsed.count(), peak_rss);
  uint64_t total = 0;
  for (uint64_t n : m.states_by_depth) total += n;
  std::cout << "Enumeration complete: YES\n"
            << "Total unique positional states: " << total << "\n"
            << "Maximum BFS depth: " << m.states_by_depth.size() - 1 << "\n"
            << "Terminal states: " << m.terminal_states << "\n"
            << "Duplicate attempts: " << m.duplicate_attempts << "\n"
            << "Legal transitions: " << m.transitions << "\n"
            << "Peak RSS: " << peak_rss << " bytes\n"
            << "Results: " << output << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "Enumeration failed: " << error.what() << "\n";
    return 1;
  }
}
