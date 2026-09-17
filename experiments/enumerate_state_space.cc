#include "Model/ayo_olopon_C++/ayo_olopon.h"

#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <filesystem>
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

void WriteRecord(std::ofstream& out, const std::string& record) {
  const uint64_t size = record.size();
  out.write(reinterpret_cast<const char*>(&size), sizeof(size));
  out.write(record.data(), static_cast<std::streamsize>(size));
  if (!out) throw std::runtime_error("Failed writing frontier record");
}

bool ReadRecord(std::ifstream& in, std::string* record) {
  uint64_t size = 0;
  if (!in.read(reinterpret_cast<char*>(&size), sizeof(size))) return false;
  if (size > (1ULL << 32)) throw std::runtime_error("Corrupt frontier record");
  record->resize(static_cast<size_t>(size));
  if (!in.read(record->data(), static_cast<std::streamsize>(size)))
    throw std::runtime_error("Truncated frontier record");
  return true;
}

void ReplaceFile(const std::string& temporary, const std::string& target) {
  std::remove(target.c_str());
  if (std::rename(temporary.c_str(), target.c_str()) != 0)
    throw std::runtime_error("Unable to replace frontier file");
}

struct Checkpoint {
  uint64_t depth = 0;
  uint64_t visited_bytes = 0;
  Metrics metrics;
};

void WriteCheckpoint(const std::string& path, const Checkpoint& checkpoint) {
  const std::string temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("Cannot create checkpoint");
  const uint64_t magic = 0x41594f434b505431ULL;  // AYOCKPT1.
  const uint64_t count = checkpoint.metrics.states_by_depth.size();
  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char*>(&checkpoint.depth), sizeof(checkpoint.depth));
  out.write(reinterpret_cast<const char*>(&checkpoint.visited_bytes), sizeof(checkpoint.visited_bytes));
  out.write(reinterpret_cast<const char*>(&checkpoint.metrics.terminal_states), sizeof(uint64_t));
  out.write(reinterpret_cast<const char*>(&checkpoint.metrics.duplicate_attempts), sizeof(uint64_t));
  out.write(reinterpret_cast<const char*>(&checkpoint.metrics.transitions), sizeof(uint64_t));
  out.write(reinterpret_cast<const char*>(&checkpoint.metrics.peak_frontier), sizeof(uint64_t));
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (uint64_t value : checkpoint.metrics.states_by_depth)
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  out.close();
  if (!out) throw std::runtime_error("Failed writing checkpoint");
  ReplaceFile(temporary, path);
}

Checkpoint ReadCheckpoint(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open checkpoint");
  uint64_t magic = 0;
  Checkpoint checkpoint;
  uint64_t count = 0;
  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&checkpoint.depth), sizeof(checkpoint.depth));
  in.read(reinterpret_cast<char*>(&checkpoint.visited_bytes), sizeof(checkpoint.visited_bytes));
  in.read(reinterpret_cast<char*>(&checkpoint.metrics.terminal_states), sizeof(uint64_t));
  in.read(reinterpret_cast<char*>(&checkpoint.metrics.duplicate_attempts), sizeof(uint64_t));
  in.read(reinterpret_cast<char*>(&checkpoint.metrics.transitions), sizeof(uint64_t));
  in.read(reinterpret_cast<char*>(&checkpoint.metrics.peak_frontier), sizeof(uint64_t));
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!in || magic != 0x41594f434b505431ULL || count > 1000000)
    throw std::runtime_error("Invalid checkpoint");
  checkpoint.metrics.states_by_depth.resize(static_cast<size_t>(count));
  for (uint64_t& value : checkpoint.metrics.states_by_depth)
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error("Truncated checkpoint");
  return checkpoint;
}

void LoadVisited(const std::string& path, uint64_t bytes,
                 absl::flat_hash_set<StateKey, StateKeyHash>* visited) {
  if (bytes % sizeof(StateKey) != 0) throw std::runtime_error("Invalid visited offset");
  std::fstream log(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!log) throw std::runtime_error("Cannot open visited log");
  log.seekp(0, std::ios::end);
  const auto actual = static_cast<uint64_t>(log.tellp());
  if (actual < bytes) throw std::runtime_error("Visited log is truncated");
  if (actual > bytes) {
    log.close();
    std::filesystem::resize_file(path, bytes);
  }
  std::ifstream input(path, std::ios::binary);
  StateKey key;
  while (input.read(reinterpret_cast<char*>(&key), sizeof(key))) visited->insert(key);
  if (!input.eof()) throw std::runtime_error("Failed reading visited log");
}

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
  if (argc > 4)
    throw std::runtime_error(
        "Usage: enumerate_state_space [output.json] [max_depth] [--resume]");
  const std::string output = argc >= 2 && std::string(argv[1]) != "--resume"
                                 ? argv[1]
                                 : "ayo_state_space_results.json";
  bool resume = false;
  int max_depth = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--resume") {
      resume = true;
    } else if (i > 1 || argument != output) {
      max_depth = std::stoi(argument);
    }
  }
  if (max_depth < -1) throw std::runtime_error("max_depth must be non-negative");
  const auto start = std::chrono::steady_clock::now();
  const std::shared_ptr<const Game> game = open_spiel::LoadGame("ayo_olopon");
  absl::flat_hash_set<StateKey, StateKeyHash> visited;
  visited.reserve(50'000'000);
  visited.max_load_factor(0.80);
  Metrics m;
  const std::string visited_path = output + ".visited.bin";
  const std::string checkpoint_path = output + ".checkpoint.bin";
  uint64_t depth = 0;
  std::string frontier_path;
  if (resume) {
    const Checkpoint checkpoint = ReadCheckpoint(checkpoint_path);
    depth = checkpoint.depth;
    m = checkpoint.metrics;
    LoadVisited(visited_path, checkpoint.visited_bytes, &visited);
    frontier_path = output + ".frontier." + std::to_string(depth) + ".bin";
    if (!std::filesystem::exists(frontier_path))
      throw std::runtime_error("Checkpoint frontier is missing");
    std::cout << "Resuming at BFS depth " << depth << ", visited="
              << visited.size() << "\n";
  } else {
    frontier_path = output + ".frontier.0.bin";
    std::ofstream initial(frontier_path, std::ios::binary | std::ios::trunc);
    std::ofstream visited_log(visited_path, std::ios::binary | std::ios::trunc);
    if (!initial || !visited_log)
      throw std::runtime_error("Cannot create initial checkpoint files");
    std::unique_ptr<State> state = game->NewInitialState();
    CheckInvariant(*state, 0, -1);
    const StateKey key = CanonicalState(*state);
    visited.insert(key);
    visited_log.write(reinterpret_cast<const char*>(&key), sizeof(key));
    WriteRecord(initial, state->Serialize());
    m.states_by_depth.push_back(1);
    m.peak_frontier = 1;
    WriteCheckpoint(checkpoint_path, {0, sizeof(StateKey), m});
  }

  bool exhausted = false;
  for (;;) {
    if (max_depth >= 0 && depth >= max_depth) break;
    std::ifstream current(frontier_path, std::ios::binary);
    if (!current) throw std::runtime_error("Cannot open frontier file");
    const std::string temporary = output + ".frontier." +
                                  std::to_string(depth + 1) + ".tmp";
    std::ofstream next(temporary, std::ios::binary | std::ios::trunc);
    std::ofstream visited_log(visited_path, std::ios::binary | std::ios::app);
    if (!next || !visited_log)
      throw std::runtime_error("Cannot create checkpoint output");
    uint64_t next_count = 0;
    std::string serialized;
    while (ReadRecord(current, &serialized)) {
      std::unique_ptr<State> state = game->DeserializeState(serialized);
      if (state->IsTerminal()) {
        ++m.terminal_states;
        continue;
      }
      for (Action action : state->LegalActions()) {
        ++m.transitions;
        std::unique_ptr<State> child = state->Child(action);
        CheckInvariant(*child, static_cast<int>(depth + 1), action);
        const StateKey key = CanonicalState(*child);
        if (!visited.insert(key).second) {
          ++m.duplicate_attempts;
        } else {
          visited_log.write(reinterpret_cast<const char*>(&key), sizeof(key));
          WriteRecord(next, child->Serialize());
          ++next_count;
        }
      }
    }
    current.close();
    next.close();
    visited_log.flush();
    visited_log.close();
    if (next_count == 0) {
      exhausted = true;
      break;
    }
    m.states_by_depth.push_back(next_count);
    m.peak_frontier = std::max<uint64_t>(m.peak_frontier, next_count);
    std::cout << "BFS depth " << depth + 1 << ": frontier=" << next_count
              << ", visited=" << visited.size() << "\n" << std::flush;
    const std::string committed = output + ".frontier." +
                                  std::to_string(depth + 1) + ".bin";
    ReplaceFile(temporary, committed);
    std::ifstream visited_size(visited_path, std::ios::binary | std::ios::ate);
    const uint64_t committed_bytes = static_cast<uint64_t>(visited_size.tellg());
    WriteCheckpoint(checkpoint_path, {depth + 1, committed_bytes, m});
    if (frontier_path != committed) std::remove(frontier_path.c_str());
    frontier_path = committed;
    ++depth;
    const auto progress_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
    WriteJson(output, m, progress_elapsed.count(), PeakResidentBytes());
  }

  m.complete = exhausted;
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  const uint64_t peak_rss = PeakResidentBytes();
  WriteJson(output, m, elapsed.count(), peak_rss);
  uint64_t total = 0;
  for (uint64_t n : m.states_by_depth) total += n;
  std::cout << "Enumeration complete: " << (m.complete ? "YES" : "NO") << "\n"
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
