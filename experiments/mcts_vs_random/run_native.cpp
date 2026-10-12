#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Algorithms/ayo_mcts_cpp.h"
#include "Model/ayo_olopon_C++/ayo_olopon.h"
#include "nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace {
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using open_spiel::Action;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::algorithms::AyoMctsBotCpp;

constexpr const char* kMcts = "MCTS_UCT_CPP";
constexpr const char* kRandom = "RAND";

struct Options {
  int games = 10000;
  int simulations = 1000;
  int rollouts_per_leaf = 1;
  double uct_c = 1.4142135623730951;
  std::uint64_t seed = 20260924;
  int max_actions = 300;
  std::int64_t max_memory_mb = 256;
  bool resume = false;
  std::filesystem::path output;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--resume") {
      options.resume = true;
      continue;
    }
    SPIEL_CHECK_LT(i + 1, argc);
    const std::string value = argv[++i];
    if (arg == "--games") options.games = std::stoi(value);
    else if (arg == "--simulations") options.simulations = std::stoi(value);
    else if (arg == "--rollouts-per-leaf") options.rollouts_per_leaf = std::stoi(value);
    else if (arg == "--uct-c") options.uct_c = std::stod(value);
    else if (arg == "--seed") options.seed = std::stoull(value);
    else if (arg == "--max-actions") options.max_actions = std::stoi(value);
    else if (arg == "--max-memory-mb") options.max_memory_mb = std::stoll(value);
    else if (arg == "--output") options.output = value;
    else open_spiel::SpielFatalError("Unknown option: " + arg);
  }
  SPIEL_CHECK_GT(options.games, 0);
  SPIEL_CHECK_EQ(options.games % 2, 0);
  SPIEL_CHECK_GE(options.simulations, 1);
  SPIEL_CHECK_GE(options.rollouts_per_leaf, 1);
  SPIEL_CHECK_GE(options.uct_c, 0.0);
  SPIEL_CHECK_GT(options.max_actions, 0);
  SPIEL_CHECK_GE(options.max_memory_mb, 1);
  if (options.output.empty()) {
    options.output = std::filesystem::path("experiments/mcts_vs_random/results") /
        ("mcts_uct_" + std::to_string(options.simulations) + "sims_games_" +
         std::to_string(options.games) + "_cpp.jsonl");
  }
  return options;
}

json PlayGame(const std::shared_ptr<const open_spiel::Game>& game,
              const Options& options, int game_id) {
  const std::uint64_t game_seed = options.seed + game_id / 2;
  const bool mcts_is_player_zero = game_id % 2 == 0;
  const Player mcts_player = mcts_is_player_zero ? 0 : 1;
  const char* player_0_policy = mcts_is_player_zero ? kMcts : kRandom;
  const char* player_1_policy = mcts_is_player_zero ? kRandom : kMcts;
  AyoMctsBotCpp bot(game, options.simulations, options.rollouts_per_leaf,
                    options.uct_c, static_cast<std::uint32_t>(game_seed),
                    options.max_memory_mb);
  std::mt19937_64 rng(game_seed);
  std::unique_ptr<State> state = game->NewInitialState();
  const auto game_started = Clock::now();
  double mcts_decision_seconds = 0.0;
  int actions = 0;
  while (!state->IsTerminal() && actions < options.max_actions) {
    const Player player = state->CurrentPlayer();
    Action action;
    if (player == mcts_player) {
      const auto search_started = Clock::now();
      action = bot.Step(*state);
      mcts_decision_seconds += std::chrono::duration<double>(
          Clock::now() - search_started).count();
    } else {
      const std::vector<Action> legal = state->LegalActions();
      std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
      action = legal[pick(rng)];
    }
    const std::vector<Action> legal = state->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action) != legal.end());
    state->ApplyAction(action);
    ++actions;
  }

  const auto* ayo = dynamic_cast<const open_spiel::ayo_olopon::AyoState*>(
      state.get());
  SPIEL_CHECK_TRUE(ayo != nullptr);
  const auto& captured = ayo->Board().score;
  std::vector<double> returns = state->Returns();
  const bool truncated = !state->IsTerminal();
  if (truncated) {
    const double player_0_return = captured[0] > captured[1] ? 1.0 :
        captured[0] < captured[1] ? -1.0 : 0.0;
    returns = {player_0_return, -player_0_return};
  }
  const double mcts_return = returns[mcts_player];
  const char* outcome = mcts_return > 0 ? "win" : mcts_return < 0 ? "loss" : "draw";
  const char* termination = truncated ? "action_limit" :
      returns[0] == 0.0 ? "terminal_draw" : "terminal_win";
  if (!truncated && ayo->LastRelayReport() != nullptr) termination = "repetition";

  return json{
      {"game_id", game_id},
      {"simulations_per_move", options.simulations},
      {"rollouts_per_leaf", options.rollouts_per_leaf},
      {"uct_c", options.uct_c},
      {"max_actions", options.max_actions},
      {"seed", game_seed},
      {"player_0_policy", player_0_policy},
      {"player_1_policy", player_1_policy},
      {"mcts_seat", mcts_player},
      {"mcts_outcome", outcome},
      {"returns_by_player", {returns[0], returns[1]}},
      {"termination_reason", termination},
      {"truncated", truncated},
      {"game_length", actions},
      {"final_captured", {captured[0], captured[1]}},
      {"mcts_decision_seconds", mcts_decision_seconds},
      {"elapsed_seconds", std::chrono::duration<double>(
          Clock::now() - game_started).count()},
  };
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = ParseOptions(argc, argv);
  const std::filesystem::path output_dir = options.output.has_parent_path()
      ? options.output.parent_path() : std::filesystem::path(".");
  std::filesystem::create_directories(output_dir);
  std::vector<json> records;
  if (options.resume && std::filesystem::exists(options.output)) {
    std::ifstream input(options.output);
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      json record = json::parse(line);
      const int expected_id = static_cast<int>(records.size());
      SPIEL_CHECK_EQ(record.at("game_id").get<int>(), expected_id);
      SPIEL_CHECK_EQ(record.at("seed").get<std::uint64_t>(),
                     options.seed + expected_id / 2);
      SPIEL_CHECK_EQ(record.at("simulations_per_move").get<int>(),
                     options.simulations);
      SPIEL_CHECK_EQ(record.at("rollouts_per_leaf").get<int>(),
                     options.rollouts_per_leaf);
      SPIEL_CHECK_EQ(record.at("uct_c").get<double>(), options.uct_c);
      SPIEL_CHECK_EQ(record.at("max_actions").get<int>(), options.max_actions);
      SPIEL_CHECK_LT(records.size(), static_cast<std::size_t>(options.games));
      records.push_back(std::move(record));
    }
  }

  const auto game = open_spiel::LoadGame("ayo_olopon");
  std::ofstream output(options.output,
                       options.resume ? std::ios::app : std::ios::trunc);
  SPIEL_CHECK_TRUE(output.good());
  for (int game_id = static_cast<int>(records.size());
       game_id < options.games; ++game_id) {
    json record = PlayGame(game, options, game_id);
    output << record.dump() << '\n';
    records.push_back(std::move(record));
    if ((game_id + 1) % 10 == 0 || game_id + 1 == options.games) {
      output.flush();
      std::cout << "native MCTS: " << game_id + 1 << '/' << options.games
                << " games\n" << std::flush;
    }
  }
  output.close();

  int wins = 0, draws = 0, losses = 0, truncated = 0;
  double total_search_seconds = 0.0, total_game_seconds = 0.0;
  for (const json& record : records) {
    const std::string result = record.at("mcts_outcome").get<std::string>();
    wins += result == "win";
    draws += result == "draw";
    losses += result == "loss";
    truncated += record.at("truncated").get<bool>();
    total_search_seconds += record.at("mcts_decision_seconds").get<double>();
    total_game_seconds += record.at("elapsed_seconds").get<double>();
  }
  const auto summary_path = output_dir /
      (options.output.stem().string() + "_summary.json");
  json summary{
      {"experiment", "ayo_mcts_vs_random_native"},
      {"engine", "OpenSpiel C++ MCTSBot (UCT) with random rollouts"},
      {"status", "completed"},
      {"games", records.size()},
      {"simulations_per_move", options.simulations},
      {"rollouts_per_leaf", options.rollouts_per_leaf},
      {"uct_c", options.uct_c},
      {"master_seed", options.seed},
      {"max_actions", options.max_actions},
      {"wins", wins}, {"draws", draws}, {"losses", losses},
      {"truncated_games", truncated},
      {"win_rate", static_cast<double>(wins) / records.size()},
      {"score_rate", (wins + 0.5 * draws) / records.size()},
      {"total_mcts_decision_seconds", total_search_seconds},
      {"average_mcts_decision_seconds_per_game",
       total_search_seconds / records.size()},
      {"total_elapsed_seconds", total_game_seconds},
      {"log_file", options.output.filename().string()},
  };
  std::ofstream summary_file(summary_path, std::ios::trunc);
  summary_file << summary.dump(2) << '\n';
  std::cout << summary.dump(2) << '\n';
  return 0;
}
