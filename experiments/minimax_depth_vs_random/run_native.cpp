#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Algorithms/ayo_minimax_cpp.h"
#include "Model/ayo_olopon_C++/ayo_olopon.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "nlohmann/json.hpp"

namespace {
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using open_spiel::Action;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::algorithms::AyoMinimaxBotCpp;

constexpr const char* kMinimax = "MINIMAX_HSTAR_CPP";
constexpr const char* kRandom = "RAND";

struct Options {
  int games = 10000;
  int depth = 3;
  std::uint64_t seed = 20260924;
  int max_actions = 300;
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
    else if (arg == "--depth") options.depth = std::stoi(value);
    else if (arg == "--seed") options.seed = std::stoull(value);
    else if (arg == "--max-actions") options.max_actions = std::stoi(value);
    else if (arg == "--output") options.output = value;
    else open_spiel::SpielFatalError("Unknown option: " + arg);
  }
  SPIEL_CHECK_GT(options.games, 0);
  SPIEL_CHECK_EQ(options.games % 2, 0);
  SPIEL_CHECK_GE(options.depth, 1);
  SPIEL_CHECK_GT(options.max_actions, 0);
  if (options.output.empty()) {
    options.output = std::filesystem::path(
        "experiments/minimax_depth_vs_random/results") /
        ("depth_" + std::to_string(options.depth) + "_games_" +
         std::to_string(options.games) + "_cpp.jsonl");
  }
  return options;
}

json PlayGame(const std::shared_ptr<const open_spiel::Game>& game,
              const AyoMinimaxBotCpp& bot, const Options& options,
              int game_id) {
  const int pair_id = game_id / 2;
  const std::uint64_t game_seed = options.seed + pair_id;
  const bool minimax_is_player_zero = game_id % 2 == 0;
  const char* player_0_policy = minimax_is_player_zero ? kMinimax : kRandom;
  const char* player_1_policy = minimax_is_player_zero ? kRandom : kMinimax;
  const Player minimax_player = minimax_is_player_zero ? 0 : 1;
  std::mt19937_64 rng(game_seed);

  std::unique_ptr<State> state = game->NewInitialState();
  const auto game_started = Clock::now();
  double minimax_seconds = 0.0;
  int actions = 0;
  while (!state->IsTerminal() && actions < options.max_actions) {
    const Player player = state->CurrentPlayer();
    Action action;
    if (player == minimax_player) {
      const auto search_started = Clock::now();
      action = bot.Step(*state);
      minimax_seconds += std::chrono::duration<double>(
          Clock::now() - search_started).count();
    } else {
      const std::vector<Action> legal = state->LegalActions();
      std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
      action = legal[pick(rng)];
    }
    const std::vector<Action> legal = state->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action) !=
                     legal.end());
    state->ApplyAction(action);
    ++actions;
  }

  std::vector<double> returns = state->Returns();
  const bool truncated = !state->IsTerminal();
  if (truncated) {
    const auto* ayo = dynamic_cast<const open_spiel::ayo_olopon::AyoState*>(
        state.get());
    SPIEL_CHECK_TRUE(ayo != nullptr);
    const auto& captured = ayo->Board().score;
    const double player_0_return = captured[0] > captured[1] ? 1.0 :
        captured[0] < captured[1] ? -1.0 : 0.0;
    returns = {player_0_return, -player_0_return};
  }
  const double score = returns[minimax_player];
  const char* outcome = score > 0 ? "win" : score < 0 ? "loss" : "draw";
  const char* termination = "action_limit";
  if (!truncated) {
    termination = returns[0] == 0.0 ? "terminal_draw" : "terminal_win";
    const auto* ayo = dynamic_cast<const open_spiel::ayo_olopon::AyoState*>(
        state.get());
    if (ayo != nullptr && ayo->LastRelayReport() != nullptr) {
      termination = "repetition";
    }
  }
  const auto* ayo = dynamic_cast<const open_spiel::ayo_olopon::AyoState*>(
      state.get());
  SPIEL_CHECK_TRUE(ayo != nullptr);
  const auto& captured = ayo->Board().score;

  return json{
      {"depth", options.depth},
      {"max_actions", options.max_actions},
      {"game_id", game_id},
      {"seed", game_seed},
      {"player_0_policy", player_0_policy},
      {"player_1_policy", player_1_policy},
      {"minimax_seat", minimax_player},
      {"minimax_outcome", outcome},
      {"returns_by_player", {returns[0], returns[1]}},
      {"termination_reason", termination},
      {"truncated", truncated},
      {"game_length", actions},
      {"final_captured", {captured[0], captured[1]}},
      {"minimax_decision_seconds", minimax_seconds},
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
  const std::filesystem::path summary_path =
      output_dir /
      (options.output.stem().string() + "_summary.json");
  std::vector<json> records;
  if (options.resume && std::filesystem::exists(options.output)) {
    std::ifstream input(options.output);
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      json record = json::parse(line);
      const int expected = static_cast<int>(records.size());
      SPIEL_CHECK_EQ(record.at("depth").get<int>(), options.depth);
      SPIEL_CHECK_EQ(record.at("game_id").get<int>(), expected);
      SPIEL_CHECK_EQ(record.at("seed").get<std::uint64_t>(),
                     options.seed + expected / 2);
      SPIEL_CHECK_EQ(record.at("max_actions").get<int>(), options.max_actions);
      records.push_back(std::move(record));
    }
    SPIEL_CHECK_LE(records.size(), static_cast<std::size_t>(options.games));
  }

  const auto game = open_spiel::LoadGame("ayo_olopon");
  const AyoMinimaxBotCpp bot(game, options.depth);
  std::ofstream output(options.output,
                       options.resume ? std::ios::app : std::ios::trunc);
  SPIEL_CHECK_TRUE(output.good());
  for (int game_id = static_cast<int>(records.size());
       game_id < options.games; ++game_id) {
    json record = PlayGame(game, bot, options, game_id);
    output << record.dump() << '\n';
    records.push_back(std::move(record));
    if ((game_id + 1) % 10 == 0 || game_id + 1 == options.games) {
      output.flush();
      std::cout << "native depth " << options.depth << ": " << game_id + 1
                << '/' << options.games << " games\n" << std::flush;
    }
  }
  output.close();

  int wins = 0, draws = 0, losses = 0, truncated = 0;
  double total_search_seconds = 0.0, total_game_seconds = 0.0;
  for (const json& record : records) {
    const std::string result = record.at("minimax_outcome").get<std::string>();
    wins += result == "win";
    draws += result == "draw";
    losses += result == "loss";
    truncated += record.at("truncated").get<bool>();
    total_search_seconds += record.at("minimax_decision_seconds").get<double>();
    total_game_seconds += record.at("elapsed_seconds").get<double>();
  }
  json summary{
      {"experiment", "ayo_minimax_depth_vs_random_native"},
      {"engine", "OpenSpiel C++ AlphaBetaSearch with in-place state undo"},
      {"status", "completed"},
      {"depth", options.depth},
      {"games", records.size()},
      {"master_seed", options.seed},
      {"max_actions", options.max_actions},
      {"wins", wins},
      {"draws", draws},
      {"losses", losses},
      {"truncated_games", truncated},
      {"win_rate", static_cast<double>(wins) / records.size()},
      {"score_rate", (wins + 0.5 * draws) / records.size()},
      {"total_minimax_decision_seconds", total_search_seconds},
      {"average_minimax_decision_seconds_per_game",
       total_search_seconds / records.size()},
      {"total_elapsed_seconds", total_game_seconds},
      {"log_file", options.output.filename().string()},
  };
  std::ofstream summary_file(summary_path, std::ios::trunc);
  summary_file << summary.dump(2) << '\n';
  std::cout << summary.dump(2) << '\n';
  return 0;
}
