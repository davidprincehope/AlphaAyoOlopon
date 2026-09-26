#include "Algorithms/ayo_minimax_cpp.h"

#include <algorithm>

#include "Model/ayo_olopon_C++/ayo_olopon.h"
#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

double HStarValue(const State& state, Player player) {
  const auto* ayo_state = dynamic_cast<const ayo_olopon::AyoState*>(&state);
  SPIEL_CHECK_TRUE(ayo_state != nullptr);
  const ayo_olopon::AyoBoard& board = ayo_state->Board();
  const int other = 1 - player;
  const int houses = board.seeds.size() / 2;
  int own_seeds = 0;
  int other_seeds = 0;
  int own_mobility = 0;
  int other_mobility = 0;
  for (int house = 0; house < houses; ++house) {
    const int own = board.seeds[player * houses + house];
    const int opponent = board.seeds[other * houses + house];
    own_seeds += own;
    other_seeds += opponent;
    own_mobility += own > 0;
    other_mobility += opponent > 0;
  }
  const int total = own_seeds + other_seeds;
  const double seed_difference = own_seeds - other_seeds;
  const int mobility_difference = own_mobility - other_mobility;
  const double score =
      0.80 * (board.score[player] - board.score[other]) +
      0.18 * seed_difference / (1 + total) +
      0.30 * mobility_difference;
  return std::max(-10.0, std::min(10.0, score));
}

}  // namespace

AyoMinimaxBotCpp::AyoMinimaxBotCpp(std::shared_ptr<const Game> game,
                                   int maximum_depth)
    : game_(std::move(game)), maximum_depth_(maximum_depth) {
  SPIEL_CHECK_TRUE(game_ != nullptr);
  SPIEL_CHECK_EQ(game_->GetType().short_name, "ayo_olopon");
  SPIEL_CHECK_GE(maximum_depth_, 1);
}

std::pair<double, Action> AyoMinimaxBotCpp::Search(const State& state) const {
  const Player maximizing_player = state.CurrentPlayer();
  return AlphaBetaSearch(
      *game_, &state,
      [maximizing_player](const State& leaf) {
        return HStarValue(leaf, maximizing_player);
      },
      maximum_depth_, maximizing_player, /*use_undo=*/true);
}

Action AyoMinimaxBotCpp::Step(const State& state) const {
  return Search(state).second;
}

}  // namespace algorithms
}  // namespace open_spiel
