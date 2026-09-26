#include "Algorithms/ayo_mcts_cpp.h"

#include <cmath>
#include <utility>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

AyoMctsBotCpp::AyoMctsBotCpp(std::shared_ptr<const Game> game,
                             int simulations, int rollouts_per_leaf,
                             double uct_c, std::uint32_t seed,
                             std::int64_t max_memory_mb)
    : game_(std::move(game)) {
  SPIEL_CHECK_TRUE(game_ != nullptr);
  SPIEL_CHECK_EQ(game_->GetType().short_name, "ayo_olopon");
  SPIEL_CHECK_GE(simulations, 1);
  SPIEL_CHECK_GE(rollouts_per_leaf, 1);
  SPIEL_CHECK_TRUE(std::isfinite(uct_c));
  SPIEL_CHECK_GE(uct_c, 0.0);
  SPIEL_CHECK_GE(max_memory_mb, 1);

  evaluator_ = std::make_shared<RandomRolloutEvaluator>(
      rollouts_per_leaf, static_cast<int>(seed));
  bot_ = std::make_unique<MCTSBot>(
      *game_, evaluator_, uct_c, simulations, max_memory_mb,
      /*solve=*/false, static_cast<int>(seed + 1), /*verbose=*/false,
      ChildSelectionPolicy::UCT);
}

Action AyoMctsBotCpp::Step(const State& state) {
  return bot_->Step(state);
}

}  // namespace algorithms
}  // namespace open_spiel
