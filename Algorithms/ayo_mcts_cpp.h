#ifndef ALPHA_AYO_OLOPON_ALGORITHMS_AYO_MCTS_CPP_H_
#define ALPHA_AYO_OLOPON_ALGORITHMS_AYO_MCTS_CPP_H_

#include <cstdint>
#include <memory>

#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Native OpenSpiel UCT agent using random-rollout leaf evaluation.
class AyoMctsBotCpp {
 public:
  AyoMctsBotCpp(std::shared_ptr<const Game> game, int simulations = 1000,
                int rollouts_per_leaf = 1, double uct_c = 1.4142135623730951,
                std::uint32_t seed = 0, std::int64_t max_memory_mb = 256);

  Action Step(const State& state);

 private:
  std::shared_ptr<const Game> game_;
  std::shared_ptr<Evaluator> evaluator_;
  std::unique_ptr<MCTSBot> bot_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // ALPHA_AYO_OLOPON_ALGORITHMS_AYO_MCTS_CPP_H_
