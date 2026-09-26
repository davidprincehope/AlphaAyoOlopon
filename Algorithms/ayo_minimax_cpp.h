#ifndef ALPHAYOOLOPON_ALGORITHMS_AYO_MINIMAX_CPP_H_
#define ALPHAYOOLOPON_ALGORITHMS_AYO_MINIMAX_CPP_H_

#include <memory>
#include <utility>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Native OpenSpiel alpha-beta agent using the repository's H* evaluator.
class AyoMinimaxBotCpp {
 public:
  AyoMinimaxBotCpp(std::shared_ptr<const Game> game, int maximum_depth);

  std::pair<double, Action> Search(const State& state) const;
  Action Step(const State& state) const;

 private:
  std::shared_ptr<const Game> game_;
  int maximum_depth_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // ALPHAYOOLOPON_ALGORITHMS_AYO_MINIMAX_CPP_H_
