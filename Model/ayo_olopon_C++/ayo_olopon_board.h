#ifndef OPEN_SPIEL_GAMES_AYO_OLOPON_AYO_OLOPON_BOARD_H_
#define OPEN_SPIEL_GAMES_AYO_OLOPON_AYO_OLOPON_BOARD_H_

#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace ayo_olopon {

inline constexpr int kNumPlayers = 2;

// A compact snapshot of the positional data that changes during play. Rules
// and repetition bookkeeping remain in AyoState, matching OpenSpiel's Oware
// organization.
struct AyoBoard {
  AyoBoard(int num_houses_per_player, int num_seeds_per_house);
  AyoBoard(Player current_player, const std::vector<int>& score,
           const std::vector<int>& seeds);

  AyoBoard(const AyoBoard&) = default;
  AyoBoard& operator=(const AyoBoard&) = default;

  bool operator==(const AyoBoard& other) const;
  bool operator!=(const AyoBoard& other) const;
  std::string ToString() const;
  std::size_t HashValue() const;
  int TotalSeeds() const;

  Player current_player;
  std::vector<int> score;
  std::vector<int> seeds;
};

std::ostream& operator<<(std::ostream& os, const AyoBoard& board);

}  // namespace ayo_olopon
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_AYO_OLOPON_AYO_OLOPON_BOARD_H_
