#include "Model/ayo_olopon_C++/ayo_olopon_board.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace open_spiel {
namespace ayo_olopon {

AyoBoard::AyoBoard(int num_houses_per_player, int num_seeds_per_house)
    : current_player(0),
      score(kNumPlayers, 0),
      seeds(kNumPlayers * num_houses_per_player, num_seeds_per_house) {}

AyoBoard::AyoBoard(Player player, const std::vector<int>& score_values,
                   const std::vector<int>& seed_values)
    : current_player(player), score(score_values), seeds(seed_values) {
  SPIEL_CHECK_EQ(score.size(), kNumPlayers);
}

bool AyoBoard::operator==(const AyoBoard& other) const {
  return current_player == other.current_player && score == other.score &&
         seeds == other.seeds;
}

bool AyoBoard::operator!=(const AyoBoard& other) const {
  return !(*this == other);
}

std::string AyoBoard::ToString() const {
  return absl::StrCat(current_player, " | ", absl::StrJoin(score, " "),
                      " | ", absl::StrJoin(seeds, " "));
}

std::size_t AyoBoard::HashValue() const {
  std::size_t hash = static_cast<std::size_t>(current_player);
  for (int value : score) {
    hash ^= static_cast<std::size_t>(value) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
  }
  for (int value : seeds) {
    hash ^= static_cast<std::size_t>(value) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
  }
  return hash;
}

int AyoBoard::TotalSeeds() const {
  int total = 0;
  for (int value : score) total += value;
  for (int value : seeds) total += value;
  return total;
}

std::ostream& operator<<(std::ostream& os, const AyoBoard& board) {
  return os << board.ToString();
}

}  // namespace ayo_olopon
}  // namespace open_spiel
