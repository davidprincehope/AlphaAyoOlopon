#ifndef OPEN_SPIEL_GAMES_AYO_OLOPON_AYO_OLOPON_H_
#define OPEN_SPIEL_GAMES_AYO_OLOPON_AYO_OLOPON_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Model/ayo_olopon_C++/ayo_olopon_board.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace ayo_olopon {

inline constexpr int kDefaultHousesPerPlayer = 6;
inline constexpr int kDefaultSeedsPerHouse = 4;
inline constexpr int kMaxRelayLaps = 10000;

struct RelayReport {
  std::string reason;
  Player player = 0;
  Action action = 0;
  int source_house = 0;
  int cycle_length = 0;
  int captured_after_0 = 0;
  int captured_after_1 = 0;
};

class AyoState : public State {
 public:
  AyoState(std::shared_ptr<const Game> game, int num_houses_per_player,
           int num_seeds_per_house);
  AyoState(std::shared_ptr<const Game> game, const AyoBoard& board);
  AyoState(const AyoState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return game_over_; }
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override { return {0.0, 0.0}; }
  std::unique_ptr<State> Clone() const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;

  const AyoBoard& Board() const { return board_; }
  const RelayReport* LastRelayReport() const {
    return last_relay_report_ ? &*last_relay_report_ : nullptr;
  }

 protected:
  void DoApplyAction(Action action) override;

 private:
  struct RelayKey {
    AyoBoard board;
    int house;
    bool operator==(const RelayKey& other) const {
      return house == other.house && board == other.board;
    }
  };
  struct BoardHash {
    std::size_t operator()(const AyoBoard& board) const {
      return board.HashValue();
    }
  };
  struct RelayKeyHash {
    std::size_t operator()(const RelayKey& key) const {
      return key.board.HashValue() ^ (static_cast<std::size_t>(key.house) << 1);
    }
  };

  int LowerHouse(int house) const;
  int UpperHouse(int house) const;
  int PlayerLowerHouse(Player player) const;
  int PlayerUpperHouse(Player player) const;
  int OpponentSeeds() const;
  bool InOpponentRow(int house) const;
  int ActionToHouse(Player player, Action action) const;
  Action HouseToAction(int house) const;
  std::vector<Action> LegalActionsForCurrentPlayer() const;
  bool SowRelay(int house, Action action);
  bool ScoreTerminal() const;
  void CollectAndTerminate();
  void SetScoreReturnsAndEnd();

  const int num_houses_per_player_;
  const int total_seeds_;
  AyoBoard board_;
  bool game_over_ = false;
  std::vector<double> returns_{0.0, 0.0};
  std::unordered_set<AyoBoard, BoardHash> positions_since_capture_;
  std::optional<RelayReport> last_relay_report_;
};

class AyoGame : public Game {
 public:
  explicit AyoGame(const GameParameters& params);
  int NumDistinctActions() const override { return num_houses_per_player_; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return 1.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  int MaxGameLength() const override { return 1000; }
  std::vector<int> ObservationTensorShape() const override { return {14}; }

 private:
  const int num_houses_per_player_;
  const int num_seeds_per_house_;
};

}  // namespace ayo_olopon
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_AYO_OLOPON_AYO_OLOPON_H_
