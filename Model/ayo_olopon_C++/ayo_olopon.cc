#include "Model/ayo_olopon_C++/ayo_olopon.h"

#include <algorithm>
#include <sstream>

#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace ayo_olopon {
namespace {

const GameType kGameType{
    "ayo_olopon", "Ayo Olopon", GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum, GameType::RewardModel::kTerminal, 2, 2,
    false, false, true, true,
    {{"num_houses_per_player", GameParameter(kDefaultHousesPerPlayer)},
     {"num_seeds_per_house", GameParameter(kDefaultSeedsPerHouse)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AyoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

AyoState::AyoState(std::shared_ptr<const Game> game, int houses, int seeds)
    : State(game),
      num_houses_per_player_(houses),
      total_seeds_(kNumPlayers * houses * seeds),
      board_(houses, seeds) {
  positions_since_capture_.insert(board_);
}

AyoState::AyoState(std::shared_ptr<const Game> game, const AyoBoard& board)
    : State(game),
      num_houses_per_player_(board.seeds.size() / kNumPlayers),
      total_seeds_(board.TotalSeeds()),
      board_(board) {
  SPIEL_CHECK_EQ(board.seeds.size() % kNumPlayers, 0);
  positions_since_capture_.insert(board_);
}

Player AyoState::CurrentPlayer() const {
  return game_over_ ? kTerminalPlayerId : board_.current_player;
}

int AyoState::LowerHouse(int house) const {
  return (house / num_houses_per_player_) * num_houses_per_player_;
}

int AyoState::UpperHouse(int house) const {
  return LowerHouse(house) + num_houses_per_player_ - 1;
}

int AyoState::PlayerLowerHouse(Player player) const {
  return player * num_houses_per_player_;
}

int AyoState::PlayerUpperHouse(Player player) const {
  return PlayerLowerHouse(player) + num_houses_per_player_ - 1;
}

int AyoState::ActionToHouse(Player player, Action action) const {
  return player * num_houses_per_player_ + action;
}

Action AyoState::HouseToAction(int house) const {
  return house % num_houses_per_player_;
}

int AyoState::OpponentSeeds() const {
  const Player opponent = 1 - board_.current_player;
  int total = 0;
  for (int house = PlayerLowerHouse(opponent);
       house <= PlayerUpperHouse(opponent); ++house) {
    total += board_.seeds[house];
  }
  return total;
}

std::vector<Action> AyoState::LegalActionsForCurrentPlayer() const {
  std::vector<Action> actions;
  if (game_over_) return actions;
  const int lower = PlayerLowerHouse(board_.current_player);
  const int upper = PlayerUpperHouse(board_.current_player);
  if (OpponentSeeds() == 0) {
    for (int house = lower; house <= upper; ++house) {
      if (board_.seeds[house] - (upper - house) > 0) {
        actions.push_back(HouseToAction(house));
      }
    }
  } else {
    for (int house = lower; house <= upper; ++house) {
      if (board_.seeds[house] > 0) actions.push_back(HouseToAction(house));
    }
  }
  return actions;
}

std::vector<Action> AyoState::LegalActions() const {
  return LegalActionsForCurrentPlayer();
}

bool AyoState::InOpponentRow(int house) const {
  return house / num_houses_per_player_ != board_.current_player;
}

bool AyoState::ScoreTerminal() const {
  const int limit = total_seeds_ / 2;
  return board_.score[0] > limit || board_.score[1] > limit ||
         (board_.score[0] == limit && board_.score[1] == limit);
}

void AyoState::SetScoreReturnsAndEnd() {
  game_over_ = true;
  if (board_.score[0] > board_.score[1]) {
    returns_ = {1.0, -1.0};
  } else if (board_.score[0] < board_.score[1]) {
    returns_ = {-1.0, 1.0};
  } else {
    returns_ = {0.0, 0.0};
  }
}

void AyoState::CollectAndTerminate() {
  for (int house = 0; house < static_cast<int>(board_.seeds.size()); ++house) {
    board_.score[house / num_houses_per_player_] += board_.seeds[house];
    board_.seeds[house] = 0;
  }
  SetScoreReturnsAndEnd();
}

bool AyoState::SowRelay(int house, Action action) {
  const int original_house = house;
  std::unordered_map<RelayKey, int, RelayKeyHash> seen;
  for (int relay = 0; relay < kMaxRelayLaps; ++relay) {
    const RelayKey signature{board_, house};
    auto [iterator, inserted] = seen.emplace(signature, relay);
    if (!inserted) {
      const int cycle_length = relay - iterator->second;
      CollectAndTerminate();
      last_relay_report_ = RelayReport{
          "repeated_relay_state", board_.current_player, action,
          original_house, cycle_length, board_.score[0], board_.score[1]};
      return true;
    }

    int to_distribute = board_.seeds[house];
    SPIEL_CHECK_NE(to_distribute, 0);
    board_.seeds[house] = 0;
    int last_house = house;
    while (to_distribute > 0) {
      last_house = (last_house + 1) % board_.seeds.size();
      if (last_house == house) continue;
      ++board_.seeds[last_house];
      --to_distribute;
      if (board_.seeds[last_house] == 4) {
        const Player capturer =
            to_distribute == 0 ? board_.current_player
                               : last_house / num_houses_per_player_;
        board_.seeds[last_house] = 0;
        board_.score[capturer] += 4;
        if (to_distribute == 0) return true;
      }
    }
    if (board_.seeds[last_house] == 1) return false;
    house = last_house;
  }

  CollectAndTerminate();
  last_relay_report_ = RelayReport{
      "relay_lap_limit_exceeded", board_.current_player, action,
      original_house, kMaxRelayLaps, board_.score[0], board_.score[1]};
  return true;
}

void AyoState::DoApplyAction(Action action) {
  const auto actions = LegalActionsForCurrentPlayer();
  SPIEL_CHECK_TRUE(std::find(actions.begin(), actions.end(), action) !=
                   actions.end());
  last_relay_report_.reset();
  const int house = ActionToHouse(board_.current_player, action);
  if (SowRelay(house, action)) positions_since_capture_.clear();
  board_.current_player = 1 - board_.current_player;

  if (ScoreTerminal()) {
    SetScoreReturnsAndEnd();
    return;
  }
  if (positions_since_capture_.find(board_) != positions_since_capture_.end()) {
    CollectAndTerminate();
    return;
  }
  positions_since_capture_.insert(board_);
  if (LegalActionsForCurrentPlayer().empty()) CollectAndTerminate();
}

std::vector<double> AyoState::Returns() const {
  return game_over_ ? returns_ : std::vector<double>{0.0, 0.0};
}

std::unique_ptr<State> AyoState::Clone() const {
  return std::unique_ptr<State>(new AyoState(*this));
}

std::string AyoState::ActionToString(Player player, Action action) const {
  return std::string(1, player == 0 ? 'A' + action : 'a' + action);
}

std::string AyoState::ToString() const {
  std::ostringstream out;
  out << "Board:";
  for (int value : board_.seeds) out << " " << value;
  out << "\nCaptured: " << board_.score[0] << " " << board_.score[1]
      << "\nCurrent player: " << CurrentPlayer()
      << "\nTerminal: " << (game_over_ ? "true" : "false");
  return out.str();
}

void AyoState::ObservationTensor(Player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), 14);
  for (int i = 0; i < 12; ++i) {
    values[i] = static_cast<float>(board_.seeds[i]) / total_seeds_;
  }
  values[12] = static_cast<float>(board_.score[0]) / total_seeds_;
  values[13] = static_cast<float>(board_.score[1]) / total_seeds_;
}

std::string AyoState::ObservationString(Player) const { return ToString(); }

AyoGame::AyoGame(const GameParameters& params)
    : Game(kGameType,
           GameInfo{kDefaultHousesPerPlayer, 0, kNumPlayers, -1.0, 1.0, 0.0,
                    1000}),
      num_houses_per_player_(ParameterValue<int>(params, "num_houses_per_player",
                                                 kDefaultHousesPerPlayer)),
      num_seeds_per_house_(ParameterValue<int>(params, "num_seeds_per_house",
                                               kDefaultSeedsPerHouse)) {}

std::unique_ptr<State> AyoGame::NewInitialState() const {
  return std::unique_ptr<State>(new AyoState(
      shared_from_this(), num_houses_per_player_, num_seeds_per_house_));
}

}  // namespace ayo_olopon
}  // namespace open_spiel
