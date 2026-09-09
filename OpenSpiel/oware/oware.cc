// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/oware/oware.h"

#include <iomanip>

#include "open_spiel/game_parameters.h"

namespace open_spiel {
namespace oware {

namespace {

// Facts about the game as required by OpenSpiel. In particular, Oware is a
// deterministic, sequential, two-player, perfect-information, zero-sum game.
const GameType kGameType{
    /*short_name=*/"oware",
    /*long_name=*/"Oware",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"num_houses_per_player", GameParameter(kDefaultHousesPerPlayer)},
     {"num_seeds_per_house", GameParameter(kDdefaultSeedsPerHouse)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new OwareGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

OwareState::OwareState(std::shared_ptr<const Game> game,
                       int num_houses_per_player, int num_seeds_per_house)
    : State(game),
      num_houses_per_player_(num_houses_per_player),
      total_seeds_(kNumPlayers * num_seeds_per_house * num_houses_per_player),
      board_(/*num_houses_per_player=*/num_houses_per_player,
             /*num_seeds_per_house=*/num_seeds_per_house) {
  // Repetition is tracked from the beginning, including the initial board.
  boards_since_last_capture_.insert(board_);
}

OwareState::OwareState(std::shared_ptr<const Game> game,
                       const OwareBoard& board)
    : State(game),
      num_houses_per_player_(board.seeds.size() / kNumPlayers),
      total_seeds_(board.TotalSeeds()),
      board_(board) {
  SPIEL_CHECK_EQ(0, board.seeds.size() % kNumPlayers);
  SPIEL_CHECK_TRUE(IsTerminal() || !LegalActions().empty());
  boards_since_last_capture_.insert(board_);
}

std::vector<Action> OwareState::LegalActions() const {
  std::vector<Action> actions;
  if (IsTerminal()) return actions;
  // Translate the current player's row into physical indices in seeds[].
  const Player lower = PlayerLowerHouse(board_.current_player);
  const Player upper = PlayerUpperHouse(board_.current_player);
  if (OpponentSeeds() == 0) {
    // In case the opponent does not have any seeds, a player must make
    // a move which gives the opponent seeds.
    // If the opponent is empty, a move is legal only when sowing reaches the
    // opponent's row and gives it at least one seed.
    for (int house = lower; house <= upper; house++) {
      const int first_seeds_in_own_row = upper - house;
      if (board_.seeds[house] - first_seeds_in_own_row > 0) {
        actions.push_back(HouseToAction(house));
      }
    }
  } else {
    // In the normal case, every non-empty house in the current player's row
    // is a legal source for sowing.
    for (int house = lower; house <= upper; house++) {
      if (board_.seeds[house] > 0) {
        actions.push_back(HouseToAction(house));
      }
    }
  }
  return actions;
}

std::string OwareState::ActionToString(Player player, Action action) const {
  // Player 0's actions are displayed as A-F and player 1's as a-f.
  return std::string(1, (player == Player{0} ? 'A' : 'a') + action);
}

void OwareState::WritePlayerScore(std::ostringstream& out,
                                  Player player) const {
  out << "Player " << player << " score = " << board_.score[player];
  if (CurrentPlayer() == player) {
    out << " [PLAYING]" << std::endl;
  } else {
    out << std::endl;
  }
}

std::string OwareState::ToString() const {
  std::ostringstream out;
  if (IsTerminal()) {
    out << "[FINISHED]" << std::endl;
  }
  WritePlayerScore(out, 1);

  // Add player 1 labels.
  for (int action = num_houses_per_player_ - 1; action >= 0; action--) {
    out << std::setw(3) << std::right << ActionToString(1, action);
  }
  out << std::endl;

  // Add player 1 house seeds.
  for (int house = kNumPlayers * num_houses_per_player_ - 1;
       house >= num_houses_per_player_; house--) {
    out << std::setw(3) << std::right << board_.seeds[house];
  }
  out << std::endl;

  // Add player 0 house seeds.
  for (int house = 0; house < num_houses_per_player_; house++) {
    out << std::setw(3) << std::right << board_.seeds[house];
  }
  out << std::endl;

  // Add player 0 labels.
  for (int action = 0; action < num_houses_per_player_; action++) {
    out << std::setw(3) << std::right << ActionToString(0, action);
  }
  out << std::endl;

  WritePlayerScore(out, 0);
  return out.str();
}

bool OwareState::IsTerminal() const {
  // Terminate when one player has more than half of the seeds
  // (works both for even and odd number of seeds), or when all seeds
  // are equally shared.
  // A player wins as soon as they have more than half of all seeds. If the
  // seeds are split equally, the game is also over as a draw.
  const int limit = total_seeds_ / 2;
  return board_.score[0] > limit || board_.score[1] > limit ||
         (board_.score[0] == limit && board_.score[1] == limit);
}

std::vector<double> OwareState::Returns() const {
  if (IsTerminal()) {
    if (board_.score[0] > board_.score[1]) {
      return {1, -1};
    } else if (board_.score[0] < board_.score[1]) {
      return {-1, 1};
    } else {
      return {0, 0};
    }
  } else {
    return {0, 0};
  }
}

std::unique_ptr<State> OwareState::Clone() const {
  return std::unique_ptr<State>(new OwareState(*this));
}

int OwareState::DistributeSeeds(int house) {
  // Sowing is deliberately implemented one seed at a time. This makes the
  // wraparound and the rule "skip the source house" explicit.
  int to_distribute = board_.seeds[house];
  SPIEL_CHECK_NE(to_distribute, 0);
  board_.seeds[house] = 0;
  int index = house;
  while (to_distribute > 0) {
    index = (index + 1) % NumHouses();
    // Seeds are never sown into the house they were drawn from, even after
    // making a complete circuit of the board.
    if (index != house) {
      board_.seeds[index]++;
      to_distribute--;
    }
  }
  return index;
}

bool OwareState::InOpponentRow(int house) const {
  // Each row is identified by integer division: indices 0..5 belong to row 0
  // and indices 6..11 belong to row 1 in the default six-house game.
  return (house / num_houses_per_player_) != board_.current_player;
}

bool OwareState::IsGrandSlam(int house) const {
  // Capturing starts at the last house sown into and moves backwards through
  // the opponent's row. A Grand Slam would remove every opponent seed, which
  // Oware forbids; the move remains valid but the capture is cancelled.
  // If there are seeds beyond the house in which the last seed was dropped,
  // it is not a Grand Slam.
  for (int index = UpperHouse(house); index > house; index--) {
    if (board_.seeds[index] > 0) {
      return false;
    }
  }
  // If not all houses are captured starting from the house in which the last
  // seed was dropped, it is not a Grand Slam. It means the opponent will still
  // have some seeds left because none of these houses can be empty due to
  // the way seeds are sown.
  const int lower = LowerHouse(house);
  for (int index = house; index >= lower; index--) {
    SPIEL_CHECK_GT(board_.seeds[index], 0);
    if (!ShouldCapture(board_.seeds[index])) {
      return false;
    }
  }
  return true;
}

int OwareState::OpponentSeeds() const {
  // Count only seeds still in the opponent's row. Captured seeds are already
  // in score[] and do not count as seeds available to receive or capture.
  int count = 0;
  const Player opponent = 1 - board_.current_player;
  const int lower = PlayerLowerHouse(opponent);
  const int upper = PlayerUpperHouse(opponent);
  for (int house = lower; house <= upper; house++) {
    count += board_.seeds[house];
  }
  return count;
}

int OwareState::DoCaptureFrom(int house) {
  // Capture consecutive opponent houses backwards while each contains 2 or
  // 3 seeds. The first house outside that range stops the capture sequence.
  const int lower = LowerHouse(house);
  int captured = 0;
  for (int index = house; index >= lower; index--) {
    if (ShouldCapture(board_.seeds[index])) {
      captured += board_.seeds[index];
      board_.seeds[index] = 0;
    } else {
      break;
    }
  }
  board_.score[board_.current_player] += captured;
  return captured;
}

void OwareState::DoApplyAction(Action action) {
  SPIEL_CHECK_LT(history_.size(), kMaxGameLength);

  // OpenSpiel supplies a row-relative action. Convert it before changing the
  // board, then sow all seeds from that physical house.
  int last_house = DistributeSeeds(ActionToHouse(CurrentPlayer(), action));

  // Captures are possible only when the final seed landed in the opponent's
  // row. Grand Slam moves sow normally but capture nothing.
  if (InOpponentRow(last_house) && !IsGrandSlam(last_house)) {
    const int captured = DoCaptureFrom(last_house);
    if (captured > 0) {
      // No need to keep previous boards for checking game repetition because
      // captured seeds do not re-enter the game.
      boards_since_last_capture_.clear();
    }
  }
  // The turn changes after sowing and any capture has completed.
  board_.current_player = 1 - board_.current_player;

  // insert() returns false when this exact position has already occurred.
  // The current player is included in board_, so turn order matters.
  if (!boards_since_last_capture_.insert(board_).second) {
    // We have game repetition, the game is ended.
    CollectAndTerminate();
  }

  // A player with no legal move cannot continue. Remaining seeds are scored
  // for their respective rows.
  if (LegalActions().empty()) {
    CollectAndTerminate();
  }
}

void OwareState::CollectAndTerminate() {
  // Seeds left on the board are awarded to the player who owns their row.
  // Setting them to zero keeps the board consistent with the scores.
  for (int house = 0; house < NumHouses(); house++) {
    const Player player = house / num_houses_per_player_;
    board_.score[player] += board_.seeds[house];
    board_.seeds[house] = 0;
  }
}

std::string OwareState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return board_.ToString();
}

void OwareState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), /*seeds*/ NumHouses() + /*scores*/ kNumPlayers);
  // The neural-network input is a normalized flat vector: all house counts
  // first, followed by both players' scores. This implementation exposes the
  // same absolute board to either player because Oware has perfect information.
  for (int house = 0; house < NumHouses(); ++house) {
    values[house] = ((double)board_.seeds[house]) / total_seeds_;
  }
  for (Player player = 0; player < kNumPlayers; ++player) {
    values[NumHouses() + player] =
        ((double)board_.score[player]) / total_seeds_;
  }
}

OwareGame::OwareGame(const GameParameters& params)
    : Game(kGameType, params),
      num_houses_per_player_(ParameterValue<int>("num_houses_per_player")),
      num_seeds_per_house_(ParameterValue<int>("num_seeds_per_house")) {}

std::vector<int> OwareGame::ObservationTensorShape() const {
  return {/*seeds*/ num_houses_per_player_ * kNumPlayers +
          /*scores*/ kNumPlayers};
}

}  // namespace oware
}  // namespace open_spiel
