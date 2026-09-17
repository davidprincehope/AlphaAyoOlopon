#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "Model/ayo_olopon_C++/ayo_olopon.h"
#include "open_spiel/spiel.h"

namespace {
using open_spiel::Action;
using open_spiel::GameParameters;
using open_spiel::State;

struct Snapshot { std::vector<int> board, captured; int player; bool terminal; };

Snapshot ReadSnapshot(const State& state) {
  const std::string text = state.ToString();
  Snapshot result;
  auto read = [&](const std::string& label, std::vector<int>* output) {
    const size_t start = text.find(label);
    SPIEL_CHECK_NE(start, std::string::npos);
    const size_t colon = text.find(':', start);
    const size_t end = text.find('\n', colon);
    SPIEL_CHECK_NE(colon, std::string::npos);
    std::string values = text.substr(colon + 1, end - colon - 1);
    for (char& c : values) if (c == ',') c = ' ';
    for (char& c : values) if (c == '[' || c == ']') c = ' ';
    std::stringstream stream(values);
    int value;
    while (stream >> value) output->push_back(value);
  };
  read("Board:", &result.board);
  read("Captured:", &result.captured);
  const size_t player_start = text.find("Current player:");
  SPIEL_CHECK_NE(player_start, std::string::npos);
  result.player = std::stoi(text.substr(player_start + 15));
  const size_t terminal_start = text.find("Terminal:");
  SPIEL_CHECK_NE(terminal_start, std::string::npos);
  const std::string terminal = text.substr(terminal_start + 9);
  result.terminal = terminal.find("true") != std::string::npos ||
                   terminal.find("1") != std::string::npos;
  return result;
}

void Expect(const State& state, const std::vector<int>& board,
            const std::vector<int>& captured, int player, bool terminal) {
  const Snapshot actual = ReadSnapshot(state);
  SPIEL_CHECK_EQ(actual.board, board);
  SPIEL_CHECK_EQ(actual.captured, captured);
  SPIEL_CHECK_EQ(actual.player, player);
  SPIEL_CHECK_EQ(actual.terminal, terminal);
}

std::shared_ptr<const open_spiel::Game> Game(const GameParameters& params = {}) {
  return open_spiel::LoadGame("ayo_olopon", params);
}

GameParameters Params(int houses, int seeds) {
  GameParameters params;
  params["num_houses_per_player"] = open_spiel::GameParameter(houses);
  params["num_seeds_per_house"] = open_spiel::GameParameter(seeds);
  return params;
}

void Play(State* state, std::initializer_list<Action> actions) {
  for (Action action : actions) state->ApplyActionWithLegalityCheck(action);
}
}  // namespace

int main() {
  // 1. Initial state and non-terminal returns.
  auto game = Game();
  auto state = game->NewInitialState();
  Expect(*state, std::vector<int>(12, 4), {0, 0}, 0, false);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({0, 1, 2, 3, 4, 5}));
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>({0.0, 0.0}));
  SPIEL_CHECK_EQ(game->ObservationTensorShape(), std::vector<int>({14}));

  // 2. Multi-lap relay ends on a one-seed landing.
  state->ApplyAction(0);
  Expect(*state, {2, 7, 1, 6, 1, 6, 6, 6, 0, 1, 6, 6}, {0, 0}, 1, false);

  // 3. Mid-relay captures go to the row owner, not necessarily the mover.
  state->ApplyAction(0);
  Expect(*state, {2, 10, 0, 2, 0, 9, 1, 10, 1, 0, 1, 0}, {8, 4}, 0, false);

  // 4. Final-seed capture goes to the mover regardless of row ownership.
  state->ApplyAction(1); state->ApplyAction(1);
  Expect(*state, {0, 1, 2, 0, 2, 11, 0, 1, 0, 0, 0, 3}, {12, 16}, 0, false);

  // 5. Starvation rule: once the opponent's row is completely empty, a
  // legal move must feed at least one seed across into that row. Houses 0
  // and 1 hold seeds but can't reach far enough, so only house 2 (whose
  // 3 seeds spill past the row boundary) is legal.
  {
    auto starved = Game(Params(3, 1))->NewInitialState();
    Play(starved.get(), {1, 0, 1, 1, 0, 1});
    Expect(*starved, {2, 1, 3, 0, 0, 0}, {0, 0}, 0, false);
    SPIEL_CHECK_EQ(starved->LegalActions(), std::vector<Action>({2}));
  }

  // 6. Capturing more than half of all seeds ends the game immediately via
  // the score check, even mid-relay -- distinct from a repetition/no-move
  // sweep, since any not-yet-collected board seeds are simply left in
  // place rather than swept to their row owners.
  {
    auto capped = Game(Params(3, 2))->NewInitialState();
    Play(capped.get(), {2, 1});
    Expect(*capped, {3, 0, 1, 0, 0, 0}, {0, 8}, open_spiel::kTerminalPlayerId,
           true);
    SPIEL_CHECK_EQ(capped->Returns(), std::vector<double>({-1.0, 1.0}));
  }

  // 7. When the score check doesn't end the game but the next player has
  // no legal actions (their whole row is empty), the board is swept to
  // its owners automatically. Here captured is 4-0 right after the sow
  // (not yet over the score limit), so the sweep -- not the score check --
  // is what ends the game.
  {
    auto stalled = Game(Params(2, 2))->NewInitialState();
    stalled->ApplyAction(1);
    Expect(*stalled, {0, 0, 0, 0}, {8, 0}, open_spiel::kTerminalPlayerId,
           true);
    SPIEL_CHECK_EQ(stalled->Returns(), std::vector<double>({1.0, -1.0}));
  }

  // 8. Action-to-string labels are uppercase for player 0 and lowercase
  // for player 1.
  {
    auto labeled = Game()->NewInitialState();
    for (Action a = 0; a < 6; ++a) {
      SPIEL_CHECK_EQ(labeled->ActionToString(0, a), std::string(1, 'A' + a));
      SPIEL_CHECK_EQ(labeled->ActionToString(1, a), std::string(1, 'a' + a));
    }
  }

  // 9. Turn-level position repetition sweeps the board and produces a draw.
  auto repetition = Game(Params(2, 1))->NewInitialState();
  Play(repetition.get(), {0, 0, 1});
  Expect(*repetition, {0, 0, 0, 0}, {2, 2}, open_spiel::kTerminalPlayerId, true);
  SPIEL_CHECK_EQ(repetition->Returns(), std::vector<double>({0.0, 0.0}));

  // 10. In-move relay cycles are detected on boards larger than the
  // trivial one-house case in test 11: the fifth move here loops back to
  // a previously seen (board, captured, house) signature mid-relay, so the
  // move resolves by sweeping the board rather than continuing forever.
  {
    auto cyclic = Game(Params(3, 1))->NewInitialState();
    Play(cyclic.get(), {2, 1, 0, 1, 2});
    Expect(*cyclic, {0, 0, 0, 0, 0, 0}, {4, 2}, open_spiel::kTerminalPlayerId,
           true);
    SPIEL_CHECK_EQ(cyclic->Returns(), std::vector<double>({1.0, -1.0}));
  }

  // 11. The smallest possible in-move relay cycle terminates by sweeping.
  auto tiny = Game(Params(1, 1));
  auto tiny_state = tiny->NewInitialState();
  tiny_state->ApplyAction(0);
  Expect(*tiny_state, {0, 0}, {0, 2}, open_spiel::kTerminalPlayerId, true);
  SPIEL_CHECK_EQ(tiny_state->Returns(), std::vector<double>({-1.0, 1.0}));

  // 12. Illegal actions are rejected.
  auto illegal = Game()->NewInitialState(); illegal->ApplyAction(0);
  const auto legal_after_first_move = illegal->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal_after_first_move.begin(),
                             legal_after_first_move.end(), 2) ==
                   legal_after_first_move.end());

  // 13. Parameterized board playthrough and legal-action filtering.
  auto parameterized = Game(Params(3, 2))->NewInitialState();
  Play(parameterized.get(), {0, 0, 1, 1, 2, 2});
  Expect(*parameterized, {0, 2, 1, 0, 1, 0}, {4, 4}, 0, false);
  SPIEL_CHECK_EQ(parameterized->LegalActions(), std::vector<Action>({1, 2}));

  // 14. Observation tensor is normalized board followed by normalized scores.
  auto observed = Game()->NewInitialState();
  const auto tensor = observed->ObservationTensor(0);
  SPIEL_CHECK_EQ(tensor.size(), 14);
  for (int i = 0; i < 12; ++i)
    SPIEL_CHECK_TRUE(std::fabs(tensor[i] - 4.0 / 48) < 1e-6);
  SPIEL_CHECK_EQ(tensor[12], 0.0f);
  SPIEL_CHECK_EQ(tensor[13], 0.0f);

  // 15. Full 60-move smallest-legal-action regression.
  auto regression = Game()->NewInitialState();
  const std::vector<Action> sequence = {
      0,0,0,0,0,0,0,1,0,1,1,0,2,1,3,0,0,2,2,3,0,5,5,0,0,0,2,1,3,0,
      0,2,1,3,0,5,0,1,2,2,3,0,0,2,2,0,4,2,5,5,0,0,1,1,2,2,3,4,0,5};
  for (Action action : sequence) {
    SPIEL_CHECK_FALSE(regression->IsTerminal());
    SPIEL_CHECK_EQ(regression->LegalActions().front(), action);
    regression->ApplyActionWithLegalityCheck(action);
  }
  Expect(*regression, std::vector<int>(12, 0), {24, 24},
         open_spiel::kTerminalPlayerId, true);
  SPIEL_CHECK_EQ(regression->Returns(), std::vector<double>({0.0, 0.0}));
  return 0;
}