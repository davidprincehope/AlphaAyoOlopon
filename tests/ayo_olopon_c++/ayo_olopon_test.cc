#include "Model/ayo_olopon_C++/ayo_olopon.h"

#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace ayo_olopon {
namespace {

std::shared_ptr<const Game> Game() { return LoadGame("ayo_olopon"); }

void InitialStateTest() {
  auto game = Game();
  auto state = game->NewInitialState();
  auto* ayo = dynamic_cast<AyoState*>(state.get());
  SPIEL_CHECK_TRUE(ayo != nullptr);
  SPIEL_CHECK_EQ(ayo->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(ayo->Board().seeds, std::vector<int>(12, 4));
  SPIEL_CHECK_EQ(ayo->Board().score, std::vector<int>({0, 0}));
  SPIEL_CHECK_EQ(ayo->LegalActions(), std::vector<Action>({0, 1, 2, 3, 4, 5}));
  SPIEL_CHECK_FALSE(ayo->IsTerminal());
}

void FirstActionsTest() {
  auto game = Game();
  AyoState player_zero(game, AyoBoard(0, {0, 0},
                                      {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}));
  player_zero.ApplyAction(0);
  SPIEL_CHECK_EQ(player_zero.Board().seeds,
                 std::vector<int>({2, 7, 1, 6, 1, 6, 6, 6, 0, 1, 6, 6}));
  SPIEL_CHECK_EQ(player_zero.Board().score, std::vector<int>({0, 0}));
  SPIEL_CHECK_EQ(player_zero.CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(player_zero.LegalActions(),
                 std::vector<Action>({0, 1, 3, 4, 5}));

  AyoState player_one(game, AyoBoard(1, {0, 0},
                                     {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}));
  player_one.ApplyAction(0);
  SPIEL_CHECK_EQ(player_one.Board().seeds,
                 std::vector<int>({6, 6, 0, 1, 6, 6, 2, 7, 1, 6, 1, 6}));
  SPIEL_CHECK_EQ(player_one.CurrentPlayer(), 0);
}

void RelayAndCaptureTests() {
  auto game = Game();
  AyoState landing_four(game,
                        AyoBoard(0, {0, 0}, {1, 3, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0}));
  landing_four.ApplyAction(0);
  SPIEL_CHECK_EQ(landing_four.Board().seeds,
                 std::vector<int>({0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0}));
  SPIEL_CHECK_EQ(landing_four.Board().score, std::vector<int>({4, 0}));

  AyoState intermediate_four(
      game, AyoBoard(0, {0, 0}, {0, 0, 0, 0, 0, 5, 3, 0, 0, 0, 0, 0}));
  intermediate_four.ApplyAction(5);
  SPIEL_CHECK_EQ(intermediate_four.Board().seeds,
                 std::vector<int>({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0}));
  SPIEL_CHECK_EQ(intermediate_four.Board().score, std::vector<int>({0, 4}));

  AyoState relay(game, AyoBoard(0, {0, 0},
                                {2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0}));
  relay.ApplyAction(0);
  SPIEL_CHECK_EQ(relay.Board().seeds,
                 std::vector<int>({0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0}));
  SPIEL_CHECK_EQ(relay.CurrentPlayer(), 1);
}

void MultipleRelayAndCycleTests() {
  auto game = Game();
  AyoState multiple_relay(
      game, AyoBoard(0, {0, 0}, {3, 0, 0, 1, 0, 1, 0, 2, 0, 0, 3, 0}));
  multiple_relay.ApplyAction(0);
  SPIEL_CHECK_EQ(multiple_relay.Board().seeds,
                 std::vector<int>({0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0}));
  SPIEL_CHECK_EQ(multiple_relay.Board().score, std::vector<int>({4, 0}));

  AyoState cycle(game, AyoBoard(
                           1, {16, 20}, {2, 1, 0, 2, 1, 0, 1, 0, 1, 3, 1, 0}));
  cycle.ApplyAction(3);
  const RelayReport* report = cycle.LastRelayReport();
  SPIEL_CHECK_TRUE(report != nullptr);
  SPIEL_CHECK_EQ(report->reason, "repeated_relay_state");
  SPIEL_CHECK_EQ(report->cycle_length, 60);
  SPIEL_CHECK_EQ(cycle.Board().seeds, std::vector<int>(12, 0));
  SPIEL_CHECK_EQ(cycle.Board().score, std::vector<int>({22, 26}));
  SPIEL_CHECK_TRUE(cycle.IsTerminal());
}

void FeedingRuleTest() {
  auto game = Game();
  AyoState state(game, AyoBoard(0, {0, 0},
                                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  SPIEL_CHECK_TRUE(state.LegalActions().empty());
}

}  // namespace
}  // namespace ayo_olopon
}  // namespace open_spiel

int main() {
  open_spiel::ayo_olopon::InitialStateTest();
  open_spiel::ayo_olopon::FirstActionsTest();
  open_spiel::ayo_olopon::RelayAndCaptureTests();
  open_spiel::ayo_olopon::MultipleRelayAndCycleTests();
  open_spiel::ayo_olopon::FeedingRuleTest();
}
