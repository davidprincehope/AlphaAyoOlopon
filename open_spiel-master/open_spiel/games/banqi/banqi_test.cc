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

#include "open_spiel/games/banqi/banqi.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace banqi {
namespace {

namespace testing = open_spiel::testing;

void BasicBanqiTests() {
  testing::LoadGameTest("banqi");
  testing::ChanceOutcomesTest(*LoadGame("banqi"));
  testing::RandomSimTest(*LoadGame("banqi"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("banqi"), 10);
}

void CaptureHierarchyTest() {
  // Test piece capture hierarchy using CanCapture via a state object.
  auto game = LoadGame("banqi");
  auto state = game->NewInitialState();
  auto* banqi_state = static_cast<BanqiState*>(state.get());

  // Red General (0) captures Red Advisor? No (same color).
  // We test cross-color captures.
  // Red General (0) vs Black Advisor (8): General beats Advisor.
  SPIEL_CHECK_TRUE(banqi_state->CanCapture(MakePiece(kRed, kGeneral),
                                           MakePiece(kBlack, kAdvisor)));
  // Red General vs Black Soldier: General CANNOT capture Soldier.
  SPIEL_CHECK_FALSE(banqi_state->CanCapture(MakePiece(kRed, kGeneral),
                                            MakePiece(kBlack, kSoldier)));
  // Red Soldier vs Black General: Soldier CAN capture General.
  SPIEL_CHECK_TRUE(banqi_state->CanCapture(MakePiece(kRed, kSoldier),
                                           MakePiece(kBlack, kGeneral)));
  // Red Soldier vs Black Cannon: Soldier CANNOT capture Cannon.
  SPIEL_CHECK_FALSE(banqi_state->CanCapture(MakePiece(kRed, kSoldier),
                                            MakePiece(kBlack, kCannon)));
  // Red Cannon vs anything via CanCapture (adjacency) should return false.
  SPIEL_CHECK_FALSE(banqi_state->CanCapture(MakePiece(kRed, kCannon),
                                            MakePiece(kBlack, kSoldier)));
  // Red Chariot vs Black Horse: Chariot (rank 3) beats Horse (rank 4).
  SPIEL_CHECK_TRUE(banqi_state->CanCapture(MakePiece(kRed, kChariot),
                                           MakePiece(kBlack, kHorse)));
  // Red Horse vs Black Chariot: Horse (rank 4) cannot beat Chariot (rank 3).
  SPIEL_CHECK_FALSE(banqi_state->CanCapture(MakePiece(kRed, kHorse),
                                            MakePiece(kBlack, kChariot)));
  // Same rank: Red Advisor vs Black Advisor.
  SPIEL_CHECK_TRUE(banqi_state->CanCapture(MakePiece(kRed, kAdvisor),
                                           MakePiece(kBlack, kAdvisor)));
}

void ChanceOutcomesProbabilitiesTest() {
  // At the start, all 32 pieces are face-down.
  // First action must be a flip. After flip, chance node determines piece.
  auto game = LoadGame("banqi");
  auto state = game->NewInitialState();

  // Player 0 flips square 0.
  auto legal = state->LegalActions();
  SPIEL_CHECK_FALSE(legal.empty());
  // All legal actions at start should be flip actions (0-31).
  for (Action a : legal) {
    SPIEL_CHECK_LT(a, kNumFlipActions);
  }

  state->ApplyAction(0);  // Flip square 0.

  // Now at a chance node.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  auto outcomes = state->ChanceOutcomes();

  // Sum of probabilities should be 1.
  double total_prob = 0;
  for (auto& [action, prob] : outcomes) {
    SPIEL_CHECK_GT(prob, 0);
    total_prob += prob;
  }
  SPIEL_CHECK_FLOAT_EQ(total_prob, 1.0);
}

void GameParametersTest() {
  // Test custom max_no_progress parameter.
  auto game = LoadGame("banqi", {{"max_no_progress", GameParameter(10)}});
  SPIEL_CHECK_TRUE(game != nullptr);
  testing::RandomSimTest(*game, 10);
}

}  // namespace
}  // namespace banqi
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::banqi::BasicBanqiTests();
  open_spiel::banqi::CaptureHierarchyTest();
  open_spiel::banqi::ChanceOutcomesProbabilitiesTest();
  open_spiel::banqi::GameParametersTest();
}
