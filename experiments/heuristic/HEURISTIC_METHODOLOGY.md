# Ayo Olopon heuristic methodology

## Checkpoint 1 — inspection and feature definitions

This checkpoint defines the features only. No heuristic feature code or
experimental policy has been added yet.

### Implementation inspected

The reference implementation is `Model/ayo_olopon/ayo_olopon.py`.

- **Board:** `state.board` is a length-12 list. Indices `0..5` are player 0's
  row and indices `6..11` are player 1's row. The board is traversed cyclically
  in increasing index order while sowing.
- **Captured seeds:** `state.captured[0]` and `state.captured[1]` are the
  players' accumulated scores. The default position contains 48 total seeds.
- **Player-row mapping:** player `p` owns the contiguous range
  `p * state.num_houses_per_player` through
  `(p + 1) * state.num_houses_per_player - 1`.
- **Current player:** `state.current_player()` returns `0` or `1` during play,
  and `pyspiel.PlayerId.TERMINAL` after the game ends.
- **Legal actions:** `state.legal_actions()` returns local house numbers
  `0..5` for the current player. Empty source houses are excluded. If the
  opponent row is empty, only feeding moves that reach that row are legal.
- **State cloning and actions:** OpenSpiel supplies `state.clone()` and
  `state.child(action)`. The latter is the preferred read-only operation for
  hypothetical feature calculations; it clones and applies one legal action.
  `state.apply_action(action)` mutates a state.
- **Relay sowing:** `_sow_relay` repeatedly picks up the final occupied landing
  house and continues sowing. A house reaching four while seeds remain in hand
  is captured by that row's owner. A four formed by the final seed is captured
  by the player who made the move.
- **Capture rules:** Captures are added in multiples of four during sowing.
  Cycle detection, the relay-lap limit, or lack of legal actions can instead
  collect all remaining board seeds by row owner before the game ends.
- **Terminal detection and utilities:** `_score_terminal` ends the game when a
  player has more than 24 seeds, or when both players have exactly 24. Other
  terminal paths include cycle/lap-limit resolution and no legal action after
  a move. `state.returns()` is `[1,-1]`, `[-1,1]`, or `[0,0]` at termination.
- **Final-eight-seeds handling:** there is no distinct final-eight-seeds rule
  in this implementation. The only score threshold is the 24-seed majority /
  24–24 condition, plus the collection-based terminal paths above. The
  heuristic must follow the implementation rather than assume an additional
  rule.
- **State serialization:** there is no custom Ayo serializer. `str(state)` is
  a human-readable board/score/turn summary; the inherited OpenSpiel
  `state.serialize()` records history and serialized state data. The internal
  `_position_key()` is a hashable repetition key containing current player,
  captured scores, and board contents.

### Proposed feature definitions

All features are evaluated from the perspective of a fixed player `p` and use
`q = state.current_player()` for the player-to-move quantities.

#### Capture advantage

\[
C(s,p) = captured(s,p) - captured(s,1-p).
\]

For the default 48-seed game, the conservative rule-based range is
`[-48, 48]`. Reachable non-terminal positions will normally occupy a smaller
range, but the wider bound keeps the feature extractor valid for serialized or
test positions.

#### Immediate tactical potential

For each legal action `a`, create `child = state.child(a)` and measure the
change in the moving player's captured score:

\[
G(s,q) = \max_{a\in A(s)}
  [captured(child,q)-captured(s,q)].
\]

Then convert to player `p`'s perspective:

\[
T(s,p) = \begin{cases}+G(s,q)&q=p\\-G(s,q)&q\ne p.\end{cases}
\]

If there are no legal actions, define `G=0` and therefore `T=0`; the state is
already terminal or has no actionable move. The conservative ranges are
`G in [0,48]` and `T in [-48,48]`. This definition deliberately measures the
actual score delta produced by the implementation, including a collection
resolution if an action causes one. That is the smallest correction needed to
keep the feature faithful to `apply_action`.

#### Mobility

\[
M(s,p) = \begin{cases}+|A(s)|&q=p\\-|A(s)|&q\ne p.\end{cases}
\]

The rule-based range is `[-6, 6]`. Mobility is retained as a hypothesis, not
as an assumed strategic benefit: feeding constraints and relay captures can
make more legal actions useful, neutral, or harmful.

#### Seed control

\[
S(s,p) = rowSeeds(s,p) - rowSeeds(s,1-p).
\]

`rowSeeds` is the sum of `state.board` over the player's six mapped houses.
The conservative range is `[-48,48]`, subject to the invariant that board
seeds plus captured seeds equal 48.

### Terminal override and normalization proposal (original, superseded)

This was the original checkpoint proposal and is retained only for audit
history. It used `W=5`; the corrected authoritative value is documented below.
Each proposed normalized feature has absolute value at most 1,
so four equal-weight features have absolute positional sum at most 4. Return
`+W` for a win by `p`, `0` for a draw, and `-W` for a loss. For a non-terminal
state, the initial comparable-scale proposal is:

\[
\hat C=C/48,\quad \hat T=T/48,\quad \hat M=M/6,\quad \hat S=S/48.
\]

The final normalization and any tighter empirical ranges are to be verified in
Checkpoint 2, after focused tests are approved.

### Strategic justification, overlap, and limitations

- `C` is the most direct progress signal because captured seeds determine the
  final score.
- `T` looks one move ahead for immediate capture opportunities and is intended
  to complement, not replace, `C`.
- `M` measures available choices but its strategic value is explicitly an
  empirical hypothesis.
- `S` measures seeds still controlled in each player's row and may overlap with
  tactical potential because row distributions influence sowing and captures.
- `C` and `S` are not independent: as seeds leave the board, captured scores
  rise. Their difference is nevertheless informative because it separates
  secured seeds from seeds remaining in the rows.
- `T` can overlap with `C` near terminal positions and can be expensive because
  it evaluates every legal child and each child may perform many relay laps.
- The features do not model deeper tactical sequences, positional pit quality,
  repetition risk directly, or opponent response beyond the sign convention.
- The conservative bounds are chosen for correctness and terminal dominance;
  Checkpoint 2 will measure observed ranges on focused and reachable states.

### Checkpoint 1 verification record

The existing Ayo, minimax, and MCTS tests pass: the Ayo rule test module's
direct self-run completed successfully, along with 2 minimax tests and 2 MCTS
tests. The project environment does not
have the `pytest` module installed, so the pytest command itself was
unavailable; the repository's `absltest` entry points were run with the
project `.venv` and local OpenSpiel package path.

Each requested feature is calculable from the implementation without changing
game rules. The only definition adjustment is interpretive: tactical potential
uses the actual captured-score delta after `child(action)`, so collection-based
terminal resolution is included when it occurs. There is no implemented
final-eight-seeds rule to encode.

**Status:** Checkpoint 1 complete. Implementation is intentionally paused until
the feature definitions are approved.

## Checkpoint 2 — implementation and verification

### Implementation decisions

The four independent raw feature functions and the evaluator are implemented
in `Algorithms/ayo_heuristic.py`:

- `capture_advantage(state, player)` implements `C`.
- `immediate_tactical_potential(state, player)` enumerates legal actions with
  `state.child(action)` and implements `T`.
- `mobility(state, player)` implements `M`.
- `seed_control(state, player)` implements `S`.
- `extract_features` returns all four raw features.
- `evaluate_state` applies normalization, weights, and terminal override.

The evaluator accepts either `HeuristicWeights` or a mapping. Missing mapping
entries are treated as zero, which permits feature-subset candidates. Terminal
values are checked before positional features, and the original default `W=5` is rejected
if it would not dominate the four-feature normalized maximum. The feature
extractor never calls a mutating operation on the supplied state.

The initial candidates use equal weights and are exposed as `H_C`, `H_CT`,
`H_CTM`, and `H_CTMS`. These are feature-combination baselines only; no weight
selection has been performed.

### Focused tests

`tests/ayo_olopon/test_ayo_heuristic.py` contains 13 tests covering:

- positive and negative capture advantage;
- no capture, one capture, multiple captures, relay capture, and automatic
  four-seed capture;
- intermediate four-seed ownership;
- both player-to-move perspectives and mobility variation;
- row-seed distributions;
- terminal win, loss, and draw overrides;
- feature and evaluator perspective antisymmetry;
- original-state immutability;
- candidate selection and invalid configuration handling.

All 13 focused tests pass. The existing Ayo rule module, 2 minimax tests, and
2 MCTS tests also pass. The repository environment lacks the `pytest` module,
so tests were run through their `absltest` entry points with `.venv` and the
local OpenSpiel package path.

### Measured ranges and normalization

Using random seed `20260921`, 25 legal reachable random games produced 6,712
player-state feature views. The observed raw ranges were:

| Feature | Observed range | Normalization |
| --- | ---: | ---: |
| `C` | `[-20, 20]` | divide by `48` |
| `T` | `[-16, 16]` | divide by `48` |
| `M` | `[-6, 6]` | divide by `6` |
| `S` | `[-28, 28]` | divide by `48` |

The rule-based denominators are retained rather than fitting to this small
sample. This preserves comparability and avoids using experimental outcomes to
define the feature scale.

### Evaluation cost and limitations

On 10 additional seeded random games (559 non-terminal evaluations), the full
`evaluate_state` call averaged approximately **1.12 ms per evaluation** on the
development machine. Tactical potential dominates the cost because it clones
and evaluates every legal child, including relay-sowing work.

The implementation intentionally remains shallow: it does not model deeper
opponent responses, pit-specific patterns, or repetition risk as a separate
feature. The observed ranges are not exhaustive, and the conservative scales
remain the source of truth for now.

**Status:** Checkpoint 2 complete. Greedy policy experiments, opening-position
generation, candidate comparison, and weight selection are intentionally paused
until approval for Checkpoint 3.

## Checkpoint 3 — greedy evaluation, selection, and finalization

### Experimental protocol

`experiments/heuristic/run_experiment.py` generated 100 unique legal,
reachable, non-terminal positions using recorded seed `20260921`. The target
opening lengths were uniformly sampled from 4–32 plies, giving early positions
(4–12 plies) and middle-game positions (13–32 plies). The positions were sorted
by recorded opening seed and split before games into 70 development and 30
verification positions. The exact snapshots are in
`experiments/heuristic/openings/opening_positions.json`.

For each decision, the greedy policy enumerated every legal action, evaluated a
cloned child state, and selected the maximum-valued action. Ties were resolved
deterministically by choosing the smallest local action number. Each matchup
used the same openings, both player assignments, and a 300-action game-length
limit. A length-limited game was scored by the final captured-seed difference;
an equal difference was recorded as a draw. Per-game action traces and metrics
are preserved in `experiments/heuristic/raw_results/`.

### Candidate comparison on development positions

The five candidates were compared pairwise, with 140 games per matchup
(70 openings × 2 player assignments). Score rates below are from the named
policy's perspective and average across its four opponents:

| Candidate | Average score rate | Average final seed differential |
| --- | ---: | ---: |
| `H_C` | 0.479 | -0.76 |
| `H_CT` | 0.501 | -0.47 |
| `H_CTM` | 0.567 | 0.68 |
| `H_CTMS` | 0.542 | 0.68 |

`H_CTM` was selected. It had the strongest average development score rate and
beat `H_CTMS` head-to-head with score rate `0.521` versus `0.479`. Its average
evaluation cost in the candidate runs was approximately `24.4 ms` per policy
decision, and no severe pathological decision pattern was observed in the
recorded games. The complete candidate aggregates are in
`candidate_development.json` and the raw per-game records are in the same
result file.

### Weight selection on development positions

Because `H_CTM` was selected, only the C/T/M entries were retained from the six
predefined vectors:

| C/T/M weights | Score rate vs RAND |
| --- | ---: |
| `[1, 1, 1]` | 0.600 |
| `[2, 1, 1]` | 0.600 |
| `[4, 2, 1]` | 0.618 |
| `[4, 2, 0.5]` | 0.646 |
| `[6, 3, 1]` | 0.621 |
| `[8, 4, 1]` | 0.646 |

This table is the original, superseded checkpoint record. `[8,4,1]` was not an
independent candidate because it is `2*[4,2,0.5]`.

### Verification result

The frozen `H_CTM` heuristic with weights `[4, 2, 0.5]` was evaluated once on
the 30 verification openings, again with both player assignments (60 games):

| Policy | Wins | Draws | Losses | Score rate | Avg. seed differential | Avg. game length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected `H_CTM` | 33 | 16 | 11 | 0.683 | 4.00 | 31.4 |
| `RAND` | 11 | 16 | 33 | 0.317 | -4.00 | 31.4 |

The selected heuristic averaged 56.94 ms per policy decision and 6.45 tied
action selections per game from its assigned player perspective. Verification
outcomes were not used to redesign the heuristic.

### Frozen final specification (original, superseded)

\[
H(s,p) = 4\hat C(s,p) + 2\hat T(s,p) + 0.5\hat M(s,p),
\]

This original checkpoint record used `C/48`, `T/48`, and `M/6` normalization,
terminal values `+5/0/-5`, and
smallest-action-number tie-breaking. Seed control `S` is excluded from the
final heuristic. The final summary is in
`experiments/heuristic/heuristic_summary.md`.

**Status:** Checkpoint 3 complete. The heuristic definition, normalization,
weights, terminal values, and tie-breaking rule are frozen.

## Correction review and corrected rerun

The original study artifacts and raw results remain preserved in `raw_results/`.
The review found four issues: `W=5` did not dominate the selected maximum
positional score of `6.5`; `[8,4,1]` was a positive scalar multiple of
`[4,2,0.5]`; unfinished 300-action games were not explicitly distinguished
from natural draws; and the runner regenerated openings by default. The full
correction rationale is recorded in `CORRECTION_LOG.md`.

The corrected implementation uses terminal values `+10/0/-10`. It validates
terminal dominance against the active weight configuration. The frozen selected
configuration has maximum positional magnitude `6.5`, so `W=10` is strict. A
temporary active configuration whose weight sum exceeds 10 receives the
smallest strictly dominating terminal value during its evaluation; this
affects only the `[8,3,1]` comparison, while the final frozen heuristic remains
`W=10`.

Weight vectors are normalized by positive scale before comparison. The old
`[8,4,1]` vector is therefore recorded as equivalent to `[4,2,0.5]` and was
not rerun as an independent candidate. The corrected six-vector set replaces
it with `[8,3,1]`.

Corrected raw game records now include `terminal`, `termination_reason`,
`winner`, `natural_draw`, `truncated`, and `action_count`. Termination reasons
are `terminal_win`, `terminal_draw`, `action_limit`, `repetition`, or `error`.
The score-rate convention is explicitly
`(wins + 0.5*naturaldraws + 0.5*scoredtruncations) / totalgames`; truncations
are not natural draws. In the corrected verification set, there were 33 wins,
16 natural draws, 0 truncations, and 11 losses for the selected heuristic.

Frozen openings are now the default. The loader validates all 100 positions,
the 70/30 split, non-terminal legal playability, duplicate absence, and state
hashes. The corrected manifest records the opening path, SHA-256 file checksum,
all state hashes, seeds, split counts, terminal value, and timestamp. Explicit
regeneration requires `--regenerate-openings --seed N` and writes a separate
opening file; it never overwrites the frozen suite.

The corrected rerun used the original frozen openings, player swapping,
deterministic tie-breaking, and 300-action limit. Corrected results are in
`raw_results_corrected/`; original results in `raw_results/` were not
overwritten. `H_CTM` remained the selected feature combination. Corrected
development weight rates were:

| C/T/M weights | Score rate vs RAND | Average seed differential |
| --- | ---: | ---: |
| `[1,1,1]` | 0.600 | 2.457 |
| `[2,1,1]` | 0.600 | 2.543 |
| `[4,2,1]` | 0.618 | 3.000 |
| `[4,2,0.5]` | 0.646 | 3.314 |
| `[6,3,1]` | 0.621 | 3.071 |
| `[8,3,1]` | 0.646 | 3.229 |

The corrected selection is `[4,2,0.5]`, using the secondary final-seed
criterion after the score-rate tie. On verification, it achieved 33 wins, 16
natural draws, 0 truncations, and 11 losses in 60 assignment-balanced games.

### Authoritative corrected evaluator

For the final selected configuration, the evaluator is defined once as:

\[
E(s,p)=
\begin{cases}
10U(s,p), & s\text{ terminal},\\[4pt]
4\hat C(s,p)+2\hat T(s,p)+0.5\hat M(s,p), & s\text{ non-terminal},
\end{cases}
\]

where `U(s,p) in {-1,0,+1}` and
`C-hat=C/48`, `T-hat=T/48`, and `M-hat=M/6`. Therefore
`-6.5 <= E_nonterminal <= 6.5` and
`-10 < -6.5 <= E_nonterminal <= 6.5 < 10`, proving terminal dominance.

For any temporary configurable weight vector `w`, the implementation validates
`W > sum(abs(w_i))`. The corrected runner uses
`W(w)=max(10, sum(abs(w_i))+1e-9)` for an active comparison vector whose
maximum exceeds 10; the frozen selected vector has `sum(abs(w_i))=6.5`, so its
authoritative terminal value is exactly 10.

The unrounded corrected development score rates for the tied leading vectors
were both exactly `0.6464285714285715`: `[4,2,0.5]` and `[8,3,1]`. The tie was
resolved by the secondary final-seed differential,
`3.3142857142857145` versus `3.2285714285714286`, selecting `[4,2,0.5]`.
