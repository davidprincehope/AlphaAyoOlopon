# Handcrafted heuristic methodology

The corrected study selected **H_CTM with weights `[4,2,0.5]`**, using development
openings only. On 30 unused verification openings in both seats it recorded
33 wins, 16 natural draws, and 11 losses against random play: **68.33% score**.
See the [short summary](heuristic_summary.md) for the frozen specification.

## Features and evaluator

The reference is [the Python game model](../../Model/ayo_olopon/ayo_olopon.py).
Player 0 owns board indices `0..5`, player 1 owns `6..11`, and sowing traverses
increasing cyclic indices. Actions use local pit numbers `0..5`, subject to
empty-pit and feeding constraints. Features use `state.child(action)` to inspect
legal successors without mutating the supplied state.

For fixed perspective player `p`, let `q` be the player to move:

| Feature | Definition | Conservative range | Scale |
| --- | --- | --- | --- |
| `C` | `captured[p] - captured[1-p]` | âˆ’48 to 48 | `/48` |
| `T` | Maximum gain in `q`'s captured score over legal children; positive if `q=p`, negative otherwise | âˆ’48 to 48 | `/48` |
| `M` | Legal-action count; positive if `q=p`, negative otherwise | âˆ’6 to 6 | `/6` |
| `S` | Seeds in `p`'s row minus seeds in the opponent's row | âˆ’48 to 48 | `/48` |

With no legal actions, tactical gain is zero. `T` includes relay sowing and any
collection-based terminal resolution caused by the action. There is no separate
final-eight-seeds rule in this implementation. Captured seeds plus board seeds
remain 48. `S` was tested and excluded from the final evaluator.

The authoritative selected evaluator is:

\[
E(s,p)=\begin{cases}
10U(s,p), & s\text{ terminal},\\
4C(s,p)/48+2T(s,p)/48+0.5M(s,p)/6, & \text{otherwise},
\end{cases}
\]

where `U` is âˆ’1, 0, or +1 for loss, draw, or win. Nonterminal magnitude is at
most 6.5, so terminal Â±10 strictly dominates it. For configurable weights,
`evaluate_state` requires `W > sum(abs(w_i))`; the corrected runner uses
`W(w)=max(10, sum(abs(w_i))+1e-9)`.

[Algorithms/ayo_heuristic.py](../../Algorithms/ayo_heuristic.py) exposes the
four feature functions, `extract_features`, and `evaluate_state`. Weights may
be a `HeuristicWeights` object or a mapping; missing entries mean zero.
Terminal values are checked before positional features. A greedy decision
evaluates every legal child from the original actor's perspective and selects
the maximum; ties choose the smallest local action.

## Frozen experimental protocol

Seed `20260921` generated 100 unique reachable nonterminal openings at 4â€“32
plies. Sorting by recorded opening seed fixed a **70 development / 30
verification** split before matches. Openings are retained in
[opening_positions.json](openings/opening_positions.json).

Each matchup plays both seats for every opening: 140 development or 60
verification games. Games have a 300-action limit. Corrected records separate
natural results, action-limit truncations, repetition, and errors. Capped games
are scored by captured-seed lead: win 1, tie 0.5, loss 0, with their cutoff status
recorded separately. Score rate is `(wins + 0.5 * draws) / games`, including
those adjudicated outcomes. Verification had no truncations, repetition
terminations, or errors.

### Feature selection

The four equal-weight feature combinations and RAND were compared pairwise.
Average development score across each candidate's four opponents:

| Candidate | Average score | Average final seed differential |
| --- | ---: | ---: |
| H_C | 0.479 | âˆ’0.76 |
| H_CT | 0.501 | âˆ’0.47 |
| H_CTM | 0.567 | 0.68 |
| H_CTMS | 0.542 | 0.68 |

H_CTM led overall and scored 0.521 against H_CTMS head-to-head.

### Weight selection and verification

Only C/T/M weights were then varied against RAND on development openings:

| Weights | Score | Average seed differential |
| --- | ---: | ---: |
| `[1,1,1]` | 0.600 | 2.457 |
| `[2,1,1]` | 0.600 | 2.543 |
| `[4,2,1]` | 0.618 | 3.000 |
| `[4,2,0.5]` | 0.646 | 3.314 |
| `[6,3,1]` | 0.621 | 3.071 |
| `[8,3,1]` | 0.646 | 3.229 |

The leading score rates tied exactly at `0.6464285714285715`. The secondary
seed-differential criterion selected `[4,2,0.5]`: `3.3142857142857145` versus
`3.2285714285714286`. Verification was not used to redesign the evaluator.

The frozen greedy policy's verification result was **33â€“16â€“11**, with average
seed differential **+4.00**, game length **31.4 actions**, and **6.45 tied
action selections per game** from its assigned-player perspective.

## Corrections, checks, and limits

The original study used terminal Â±5 (below the selected 6.5 feature bound),
counted `[8,4,1]` as independent despite being `2 Ã— [4,2,0.5]`, conflated
unfinished games with natural draws, and regenerated openings by default.
The corrected rerun used Â±10, replaced that redundant vector with `[8,3,1]`,
added explicit termination fields, and loaded the original frozen suite.
Authoritative corrected records and manifests are in `raw_results_corrected/`.

Focused checks cover captures, both perspectives, terminal dominance,
immutability, antisymmetry, feature subsets, and invalid settings. A 25-game
sample (seed `20260921`, 6,712 player-state views) observed `C=[-20,20]`,
`T=[-16,16]`, `M=[-6,6]`, and `S=[-28,28]`; normalization retains conservative
rule-based bounds. Direct full evaluation averaged 1.12 ms on an earlier
559-position sample; the corrected greedy policy averaged 63.35 ms per decision
on the development machine. These measure different workloads.

Tactical potential is expensive because it explores every legal child and
relay sequence. The evaluator does not model deeper replies, pit-specific
patterns, or repetition risk directly. Mobility's value is empirical; the
modest opening sample and random opponent do not establish global optimality.

## Reproduce

From the repository root in an activated environment:

```powershell
$env:PYTHONPATH = "$PWD\open_spiel;$PWD"
python experiments/heuristic/run_experiment.py --use-frozen-openings
```

The loader checks legality, uniqueness, the 70/30 split, and saved state hashes.
The frozen opening file's SHA-256 is
`971dba4079f260d733418fed6f008130b56778fd947584f8292669b2c92b2735`.
To generate a separate suite explicitly:

```sh
python experiments/heuristic/run_experiment.py --regenerate-openings --seed 20260921
```

Regeneration writes a separate opening file and preserves the frozen suite.
