# Ayo Olopon heuristic summary

## 1. Final heuristic formula

The authoritative fixed greedy evaluation heuristic is:

\[
E(s,p)=
\begin{cases}
10U(s,p), & s\text{ terminal},\\[4pt]
4\hat C(s,p)+2\hat T(s,p)+0.5\hat M(s,p), & s\text{ non-terminal},
\end{cases}
\]

where `U(s,p) in {-1,0,+1}`. Here `C-hat=C/48`, `T-hat=T/48`, and
`M-hat=M/6`. Thus `-6.5 <= E_nonterminal <= 6.5` and
`-10 < -6.5 <= E_nonterminal <= 6.5 < 10`, proving terminal dominance.

For a temporary active weight vector `w`, the evaluator requires
`W > sum(abs(w_i))`; the corrected runner uses the exact rule
`W(w)=max(10, sum(abs(w_i))+1e-9)`. The frozen selected vector sums to `6.5`,
so its authoritative terminal value is exactly `10`.

## 2. Feature definitions

- `C`: captured seeds of `p` minus captured seeds of the opponent.
- `T`: the maximum captured-score gain from one legal action for the player to
  move, signed from `p`'s perspective.
- `M`: legal-action count, signed positively when `p` is to move.
- `S`: row-seed differential was tested but excluded from the final heuristic.

Tactical potential evaluates actual cloned child states, including any relay
sowing and collection-based terminal resolution caused by the action.

## 3. Normalization

\[
\hat C=C/48,\qquad \hat T=T/48,\qquad \hat M=M/6.
\]

These are conservative rule-based scales. A 25-game reachable-state sample
observed ranges `C=[-20,20]`, `T=[-16,16]`, and `M=[-6,6]`.

## 4. Tested candidate heuristics

The equal-weight candidates were:

`RAND`, `H_C`, `H_CT`, `H_CTM`, and `H_CTMS`.

On 70 development openings with both player assignments, average score rates
across opponents were:

| Candidate | Average score rate |
| --- | ---: |
| `H_C` | 0.479 |
| `H_CT` | 0.501 |
| `H_CTM` | 0.567 |
| `H_CTMS` | 0.542 |

`H_CTM` was selected for weight testing. It also beat `H_CTMS` head-to-head,
0.521 to 0.479.

## 5. Tested weight configurations

The six predefined C/T/M configurations were:

| Weights | Development score rate vs RAND |
| --- | ---: |
| `[1, 1, 1]` | 0.600 |
| `[2, 1, 1]` | 0.600 |
| `[4, 2, 1]` | 0.618 |
| `[4, 2, 0.5]` | 0.646 |
| `[6, 3, 1]` | 0.621 |
| `[8, 3, 1]` | 0.646 |

`[8,4,1]` was removed because it is exactly `2*[4,2,0.5]`; positive scalar
multiples produce identical greedy action rankings. The corrected replacement
was `[8,3,1]`. The corrected rerun selected `[4,2,0.5]`: the unrounded score
rates tied exactly at `0.6464285714285715`, while final seed differential
favored `[4,2,0.5]` (`3.3142857142857145` versus `3.2285714285714286`).

## 6. Greedy evaluation method

At every decision, enumerate all legal actions, clone and apply each action,
evaluate the resulting state, and choose the action with maximum value. Ties
choose the smallest local action number. Games use a 300-action limit and both
player assignments for every opening.

## 7. Results

The verification set contained 30 previously unused openings and 60 games:

| Policy | Wins | Natural draws | Truncations | Losses | Score rate | Final seed differential | Game length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected `H_CTM` | 33 | 16 | 0 | 11 | 0.683 | 4.00 average | 31.4 average |
| `RAND` | 11 | 16 | 0 | 33 | 0.317 | -4.00 average | 31.4 average |

The 16 draws were natural terminal draws; there were zero action-limit
truncations. Repetition terminations and errors were also zero in verification.
Score rate uses the explicit convention
`(wins + 0.5*naturaldraws + 0.5*scoredtruncations) / total_games`. A
truncation is not described as a game-theoretic draw.

The corrected selected policy averaged 63.35 ms per evaluation decision and
6.45 tied action selections per game from its assigned-player perspective.

## 8. Selection justification

`H_CTM` had the strongest average development score rate among the predefined
candidates. The `[4,2,0.5]` vector was selected from the six predefined vectors
using development positions only. It tied `[8,3,1]` on score rate, and the
secondary final-seed criterion selected `[4,2,0.5]`.

Among the predefined candidates, the selected heuristic produced the strongest
or most suitable greedy policy and was adopted as the fixed Ayo Olopon
evaluation heuristic.

## 9. Computational cost

The feature implementation averaged approximately 1.12 ms for a direct full
evaluation on earlier random-state measurements. In the corrected complete
greedy matchups, the selected policy averaged 63.35 ms per decision because each
decision evaluates all legal child states and relay sowing can be expensive.

## 10. Limitations

This is a shallow one-ply greedy heuristic. It does not prove global
optimality, model deeper opponent responses, explicitly model repetition risk,
or establish that mobility is universally beneficial. The opening sample and
verification set are modest, and random-policy comparison is not a substitute
for a strong game-theoretic baseline.

## 11. Reproduction commands

From the repository root on Windows PowerShell, using the exact frozen openings:

```powershell
$env:PYTHONPATH="$PWD\open_spiel;$PWD"
.venv\Scripts\python.exe experiments\heuristic\run_experiment.py --use-frozen-openings
```

 To
explicitly generate a separate suite, use:

```powershell
.venv\Scripts\python.exe experiments\heuristic\run_experiment.py --regenerate-openings --seed 20260921
```

Original results remain in `experiments/heuristic/raw_results/`; corrected raw
results are in `experiments/heuristic/raw_results_corrected/`. The frozen
opening SHA-256 is
`971dba4079f260d733418fed6f008130b56778fd947584f8292669b2c92b2735`.
