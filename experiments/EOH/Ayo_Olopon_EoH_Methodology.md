# EoH methodology for the Ayo Olopon evaluation heuristic

**Purpose.** Develop one reproducible, non-terminal evaluation function for Ayo Olopon by adapting Evolution of Heuristics (EoH) [1]. This is a heuristic-development experiment conducted before any Minimax depth sweep. The EoH paper establishes the *generation and evolution mechanism* on combinatorial optimization tasks; competitive two-player gameplay is our adaptation, not a result established by that paper.

**Research question.** Can iterative LLM-generated heuristic design, selected by controlled gameplay, produce an effective Ayo evaluation function under a fixed one-ply greedy policy?

## 1. Experimental boundary and controls

Freeze the OpenSpiel game version, implemented rules, terminal utility, state serialization, legal-action ordering, maximum-game-length convention, and player-perspective convention. Record the exact versions and prompts of any LLMs used. One LLM can run EoH; if several LLMs are used, report each model's candidate count and budget, and either run separate matched evolutions or explicitly label a mixed-model population. Do not attribute the result to one model when offspring have multiple model ancestries.

**Reuse the earlier frozen states.** The corrected earlier experiment reports `experiments/heuristic/openings/opening_positions.json`: 100 unique reachable, non-terminal positions sampled from 4–32 plies with seed `20260921`, split into 70 development and 30 verification positions. The loader checks the split, state hashes, legality, and file checksum. Locate this file in the actual game repository, verify the checksum and game compatibility, and retain the recorded split. Use the **70 former development positions for EoH generation and fitness**. The **30 former verification positions** may be used once to choose among an EoH shortlist, but they are already examined data from the previous experiment: label them `EoH_selection`, not a fresh held-out test. Do not regenerate, overwrite, or reshuffle either split. Both seat assignments should be played from each position. A later independent performance claim requires a **new, previously unseen** test suite constructed and frozen before evaluating the selected EoH heuristic on it.

**Checkpoint 0 — experimental contract.** Advance when rule/invariant tests pass; all 100 frozen states, their 70/30 membership and hashes are verified; opponent agents, tie-breaking, match budgets, fitness weights, compute cap, random seeds, and stopping rule have been recorded before evolution. An unexpected game loop, invalid terminal result, or player-perspective mismatch blocks progress.

## 2. Candidate representation and fixed decision policy

Following EoH, represent each candidate by (i) a short natural-language strategic idea, (ii) its executable Python evaluation function, and (iii) its measured fitness [1]. The function has the conceptual interface

```python
def evaluate_state(state_view, player_id: int) -> float:
    """Return a finite estimate of the advantage for player_id."""
```

`state_view` is a read-only, documented snapshot of information available in the actual game, including pit counts, captured counts, player to move, and any rule-relevant counters. The LLM may change the computation *inside* `evaluate_state`, but not the game rules, legal-action generator, greedy policy, opponents, or evaluation protocol. No terminal reward is invented by a candidate: terminal states always use the game's exact utility on a scale that strictly dominates the allowed non-terminal output range.

For a state where player \(p\) acts, every candidate is attached to the identical one-ply policy:

\[
\pi_h(s)=\underset{a\in A(s)}{\arg\max}\;V_h(T(s,a),p),
\quad V_h(s',p)=
\begin{cases}
U(s',p),&s'\text{ terminal},\\
h(s',p),&\text{otherwise}.
\end{cases}
\]

Use fixed legal-action-order tie-breaking. The evaluation always remains from the **original acting player's** perspective, even if the successor is the opponent's turn. Minimax is *not* used during EoH fitness measurement; it is introduced only after the heuristic is frozen.

**Checkpoint 1 — interface.** Test the greedy wrapper with a simple known-good function. Verify legal actions, exact terminal preference, reproducibility, unchanged input states, perspective consistency on hand-built positions, and seat-swapped games. Advance only if every test passes.

## 3. Candidate safety and correctness screening

Give EoH a fixed function template, full Ayo rules, data schema, terminal/perspective convention, and explicit prohibition on searching, mutating game state, file/network access, or consulting opponent policy. Generated Python is untrusted: execute it in an isolated worker with CPU, memory, and wall-clock limits and no unnecessary file or network privileges. Do not rely on a Python exception handler as a sandbox.

Before gameplay, reject functions that fail to compile, throw exceptions, mutate inputs, return NaN/infinity or non-numeric values, exceed time limits, depend on hidden data, or produce non-deterministic scores. Test both player identities and curated tactical, terminal-adjacent, empty-pit, and relay-sowing cases. If a known player-swap transform preserves the rule state, verify the value changes sign (within tolerance); otherwise use paired perspective test fixtures rather than asserting an invalid symmetry. Enforce or clip to a preregistered non-terminal range below terminal utility.

**Checkpoint 2 — candidate validator.** Known valid fixture code passes and a deliberate suite of invalid/mutating/slow code fails. At least one LLM-generated candidate must pass. If none does, repair the *interface or prompt contract* using development cases, document the revision, and restart the generation budget; do not use the 30 selection positions as prompt feedback.

## 4. EoH generation and evolution

Initialize a population of \(N\) LLM-generated **thought–code** pairs. Evaluate each valid candidate on the frozen evolution arena. Later generations select parent candidates by fitness and ask the LLM to produce new thought–code pairs using EoH-style operators [1]:

| Operator | Instruction to the LLM | Research role |
|---|---|---|
| E1 | Propose a substantially different idea from several parents | Explore new strategies |
| E2 | Extract a shared idea from several parents, then extend it | Recombine useful ideas |
| M1 | Improve one parent | Modify structure |
| M2 | Change the parameters of one parent | Tune weights or thresholds |
| M3 | Remove redundant components | Simplify and reduce cost |

The published EoH algorithm uses all five evolutionary strategies and retains the best feasible individuals after each generation [1]. The current repository example exposes a configurable operator list and illustrates `e1`, `e2`, `m1`, and `m2`; check whether `m3` is supported in the pinned version before claiming that all five were run [2]. Save every generated thought, source code, prompt/model identifier, parent IDs, validation outcome, fitness, and run seed. An LLM suggestion is **not** presumed to improve its parent; only match results assign fitness.

Pre-register population size \(N\), number of generations \(G\), operator set, calls per operator, and total LLM/game budgets after a short throughput pilot. For example, \(N=5\), four operators, and three generations permit up to \(5+3(4)(5)=65\) candidates before invalid-output retries. This is an *illustrative Ayo budget*, not a parameter taken from EoH. Pin a repository commit because its current `LLMConfig` / `EoH` / `BaseProblem` API differs from the earlier v0.1 API [2].

**Checkpoint 3 — evolution loop.** Complete one miniature generation and verify parent attribution, feasible-candidate selection, reproducible fitness for identical code, preservation of the current best candidate, and exact enforcement of the query/game budget. A lack of score improvement does not fail this engineering checkpoint.

## 5. Gameplay fitness and selection

First generate and validate an initial EoH population. From this population, choose a prespecified number of *distinct*, valid thought–code heuristics as a **frozen anchor pool** before any subsequent evolution; document the deterministic selection/diversity rule and their hashes. Anchors remain unchanged throughout EoH, even if they leave the evolving parent population. Add random legal play (`RND`) as a low-weight sanity anchor. This gives each new candidate the same fixed adversaries without evaluating against the earlier handcrafted heuristic. If too few valid or meaningfully distinct initial candidates exist, fix the initialization procedure in a labeled pilot before freezing the pool.

On the 70 evolution positions, play candidate \(h\) against **every member of this same frozen panel**. Use identical position IDs, both seat assignments, and predefined random seeds for `RND`. Self-matches with an anchor that is also a candidate remain in the schedule so every candidate has the same opponent set; identify them in the logs. Random-only performance is insufficient to distinguish stronger heuristics.

For opponent \(j\), define the match score after the frozen game-length and repetition scoring rules assign each game exactly one win, draw, or loss. A scored truncation counts once in its assigned outcome and is *also* reported separately by termination reason; do not double-count it.

\[
S_j(h)=\frac{W_j+0.5D_j}{N_j}.
\]

Pre-register nonnegative panel weights \(\alpha_j\) summing to one, with most weight on generated-heuristic anchors rather than `RND`, and maximize \(S_{\mathrm{dev}}(h)=\sum_j\alpha_jS_j(h)\). Since the repository's `BaseProblem.evaluate_program` expects **lower** fitness to be better, return \(f(h)=1-S_{\mathrm{dev}}(h)\) for feasible candidates; return a failure value for invalid ones [2]. Use a fixed runtime cap rather than changing the primary fitness mid-run. Report cost per evaluation as a separate metric. Preserve the corrected runner's distinction between natural draws and action-limit/repetition truncations and verify its existing scoring rule before adopting it. Duplicate code or behaviorally identical candidates should not receive extra games under the same budget.

After the preregistered generation/query budget, rank candidates using evolution results and evaluate only the top prespecified shortlist on the **30 earlier verification positions**, now named `EoH_selection`, against the same panel and, if preregistered, head-to-head among shortlisted candidates. Choose one candidate using the prespecified selection rule (highest aggregate score, with simpler/faster code as a declared tie-break). Do not repeatedly probe those 30 positions during evolution. Save its code hash, thought, prompts, opponent versions, and all match IDs; freeze it as \(H^*\).

**Checkpoint 4 — selection.** Advance when the planned budget and matches are complete, all candidate outcomes and computation costs are logged, and exactly one EoH candidate \(H^*\) is frozen. Document its performance against the fixed anchors, including any failure to improve across generations. Do not substitute the earlier handcrafted heuristic in this selection stage.

## 6. Independent final test (later) and handoff

The current task ends with a frozen EoH heuristic and its results on development/selection states. Those scores demonstrate the search and selection process; **they are not an independent estimate of final playing strength**. If a later study needs such an estimate, generate and freeze a separate untouched position set before evaluating \(H^*\) on it. Compare against the frozen anchor pool, random, and any subsequently selected external baselines only under a separate preregistered protocol. A proposed starting target is 200 new position pairs for a primary head-to-head (400 games); finalize this from a throughput/power pilot before seeing those results. Report W–D–L, seat breakdown, paired confidence intervals, game truncations, median evaluation time, LLM calls, and total compute. Do not revise \(H^*\) after seeing the independent results.

**Checkpoint 5 — handoff.** Deliver the selected EoH function, its code hash, full thought/code lineage, exact candidate-selection results, runtime, and all logs. Minimax depth and decision cost belong to a subsequent experiment; no Minimax result feeds back into EoH. If an independent final test is later run, add its report as a distinct checkpoint without revising the frozen heuristic.

## Minimal results record

| Stage | Required output |
|---|---|
| Contract | Rule/perspective specification; original frozen file checksum, all 100 state hashes and 70/30 membership, opponents, seeds, budgets |
| Validator | Test fixtures; pass/fail reasons; isolation/time-limit checks |
| Evolution | Candidate thought/code/parent/operator/model; score by frozen anchor; cost by generation |
| Selection | W–D–L and match score on the reused 30 positions; chosen code hash |
| Independent test, if later run | New suite provenance, paired score matrix, intervals, seat breakdown, runtime, and full logs |
| Handoff | Frozen EoH heuristic function and exact future search input interface |

## References

[1] Liu, F., Tong, X., Yuan, M., Lin, X., Luo, F., Wang, Z., Lu, Z., & Zhang, Q. (2024). *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model.* ICML, PMLR 235, 32201–32223. https://proceedings.mlr.press/v235/liu24bs.html ; methodology: https://arxiv.org/html/2401.02051v3

[2] Liu et al. *EoH official repository* (pin the exact revision used for the experiment). https://github.com/FeiLiu36/EoH
