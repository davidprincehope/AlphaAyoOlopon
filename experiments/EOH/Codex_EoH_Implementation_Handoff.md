# Codex handoff: implement and audit EoH for Ayo Olopon

## Goal and authority

Implement the experiment in `Ayo_Olopon_EoH_Methodology.md` in the user's **existing Ayo Olopon repository**. Deliver a reproducible pipeline that uses an LLM to generate and evolve *evaluation functions* and measures their strength under one fixed greedy policy. Use the already frozen states from the earlier heuristic evaluation. The goal in this task is to obtain and document a selected EoH heuristic; the prior handcrafted heuristic is **not an opponent, fitness component, selection criterion, or fallback**. Do not claim to have run the full evolutionary search or an independent final test if only a pilot was run. Do not build a new game or change its rules to fit EoH.

Reference methodology: Liu et al. (2024), *Evolution of Heuristics*, ICML, especially §§3.1–3.4: https://proceedings.mlr.press/v235/liu24bs.html and https://arxiv.org/html/2401.02051v3 . Official implementation: https://github.com/FeiLiu36/EoH . Pin the exact upstream commit. The paper uses an LLM-generated natural-language **thought** plus executable **code** for every candidate, a fitness evaluation on fixed problem instances, parent selection, evolution prompts, and elite population selection. Its original operators are E1, E2, M1, M2, and M3, normally producing at most `5 × population_size` proposals per generation. The README's example shows only four operators; audit the pinned code rather than assuming the README example replicates the paper. Our game-based fitness against a frozen initial EoH anchor pool is an **explicit adaptation**.

If `Ayo_Olopon_EoH_Methodology.md` is not available in the repository, have the user attach it before implementing choices not fully specified here. If the game repository is not available, report that specific blocker; do not simulate completion with an invented Ayo environment.

## How to work and how to report

Work in numbered checkpoints. At each checkpoint, commit or otherwise preserve reviewable code and write a short entry in `docs/eoh/implementation_log.md` containing: objective, files changed, commands run, test results, observed evidence, deviations from the paper or methodology (with reason), and the next step. Record failures too. Do not mark a gate passed merely because code compiles. Keep unrelated research code unchanged.

Maintain these review artifacts in the existing repo:

- `docs/eoh/README.md`: architecture, how to install, test, pilot, run, resume, and reproduce;
- `docs/eoh/methodology_mapping.md`: paper § / EoH component → concrete code → Ayo adaptation → verification evidence;
- `docs/eoh/experiment_manifest.yaml`: frozen rules/version, heuristic contract, original opening file checksum and 70/30 membership, EoH anchor IDs/hashes, seeds, match protocol, model and prompt IDs, upstream EoH commit, budgets, stopping rule, fitness weights, and selection rule;
- `docs/eoh/implementation_log.md`: checkpoint-by-checkpoint audit trail;
- machine-readable candidate catalog and match logs with documented schemas; and
- a results report that distinguishes implementation tests, the pilot, evolution on the reused 70 positions, selection on the reused 30 positions, and any later independent test.

Never place LLM credentials or full secret-bearing request payloads in source control or artifacts. Generated heuristic programs are untrusted code: run them in isolated, resource-limited workers with restricted filesystem/network access. Clearly describe the actual isolation boundary; a Python try/except or thread timeout alone is insufficient.

## Checkpoint 0 — inspect and freeze the experimental contract

1. Read the actual OpenSpiel game, its tests, existing greedy and handcrafted heuristic implementations, any existing match runner, and repo guidance. Document the precise Ayo rules used, score/terminal utilities (including `+10/0/-10` only if verified in code), draw or move-limit policy, turn/perspective convention, and clone/serialization behavior. Do not substitute generic Oware rules.
2. Locate the earlier *corrected* heuristic experiment's `experiments/heuristic/openings/opening_positions.json` and its manifest. Its record describes 100 unique legal positions, seed `20260921`, a 70 development / 30 verification split, state hashes, a 300-action limit, deterministic smallest-action tie-breaking, and explicit natural-draw versus truncation records. Verify these facts against the actual artifacts. Do not rerun or compare the earlier handcrafted heuristic during EoH.
3. Implement the official EoH dependency at a pinned revision; check the actual problem-extension API, objective direction, candidate serialization, operator support, parent selection, elite selection, retries, and generated-code execution path. Record the paper-to-code differences, especially M3 and the number of proposals per generation. Prefer a small adapter around the official code rather than a replacement algorithm.
4. Load the existing frozen file by default, verify its checksum and every state hash, game compatibility, legality, uniqueness and recorded 70/30 membership. Do not regenerate, overwrite, or reshuffle it. Use the 70 former development positions for evolution and the 30 former verification positions **once** for EoH selection. Both sets were used or inspected in the earlier study; label the 30 as reused selection data, **not** a new held-out final test. Play each position with seats swapped. If independent final strength is later required, create a separate untouched suite in a separately scoped evaluation task.

**Evidence to show:** game-contract note, passing existing game tests, checksum/hash and 70/30 membership verification, pinned EoH revision and operator audit. **Gate:** resolve incorrect rules, terminal/perspective ambiguity, frozen-suite mismatch, or cross-split leakage before moving on.

## Checkpoint 1 — one fixed greedy-policy interface

Implement `evaluate_state(state_view, player_id) -> finite float`, where `state_view` is a documented, read-only view of information available in the real game. For each legal action at state `s` with original acting player `p`, clone and apply the action, evaluate its successor from `p`'s perspective, and select the largest value. Use exact game utility on terminal successors; otherwise use the candidate heuristic. Break ties by the earlier corrected runner's fixed smallest-action-number rule. This policy is identical for all EoH candidates and frozen EoH anchors. No heuristic may replace legal moves, choose actions directly, invoke minimax, or access the opponent's implementation.

**Evidence to show:** tests on hand-worked Ayo positions for action legality, clone immutability, the original player's perspective after turn changes, terminal move preference, ties, and seat swaps. Include at least one example that lists each legal action, successor features, heuristic score, and chosen action. **Gate:** tests pass; a simple test fixture completes paired games deterministically under fixed seeds.

## Checkpoint 2 — generated-code contract and validator

Build the LLM prompt from the verified game rules and the frozen function signature. Require one concise thought and its matching code. Keep the game and greedy-policy scaffold outside the editable candidate code. Validate imports/API use, execution, finite bounded outputs, no input mutation, deterministic scores on identical states, speed and resource limits, both player IDs, terminal-adjacent and relay-sowing fixtures, and valid player symmetry where a verified transform exists. Reject unsafe or invalid candidates with recorded reasons and without altering the outcome data. Make the terminal positive win utility strictly outrank any allowed nonterminal score; define handling of draw/loss utility explicitly.

**Evidence to show:** validator specification, pass/fail cases including deliberately faulty code, one successful candidate dry run, CPU/memory/time enforcement, exact number of LLM retries. **Gate:** known-good fixture code passes and known-invalid programs fail. If no generated candidate passes, document prompt/interface changes and restart a clearly labeled development pilot without looking at the 30 selection outcomes.

## Checkpoint 3 — gameplay fitness, reproducibility, and throughput pilot

Initialize an EoH population, then freeze a prespecified number of distinct, valid **LLM-generated initial heuristics** as the permanent opponent anchors before subsequent evolution. Document the selection/diversity rule and source hashes. These anchors stay fixed even if they leave the evolving parent population. Add random legal as a low-weight sanity anchor. If too few distinct initial heuristics pass validation, revise the initialization prompt in a labeled pilot before freezing the panel. Use the same 70 evolution positions, both seats, deterministic action ties and recorded random-opponent seeds for all candidate evaluations. Verify and freeze the corrected runner's existing rule for determining the result of an action-limit or repetition truncation; record truncations separately from natural draws. Each game must contribute to **exactly one** win, draw, or loss. For anchor `j`, compute `score_j = (wins + 0.5*draws)/games`, counting an assigned truncated outcome once and recording its termination reason separately. Pre-register `alpha_j >= 0` summing to 1, with the larger share on generated-heuristic anchors; `S = sum_j alpha_j*score_j`. Verify objective direction in the pinned official API, then return `fitness = 1 - S` if lower is better. Include matches against oneself in the same fixed anchor schedule when a candidate is an anchor. Report seat splits, game lengths, terminal reasons and runtime separately. Avoid rewarding duplicate code or behavior with extra evaluations.

Run a small *labeled pilot only* to benchmark games per candidate and LLM cost and to verify deterministic replay. Propose feasible exact opening counts, shortlist size, population size, generations, operator calls, query/game budgets, timeout, and stopping condition in the manifest. Keep default proposals clearly labeled as proposals until frozen. **Do not launch a paid or lengthy full EoH run before presenting its projected calls, matches, runtime, and cost for user review.**

**Evidence to show:** frozen initial anchor catalog, one fully replayable candidate match log, repeated-fitness consistency, treatment of draws/truncations, panel weighting calculation, throughput/cost estimate. **Gate:** identical code and seeds reproduce game outcomes and fitness; all proposed full-run settings are explicit and ready to review.

## Checkpoint 4 — EoH evolution loop

After review of the frozen full-run budget, initialize `N` LLM-generated thought/code pairs. For each generation, select parents according to the pinned EoH implementation, generate offspring using the declared operator set, validate and evaluate feasible offspring, and select the best `N` feasible candidates for the next generation. Use E1 to seek distinct ideas; E2 to extend shared ideas; M1 to modify structure; M2 to tune parameters; and M3 to simplify **if supported and chosen**. If the official pinned version lacks M3, either add it in a documented, separately tested patch faithful to the paper or run four operators and report that variation explicitly. Do not claim paper-exact five-operator EoH after running four.

Record for **every attempted generation**: ID, generation, operator, parent IDs, LLM provider/model/version, prompt and response (redacted for secrets), thought, source hash, validation reason, per-anchor W–D–L and truncations, fitness, compute, and whether selected into the next population. Make retries, duplicate handling, interruption and resume deterministic and auditable. No evolution runs on the 30 selection positions.

**Evidence to show:** successful miniature-generation audit before full run; then generation-by-generation population table, best-so-far curve, discarded/error counts, final candidate catalog. **Gate:** actual queries/matches stay within the reviewed budget, elite retention works, and outcomes can be reconstructed from saved logs. Score improvement is an empirical result, not an engineering gate.

## Checkpoint 5 — selection on reused positions and heuristic handoff

Use the preregistered shortlist rule to select candidates from evolution results. Compare that shortlist **once** on the 30 earlier verification positions using the same frozen panel and any preregistered shortlist head-to-head matches. These positions are reused **selection data**, not an independent final test. Freeze the winning EoH candidate's code and thought as `H*` based on the preregistered score and tie-break rule. Seal configuration, source hashes, analysis script, opponents, and match count. Do not substitute the earlier handcrafted heuristic if EoH performance is disappointing; report the result.

Report the 70-position evolution and 30-position selection results separately, including W–D–L, truncations, match score by anchor, seat breakdown, optional opening-pair intervals, runtime, generated/valid candidate counts, LLM calls and cost. Describe the selected function and its lineage and show how it scores one real decision. This finishes the current EoH task. A new untouched final-test suite and comparisons with external baselines can be designed later as a separate evaluation; no result from the 70/30 reused suite should be advertised as an independent generalization estimate.

**Evidence to show:** frozen pre-selection manifest and hashes, one logged selection pass, raw paired logs, reproducible analysis output, frozen `H*` source and code hash. **Gate:** the EoH heuristic is selected and documented; Minimax depth experiments are a separate next task and must never feed back into EoH selection.

## Review checklist for the user

Before calling the implementation complete, answer these in the results report with file or commit references:

1. Which parts reproduce Liu et al.'s mechanism, and which parts adapt it to Ayo?
2. Which EoH revision and which of E1/E2/M1/M2/M3 were actually used? How were parents and elites selected?
3. What exact game rules, player perspective, heuristic range, and terminal outcomes were verified?
4. Was the original 100-position frozen file reused unchanged, with its checksum, 70/30 membership and state hashes verified?
5. What does one candidate actually return for a worked Ayo state, and how does greedy choose the action?
6. How are failed/unsafe candidates handled? Can a candidate inspect opponent code or the 30 selection positions?
7. What is the precise fitness formula and per-opponent/seat breakdown? Can an existing candidate's score be reproduced from logs?
8. What was the LLM call/game/runtime cost per generation and in total?
9. Was the selection rule frozen before using the 30 earlier verification positions? Which EoH candidate was selected, and why?

The handoff is implementation-ready once the existing game repository, the original frozen opening file, and the methodology document are available. The full-run population, game counts, LLM provider, and spending budget are determined from the checkpoint-3 pilot, then recorded before full evolution. Do not invent those as if they came from the EoH paper. Do not run the old handcrafted baseline or generate a new opening suite in this task.
