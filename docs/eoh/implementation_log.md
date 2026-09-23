# EoH checkpoint log

## Checkpoint 0 — contract audit (partial; gate not passed)

- **Objective:** inspect game, corrected runner, and frozen suite before changes.
- **Files changed:** none in this checkpoint.
- **Evidence:** actual game is `Model/ayo_olopon/ayo_olopon.py`; corrected runner is `experiments/heuristic/run_experiment.py`. Game return vector is ±1/0, not ±10. Frozen file is `experiments/heuristic/openings/opening_positions.json`; corrected manifest gives SHA-256 `971dba4079f260d733418fed6f008130b56778fd947584f8292669b2c92b2735`, counts 70/30, 300-action limit, and random game seed 20260922. Runner labels action-limit seed-lead outcomes separately and resolves repeated relay reports as `repetition`.
- **Commands:** `rg ...` audit; `git -C open_spiel status --short`; `git -C open_spiel rev-parse HEAD`.
- **OpenSpiel checkout:** `48401890ee9857e611678302371378175a8e4c6b`; working tree already has modified `CMakeLists.txt` files and was left untouched.
- **Failure/limitation:** `git ls-remote https://github.com/FeiLiu36/EoH.git refs/heads/main` failed because shell network access to github.com is blocked. Public repository page is accessible via web and documents the current API and an example configured with e1/e2/m1/m2. This is not source verification of the actual supported set. No exact full commit SHA/source tree could be retrieved; therefore the required pinned dependency and full operator/objective/selection/execution audit are incomplete. Upstream v0.1 release page shows abbreviated `4b2338f`, insufficient to pin or audit here.
- **Deviation:** no rules altered. Do not proceed to evolution until upstream is pinned and frozen data compatibility verification passes end-to-end.

## Checkpoint 1 — fixed greedy policy (implemented; pilot evidence pending)

- **Files changed:** `Algorithms/eoh_ayo.py`; `experiments/EOH/pilot.py`.
- **Contract:** immutable snapshot, actor perspective retained after turn switch, `state.child`, first-action tie, terminal utility ±11/0, candidate range ±10.
- **Tests/commands:** recorded after pilot run below.
- **Deviation:** terminal utilities follow actual OpenSpiel ±1 with strictly dominant policy sentinels ±11; differs numerically from prior heuristic's experimental ±10 convention but preserves ordering.

## Checkpoint 2 — candidate validator (partial; gate not passed)

- **Files changed:** `experiments/EOH/candidate_validator.py`.
- **Checks:** required function/signature, restricted AST syntax/calls, numeric finite bounded outputs, frozen dataclass input, runtime check, exception reason.
- **Failure/limitation:** not an isolated worker; no enforceable CPU/memory/process/network restrictions. A generated program that escapes the AST policy is not safely contained. No LLM-generated candidate exists because provider/model is not configured.
- **Deviation:** source-level prefilter only, explicitly weaker than requested sandbox. Full evolution is prohibited until OS-level isolation is added.

## Checkpoint 3 — fitness/reproducibility/throughput pilot (fixture pilot only)

- **Files changed:** `experiments/EOH/pilot.py`, `docs/eoh/*`.
- **Commands/tests and measurements:** populated by executing the commands below; fixture pilot deliberately makes zero LLM calls and plays only the first development opening with both seats. It is not a candidate fitness or EoH pilot.
- **Deviation:** cannot freeze a generated anchor panel without LLM output, and cannot audit upstream adapter. No 30-position selection data are evaluated.

## Checkpoints 4–5 — not started

Full evolution and selection are intentionally not started. The LLM provider/model/cost is unspecified, official full-SHA pin unavailable from this environment, and validator lacks an OS sandbox. No candidate `H*` is selected. Minimax has not been used.

## Executed checks and pilot evidence (2026-09-22)

- `.\.venv\Scripts\python.exe experiments\EOH\pilot.py` — **passed**. Loader verified 100 records, 70 development / 30 reused selection membership, checksum `971dba4079f260d733418fed6f008130b56778fd947584f8292669b2c92b2735`, and stored hashes via corrected runner. No selection positions were used for game outcomes. Candidate fixture passed six player/action successor checks. 600 fixture calls in 0.001931 sec (310,720 calls/sec on this run; timing varies). Paired pilot on development opening 0: candidate seat 0 draw in 33 actions; candidate seat 1 win in 15 actions; both terminal, seeds 20260922 and 20260923. Detailed raw actions are in `experiments/EOH/pilot_results.json`.
- `.\.venv\Scripts\python.exe -m unittest tests.ayo_olopon.test_eoh_interface tests.ayo_olopon.test_ayo_olopon tests.ayo_olopon.test_heuristic_corrections` — **passed**, 8 tests. Covers game basics, prior correction invariants, legal greedy choice, source-state immutability, post-move perspective, terminal preference, seat-swapped turn, tie-breaking, and valid/invalid/looping candidate source.
- `.\.venv\Scripts\python.exe -m pytest ...` — **failed to start**, `No module named pytest`; retried equivalent selected suite under `unittest`, which passed.
- Pilot worked example: opening 0 has actor 1 and legal actions 2, 3, 5. Fixture function scores `capture_lead/48 + row_seed_delta/96`: action 2 = `8/48 - 6/96 = 0.1041667`; action 3 = `8/48 - 2/96 = 0.1458333`; action 5 = `8/48 - 4/96 = 0.125`. The greedy wrapper chooses action **3**. Each successor changes player-to-move to 0, but all three scores remain from original actor 1's perspective. Exact successor snapshots are in the pilot JSON.
- Proposal for later review only: 5 initial + 3 generations × 4 operators × 5 proposals = 65 candidate slots; one retry allows up to 130 requests. 65 × 70 openings × 2 seats × 6 anchors = 54,600 evolution games; 3 × 30 × 2 × 6 = 1,080 once-only selection games, total 55,680. Naive observed match-time extrapolation is 6.6 minutes; planning range 7–30 minutes plus LLM latency, weakly based on two short pilot games. Numeric provider cost is unavailable: no provider/model/token price was specified. Cost formula is 130 × per-request price (or token pricing). Exact configuration is explicitly proposal-only in manifest; do not launch on it yet.
- **Checkpoint status:** Checkpoint 1 passes for policy properties exercised. Checkpoint 0 remains partial due official full-SHA pin and serialization audit. Checkpoint 2 remains partial due non-isolated validator and absence of an LLM-generated passing candidate. Its AST rejects imports, attribute writes are blocked by the frozen dataclass at runtime, and loops/comprehensions are rejected; the runtime check is not a hard kill and memory/CPU isolation is absent. Checkpoint 3 is only a fixture throughput pilot, not the required fixed-anchor fitness pilot. Checkpoints 4–5 not started.

## User authorization to proceed; upstream revision pin update (2026-09-22)

- User instructed: “Okay lets go ahead with the EOH experiment.” This authorizes moving into the full experiment after readiness gates; it does not resolve the exact GPT-6 Luna billing endpoint or count the fixture pilot as a fixed-anchor fitness pilot.
- Official EoH main history page resolves current head at audit time to full commit `472545785c936dcfc863d2bc0d6109cf23c7ce62` (commit dated 2026-08-06). This replaces the earlier incomplete pin status in the manifest.
- Pinned source paths are not checked out locally because command-line network access remains unavailable. The browsable source page at this revision family shows `EVOL` and a `Methods` factory with parent selection and population-management strategies; the public current README example uses `e1/e2/m1/m2`. The exact pinned EOH proposal/mutation implementation, M3 availability, objective direction, retries, candidate execution and elite behavior still require the source tree. Therefore “revision pinned” is complete, but “dependency source installed and operator/API audit” is not.
- Full model calls have not begun. GPT-6 Luna is the user's named model; current official API rate page does not list its rate or a matching API model ID. GPT-5.6 Luna calculations remain proxy estimates only. No substitution was made.
- Source install: `git clone --filter=blob:none --no-checkout https://github.com/FeiLiu36/EoH.git experiments/EOH/vendor/EoH`, then detached checkout of the full SHA. `git rev-parse HEAD` matched `472545785c936dcfc863d2bc0d6109cf23c7ce62`. `python -m pip install -e .` from the pinned `eoh/` package succeeded, installing eoh 0.2 plus joblib 1.6.0, requests 2.34.2, cloudpickle and urllib3 dependencies. These commands required shell network permission; no source edits to the upstream checkout.
- Source audit: operators include i1/E1/E2/M1/M2/M3, with M3 explicitly handled in `_generate` and M3 simplification prompt. E1/E2 select two parents; M1/M2/M3 select one. `parent_selection` ranks sorted population and samples with inverse-rank weights `1/(rank+1+population_size)`. Fitness is rounded to five decimals, non-finite/None discarded; population manager deduplicates equal rounded objectives and retains the smallest `pop_size`. `BaseProblem.evaluate` compiles through unrestricted `exec`; `_eval_with_timeout` uses a spawned subprocess and kills it after timeout but has no memory or network sandbox. `InterfaceLLM` performs an unbudgeted connectivity request; its HTTP implementation may retry five times by default; EoH extraction makes up to four response calls and duplicate code may trigger two additional generation attempts.
- **Budget correction:** the earlier 65-candidate / 130-request proposal was inaccurate for this pinned upstream path. Corrected proposal is 10 initial candidates (2N) plus 25 offspring × 3 generations = 85 candidate slots; exact fixed operator schedule is 5 each of E1/E2/M1/M2/M3 per generation. With four extraction calls and two duplicate retries per sample, cap is 85×3×4 + 1 connection probe = 1,021 API requests (adapter will set one HTTP transport attempt each). At 840 games per candidate, maximum evolution matches = 71,400; K=3 selection uses 1,080 matches; maximum 72,480 total. Runtime projection: 8.6 min gameplay-only by 2-game extrapolation; nominal 10–60 min at assumed 5–30 sec/sample LLM latency, retries/timeouts may extend to hours.
- GPT-5.6 Luna public API rates used as proxy only: 4k input/1k output gives $2.04 at 1,021 calls; 8k/2k gives $4.08, +10% = $4.49. At no retries (86 calls including connection probe), proxy $0.17–$0.34. GPT-6 Luna exact API identifier/rate remains unverified.
- **Full evolution still not started.** This workspace has no `OPENAI_API_KEY` or other API credential environment variable configured. The host tools expose GPT-6 Luna as a Codex model choice, but that does not configure the EoH REST client. A local API endpoint/model key and user review of corrected cap ($4.49 GPT-5.6 pricing proxy, not an actual GPT-6 rate) are needed before API calls.
- Validator hardening: added a bounded arithmetic candidate language. It rejects imports, loops, comprehensions, nested/dynamic attribute access, non-numeric expressions, unknown names, excessive AST/source size, and mutation targets. Only immutable tuple/scalar StateView fields and a small numeric builtin set are available. The pinned evaluator's subprocess timeout remains the last-resort wall-clock limit. This reduces the source-reachable surface but is not a general OS security sandbox; it still lacks OS memory quotas/network policy.
- Latest tests: `python -m unittest tests.ayo_olopon.test_eoh_interface tests.ayo_olopon.test_eoh_upstream_contract tests.ayo_olopon.test_ayo_olopon tests.ayo_olopon.test_heuristic_corrections` — **passed, 11 tests**. Includes pinned EoH prompt/operator/config/parent/elite contracts. The `pilot.py` fixture run passed again after validator hardening; it remains a fixture-only throughput pilot, not generated-anchor fitness.
- Remaining gates before actual LLM work: no GPT-6 Luna API-compatible endpoint/model availability has been demonstrated and no local API credential is configured; GPT-6 Luna rate is absent from public pricing. Earlier GPT-5.6 Luna proxy was $0.58 max, but source audit changes true request cap and proxy budget to $4.49. Present corrected cap before requesting any paid call. The generation/match adapter and true anchor fitness pilot are still not implemented/run.

## Stateless chat-generation pilot (2026-09-22)

- User asked whether GPT-6 Luna could generate candidates through chat without an API key and without inheriting this conversation's memory. A fresh collaboration agent was launched with `fork_turns:none`, model `gpt-6-luna`, medium reasoning, and a self-contained Ayo interface/DSL prompt. It was instructed not to inspect repo files or prior conversation. It returned one idea and `evaluate_state` function. It has no access to this conversation's remembered history through the fork, but operates under the same system/developer instructions and shared workspace environment; this is context isolation, not an independent blinded experimental organization.
- Candidate code: captured-seed advantage weighted 2.5, row seed difference weighted 0.5, plus a small current-player/mobility signal weighted 0.08, scaled by 48 and clamped to [-10, 10]. Full source is retained in the handoff transcript and candidate log format below; no evolutionary fitness was assigned.
- **Validation:** `validate_source` returned `(True, 'passed')`; `validate` on two synthetic immutable views returned `ok=True`, scores `[0.01, 0.07833333333333332]`. A first command failed because the validator exports `validate`/`validate_source`, not the incorrectly guessed `validate_candidate`; corrected without code changes. This source validation is not an OS security sandbox.
- **Artifact:** raw thought/code and SHA-256 are logged in `experiments/EOH/chat_generation_pilot.jsonl`. The final selected regression command `python -m unittest tests.ayo_olopon.test_eoh_interface tests.ayo_olopon.test_eoh_upstream_contract tests.ayo_olopon.test_ayo_olopon tests.ayo_olopon.test_heuristic_corrections` passed all 11 tests. The test-generated tracked bytecode change was restored; pre-existing OpenSpiel modifications and the untracked PDF were left untouched.
- **Deviation:** one fresh chat agent call substitutes for a local API request for this tiny pilot only. It does not implement the EoH scheduler, parent selection, fitness evaluation, or full-run retry accounting; it is not counted as one of 85 accepted/evaluated evolution candidates and no further generations were started. Chat-plan usage and a GPT-6 Luna API price are not exposed to this repo, so cannot be costed in dollars from public API rates.
- **Scope boundary:** no held-out 30-position outcomes read; the agent prompt included only game rules and the permitted interface. The 70-position development data was not used to evaluate the candidate.

## Budget update — requested primary model (2026-09-22)

- User specified GPT-6 Luna as the research model. The public OpenAI API pricing/model pages currently retrieved list GPT-5.6 Luna, not GPT-6 Luna; no exact GPT-6 Luna API price was found. No model substitution is made in the manifest.
- For planning only, GPT-5.6 Luna's public standard short-context rates ($0.20/M uncached input, $1.20/M output) yield $0.26–$0.52 for 130 calls assuming 4k–8k input and 1k–2k output tokens per call; upper case plus 10% contingency is $0.58. 65-call no-retry equivalent is $0.13–$0.26. These token counts are assumptions, not measurements.
- **No API calls or paid evolution were run.** Confirm GPT-6 Luna's billable endpoint/rate and measure representative tokens before treating the proxy as a budget.

## Completion record — actual run (2026-09-22; supersedes pre-run status above)

The earlier “not started,” “do not launch,” and proposal-only statements are historical planning notes and are superseded by this execution record. The run used fresh stateless GPT-6 Luna chat agents after the user explicitly asked to start full evolution. No API key or REST API calls were used. Cost in USD is unavailable because chat usage and billing are not exposed; the former API proxy is not actual run cost.

### Checkpoint 0 — game-contract and upstream audit

- **Code/artifacts:** `experiments/EOH/vendor/EoH` pinned at `472545785c936dcfc863d2bc0d6109cf23c7ce62`; contract captured in `docs/eoh/README.md` and `experiment_manifest.yaml`.
- **Audit:** inspected OpenSpiel Ayo registration, implementation, frozen opening loader, and corrected baseline. Two-player deterministic sequential game; local actions 0–5; captured majority/empty-side/repetition/cycle termination; terminal returns +1/0/-1. The adapter uses actual legal actions and `state.child(action)`.
- **Upstream operators verified:** `i1`, `E1`, `E2`, `M1`, `M2`, `M3`; official `parent_selection` and `population_management` used. Objective is lower-is-better, rank-weighted parent selection, five-decimal fitness, unique-objective elite selection.
- **Checks/results:** `test_eoh_upstream_contract` and final four-module unittest command passed (details below). Pinned checkout revision confirmed. No game rules were changed.
- **Failures/deviations:** EoH's general-purpose Python `exec` worker is not used for candidate source; candidates are restricted by the local AST DSL. Official asynchronous weighted operator dispatch was replaced by the fixed five-per-operator schedule. This increases schedule reproducibility but is a methodology deviation.

### Checkpoint 1 — greedy policy interface

- **Code changed:** `Algorithms/eoh_ayo.py`; immutable state snapshot, actual successor evaluation, original actor perspective, exact terminal ranking ±11/0, finite bounded nonterminal values, deterministic earliest-legal-action tie.
- **Tests/results:** legality, nonmutation, terminal ranking, player-perspective handling, seat swap, and tie behavior passed through `test_eoh_interface`.
- **Failure/deviation:** none observed. Minimax was excluded from candidate scoring, fitness, and selection.

### Checkpoint 2 — candidate validator

- **Code changed:** `experiments/EOH/candidate_validator.py`; constrained AST/source limits, immutable tuple/scalar fields, numeric operations, finite [-10,10] score enforcement, bounded runtime check, fixture validation on actual successor snapshots and both `player_id` values. During the run, tuple aliases/conditional tuple expressions, boolean arithmetic, and `int(bool)` were enabled to support valid generated programs.
- **Tests/results:** accepted candidates validated successfully; generated invalid candidates were rejected and retried. All 85 finally scored candidates passed validation. Four-module final unit suite passed.
- **Failures:** early proposal batches contained rejected syntax/operators, including power (`**`) in one generation. Corrections were requested before admitting them. `rejected_generation_1.jsonl` archives three early failures, but not every failed/corrected raw response is preserved; accepted candidate provenance is complete, rejected raw provenance is partial.
- **Deviation/limitation:** validation is an in-process restricted DSL, not an adversarially secure OS sandbox. This experimental adapter only executes accepted restricted sources; no resource-isolation claim is made.

### Checkpoint 3 — reproducibility and throughput pilot

- **Artifacts/code:** `experiments/EOH/pilot.py`, `pilot_results.json`, `fitness.py`, frozen `anchor_panel.json`, and deterministic generation plans.
- **Pilot:** opening checksum verified; 100 opening records split 70 development/30 reused selection; fixture exercised six successors; paired-seat fixture/random games completed. Reported benchmark rate varies by host and is documented in `pilot_results.json`.
- **Tests/results:** pilot passed; fixed-anchor gameplay pilot and full fitness protocol were then run. All candidate rows use fixed panel/opponent/seat/opening ordering and deterministic seeds.
- **Deviation:** chat proposal generation cannot use EoH's REST `InterfaceLLM`; five outputs were batched per operator chat interaction. Approximate 23 counted proposal-batch interactions, 103 raw proposals, 85 admitted/scored candidates. This is an interaction count, not billable request/token count.

### Checkpoint 4 — full evolution

- **Frozen exact configuration:** population 5; three generations; 10 initial candidates; 25 offspring per generation, schedule E1/E2/M1/M2/M3 = five each; five generated frozen anchors plus seeded random at weights 0.18 each/0.10; 70 development positions × two seats × six opponents = 840 games per candidate; 85 candidates = 71,400 games. Anchor panel frozen before any fitness evaluation. Full parameters in `experiment_manifest.yaml`.
- **Results:** 85/85 candidate slots validated and scored; `matches.jsonl` contains 71,400 records, zero match errors and zero action-limit truncations. Generation best fitness: 0.404929, 0.401714, 0.397857, 0.397500 for generations 0–3.
- **Failure/retry record:** invalid generation responses were retried/corrected; full failed raw-response archive is incomplete, disclosed in `results.md`. No generation match failures.
- **Deviation:** fixed operator schedule and chat interaction batching differ from upstream's live asynchronous REST sampling. Fitness is adapted to fixed Ayo gameplay WDL as specified.

### Checkpoint 5 — frozen shortlist selection

- **Protocol freeze:** `selection_protocol_frozen.json` and top-three `selection_shortlist.json` were written before selection outcomes. Pinned population management determined top three unique rounded-fitness candidates. Selection used the reused 30-position subset, both seats and six opponents (360 games per finalist, 1,080 total). Tie-break fixed before outcomes: fewer AST nodes, then lower median fixture evaluation time, then lexical source hash.
- **Results:** all finalists tied at weighted score 0.5473333333. Tie-break selected `g2-m2-05`, SHA `1be272704c60811ba53ffe28e47c90ea21d50e04e8e3371ab0e01fd291bd2880`; its selection W/D/L was 147/110/103. H* frozen in `experiments/EOH/H_star.json`, executable source copied to `Algorithms/H_star_ayo.py`.
- **Held-out data:** independent test set was not opened. Reused selection outcomes are not represented as an independent test.
- **Worked example:** opening 0, actor 1, legal `[2,3,5]`, H* scores `[4.78,6.06,5.42]`, chooses action 3; full snapshots in `H_star_worked_example.json`.

### Final verification and run cost

- **Command:** `\.venv\Scripts\python.exe -m unittest tests.ayo_olopon.test_eoh_interface tests.ayo_olopon.test_eoh_upstream_contract tests.ayo_olopon.test_ayo_olopon tests.ayo_olopon.test_heuristic_corrections` — **passed**, 11 tests in 0.047 seconds.
- **Command:** `\.venv\Scripts\python.exe experiments\EOH\pilot.py` — **passed**; verified all 100 opening entries/checksum/split, candidate fixture, and paired-seat pilot; 600 evaluator calls measured 0.001444 sec (415,484 calls/sec on final verification; machine/timing dependent).
- Verification touched a tracked Python bytecode cache; it was restored immediately. Pre-existing OpenSpiel submodule modifications and the untracked research PDF were left untouched.
- **Runtime:** captured wall window about 47m47s from first accepted catalog file through H* artifact timestamp; first generation prompt may predate the file window. Sum of per-match timers about 19m34s.
- **Cost:** not available in dollars. GPT-6 Luna ran in chat agents without API billing/token telemetry; do not interpret any GPT-5.6 Luna API price estimate as experiment cost.
