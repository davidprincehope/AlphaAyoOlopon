# H* versus handcrafted on 400 fresh openings

This study compares frozen EoH **H*** (`g2-m2-05`) with the handcrafted
**H_CTM** evaluator (capture/tactical/mobility weights 4/2/0.5).
The [opening set](opening_set_400.json) contains 400 distinct reachable,
nonterminal positions, generated with seed `20260924` after 4–32 random
legal actions, after both evaluators were frozen.

Each opening is played in both seats at depths 1, 2 and 3: 800 games per
depth, **2,400 games** overall. Depth counts complete player actions. The
policy maximizes its evaluator and backs up the opponent's reply by minimum;
ties retain legal-action order. No alpha-beta pruning is used.

| Depth | H_CTM W–D–L | H_CTM score | H* W–D–L | H* score |
| --- | ---: | ---: | ---: | ---: |
| 1 | 191–251–358 | 39.56% | 358–251–191 | 60.44% |
| 2 | 330–150–320 | 50.63% | 320–150–330 | 49.38% |
| 3 | 310–102–388 | 45.13% | 388–102–310 | 54.88% |

All games ended naturally: 1,897 terminal wins and 503 draws, with no
truncations. Score credits a draw as half a win. The
[result summary](results_400.json) retains the full protocol and aggregate
counts. Raw game traces remain local.

```sh
python -m experiments.handcrafted_vs_eoh_400.run_opening_comparison --run-only --output-dir runs/handcrafted-vs-eoh-rerun
```

Snapshots restore the board, captured scores and player to move, while
resetting repetition history to the restored position. This limitation
applies to both policies. Equal outer depth does not mean equal compute:
the handcrafted tactical feature examines child states internally.
The 300-action cap would be adjudicated by captured-seed lead; it was not
reached in this run.
