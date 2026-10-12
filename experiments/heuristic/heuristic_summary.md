# Handcrafted heuristic summary

The selected **H_CTM** greedy evaluator uses:

\[
E(s,p)=\begin{cases}
10U(s,p), & s\text{ terminal},\\
4C(s,p)/48+2T(s,p)/48+0.5M(s,p)/6, & \text{otherwise}.
\end{cases}
\]

`U` is the win/draw/loss return for player `p`. Nonterminal values are bounded
by Â±6.5, so terminal Â±10 dominates them.

| Feature | Meaning |
| --- | --- |
| C | Captured-seed lead from `p`'s perspective |
| T | Best immediate captured-score gain for the player to move, signed from `p`'s perspective |
| M | Legal-action count, positive when `p` is to move |

T uses actual cloned children, including relay sowing and collection at
termination. Row-seed control was tested and excluded. Greedy play evaluates
all legal successors from the actor's perspective; ties choose the lowest
action ID.

## Selection and result

The study froze 100 openings at 4â€“32 plies with seed `20260921`, split into 70
development and 30 verification positions. Every opening was played in both
seats with a 300-action limit. H_CTM led the four tested feature combinations;
its `[4,2,0.5]` weights tied `[8,3,1]` on development score and won the secondary
final-seed-differential criterion.

On the unused verification positions, the selected policy scored **68.33%**
against random: **33 wins, 16 natural draws, 11 losses in 60 games**, with zero
truncations. Average final seed differential was **+4.00**, game length
**31.4 actions**, and decision time **63.35 ms** on the development machine.
Score rate credits draws as half a win. Action-limit results use captured-seed
lead (win, tie, or loss) and are recorded separately from natural results.

This is a shallow evaluator tested on a modest opening sample. It does not
establish optimal play or model repetition risk explicitly. Tactical capture
lookahead makes equal outer search depths unequal in compute.

## Reproduce and inspect

From an activated environment at the repository root:

```powershell
$env:PYTHONPATH = "$PWD\open_spiel;$PWD"
python experiments/heuristic/run_experiment.py --use-frozen-openings
```

The [methodology](HEURISTIC_METHODOLOGY.md) contains the feature definitions,
selection tables, corrected specification and frozen-opening provenance.
Authoritative compact results and manifests are in `raw_results_corrected/`.
