# Growth of the discovered position space

Figure 2 plots newly discovered position keys by discovery depth on a log scale and their cumulative count on a linear scale in a second panel. The completed enumeration has 282,850,985 keys over depths 0–103. The discovery count peaks at depth 20 with 15,003,812 new keys.

Position key: 12 pit counts + 2 captured scores + player to move / terminal marker. Repetition history and elapsed move count are excluded. Discovery depth is the first BFS level; a complete player action includes all relay laps. These counts describe the enumerator's position keys, not every history-dependent state or a solution of the game. No smoothing or uncertainty estimates are applied.

Source: [enumeration results](../../../experiments/ayo_state_space_results.json). The source is included in Git, so the figure can be regenerated from a fresh clone.

[PNG](state_space_growth.png) is 300 dpi; [SVG](state_space_growth.svg) retains editable text. [chart_data.csv](chart_data.csv) contains all 104 depth counts and their cumulative sums. [figure_provenance.json](figure_provenance.json) records source, script, and export hashes.

```powershell
.venv/Scripts/python.exe -B -m visualizations.state_space_growth
```
