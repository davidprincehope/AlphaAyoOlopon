# MLP600 training losses

Three separate figures show mean minibatch losses in each of the 600 learner rounds. Values are recorded optimizer statistics from evolving replay data; they measure fitting, not held-out generalization or playing strength. No smoothing is applied.

| Figure | Experimental measure |
| --- | --- |
| [Total training loss](total_training_loss.png) | Policy + value + weight-only L2 regularization |
| [Value loss](value_loss.png) | Mean squared error against final game outcomes |
| [Policy loss](policy_loss.png) | Cross-entropy against self-play MCTS targets |

The standalone plots and the [merged three-panel comparison](training_losses_combined.png) are exported at 600 dpi. Vertical axes identify total training loss, value loss and policy loss. Value loss uses a 0.10 to 0.70 range and policy loss uses 0.65 to 1.10, with finer ticks. These linear axes start above zero to resolve variation; all 600 observations remain visible. The merged image keeps the same separate scales; compare trends using the tick values. [Recorded values and audit](loss_data.json) and [provenance](figure_provenance.json) retain the numerical evidence and source hashes.

```powershell
.venv/Scripts/python.exe -B -m visualizations.plot_training_losses
```
