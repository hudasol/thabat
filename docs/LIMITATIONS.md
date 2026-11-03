# Limitations

Author: Huda Mueen. This file has no result numbers on purpose. Numbers are in RESULTS.md, which is rendered from the result files.

## What the study does not show

* It does not show anything about large models or long training runs. The RL policies are two layer networks with 64 units, trained for thousands of steps. Ootani's open question about scale (arXiv 2607.05104) is not answered here. Our setting is a different small regime.
* It covers three tasks and two algorithms (TD3 and DQN). Other algorithms (PPO, SAC), larger networks, other tasks and other numerical settings may behave differently.
* It does not identify a mechanism. We observe that outcomes flip and where, not why. The account that a rounding difference grows through the trajectory and sends the policy into another basin is a hypothesis that fits the data, not a result.
* It does not show that float64 or a different kernel path is worse or better. The average success rate is compared only for a detectable change.

## Choices that favour seeing flips

* The training budget of each task was chosen on pilot seeds so that the reference success rate is intermediate. This puts every task where a small change is most likely to matter. A task trained until success is almost certain would show fewer flips by construction. The flip rates are therefore a statement about sensitivity in an intermediate regime and not about training in general.
* The binary outcome depends on a threshold set before the experiments. A different threshold changes which seeds count as flipped. The table `tables/threshold_sweep_exploratory.csv` moves the threshold after the fact, as an exploratory check only, and was not part of the predictions.

## Statistical limits

* With 30 seeds per RL task the intervals are wide. A hit on H2 means that no change was detected, not that the rates are equal. The paired difference interval in `tables/summary.csv` shows what the data can and cannot exclude.
* Many comparisons were made. Holm adjustment is applied to the McNemar tests of H2 across all RL comparisons, but not to H1 and H3, which are scored against fixed thresholds.
* H3 pools seed and arm pairs. Pairs from the same seed in different arms are not independent, so the one sided p values for H3 are optimistic and the bootstrap intervals, which resample pairs, are too narrow.
* The number of seeds actually run is smaller than the target of Amendment 1 for some environments, because of limited free compute. The numbers used are stated in RESULTS.md.

## Outcome definition

* The outcome is the score at the last training step. Learning curves of both TD3 and the grokking control oscillate (Figure 4 in RESULTS.md), so a part of the flips is the evaluation landing on a different phase of an oscillation, and not a lasting difference in what was learned. We did not measure how many flips persist if the evaluation were taken a few hundred steps earlier or later.
* The grokking control ran on fewer seeds than planned (compute), and the 4 thread arm of the RL tasks was stopped early once it proved bit identical.

## Instrument limits

* Thread count did not change the final parameters of the small MLP policies at all in our setting. For those tasks the thread arm is not a perturbation, and its predictions are NOT TESTABLE. This may differ for other BLAS libraries, other CPUs, larger layers or convolutional models.
* The kernel path arm uses an Intel MKL setting (MKL_CBWR set to COMPATIBLE). It is a documented, benign change, but it is specific to this library and this CPU family.
* All CPU arms ran on one machine with one library stack. The captured environments are in RESULTS.md. Results on other CPUs may differ.
* The CPU versus GPU arm needs the notebook run on a free GPU. If its result files are absent from `results/raw/`, the GPU predictions are NOT RUN, and nothing in this repository stands in for them. The CPU versus GPU question is open until that run is done.
* Reference environment determinism was checked on 10 seeds per task, not all seeds. A bitwise identical repeat on those seeds is evidence, not proof, that the rest are repeatable.

## Design changes after the pilot

The pilot showed that three pre registered settings (CartPole observation noise, the grokking learning rate, the InvertedPendulum batch size) could not reach the target success band within the grid. They were changed in dated amendments, on pilot seeds only and before any evaluation run. The consequence is that CartPole is the standard task with a stricter threshold and not a noisy variant, and the grokking control is a conceptual replication that differs from Ootani in task, size, learning rate and threshold. Some chosen budgets have pilot rates at the edge of the target band, and the pilot used only 10 seeds, so the reference rates in the evaluation can fall outside the band that the predictions require (0.2 to 0.8). The scorer then reports NOT TESTABLE and does not count it as a miss.

## Prior art review

The review used abstract pages and summaries produced by a web fetch tool. Quoted figures from other papers should be checked against the PDFs before submission. The search for earlier work on the same question found none, which is weak evidence of absence.
