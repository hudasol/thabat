# Results

Author: Huda Mueen. This file is rendered by `scripts/render_results.py` from `results/summary.json` and the tables written by `scripts/analyze.py`. Every number below comes from those files. The predictions are in PREDICTIONS.md, written before any experiment, and amended only in the dated amendment log there.

## Scope of what was run

* Tasks: Pendulum (TD3), CartPole (DQN), InvertedPendulum (TD3, MuJoCo), and the grokking control (small transformer).
* Seeds per environment: 30 reference seeds for Pendulum, 30 for CartPole, 30 for InvertedPendulum and 42 for the grokking control. The paired comparisons use the seeds present in both environments: float64 and kernel path arms 30 (Pendulum), 30 (CartPole) and 30 (InvertedPendulum). The 4 thread arm was run on 30 seeds (Pendulum), 10 (CartPole) and 5 (InvertedPendulum), where it was stopped early to save compute after its runs proved bit identical to the reference. Amendment 1 aimed at 60 seeds per RL task, and compute allowed 30. The minimum of 30 seeds per environment is met except for those capped thread arms.
* The GPU arm was not run. It needs notebooks/gpu_arm.ipynb on a free GPU, and its result files are not in the repository. Its predictions are NOT RUN.
* The determinism control used 10 seeds per task.

Pinned environments of the arms that ran. The BLAS column is the library numpy was built with. The matrix products of the torch CPU build use its own MKL, which the kernel path setting changes. The machine has 2 cores, so the 4 thread arm was oversubscribed.

| Arm | CPU | Torch | Threads | BLAS | MKL_CBWR | Device | Dtype |
|---|---|---|---|---|---|---|---|
| fp64 | Intel(R) Xeon(R) Processor @ 2.80GHz | 2.15.0+cpu | 1 | scipy-openblas | None | cpu | float64 |
| ref | Intel(R) Xeon(R) Processor @ 2.80GHz | 2.15.0+cpu | 1 | scipy-openblas | None | cpu | float32 |
| thr | Intel(R) Xeon(R) Processor @ 2.80GHz | 2.15.0+cpu | 4 | scipy-openblas | None | cpu | float32 |
| isa | Intel(R) Xeon(R) Processor @ 2.80GHz | 2.15.0+cpu | 1 | scipy-openblas | COMPATIBLE | cpu | float32 |
| ref_repeat | Intel(R) Xeon(R) Processor @ 2.80GHz | 2.15.0+cpu | 1 | scipy-openblas | None | cpu | float32 |

## Instrument checks

* Determinism: repeating the reference environment gave identical final parameters for every repeated seed, with rates 1.00 (Pendulum), 1.00 (CartPole), 1.00 (InvertedPendulum) and 1.00 (grokking control). Flip counts are therefore not caused by run to run noise in the reference.
* Perturbation: in the float64 and kernel path arms the final parameters differ from the reference in every seed (rates 1.00 and 1.00 for Pendulum, and equally 1.00 in the other tasks, see the table). In the 4 thread arm of the three RL tasks the final parameters were bit identical to the reference in every seed (rates 0.00, 0.00 and 0.00). Thread count is therefore no perturbation for these small MLPs, and the thread predictions are NOT TESTABLE, as Amendment 1 anticipated. For the small transformer, 4 threads changed the parameters in rate 1.00 of the seeds.

## Paired comparison by seed

Lost is a seed that succeeded in the reference and failed in the arm, gained is the reverse. Intervals are Wilson 95 percent. Holm adjusted p values cover all RL comparisons. Bits differ is the share of seeds with different final parameters. The grokking control uses the unadjusted p value.

| Task | Arm | n | Reference rate | Arm rate | Lost | Gained | Flip rate (95% CI) | McNemar p | Holm p | Bits differ |
|---|---|---|---|---|---|---|---|---|---|---|
| cartpole | float64 | 30 | 0.233 | 0.267 | 3 | 4 | 0.233 [0.118, 0.409] | 1.000 | 1.000 | 1.000 |
| cartpole | kernel path | 30 | 0.233 | 0.167 | 4 | 2 | 0.200 [0.095, 0.373] | 0.688 | 1.000 | 1.000 |
| cartpole | 4 threads | 10 | 0.100 | 0.100 | 0 | 0 | 0.000 [0.000, 0.278] | 1.000 | 1.000 | 0.000 |
| grok | 4 threads | 42 | 0.310 | 0.214 | 6 | 2 | 0.190 [0.100, 0.333] | 0.289 | n/a | 1.000 |
| invpend | float64 | 30 | 0.367 | 0.533 | 3 | 8 | 0.367 [0.219, 0.545] | 0.227 | 1.000 | 1.000 |
| invpend | kernel path | 30 | 0.367 | 0.567 | 2 | 8 | 0.333 [0.192, 0.512] | 0.109 | 0.984 | 1.000 |
| invpend | 4 threads | 5 | 0.400 | 0.400 | 0 | 0 | 0.000 [0.000, 0.434] | 1.000 | 1.000 | 0.000 |
| pendulum | float64 | 30 | 0.467 | 0.433 | 6 | 5 | 0.367 [0.219, 0.545] | 1.000 | 1.000 | 1.000 |
| pendulum | kernel path | 30 | 0.467 | 0.433 | 6 | 5 | 0.367 [0.219, 0.545] | 1.000 | 1.000 | 1.000 |
| pendulum | 4 threads | 30 | 0.467 | 0.467 | 0 | 0 | 0.000 [0.000, 0.114] | 1.000 | 1.000 | 0.000 |

![Share of seeds that flip](../figures/flip_rates.png)

Figure 1. Flip rate per task and arm with Wilson 95 percent intervals. The dotted line is the 0.10 level used in the H1 predictions.

![Success rates](../figures/success_rates.png)

Figure 2. Success rate in the reference and in each arm, with Wilson 95 percent intervals.

![Paired final scores](../figures/paired_scatter.png)

Figure 3. Final score of the reference run against the final score of the arm for the same seed. Dotted lines mark the success threshold. Rose points are seeds whose outcome flipped. Points far from the diagonal are runs that ended in a different place, not runs that missed the threshold by a small amount.

![Grokking curves of flipped seeds](../figures/grok_curves.png)

Figure 4. Held out accuracy during training for seeds of the grokking control whose outcome flipped between 1 and 4 threads.

## Where flips occur (H3)

The AUC compares the distance to the threshold of flipped and unflipped seed and arm pairs, pooled over arms with a valid perturbation check. A value of 0.5 means no relation and 1.0 means flipped seeds are always the closest to the threshold.

| Task | Pooled pairs | Flipped | AUC of distance (95% CI) | One sided p |
|---|---|---|---|---|
| pendulum | 60 | 22 | 0.620 [0.459, 0.761] | 0.063 |
| cartpole | 60 | 13 | 0.659 [0.507, 0.805] | 0.041 |
| invpend | 60 | 21 | 0.447 [0.307, 0.594] | 0.755 |
| all | 180 | 56 | 0.587 [0.496, 0.671] | 0.031 |

## All predictions scored

HIT: 18. MISS: 6. NOT TESTABLE: 6. NOT RUN: 6. Total: 36. The scoring is done by `thabat_kit.scorer` on `results/summary.json`, not by hand. The totals include instrument and software checks (C0, B1, B2) next to the hypotheses.

| Prediction | Check | Observed | Status |
|---|---|---|---|
| H1.pendulum.thr | flip_rate.pendulum.thr >= 0.1 | flip_rate = 0 | NOT TESTABLE |
| H1.pendulum.fp64 | flip_rate.pendulum.fp64 >= 0.1 | flip_rate = 0.367 | HIT |
| H1.pendulum.gpu | flip_rate.pendulum.gpu >= 0.1 | flip_rate = n/a | NOT RUN |
| H1.cartpole.thr | flip_rate.cartpole.thr >= 0.1 | flip_rate = 0 | NOT TESTABLE |
| H1.cartpole.fp64 | flip_rate.cartpole.fp64 >= 0.1 | flip_rate = 0.233 | HIT |
| H1.cartpole.gpu | flip_rate.cartpole.gpu >= 0.1 | flip_rate = n/a | NOT RUN |
| H1.invpend.thr | flip_rate.invpend.thr >= 0.1 | flip_rate = 0 | NOT TESTABLE |
| H1.invpend.fp64 | flip_rate.invpend.fp64 >= 0.1 | flip_rate = 0.367 | HIT |
| H1.invpend.gpu | flip_rate.invpend.gpu >= 0.1 | flip_rate = n/a | NOT RUN |
| H2.pendulum.thr | mcnemar_holm_p.pendulum.thr >= 0.05; abs_delta.pendulum.thr <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0 | NOT TESTABLE |
| H2.pendulum.fp64 | mcnemar_holm_p.pendulum.fp64 >= 0.05; abs_delta.pendulum.fp64 <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0.0333 | HIT |
| H2.pendulum.gpu | mcnemar_holm_p.pendulum.gpu >= 0.05; abs_delta.pendulum.gpu <= 0.1 | mcnemar_holm_p = n/a; abs_delta = n/a | NOT RUN |
| H2.cartpole.thr | mcnemar_holm_p.cartpole.thr >= 0.05; abs_delta.cartpole.thr <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0 | NOT TESTABLE |
| H2.cartpole.fp64 | mcnemar_holm_p.cartpole.fp64 >= 0.05; abs_delta.cartpole.fp64 <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0.0333 | HIT |
| H2.cartpole.gpu | mcnemar_holm_p.cartpole.gpu >= 0.05; abs_delta.cartpole.gpu <= 0.1 | mcnemar_holm_p = n/a; abs_delta = n/a | NOT RUN |
| H2.invpend.thr | mcnemar_holm_p.invpend.thr >= 0.05; abs_delta.invpend.thr <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0 | NOT TESTABLE |
| H2.invpend.fp64 | mcnemar_holm_p.invpend.fp64 >= 0.05; abs_delta.invpend.fp64 <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0.167 | MISS |
| H2.invpend.gpu | mcnemar_holm_p.invpend.gpu >= 0.05; abs_delta.invpend.gpu <= 0.1 | mcnemar_holm_p = n/a; abs_delta = n/a | NOT RUN |
| H3.pendulum | auc_dist.pendulum >= 0.7 | auc_dist = 0.62 | MISS |
| H3.cartpole | auc_dist.cartpole >= 0.7 | auc_dist = 0.659 | MISS |
| H3.invpend | auc_dist.invpend >= 0.7 | auc_dist = 0.447 | MISS |
| H3.all | auc_dist.all >= 0.7 | auc_dist = 0.587 | MISS |
| PC1.grok.thr | flip_rate.grok.thr >= 0.08 | flip_rate = 0.19 | HIT |
| PC2.grok.thr | mcnemar_p.grok.thr >= 0.05; abs_delta.grok.thr <= 0.1 | mcnemar_p = 0.289; abs_delta = 0.0952 | HIT |
| C0.grok | determinism_rate.grok >= 1 | determinism_rate = 1 | HIT |
| C0.pendulum | determinism_rate.pendulum >= 1 | determinism_rate = 1 | HIT |
| C0.cartpole | determinism_rate.cartpole >= 1 | determinism_rate = 1 | HIT |
| C0.invpend | determinism_rate.invpend >= 1 | determinism_rate = 1 | HIT |
| B1.kit | tests_failed.kit <= 0 | tests_failed = 0 | HIT |
| B2.kit | coverage_percent.kit >= 85 | coverage_percent = 95.8 | HIT |
| H1.pendulum.isa | flip_rate.pendulum.isa >= 0.1 | flip_rate = 0.367 | HIT |
| H2.pendulum.isa | mcnemar_holm_p.pendulum.isa >= 0.05; abs_delta.pendulum.isa <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0.0333 | HIT |
| H1.cartpole.isa | flip_rate.cartpole.isa >= 0.1 | flip_rate = 0.2 | HIT |
| H2.cartpole.isa | mcnemar_holm_p.cartpole.isa >= 0.05; abs_delta.cartpole.isa <= 0.1 | mcnemar_holm_p = 1; abs_delta = 0.0667 | HIT |
| H1.invpend.isa | flip_rate.invpend.isa >= 0.1 | flip_rate = 0.333 | HIT |
| H2.invpend.isa | mcnemar_holm_p.invpend.isa >= 0.05; abs_delta.invpend.isa <= 0.1 | mcnemar_holm_p = 0.984; abs_delta = 0.2 | MISS |

## Reading the results

H1 (flips occur). The float64 and kernel path arms flipped between 0.20 and 0.37 of the seeds for Pendulum and CartPole and about a third for InvertedPendulum (see table), all above the 0.10 level, and all HIT. This is a result about magnitude. That RL runs diverge under tiny numerical differences was already known, and we do not claim it as new.

H2 (average rate unchanged). It holds for Pendulum and CartPole in both arms, with absolute differences of rate between 0.03 and 0.07. It is a MISS for InvertedPendulum in both arms: the absolute difference of rates was 0.17 (float64) and 0.20 (kernel path), above the 0.10 limit, although the Holm adjusted McNemar p values (1.00 and 0.98) did not reach 0.05. In both arms the success rate moved up (from 0.37 to 0.53 and 0.57), with more seeds gained than lost. With 30 seeds this is compatible with a true change and with none. We keep it as a miss, because the prediction was written with the 0.10 limit, and we do not explain it away. A sound reading is that an unchanged average cannot be assumed, and that a larger sample is needed for this task.

H3 (flips near the threshold). All four H3 predictions are MISS. For Pendulum and CartPole and the pooled set the AUC lies between 0.55 and 0.70, which the predictions define as inconclusive and score as a miss. For InvertedPendulum it is below 0.55, so there is no support. Figure 3 shows why: in CartPole and InvertedPendulum many flipped seeds jump between a failed policy and a solved one, far from the threshold, and the scores of these tasks are close to bimodal. This agrees better with the competing account named in PREDICTIONS.md, that a rounding difference early in training can send a run to a different basin whatever its reference score was, than with the account that flips are near misses. Pendulum scores bunch just below and above the threshold, so some flips there are near misses.

Positive control. The grokking control with 4 threads against 1 thread flipped 0.19 of 42 seeds, which is a HIT of PC1 (at least 0.08), and the aggregate rate moved from 0.31 to 0.21 with an unadjusted McNemar p of 0.29, a HIT of PC2. The absolute difference (0.095) is close to the 0.10 limit, so this hit is narrow. The control reproduces the pattern reported by Ootani in kind, on a different task, with a smaller model setup and a different threshold, and with a flip rate in the same range.

Coincidence to note. In Pendulum the float64 arm and the kernel path arm have the same counts of lost and gained seeds. The two arms differ in the final parameters of every seed and in the outcomes of several seeds, so they are different runs, and the equal counts are a coincidence.

What these results do not show is in LIMITATIONS.md. An exploratory table of flip rates at moved thresholds (not part of the predictions) is in `tables/threshold_sweep_exploratory.csv`.
