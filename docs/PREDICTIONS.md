# Predictions

Author: Huda Mueen. Written and committed on 3 November 2026, before any experiment was run. Nothing below may be changed after results exist, except by a dated entry in the amendment log at the end of this file that keeps the original text.

## Research question

Does the numerical knife edge reported for grokking in a small transformer (Ootani, arXiv 2607.05104) appear in small robot policy training? Concretely: for the same seed, do benign numerical changes (thread count, CPU versus GPU, float32 versus float64) flip whether a policy reaches a fixed success threshold, without moving the average success rate?

RL training is already known to be seed sensitive (Henderson et al. 2018), and GPU nondeterminism alone is known to make RL runs diverge (Nagarajan et al. 2018). The new part is the paired by seed test of a binary outcome under named numerical environments. See PRIOR_ART.md.

## Hypotheses

* H1. Same seed outcome flips occur in small RL training under benign numerical changes. Scored on magnitude: a flip rate of at least 0.10 for a task and arm.
* H2. The average success rate does not change across numerical environments. Scored as: Holm adjusted exact McNemar p at least 0.05 and absolute difference of rates at most 0.10.
* H3. Flips concentrate in runs that end close to the success threshold. Scored as: AUC of at least 0.70 for the distance to threshold of flipped versus unflipped seeds.

Each prediction block below states the expected effect size and what would falsify it.

## Definitions

* Numerical environment: the triple (device, thread count, dtype). The reference environment is CPU, 1 thread, float32.
* Arms: `thr` is CPU with 4 threads (float32). `fp64` is CPU with 1 thread and float64. `gpu` is one CUDA device, float32, TF32 disabled, deterministic algorithms not forced. The reference environment is compared with each arm.
* Same seed: one integer seed fixes the initial weights (drawn in float32 on the CPU, then cast), the replay sampling, the exploration noise, the environment reset seeds and the evaluation seeds. All of these are drawn from generators whose output does not depend on network arithmetic. Only the arithmetic of the network differs between arms. Trajectories can still diverge, because actions depend on the network.
* Outcome: success is 1 when the final evaluation score reaches the task threshold, else 0. The evaluation uses a deterministic policy and fixed evaluation seeds that are the same in all arms.
* Flip: a seed whose outcome in the arm differs from its outcome in the reference environment.
* Metrics (names are used by the scorer, with the suffix `.task.arm`):
  * `flip_rate`: share of seeds that flip.
  * `ref_rate`: success rate in the reference environment.
  * `abs_delta`: absolute difference between the arm rate and the reference rate.
  * `mcnemar_p`: exact two sided McNemar test on the discordant pairs. `mcnemar_holm_p`: the same, Holm adjusted over all RL task and arm comparisons that were run.
  * `perturbation_rate`: share of seeds whose final parameter vectors (cast to float64) are not bitwise equal between reference and arm. This is the instrument check. Below 0.5 the arm is reported as perturbation mostly absent and its H1 and H2 blocks are NOT TESTABLE.
  * `determinism_rate`: share of 10 repeated reference runs with an identical final parameter hash.
  * `auc_dist`: area under the ROC curve of the score `-distance`, separating flipped from unflipped seeds, where distance is the absolute gap between the reference run's final evaluation score and the threshold. For `auc_dist.all` the gap is divided by the standard deviation of the reference scores of its task. `n_flips`: number of flipped seeds pooled over arms.

## Tasks, algorithms and thresholds

| Task key | Environment | Algorithm | Success when |
|----------|-------------|-----------|--------------|
| pendulum | Pendulum-v1 | TD3 | mean return over 20 evaluation episodes at least -250 |
| cartpole | CartPole-v1 with seeded Gaussian observation noise (standard deviation 0.1), cap 500 steps | DQN | mean return over 20 evaluation episodes at least 400 |
| invpend | InvertedPendulum-v5 (MuJoCo), cap 1000 steps | TD3 | mean return over 10 evaluation episodes at least 800 |
| grok | modular addition, small transformer of roughly 12,000 parameters | AdamW, full batch | held out accuracy at the final step at least 0.90 |

Thresholds are fixed here. Training budgets (steps, and for grok the training fraction) are chosen on pilot seeds only, see below.

## Seeds and the pilot rule

* Evaluation seeds: 0 to 29 for each RL task (at least 30 per environment), 0 to 59 for grok.
* Determinism control: seeds 0 to 9, reference environment repeated.
* Pilot seeds: 1000 to 1009. The pilot runs only in the reference environment. Training budget is chosen from a small preset grid (written in configs/pilot_grid.json before the pilot) as the smallest budget whose pilot success rate lies between 0.3 and 0.7. If none does, the budget closest to 0.5 is used and the choice is flagged. Pilot seeds are never used for evaluation, and evaluation seeds are never used for tuning. The pilot outcome is recorded in docs/PILOT.md and the chosen budgets in configs/.
* Hyperparameters other than the budget are fixed in configs/ before the pilot and are not tuned.

## Weaknesses of the design, stated in advance

* Choosing a budget that gives an intermediate success rate puts each task where flips are easy to see. A task with a saturated success rate would show fewer flips by construction. Results describe sensitivity at an intermediate regime, not in general.
* With 30 seeds, interval estimates are wide. H2 can only show no detectable change.
* Small matrix products may give identical bits for different thread counts. Then the thread arm tests nothing, and we report that.
* The GPU arm needs a notebook run on a free Kaggle or Colab GPU. Until its result files are added, its predictions are NOT RUN. Nothing is simulated in their place.
* The grokking positive control is a conceptual replication. Its task, size and threshold differ from Ootani.
* A flip rate near zero in float64 would also be informative: it would mean the knife edge is a property of the transformer setting and not of RL.

## Prediction blocks

The scorer in thabat_kit reads the blocks below. A block is scored HIT when every `check` line holds, MISS when a `requires` line holds but a `check` line fails, NOT TESTABLE when a `requires` line fails, and NOT RUN when a needed metric is absent from the results file.

### H1.pendulum.thr

- hypothesis: H1
- statement: Pendulum, thread count arm (CPU, 4 threads versus 1 thread, float32): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.10, plausible range 0.03 to 0.30. This is the arm most likely to show no perturbation at all, because small matrix products can be reduced in the same order for 1 and 4 threads. If fewer than half of the paired seeds differ bitwise, the prediction is NOT TESTABLE, not a miss.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.pendulum.thr >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: flip_rate.pendulum.thr >= 0.10

### H1.pendulum.fp64

- hypothesis: H1
- statement: Pendulum, float64 arm (CPU, 1 thread, float64 versus float32): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.20, plausible range 0.10 to 0.40. This is a larger perturbation than a change of reduction order, since every operation is rounded differently.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.pendulum.fp64 >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: flip_rate.pendulum.fp64 >= 0.10

### H1.pendulum.gpu

- hypothesis: H1
- statement: Pendulum, CPU versus GPU arm (one CUDA GPU versus CPU, 1 thread, float32, TF32 disabled): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.20, plausible range 0.10 to 0.40. This arm needs the free Kaggle or Colab notebook. It stays NOT RUN until result files from that notebook are added to the repository.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.pendulum.gpu >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: flip_rate.pendulum.gpu >= 0.10

### H1.cartpole.thr

- hypothesis: H1
- statement: CartPole variant, thread count arm (CPU, 4 threads versus 1 thread, float32): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.10, plausible range 0.03 to 0.30. This is the arm most likely to show no perturbation at all, because small matrix products can be reduced in the same order for 1 and 4 threads. If fewer than half of the paired seeds differ bitwise, the prediction is NOT TESTABLE, not a miss.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.cartpole.thr >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: flip_rate.cartpole.thr >= 0.10

### H1.cartpole.fp64

- hypothesis: H1
- statement: CartPole variant, float64 arm (CPU, 1 thread, float64 versus float32): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.20, plausible range 0.10 to 0.40. This is a larger perturbation than a change of reduction order, since every operation is rounded differently.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.cartpole.fp64 >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: flip_rate.cartpole.fp64 >= 0.10

### H1.cartpole.gpu

- hypothesis: H1
- statement: CartPole variant, CPU versus GPU arm (one CUDA GPU versus CPU, 1 thread, float32, TF32 disabled): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.20, plausible range 0.10 to 0.40. This arm needs the free Kaggle or Colab notebook. It stays NOT RUN until result files from that notebook are added to the repository.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.cartpole.gpu >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: flip_rate.cartpole.gpu >= 0.10

### H1.invpend.thr

- hypothesis: H1
- statement: InvertedPendulum (MuJoCo), thread count arm (CPU, 4 threads versus 1 thread, float32): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.10, plausible range 0.03 to 0.30. This is the arm most likely to show no perturbation at all, because small matrix products can be reduced in the same order for 1 and 4 threads. If fewer than half of the paired seeds differ bitwise, the prediction is NOT TESTABLE, not a miss.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.invpend.thr >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: flip_rate.invpend.thr >= 0.10

### H1.invpend.fp64

- hypothesis: H1
- statement: InvertedPendulum (MuJoCo), float64 arm (CPU, 1 thread, float64 versus float32): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.20, plausible range 0.10 to 0.40. This is a larger perturbation than a change of reduction order, since every operation is rounded differently.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.invpend.fp64 >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: flip_rate.invpend.fp64 >= 0.10

### H1.invpend.gpu

- hypothesis: H1
- statement: InvertedPendulum (MuJoCo), CPU versus GPU arm (one CUDA GPU versus CPU, 1 thread, float32, TF32 disabled): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.20, plausible range 0.10 to 0.40. This arm needs the free Kaggle or Colab notebook. It stays NOT RUN until result files from that notebook are added to the repository.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.invpend.gpu >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: flip_rate.invpend.gpu >= 0.10

### H2.pendulum.thr

- hypothesis: H2
- statement: Pendulum, thread count arm (CPU, 4 threads versus 1 thread, float32): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.pendulum.thr >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: mcnemar_holm_p.pendulum.thr >= 0.05
- check: abs_delta.pendulum.thr <= 0.10

### H2.pendulum.fp64

- hypothesis: H2
- statement: Pendulum, float64 arm (CPU, 1 thread, float64 versus float32): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.pendulum.fp64 >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: mcnemar_holm_p.pendulum.fp64 >= 0.05
- check: abs_delta.pendulum.fp64 <= 0.10

### H2.pendulum.gpu

- hypothesis: H2
- statement: Pendulum, CPU versus GPU arm (one CUDA GPU versus CPU, 1 thread, float32, TF32 disabled): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.pendulum.gpu >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: mcnemar_holm_p.pendulum.gpu >= 0.05
- check: abs_delta.pendulum.gpu <= 0.10

### H2.cartpole.thr

- hypothesis: H2
- statement: CartPole variant, thread count arm (CPU, 4 threads versus 1 thread, float32): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.cartpole.thr >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: mcnemar_holm_p.cartpole.thr >= 0.05
- check: abs_delta.cartpole.thr <= 0.10

### H2.cartpole.fp64

- hypothesis: H2
- statement: CartPole variant, float64 arm (CPU, 1 thread, float64 versus float32): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.cartpole.fp64 >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: mcnemar_holm_p.cartpole.fp64 >= 0.05
- check: abs_delta.cartpole.fp64 <= 0.10

### H2.cartpole.gpu

- hypothesis: H2
- statement: CartPole variant, CPU versus GPU arm (one CUDA GPU versus CPU, 1 thread, float32, TF32 disabled): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.cartpole.gpu >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: mcnemar_holm_p.cartpole.gpu >= 0.05
- check: abs_delta.cartpole.gpu <= 0.10

### H2.invpend.thr

- hypothesis: H2
- statement: InvertedPendulum (MuJoCo), thread count arm (CPU, 4 threads versus 1 thread, float32): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.invpend.thr >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: mcnemar_holm_p.invpend.thr >= 0.05
- check: abs_delta.invpend.thr <= 0.10

### H2.invpend.fp64

- hypothesis: H2
- statement: InvertedPendulum (MuJoCo), float64 arm (CPU, 1 thread, float64 versus float32): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.invpend.fp64 >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: mcnemar_holm_p.invpend.fp64 >= 0.05
- check: abs_delta.invpend.fp64 <= 0.10

### H2.invpend.gpu

- hypothesis: H2
- statement: InvertedPendulum (MuJoCo), CPU versus GPU arm (one CUDA GPU versus CPU, 1 thread, float32, TF32 disabled): the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. With 30 seeds the 95 percent interval of a paired difference is wide (a half width of roughly 0.15 to 0.25), so a hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.invpend.gpu >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: mcnemar_holm_p.invpend.gpu >= 0.05
- check: abs_delta.invpend.gpu <= 0.10

### H3.pendulum

- hypothesis: H3
- statement: Pendulum: flipped seeds are the ones whose reference score ends close to the success threshold.
- expected: Area under the curve of the distance to threshold, flipped seeds versus unflipped seeds (0.5 means no relation, 1.0 means flipped seeds are always closer), of about 0.75, pooled over the arms that were run. The competing account is chaotic early divergence, where a flip can happen to any seed and the AUC stays near 0.5.
- falsified_if: The AUC is below 0.70. A value between 0.55 and 0.70 is scored as a miss and described as inconclusive.
- requires: n_flips.pendulum >= 5
- check: auc_dist.pendulum >= 0.70

### H3.cartpole

- hypothesis: H3
- statement: CartPole variant: flipped seeds are the ones whose reference score ends close to the success threshold.
- expected: Area under the curve of the distance to threshold, flipped seeds versus unflipped seeds (0.5 means no relation, 1.0 means flipped seeds are always closer), of about 0.75, pooled over the arms that were run. The competing account is chaotic early divergence, where a flip can happen to any seed and the AUC stays near 0.5.
- falsified_if: The AUC is below 0.70. A value between 0.55 and 0.70 is scored as a miss and described as inconclusive.
- requires: n_flips.cartpole >= 5
- check: auc_dist.cartpole >= 0.70

### H3.invpend

- hypothesis: H3
- statement: InvertedPendulum (MuJoCo): flipped seeds are the ones whose reference score ends close to the success threshold.
- expected: Area under the curve of the distance to threshold, flipped seeds versus unflipped seeds (0.5 means no relation, 1.0 means flipped seeds are always closer), of about 0.75, pooled over the arms that were run. The competing account is chaotic early divergence, where a flip can happen to any seed and the AUC stays near 0.5.
- falsified_if: The AUC is below 0.70. A value between 0.55 and 0.70 is scored as a miss and described as inconclusive.
- requires: n_flips.invpend >= 5
- check: auc_dist.invpend >= 0.70

### H3.all

- hypothesis: H3
- statement: All RL tasks pooled, with distance to threshold divided by the standard deviation of the reference scores of that task: flipped seeds end closer to the threshold.
- expected: AUC of about 0.75.
- falsified_if: The AUC is below 0.70.
- requires: n_flips.all >= 10
- check: auc_dist.all >= 0.70

### PC1.grok.thr

- hypothesis: PC
- statement: Positive control, modular addition with a small transformer: changing the CPU thread count from 1 to 4 flips a minority of paired seeds.
- expected: Flip rate about 0.16 (Ootani reports 49 of 300), plausible range 0.08 to 0.30. Our task, threshold and size differ from Ootani, so this is a conceptual replication.
- falsified_if: The flip rate is below 0.08 although the perturbation check passed.
- requires: perturbation_rate.grok.thr >= 0.5
- requires: ref_rate.grok >= 0.2
- requires: ref_rate.grok <= 0.8
- check: flip_rate.grok.thr >= 0.08

### PC2.grok.thr

- hypothesis: PC
- statement: Positive control: the aggregate grokking rate does not change detectably between 1 and 4 threads.
- expected: Absolute difference of at most 0.07 (Ootani: 74 versus 81 of 300).
- falsified_if: The exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.grok.thr >= 0.5
- requires: ref_rate.grok >= 0.2
- requires: ref_rate.grok <= 0.8
- check: mcnemar_p.grok.thr >= 0.05
- check: abs_delta.grok.thr <= 0.10

### C0.grok

- hypothesis: C0
- statement: Negative control, task grok: repeating the reference environment with the same seed gives bitwise identical final parameters on every one of 10 seeds.
- expected: Determinism rate of exactly 1.0.
- falsified_if: Any repeat differs. In that case all flip counts for this task are uninterpretable and are reported as such.
- check: determinism_rate.grok >= 1.0

### C0.pendulum

- hypothesis: C0
- statement: Negative control, task pendulum: repeating the reference environment with the same seed gives bitwise identical final parameters on every one of 10 seeds.
- expected: Determinism rate of exactly 1.0.
- falsified_if: Any repeat differs. In that case all flip counts for this task are uninterpretable and are reported as such.
- check: determinism_rate.pendulum >= 1.0

### C0.cartpole

- hypothesis: C0
- statement: Negative control, task cartpole: repeating the reference environment with the same seed gives bitwise identical final parameters on every one of 10 seeds.
- expected: Determinism rate of exactly 1.0.
- falsified_if: Any repeat differs. In that case all flip counts for this task are uninterpretable and are reported as such.
- check: determinism_rate.cartpole >= 1.0

### C0.invpend

- hypothesis: C0
- statement: Negative control, task invpend: repeating the reference environment with the same seed gives bitwise identical final parameters on every one of 10 seeds.
- expected: Determinism rate of exactly 1.0.
- falsified_if: Any repeat differs. In that case all flip counts for this task are uninterpretable and are reported as such.
- check: determinism_rate.invpend >= 1.0

### B1.kit

- hypothesis: B
- statement: thabat_kit passes its full test suite.
- expected: Zero failed tests.
- falsified_if: Any test fails at the commit that is tagged as the release.
- check: tests_failed.kit <= 0

### B2.kit

- hypothesis: B
- statement: Line coverage of thabat_kit is at least 85 percent.
- expected: Coverage between 85 and 95 percent.
- falsified_if: Coverage below 85 percent.
- check: coverage_percent.kit >= 85

## Amendment log

#### Amendment 1, 3 November 2026, made before the pilot and before any evaluation run

The original text above is kept unchanged. Reasons and changes:

1. Speed. A smoke test of 1,500 step runs (seed 0, no results used) showed about one second per thousand training steps on this machine. Thirty seeds are therefore cheap, and we raise the number of evaluation seeds for each RL task from 0 to 29 to 0 to 59, to narrow the intervals for H2. The grokking control keeps 0 to 59. The minimum of 30 seeds per environment is still met.
2. Thread count and small MLPs. The same smoke test showed bitwise identical final parameters for 1 versus 4 threads on the Pendulum TD3 network, with hidden width 64, 128 and 256 (batch equal to width). The likely reason is that matrix multiplication threads split the output and keep the order of summation along the inner dimension fixed. We expect the `thr` blocks for the three RL tasks to be NOT TESTABLE, and we will report that as a finding about the instrument, not as a miss. The same test showed that the small transformer of the grokking control does change under 4 threads.
3. New arm `isa`. Because thread count gives no perturbation for these MLPs, we add one CPU arm that does change the bits in the smoke test: the same code with the environment variable MKL_CBWR set to COMPATIBLE before the process starts, which selects a different kernel path in Intel MKL. It is CPU, 1 thread, float32, and a benign change of the same kind as the others. It is added to the Holm family for H2. Its blocks follow. It is an exploratory addition made after seeing a smoke test of bit differences, and not before seeing anything, which is why it is recorded here and not above.
4. Machine control for the GPU arm. The Kaggle or Colab notebook runs a CPU reference (`ref_k`, identical settings to `ref`) and the GPU arm (`gpu`) on the same machine, and the GPU comparison is against `ref_k`. The comparison of `ref` with `ref_k` is reported as an exploratory machine arm without prediction blocks.

### H1.pendulum.isa

- hypothesis: H1
- statement: Pendulum, kernel path arm (CPU, 1 thread, float32, MKL conditional reproducibility set to COMPATIBLE versus the default path): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.15, plausible range 0.05 to 0.35. The smoke test showed that this setting changes the final bits of a short Pendulum run, so the perturbation is expected to be present.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.pendulum.isa >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: flip_rate.pendulum.isa >= 0.10

### H2.pendulum.isa

- hypothesis: H2
- statement: Pendulum, kernel path arm: the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. A hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.pendulum.isa >= 0.5
- requires: ref_rate.pendulum >= 0.2
- requires: ref_rate.pendulum <= 0.8
- check: mcnemar_holm_p.pendulum.isa >= 0.05
- check: abs_delta.pendulum.isa <= 0.10

### H1.cartpole.isa

- hypothesis: H1
- statement: CartPole variant, kernel path arm (CPU, 1 thread, float32, MKL conditional reproducibility set to COMPATIBLE versus the default path): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.15, plausible range 0.05 to 0.35. The smoke test showed that this setting changes the final bits of a short Pendulum run, so the perturbation is expected to be present.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.cartpole.isa >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: flip_rate.cartpole.isa >= 0.10

### H2.cartpole.isa

- hypothesis: H2
- statement: CartPole variant, kernel path arm: the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. A hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.cartpole.isa >= 0.5
- requires: ref_rate.cartpole >= 0.2
- requires: ref_rate.cartpole <= 0.8
- check: mcnemar_holm_p.cartpole.isa >= 0.05
- check: abs_delta.cartpole.isa <= 0.10

### H1.invpend.isa

- hypothesis: H1
- statement: InvertedPendulum (MuJoCo), kernel path arm (CPU, 1 thread, float32, MKL conditional reproducibility set to COMPATIBLE versus the default path): for the same seed, the success outcome flips for a substantial share of seeds.
- expected: Flip rate about 0.15, plausible range 0.05 to 0.35. The smoke test showed that this setting changes the final bits of a short Pendulum run, so the perturbation is expected to be present.
- falsified_if: The flip rate is below 0.10 although the perturbation check passed.
- requires: perturbation_rate.invpend.isa >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: flip_rate.invpend.isa >= 0.10

### H2.invpend.isa

- hypothesis: H2
- statement: InvertedPendulum (MuJoCo), kernel path arm: the average success rate does not change detectably.
- expected: Absolute difference between arm rate and reference rate of at most 0.07. A hit means no detectable change, not equivalence.
- falsified_if: The Holm adjusted exact McNemar p value is below 0.05, or the absolute difference exceeds 0.10.
- requires: perturbation_rate.invpend.isa >= 0.5
- requires: ref_rate.invpend >= 0.2
- requires: ref_rate.invpend <= 0.8
- check: mcnemar_holm_p.invpend.isa >= 0.05
- check: abs_delta.invpend.isa <= 0.10

#### Amendment 2, 3 November 2026, a clarification made before the main runs

Two points in the original text were open to more than one reading. We fix the reading now, before any evaluation run:

1. H3 pools flipped and unflipped seed and arm pairs over the arms of a task whose `perturbation_rate` is at least 0.5. An arm with no perturbation has no flips by construction, and including its pairs would only dilute the comparison. The pairs of one seed in different arms are not independent, which we note in the write up.
2. The Holm family for H2 consists of every RL task and arm comparison for which result files exist, including the `thr` comparisons (whose p value is 1 when no seed differs) and the `isa` comparisons. The grokking control uses the unadjusted McNemar p value.

#### Amendment 3, 3 November 2026, pilot grid refinement for Pendulum, made before any evaluation run

The pilot (pilot seeds 1000 to 1009, reference environment only) gave a success rate of 0.0 for Pendulum at 6,000 steps and 1.0 at 9,000 steps, so the original grid has no entry in the target band of 0.3 to 0.7, and the fallback of the closest entry to 0.5 is a tie between 0.0 and 1.0. We insert 7,000 and 8,000 steps into the Pendulum grid, between the two entries that bracket the band. The rule is unchanged, and the entries already run keep their results. No evaluation seed has been used.

#### Amendment 4, 3 November 2026, CartPole and grokking settings changed after the pilot, made before any evaluation run

The pilot (pilot seeds only, reference environment) showed that two pre registered settings cannot reach the target band of 0.3 to 0.7 within the grid. Pilot rates and scores are kept in results/pilot/original_grid/ and results/pilot/explore/, and are tabulated in PILOT.md.

1. CartPole. With seeded observation noise of standard deviation 0.1 the pilot success rate was 0.0 at 10,000 steps and 0.0 at 18,000 steps, and the first four seeds at 28,000 steps scored between 13 and 225, far from the threshold of 400. The exploration at 20,000 steps and 4 pilot seeds gave 0.25 without noise and 0.0 with noise of 0.02. We therefore remove the observation noise (standard deviation 0.0). The task is now CartPole with its standard dynamics. The words "CartPole variant" in the blocks above refer to this setting, which differs from the standard task only in the success threshold (400 of a possible 500) and in the use of a small DQN. The threshold is unchanged. The new budget grid is 15,000 to 40,000 steps.
2. Grokking control. At the pre registered learning rate of 0.001 the held out accuracy stayed on a memorisation plateau (about 0.48 to 0.57) for all six completed grid entries (training fractions 0.4, 0.5, 0.6 at 2,000 steps, and 0.4, 0.5, 0.6 at 4,000 steps), with a pilot rate of 0.0 each. In an exploration with 4 pilot seeds and 3,000 steps, a learning rate of 0.003 gave a rate of 0.25 for a training fraction of 0.5 on the same task (modulus 23, 12,311 parameters), and 0.0 for modulus 13 at 0.5, while modulus 13 at a training fraction of 0.7 gave 1.0. We keep modulus 23, raise the learning rate to 0.003, and use a new grid over training fraction and steps. The threshold of 0.90 held out accuracy is unchanged.

Everything else, including the thresholds, arms, metrics and checks of all prediction blocks, is unchanged. Only pilot seeds were used for these decisions.

