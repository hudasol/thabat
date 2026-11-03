# Prior art

Author: Huda Mueen. Date of review: 3 November 2026.

## Verdict

The gap is not closed, but it is narrower than the project brief suggests. We continue, with the claims reduced as described below.

What is already known, with evidence:

1. In a roughly 12,000 parameter transformer on modular arithmetic, changing the CPU thread count or moving from CPU to GPU flips a minority of same seed grokking outcomes while the aggregate rate does not shift detectably (Ootani, arXiv 2607.05104).
2. In deep Q learning on one Atari game, leaving only GPU nondeterminism on, with every other seed fixed, is enough to make training runs diverge in score (Nagarajan, Warnell and Stone, 2018).
3. In supervised training, seed, library kernel choice and reduction order each produce about the same run to run variance (Summers and Dinneen, 2021, as reported in Ootani).
4. Tiny perturbations of a single weight early in training reliably send two otherwise identical supervised runs to different solutions (Altintas, Kwok, Raffel and Rolnick, ICML 2025).

Consequence for our hypotheses. H1 in its weak form (numerical noise alone makes RL runs diverge) is already supported by items 2 and 4, so a confirmation of H1 is not by itself a contribution. We therefore score H1 on magnitude (a flip rate of at least 10 percent), not on existence. The parts we could not find in the literature we read are:

* a paired by seed analysis of a binary success outcome in RL under controlled, named numerical environments (thread count, CPU versus GPU, float32 versus float64), with exact McNemar tests;
* a test whether the average success rate stays put while individual seeds flip (H2);
* a test whether flips concentrate near the success threshold or are spread over the score range (H3);
* small robot control tasks (Pendulum, a CartPole variant, InvertedPendulum) as the setting.

Ootani lists persistence of the knife edge beyond the 12K parameter regime as open. Our study addresses a different regime (small RL policies, not larger models), so it is a neighbouring question and not a direct answer to the scale question. We say so in the write up.

## The five closest papers

| # | Paper | What it did | How this work differs |
|---|-------|-------------|-----------------------|
| 1 | Ootani, arXiv 2607.05104 (2026) | Grokking at 12K parameters; 300 paired seeds for thread count (49 flipped, 74 versus 81 of 300 grokked, McNemar p 0.39) and 100 paired seeds for CPU versus GPU (19 flipped, 86 versus 85 of 100). | We ask the same paired question for RL policies reaching a success threshold, add float32 versus float64, and test where in the score range flips occur. |
| 2 | Nagarajan, Warnell and Stone, Deterministic implementations for reproducibility in deep RL (2018) | DQN on one Atari game, 4 groups of 5 runs; leaving only GPU nondeterminism on makes score variance grow after about 2 million frames. | Their outcome is a score distribution with unpaired groups of 5 and no tests. Ours is a paired binary outcome per seed over at least 30 seeds, on small robot control tasks, with several named numerical environments. |
| 3 | Summers and Dinneen, Nondeterminism and instability in neural network optimization (2021), as cited by Ootani | Supervised training; the sources of nondeterminism (seed, kernel, reduction order) give similar variance. | Supervised accuracy variance, not a threshold crossing in control tasks, and not paired by seed across numerical environments. We have not read the full text; we rely on the description in Ootani. |
| 4 | Altintas, Kwok, Raffel and Rolnick, The butterfly effect (ICML 2025, arXiv 2506.13234) | Single weight perturbations early in supervised training make runs diverge in weight space and loss barrier. | They perturb weights by a controlled amount in supervised learning. We change only the arithmetic environment and measure a task outcome in RL. |
| 5 | Agarwal et al., Deep RL at the edge of the statistical precipice (NeurIPS 2021) | Argues for interval estimates over few runs; notes in a footnote that fixing a seed may not control framework nondeterminism on GPUs. | A methodology paper for comparing algorithms. We use its advice (intervals, many seeds) and ask a different question, about outcome flips at a fixed seed. |

## Further context read

* Henderson et al., Deep reinforcement learning that matters (AAAI 2018). Established that RL results depend strongly on seeds and implementation details. We rely on this from prior knowledge and did not read again the full text for this review. It is the reason we state plainly that seed sensitivity in RL is old news; our question is only about benign numerical changes at a fixed seed.
* arXiv 2609.25634 (Opus and Lu), a replica overlap probe of grokking. The abstract page reports that a preregistered numerical precision requirement failed, but gives no details of the fp32 versus fp64 gate. We read only the abstract page. We take from it a caution: a precision comparison needs a validated instrument, so our harness records a bitwise perturbation check for every arm (see PREDICTIONS.md).
* arXiv 2609.10657 (Kataria), scaling and phase structure of grokking. The abstract page does not discuss seeds or numerical precision. Not a close competitor.
* arXiv 2605.18845 (Truong et al.), first passage law for grokking delay. The abstract page does not discuss seeds or numerical precision. Not a close competitor.
* Prieto et al. (2025), grokking at the edge of numerical stability, as cited by Ootani. Describes a directional effect (softmax collapse) which differs from non directional flips. Not read directly.

## Limits of this review

* The reading used web fetches of abstract pages and of full text summaries made from the fetched pages by an automated step. Figures quoted from Ootani and Nagarajan et al. come from those summaries and should be checked against the PDFs before any submission.
* A web search for 2026 RL work on thread count, device or precision induced outcome flips returned nothing that matches. Absence of search results is weak evidence of absence, not proof.
* We did not search the full body of literature on reproducibility in deep RL beyond the items above.
