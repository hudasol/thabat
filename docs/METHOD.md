# Method

Author: Huda Mueen. The hypotheses, thresholds, metrics and scoring rules are fixed in PREDICTIONS.md before any experiment. This file describes how the experiments and the analysis are carried out.

## Design

For each task and seed, the same training run is repeated under several numerical environments. The reference environment is CPU, 1 thread, float32. Each arm changes one thing:

| Arm | Change from the reference | Where it runs |
|-----|---------------------------|---------------|
| thr | 4 threads | this repository, CPU |
| fp64 | float64 network arithmetic | this repository, CPU |
| isa | MKL kernel path set to COMPATIBLE through MKL_CBWR | this repository, CPU, separate process |
| gpu | CUDA GPU, float32, TF32 off | free Kaggle or Colab notebook |

Everything else is shared: the seed fixes the initial weights, replay sampling, exploration noise, target policy noise, environment reset seeds and evaluation seeds. All of these come from numpy generators derived from the seed by name, so no torch random number generator is consumed and the draws are identical in every arm. The initial weights are drawn in float32 on the CPU and then cast. The only difference between arms is the arithmetic of the networks. Trajectories can still diverge after the first rounding difference, because the actions depend on the networks. That divergence is the object of study.

## Tasks

| Task | Algorithm | Success threshold | Budget |
|------|-----------|-------------------|--------|
| Pendulum | TD3, two layers of 64 units | mean return over 20 episodes of at least -250 | chosen by pilot |
| CartPole variant (seeded observation noise, standard deviation 0.1) | DQN, two layers of 64 units | mean return over 20 episodes of at least 400 | chosen by pilot |
| InvertedPendulum (MuJoCo) | TD3, two layers of 64 units | mean return over 10 episodes of at least 800 | chosen by pilot |
| Modular addition (positive control), 12,311 parameter transformer | full batch AdamW | held out accuracy of at least 0.90 | chosen by pilot |

The final evaluation uses a deterministic policy and fixed evaluation seeds (100000 and up), identical for all training seeds and arms. Learning curves use different evaluation seeds (900000 and up) and do not influence training.

## Pilot

The training budget is chosen on pilot seeds 1000 to 1009, in the reference environment only, by the rule in configs/pilot_grid.json: the cheapest budget in an ordered list whose pilot success rate lies between 0.3 and 0.7. Evaluation seeds are 0 to 59. The pilot is documented in PILOT.md.

## Controls and instrument checks

* Determinism control: the reference environment is run twice for seeds 0 to 9, and the final parameter hashes must be identical.
* Perturbation check: for every arm, the share of seeds whose final parameters differ bitwise from the reference (after casting to float64). If it is below 0.5, the arm is reported as perturbation mostly absent, and its H1 and H2 predictions are NOT TESTABLE.
* Positive control: the grokking transformer under 1 versus 4 threads.

## Analysis

For each task and arm, paired by seed: success rates with Wilson 95 percent intervals; the number of seeds lost (success in the reference, failure in the arm) and gained; the flip rate with a Wilson interval; the paired difference of rates with a bootstrap interval over seeds (10,000 resamples); the exact two sided McNemar test, with Holm adjustment over all RL comparisons; the distance to the threshold of the reference score and the AUC that separates flipped from unflipped seeds, with a bootstrap interval. The thresholds of the predictions are applied by the scorer in thabat_kit, never by hand.

## Reproduce

    pip install -e ".[study,dev]"
    python scripts/pilot.py --task pendulum        # likewise cartpole, invpend, grok
    python scripts/run_task.py --task pendulum --arms ref,ref_repeat,thr,fp64 --seeds 0:60 --shard a
    python scripts/run_task.py --task pendulum --arms isa --mkl-cbwr COMPATIBLE --seeds 0:60 --shard isa
    python scripts/analyze.py
    python scripts/render_results.py

The GPU arm is run with notebooks/gpu_arm.ipynb, and its result files are added to results/raw/.
