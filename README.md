# Thabat (ثبات)

A small reproducibility harness, `thabat-kit`, and a study of numerical sensitivity in small training runs. Author: Huda Mueen.

Training is not only random, it is also arithmetic. A seed fixes the random numbers, but the order in which a processor adds floating point numbers, the number type, and the device can still differ, and those differences are tiny. Ootani (arXiv 2607.05104) reported that in a 12,000 parameter transformer on modular arithmetic these tiny differences flip a minority of same seed grokking outcomes without moving the average rate. The open question that paper names is whether this survives elsewhere.

This repository asks it for small robot policy training. For the same seed, do benign numerical changes (thread count, kernel path, CPU versus GPU, float32 versus float64) flip whether a policy reaches a fixed success threshold, without moving the average success rate? RL training is already known to be seed sensitive (Henderson et al. 2018) and GPU nondeterminism is already known to make RL runs diverge (Nagarajan et al. 2018). The new part is the paired by seed test of a binary outcome under named numerical environments. See `docs/PRIOR_ART.md`.

## Status

Results, with every prediction scored as hit or miss, are in `docs/RESULTS.md`. Limitations are in `docs/LIMITATIONS.md`. A short paper style write up is in `docs/PAPER.md`.

## Layout

    src/thabat_kit/     reusable harness (no dependency on the study)
    src/thabat_study/   grokking control, TD3, DQN, analysis, figures
    configs/            task settings, pilot grid, chosen budgets
    scripts/            pilot, run, analyze, render
    notebooks/          free GPU arm (Kaggle or Colab)
    results/            raw rows (one JSON line per environment and seed), summary
    tables/ figures/    CSV tables and plots written by scripts/analyze.py
    docs/               prior art, predictions, pilot, method, results, limitations, paper
    tests/              kit tests, study tests, contract and document tests
    examples/           a runnable adoption example (synthetic, not a result)

## Quick start

    pip install -e ".[study,dev]"
    python -m pytest
    python scripts/analyze.py && python scripts/render_results.py

The analysis reads `results/raw/` and writes the summary, tables, figures and the scored predictions. To rerun the experiments see `docs/METHOD.md`.

## Using thabat-kit in another project

    from thabat_kit.env import NumericalEnv
    from thabat_kit.runner import PairedRunner
    from thabat_kit.stats import compare_arms

    envs = [NumericalEnv("ref"), NumericalEnv("fp64", dtype="float64")]
    table = PairedRunner(run, envs, "results/raw/mytask.jsonl").run(range(30))
    compare_arms(table.rows("ref"), table.rows("fp64"))

`run(seed, env)` is your training function and returns a dict with `outcome`. The kit seeds every random source, records library versions, thread count, BLAS backend, device, dtype and determinism flags, runs the same seeds under each environment, gives Wilson and bootstrap intervals, exact McNemar tests and AUC, scores a predictions file, and fills documents from result files. Steps for khaliya, murouna, abad and jidhr are in `docs/ADOPTING.md`.

## What is real and what is not

* All experimental results come from real training runs whose raw rows are in `results/raw/`.
* Tests use small hand built rows. Where rows are hand built to check statistics they are labelled SYNTHETIC in the test files. None of them are results.
* `examples/adopt_minimal.py` is a synthetic demonstration of the kit calls and is not a result.
* The GPU arm needs the notebook. If its result files are absent from `results/raw/`, its predictions are scored NOT RUN.
* This is a student research project. It makes no claim about large models and no claim about any organisation.

## Licence

MIT. See `LICENSE`.
