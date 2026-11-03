# Adopting thabat-kit in another project

thabat-kit has no dependency on this study. A project such as khaliya, murouna, abad or jidhr adds it in four steps.

## 1. Install

    pip install "git+https://github.com/hudasol/thabat"

PyTorch is optional. Without it the kit still seeds numpy and python and captures the environment.

## 2. Wrap one training run in a function

The function takes a seed and a `NumericalEnv` and returns a dict with `outcome` (0 or 1). Return `score` and `param_hash` as well when you have them.

    from thabat_kit.env import NumericalEnv
    from thabat_kit.seeding import seed_everything
    from thabat_kit.fingerprint import hash_state_dict

    def run(seed, env):
        bundle = seed_everything(seed)          # python, numpy, torch
        rng = bundle.rng("data")                # a private named stream, independent of the others
        ...                                     # build, train, evaluate
        return {"outcome": int(score >= TARGET), "score": score,
                "param_hash": hash_state_dict(model.state_dict())}

Rules that keep a paired comparison honest:

* Draw every random number from a stream derived from the seed, and never from a generator whose output depends on network arithmetic.
* Initialise the model in float32 and cast, so the dtype arm starts from the same point.
* Evaluate with fixed evaluation seeds that are the same in every environment.

## 3. Run the same seeds under several environments

    from thabat_kit.runner import PairedRunner
    envs = [NumericalEnv("ref"), NumericalEnv("thr", threads=4), NumericalEnv("fp64", dtype="float64")]
    table = PairedRunner(run, envs, "results/raw/mytask.jsonl").run(range(30))

Each row is one JSON line. The environment capture (library versions, BLAS backend, threads, device, determinism flags) is written next to it as `mytask.envs.json`. An interrupted run resumes.

## 4. Compare, plot and score

    from thabat_kit.stats import compare_arms
    c = compare_arms(table.rows("ref"), table.rows("thr"))
    c["flip_rate"], c["flip_ci"], c["mcnemar_p"], c["perturbation_rate"]

Always look at `perturbation_rate` first. If two environments give bitwise identical parameters, a flip rate of zero says nothing about sensitivity.

To score predictions, write blocks like the ones in `docs/PREDICTIONS.md` before you run, collect your numbers into one flat dict with names such as `flip_rate.mytask.thr`, and call `thabat_kit.scorer.score(parse_predictions(text), metrics)`. To keep documents honest, write them as templates with `{{key:.2f}}` placeholders and render them with `thabat_kit.report`.

A runnable toy that shows these calls end to end is `examples/adopt_minimal.py`. Its data is synthetic and it is not a result.

## What the kit does not do

It does not make training deterministic. It records and varies the numerical environment so that the sensitivity of an outcome can be measured. For bitwise reproducibility on one machine, pin the library versions it records and run the reference environment twice, as the determinism control in this study does.
