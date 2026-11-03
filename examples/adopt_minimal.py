"""Minimal adoption example for thabat_kit.

SYNTHETIC DEMONSTRATION. The "training run" below is a toy: it adds 20000 float32
numbers in chunks, and the number of chunks follows the thread count of the
environment. It exists only to show the calls a project makes. It is not part of
the study and its numbers are not results.

    python examples/adopt_minimal.py
"""
import tempfile
from pathlib import Path

import numpy as np

from thabat_kit.env import NumericalEnv
from thabat_kit.fingerprint import hash_arrays
from thabat_kit.runner import PairedRunner
from thabat_kit.stats import compare_arms


def run(seed: int, env: NumericalEnv) -> dict:
    rng = np.random.default_rng(seed)               # all draws come from the seed
    x = rng.normal(size=20000).astype(env.dtype)
    partial = [chunk.sum(dtype=env.dtype) for chunk in np.array_split(x, env.threads)]
    total = float(np.sum(partial, dtype=env.dtype))  # order of summation depends on env.threads
    return {"outcome": int(total > 0.0), "score": total, "param_hash": hash_arrays([[total]])}


def main() -> None:
    envs = [NumericalEnv("ref", threads=1), NumericalEnv("thr", threads=4), NumericalEnv("fp64", dtype="float64")]
    with tempfile.TemporaryDirectory() as tmp:
        table = PairedRunner(run, envs, Path(tmp) / "demo.jsonl").run(range(200))
    for arm in ("thr", "fp64"):
        c = compare_arms(table.rows("ref"), table.rows(arm), n_boot=500)
        print(f"{arm}: flips {c['flips']} of {c['n']}, differs in bits for {c['perturbation_rate']:.2f} of seeds")


if __name__ == "__main__":
    main()
