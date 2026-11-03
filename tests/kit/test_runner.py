import json

import pytest

from thabat_kit.env import NumericalEnv
from thabat_kit.fingerprint import hash_arrays
from thabat_kit.runner import PairedRunner, PairedTable


def toy_run(seed, env):
    # SYNTHETIC deterministic toy: outcome depends on the seed only
    return {"outcome": int(seed % 2 == 0), "score": float(seed), "param_hash": hash_arrays([[seed]])}


def test_runner_pairs_resumes_and_records_env(tmp_path):
    envs = [NumericalEnv("ref"), NumericalEnv("fp64", dtype="float64")]
    calls = []

    def fn(seed, env):
        calls.append((seed, env.name))
        return toy_run(seed, env)

    path = tmp_path / "runs.jsonl"
    runner = PairedRunner(fn, envs, path)
    table = runner.run(range(4))
    assert len(calls) == 8
    assert table.envs() == ["ref", "fp64"] and table.seeds("ref") == [0, 1, 2, 3]
    runner.run(range(6))  # resume: only seeds 4 and 5 are new
    assert len(calls) == 12
    caps = json.loads(path.with_suffix(".envs.json").read_text())
    assert set(caps) == {"ref", "fp64"} and caps["ref"]["requested"]["name"] == "ref"
    assert len(PairedTable.load(path, tmp_path / "missing.jsonl").rows("ref")) == 6


def test_runner_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        PairedRunner(toy_run, [NumericalEnv("a"), NumericalEnv("a")], tmp_path / "x.jsonl")
    with pytest.raises(KeyError):
        PairedRunner(lambda s, e: {"score": 1.0}, [NumericalEnv("a")], tmp_path / "y.jsonl").run([0])


def test_runner_logs(tmp_path):
    lines = []
    PairedRunner(toy_run, [NumericalEnv("a")], tmp_path / "z.jsonl").run([1], log=lines.append)
    assert lines and "seed=1" in lines[0]
