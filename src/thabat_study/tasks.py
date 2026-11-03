"""Task registry: configs, budgets and the numerical environments (arms)."""
from __future__ import annotations

import json
from pathlib import Path

from thabat_kit.env import NumericalEnv

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("pendulum", "cartpole", "invpend", "grok")

ARMS = {
    "ref": NumericalEnv("ref", device="cpu", threads=1, dtype="float32"),
    "ref_repeat": NumericalEnv("ref_repeat", device="cpu", threads=1, dtype="float32"),
    "thr": NumericalEnv("thr", device="cpu", threads=4, dtype="float32"),
    "fp64": NumericalEnv("fp64", device="cpu", threads=1, dtype="float64"),
    # isa is the reference settings, run in a process started with MKL_CBWR=COMPATIBLE
    "isa": NumericalEnv("isa", device="cpu", threads=1, dtype="float32"),
    # ref_k and gpu are run together on the same free GPU machine (see notebooks/gpu_arm.ipynb)
    "ref_k": NumericalEnv("ref_k", device="cpu", threads=1, dtype="float32"),
    "gpu": NumericalEnv("gpu", device="cuda", threads=1, dtype="float32", tf32=False),
}


def load_cfg(task: str, need_budget: bool = True) -> dict:
    cfg = json.loads((ROOT / "configs" / f"{task}.json").read_text())
    budget_path = ROOT / "configs" / "budget.json"
    budget = json.loads(budget_path.read_text()).get(task) if budget_path.exists() else None
    if budget:
        cfg.update(budget)
    elif need_budget:
        raise FileNotFoundError(f"no budget for {task}; run scripts/pilot.py first")
    return cfg


def runner_for(task: str):
    if task == "grok":
        from .grok import run_grok as fn
    elif task in ("pendulum", "invpend"):
        from .td3 import run_td3 as fn
    elif task == "cartpole":
        from .dqn import run_dqn as fn
    else:
        raise KeyError(task)
    return fn
