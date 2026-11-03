"""Turn raw paired rows into the flat metrics that docs/PREDICTIONS.md refers to."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thabat_kit import stats
from thabat_kit.runner import PairedTable

RL_TASKS = ("pendulum", "cartpole", "invpend")
RL_ARMS = ("thr", "fp64", "isa", "gpu")
PERTURBATION_MIN = 0.5


def load_task(raw_dir, task: str) -> PairedTable:
    return PairedTable.load(*sorted(Path(raw_dir).glob(f"{task}.*.jsonl")))


def _score_pairs(task, arm, comp, base_rows, arm_rows, threshold):
    out = []
    for s in comp["seeds"]:
        flipped = int(base_rows[s]["outcome"] != arm_rows[s]["outcome"])
        out.append({"task": task, "arm": arm, "seed": s, "flipped": flipped,
                    "ref_score": base_rows[s]["score"], "arm_score": arm_rows[s]["score"],
                    "distance": abs(base_rows[s]["score"] - threshold)})
    return out


def analyse(raw_dir, thresholds: dict, n_boot: int = 10000) -> dict:
    metrics: dict = {}
    comps: dict = {}
    pairs: list = []
    ref_sd: dict = {}
    for task in ("grok",) + RL_TASKS:
        table = load_task(raw_dir, task)
        envs = table.envs()
        if "ref" not in envs:
            continue
        ref = table.rows("ref")
        metrics[f"n_seeds.{task}"] = len(ref)
        metrics[f"ref_rate.{task}"] = float(np.mean([r["outcome"] for r in ref.values()]))
        ref_sd[task] = float(np.std([r["score"] for r in ref.values()], ddof=1)) if len(ref) > 1 else float("nan")
        if "ref_repeat" in envs:
            both = sorted(set(ref) & set(table.rows("ref_repeat")))
            same = [ref[s]["param_hash"] == table.rows("ref_repeat")[s]["param_hash"] for s in both]
            metrics[f"determinism_rate.{task}"] = float(np.mean(same)) if same else float("nan")
            metrics[f"n_determinism.{task}"] = len(same)
        arms = ("thr",) if task == "grok" else RL_ARMS
        for arm in arms:
            if arm not in envs:
                continue
            base = "ref_k" if (arm == "gpu" and "ref_k" in envs) else "ref"
            c = stats.compare_arms(table.rows(base), table.rows(arm), n_boot=n_boot)
            comps[(task, arm)] = c
            for k in ("flip_rate", "abs_delta", "mcnemar_p", "perturbation_rate", "n", "arm_rate"):
                if k in c:
                    metrics[f"{k}.{task}.{arm}"] = c[k]
            if task != "grok":
                pairs += _score_pairs(task, arm, c, table.rows(base), table.rows(arm), thresholds[task])
        if task != "grok" and "ref_k" in envs:
            c = stats.compare_arms(ref, table.rows("ref_k"), n_boot=n_boot)
            comps[(task, "machine")] = c
            metrics[f"flip_rate.{task}.machine"] = c["flip_rate"]
            metrics[f"perturbation_rate.{task}.machine"] = c.get("perturbation_rate", float("nan"))
    holm_in = {f"{t}.{a}": c["mcnemar_p"] for (t, a), c in comps.items() if t != "grok" and a != "machine"}
    for key, p in stats.holm(holm_in).items():
        metrics[f"mcnemar_holm_p.{key}"] = p
    # H3, pooled over arms whose perturbation check passed
    valid = {(t, a) for (t, a), c in comps.items()
             if t != "grok" and a != "machine" and c.get("perturbation_rate", 0) >= PERTURBATION_MIN}
    h3 = [p for p in pairs if (p["task"], p["arm"]) in valid]
    for task in RL_TASKS:
        _auc_metrics(metrics, f"{task}", [p for p in h3 if p["task"] == task], 1.0, n_boot)
    scale = {t: ref_sd.get(t, float("nan")) for t in RL_TASKS}
    std_pairs = [{**p, "distance": p["distance"] / scale[p["task"]]} for p in h3 if scale[p["task"]] > 0]
    _auc_metrics(metrics, "all", std_pairs, 1.0, n_boot)
    return {"metrics": metrics, "comps": comps, "pairs": pairs, "h3_pairs": h3, "ref_sd": ref_sd}


def _auc_metrics(metrics: dict, key: str, pairs: list, scale: float, n_boot: int) -> None:
    flipped = [-p["distance"] for p in pairs if p["flipped"]]
    kept = [-p["distance"] for p in pairs if not p["flipped"]]
    metrics[f"n_flips.{key}"] = len(flipped)
    metrics[f"n_pairs.{key}"] = len(pairs)
    if flipped and kept:
        a, pval = stats.auc(flipped, kept)
        lo, hi = stats.auc_ci(flipped, kept, n_boot=min(n_boot, 3000))
        metrics[f"auc_dist.{key}"], metrics[f"auc_p.{key}"] = a, pval
        metrics[f"auc_lo.{key}"], metrics[f"auc_hi.{key}"] = lo, hi
    else:
        metrics[f"auc_dist.{key}"] = float("nan")


def threshold_sweep(pairs: list, thresholds: dict, ref_sd: dict, offsets=(-0.5, -0.25, 0.0, 0.25, 0.5)) -> list:
    """Exploratory, after the fact: flip rate if the threshold were moved by a multiple of the
    standard deviation of the reference scores. Not part of the predictions."""
    out = []
    keys = sorted({(p["task"], p["arm"]) for p in pairs})
    for task, arm in keys:
        P = [p for p in pairs if p["task"] == task and p["arm"] == arm]
        for off in offsets:
            tau = thresholds[task] + off * ref_sd[task]
            flips = sum((p["ref_score"] >= tau) != (p["arm_score"] >= tau) for p in P)
            out.append({"task": task, "arm": arm, "offset_sd": off, "threshold": tau, "n": len(P),
                        "flip_rate": flips / len(P), "ref_rate": float(np.mean([p["ref_score"] >= tau for p in P]))})
    return out


def jsonable(obj):
    if isinstance(obj, dict):
        return {(f"{k[0]}.{k[1]}" if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_summary(result: dict, path) -> None:
    Path(path).write_text(json.dumps(jsonable({"metrics": result["metrics"], "comps": result["comps"]}),
                                     indent=1, sort_keys=True))
