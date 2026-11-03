"""Rates, intervals, paired tests and AUC. Everything takes a seed where it resamples."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

Z95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion k out of n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))


def rate_with_ci(outcomes) -> tuple[float, float, float]:
    """Success rate with a Wilson 95 percent interval."""
    o = np.asarray(outcomes, dtype=int)
    lo, hi = wilson_interval(int(o.sum()), len(o))
    return (float(o.mean()) if len(o) else float("nan"), lo, hi)


def bootstrap_ci(x, stat=np.mean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval of stat(x) over resamples of x."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def mcnemar_exact(lost: int, gained: int) -> float:
    """Exact two sided McNemar p value from the two discordant counts."""
    n = lost + gained
    if n == 0:
        return 1.0
    return float(sps.binomtest(min(lost, gained), n, 0.5).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step down adjustment. Returns adjusted p values under the same keys."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return {k: adjusted[k] for k in pvals}


def auc(pos, neg) -> tuple[float, float]:
    """Probability that a random pos value exceeds a random neg value (ties count half).

    Returns (auc, one sided Mann Whitney p value for pos being larger).
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    res = sps.mannwhitneyu(pos, neg, alternative="greater")
    return (float(res.statistic / (len(pos) * len(neg))), float(res.pvalue))


def auc_ci(pos, neg, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Bootstrap interval for the AUC, resampling the two groups separately."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        p = pos[rng.integers(0, len(pos), len(pos))]
        q = neg[rng.integers(0, len(neg), len(neg))]
        vals.append(auc(p, q)[0])
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def compare_arms(ref: dict, arm: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of two numerical environments.

    ref and arm map seed to a row with keys outcome (0 or 1) and, optionally,
    score and param_hash. Only seeds present in both are used.
    """
    seeds = sorted(set(ref) & set(arm))
    r = np.array([int(ref[s]["outcome"]) for s in seeds])
    a = np.array([int(arm[s]["outcome"]) for s in seeds])
    n = len(seeds)
    lost = int(((r == 1) & (a == 0)).sum())
    gained = int(((r == 0) & (a == 1)).sum())
    flips = lost + gained
    out = {
        "n": n,
        "seeds": seeds,
        "ref_rate": float(r.mean()) if n else float("nan"),
        "arm_rate": float(a.mean()) if n else float("nan"),
        "lost": lost,
        "gained": gained,
        "flips": flips,
        "flip_rate": flips / n if n else float("nan"),
        "flip_ci": wilson_interval(flips, n),
        "mcnemar_p": mcnemar_exact(lost, gained),
    }
    out["delta"] = out["arm_rate"] - out["ref_rate"]
    out["abs_delta"] = abs(out["delta"])
    out["delta_ci"] = bootstrap_ci((a - r), n_boot=n_boot, seed=seed)
    if n and all("param_hash" in ref[s] and "param_hash" in arm[s] for s in seeds):
        differs = [ref[s]["param_hash"] != arm[s]["param_hash"] for s in seeds]
        out["perturbation_rate"] = float(np.mean(differs))
    return out
