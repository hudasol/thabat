"""Plain figures. No titles beyond what is needed, direct labels, one accent colour."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from thabat_kit import plotstyle, stats
from thabat_kit.plotstyle import GREY, INK, ROSE, SAKURA

ARM_LABEL = {"thr": "4 threads", "fp64": "float64", "isa": "kernel path", "gpu": "GPU", "machine": "other machine"}
MARK = {"thr": "o", "fp64": "s", "isa": "^", "gpu": "D", "machine": "v"}


def flip_rates(comps: dict, path) -> None:
    plotstyle.apply_style()
    keys = [k for k in comps if k[1] != "machine"]
    labels = [f"{t}  {ARM_LABEL.get(a, a)}" for t, a in keys]
    est = [comps[k]["flip_rate"] for k in keys]
    lo = [comps[k]["flip_ci"][0] for k in keys]
    hi = [comps[k]["flip_ci"][1] for k in keys]
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(keys) + 1.0))
    plotstyle.interval_plot(ax, labels, est, lo, hi, ref=0.10, xlabel="share of seeds whose outcome flips (Wilson 95%)")
    ax.set_xlim(0, max(0.5, max(hi) + 0.05))
    plotstyle.save(fig, path)


def success_rates(comps: dict, path) -> None:
    plotstyle.apply_style()
    labels, est, lo, hi = [], [], [], []
    for task in dict.fromkeys(t for t, _ in comps):
        arms = [a for t, a in comps if t == task and a != "machine"]
        c0 = comps[(task, arms[0])]
        rows = [("ref", c0["ref_rate"], c0["n"])] + [(a, comps[(task, a)]["arm_rate"], comps[(task, a)]["n"]) for a in arms]
        for name, rate, n in rows:
            l, h = stats.wilson_interval(round(rate * n), n)
            labels.append(f"{task}  {ARM_LABEL.get(name, name)}")
            est.append(rate), lo.append(l), hi.append(h)
    fig, ax = plt.subplots(figsize=(5.6, 0.26 * len(labels) + 1.0))
    plotstyle.interval_plot(ax, labels, est, lo, hi, xlabel="success rate (Wilson 95%)")
    ax.set_xlim(0, 1)
    plotstyle.save(fig, path)


def paired_scatter(pairs: list, thresholds: dict, path) -> None:
    plotstyle.apply_style()
    tasks = [t for t in ("pendulum", "cartpole", "invpend") if any(p["task"] == t for p in pairs)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        P = [p for p in pairs if p["task"] == task]
        lim = [min(min(p["ref_score"], p["arm_score"]) for p in P), max(max(p["ref_score"], p["arm_score"]) for p in P)]
        ax.plot(lim, lim, color=GREY, lw=0.6)
        ax.axvline(thresholds[task], color=INK, lw=0.5, ls=":")
        ax.axhline(thresholds[task], color=INK, lw=0.5, ls=":")
        for arm in dict.fromkeys(p["arm"] for p in P):
            for flipped, colour, size in ((0, GREY, 14), (1, ROSE, 22)):
                Q = [p for p in P if p["arm"] == arm and p["flipped"] == flipped]
                ax.scatter([p["ref_score"] for p in Q], [p["arm_score"] for p in Q], s=size, marker=MARK.get(arm, "o"),
                           color=colour, alpha=0.8 if flipped else 0.5, linewidths=0, label=f"{ARM_LABEL.get(arm, arm)}" if flipped == 0 else None)
        ax.set_xlabel("reference final score")
        ax.set_ylabel("arm final score")
        ax.text(0.02, 0.97, task, transform=ax.transAxes, va="top")
        if task == tasks[0]:
            ax.legend(loc="lower right", fontsize=6.5, markerscale=0.8)
    plotstyle.save(fig, path)


def grok_curves(table, flipped_seeds: list, threshold: float, path, max_panels: int = 6) -> None:
    plotstyle.apply_style()
    seeds = flipped_seeds[:max_panels]
    if not seeds:
        return
    fig, axes = plt.subplots(1, len(seeds), figsize=(2.3 * len(seeds), 2.6), squeeze=False, sharey=True)
    for ax, s in zip(axes[0], seeds):
        for env, colour in (("ref", GREY), ("thr", ROSE)):
            c = np.array(table.rows(env)[s]["curve"])
            ax.plot(c[:, 0], c[:, 1], color=colour, lw=1.0, label="1 thread" if env == "ref" else "4 threads")
        ax.axhline(threshold, color=INK, lw=0.5, ls=":")
        ax.set_xlabel("step")
        ax.text(0.03, 0.96, f"seed {s}", transform=ax.transAxes, va="top")
    axes[0][0].set_ylabel("held out accuracy")
    axes[0][0].legend(loc="center left", fontsize=6.5)
    plotstyle.save(fig, path)
