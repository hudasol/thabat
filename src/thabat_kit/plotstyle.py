"""One plain plotting style. Pearl background, one rose accent, thin lines, no decoration."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PEARL = "#FBF8F6"
SAKURA = "#E8B4C4"
ROSE = "#B04A6B"
INK = "#2B2B2B"
GREY = "#8C8C8C"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PEARL,
            "axes.facecolor": PEARL,
            "savefig.facecolor": PEARL,
            "font.family": "DejaVu Sans Mono",
            "font.size": 8,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def interval_plot(ax, labels, est, lo, hi, ref=None, xlabel="") -> None:
    """Horizontal point and interval plot, one row per label (first label at the top)."""
    n = len(labels)
    ys = list(range(n))[::-1]
    for y, e, a, b in zip(ys, est, lo, hi):
        ax.plot([a, b], [y, y], color=GREY, lw=1.0)
        ax.plot([e], [y], "o", color=ROSE, ms=4)
    if ref is not None:
        ax.axvline(ref, color=GREY, lw=0.6, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, metadata={"Software": "thabat_kit"})
    plt.close(fig)
