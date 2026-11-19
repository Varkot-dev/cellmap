"""One shared figure style, so every figure in the repo reads as one hand.

Same direction as the owner's earlier DCMvsACM project: off-white canvas, ink plus one warm
accent and one cool secondary, restrained type, no gridlines unless they carry information.
Headlines state the finding; the explanation goes in the caption, not inside the plot.
"""

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

INK = "#1A1A1A"
PAPER = "#FBFAF7"
MUTED = "#9A9791"
GRID = "#E7E4DD"
ACCENT = "#C2703D"  # terracotta: the thing of interest
ACCENT_2 = "#3C5A78"  # slate blue: the second group, used sparingly

# Train is the reference (ink); test is what every result is about (accent).
SPLIT_COLORS = {"train": INK, "val": ACCENT_2, "test": ACCENT}
METHOD_COLORS = {"pca": INK, "pca_harmony": ACCENT}
METHOD_NAMES = {"pca": "HVG + PCA", "pca_harmony": "PCA + Harmony"}


def apply() -> None:
    """Set the shared matplotlib style. Call once per script."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def title(ax: Axes, headline: str, sub: str | None = None) -> None:
    """Bold left-aligned headline that states the finding, with an optional muted subline."""
    ax.set_title("")
    ax.text(0.0, 1.10, headline, transform=ax.transAxes, fontsize=12.5, fontweight="bold",
            ha="left", va="bottom", color=INK)  # fmt: skip
    if sub:
        ax.text(0.0, 1.04, sub, transform=ax.transAxes, fontsize=9.5, ha="left", va="bottom",
                color=MUTED)  # fmt: skip


def figure_caption(fig: Figure, text: str, y: float = -0.02) -> None:
    """Muted caption under the figure. The explanation lives here, not inside the plot."""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=9, color=MUTED, wrap=True)
