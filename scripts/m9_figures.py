"""Figure for the conformal prediction results.

Usage:
    python scripts/m9_figures.py

Writes assets/m9_conformal_m4_blood.png from results/m9/m4_blood/ and results/m9/pbmc/:
    left   coverage on each held-out lab under four ways of calibrating (target 90 percent)
    right  how often a type the classifier never saw gets an empty set, against known cells
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cellmap.plotting import ACCENT, ACCENT_2, INK, MUTED, apply, figure_caption, title  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "m9"
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

CALIBRATIONS = {
    "random": ("Random cells (same people as training)", MUTED),
    "donor": ("Held-out donors", ACCENT_2),
    "lab": ("Other training labs (heuristic)", INK),
    "donor_hier": ("Held-out donors, hierarchical", ACCENT),
}
LABS = {"aida": "AIDA", "combat": "COMBAT", "hao": "Hao", "onek1k": "OneK1K",
        "schulte": "Schulte-Schrepping", "stephenson": "Stephenson"}  # fmt: skip
NOVEL = {"dendritic cell": "dendritic cell", "plasma cell": "plasma cell", "B cell": "B cell",
         "plasmablast": "plasmablast", "regulatory T cell": "regulatory T",
         "CD16-negative, CD56-bright natural killer cell, human": "CD56-bright NK"}  # fmt: skip


def coverage_by_lab(ax: plt.Axes) -> None:
    c = pd.read_csv(RES / "m4_blood" / "conformal.csv")
    c = c[(c["scheme"] == "dataset") & (~c["mondrian"]) & (c["alpha"] == 0.1)
          & (c["score"] == "aps")]  # fmt: skip
    mean = c.groupby(["fold", "calibration"])["coverage"].mean().unstack()
    order = mean["random"].sort_values().index
    y = np.arange(len(order))
    offsets = np.linspace(-0.27, 0.27, len(CALIBRATIONS))
    for (cal, (label, color)), dy in zip(CALIBRATIONS.items(), offsets, strict=True):
        ax.scatter(mean.loc[order, cal], y + dy, s=34, color=color, label=label, zorder=3)
    ax.axvline(0.9, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)), label="target 0.90")
    ax.set_yticks(y, [LABS[f] for f in order])
    ax.set_xlabel("coverage on the held-out lab (mean of 6 embeddings x 2 classifiers)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, fontsize=8.5)
    random_mean = c[c["calibration"] == "random"]["coverage"].mean()
    worst = mean["random"].idxmin()
    title(ax, f"A new lab erodes the guarantee: {random_mean:.3f} instead of 0.90",
          f"Six blood labs, one held out at a time; worst lab {LABS[worst]} "
          f"at {mean.loc[worst, 'random']:.3f} with random-cell calibration")  # fmt: skip


def novel_types(ax: plt.Axes) -> None:
    novel = pd.concat([pd.read_csv(RES / n / "novel.csv") for n in ("m4_blood", "pbmc")])
    lac = novel[novel["score"] == "lac"]
    share = lac.groupby("novel_type")[["novel_empty_share", "known_empty_share"]].mean()
    share = share.loc[[t for t in NOVEL if t in share.index]].sort_values("novel_empty_share")
    y = np.arange(len(share))
    for i, (_, row) in enumerate(share.iterrows()):
        ax.plot([row["known_empty_share"], row["novel_empty_share"]], [i, i], color=MUTED,
                linewidth=1.2, zorder=1)  # fmt: skip
    ax.scatter(share["known_empty_share"], y, s=34, color=INK, label="known cells", zorder=3)
    ax.scatter(share["novel_empty_share"], y, s=34, color=ACCENT, label="the unseen type",
               zorder=3)  # fmt: skip
    ax.set_yticks(y, [NOVEL[t] for t in share.index])
    ax.set_xlabel("share of cells whose prediction set is empty (LAC, alpha 0.1)")
    ax.legend(loc="lower right", fontsize=8.5)
    title(ax, "Distinct unseen types get flagged; close siblings do not",
          "One type removed from the classifier's labels; mean over embeddings")  # fmt: skip


def main() -> None:
    apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.55})
    coverage_by_lab(ax1)
    novel_types(ax2)
    figure_caption(
        fig,
        "Split conformal prediction with APS scores on a linear probe or kNN classifier. "
        "Right: dendritic, plasma and B cells from the six-lab data (6 embeddings), the rest "
        "from PBMC (10 embeddings); linear probe, donor calibration.",
        y=-0.17,
    )
    out = ROOT / "assets" / "m9_conformal_m4_blood.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
