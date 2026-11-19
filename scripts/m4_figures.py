"""M4 figure: how much each method loses when a whole lab is held out.

Usage:
    python scripts/m4_figures.py --name m4_blood

For every method with results in results/m4/<name>/metrics_<method>.csv: kNN macro F1 over the
12 lineages, with held-out donors from labs seen in training (5 donor splits) next to a whole
held-out lab (6 folds, one per lab). One dot per fold, a bar at the mean.

Writes:
    assets/m4_lab_holdout_<name>.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellmap.plotting import ACCENT, ACCENT_2, GRID, PAPER, apply, figure_caption, title

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
NAMES = {
    "pca": "HVG + PCA",
    "pca_harmony": "PCA + Harmony",
    "vae": "VAE (hand-written)",
    "transformer": "Transformer (hand-written)",
    "census_scvi": "Census scVI",
    "geneformer_ft": "Geneformer (fine-tuned)",
    "tf_sapiens": "TranscriptFormer",
}
METRIC = "knn_macro_f1_lineage"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="m4_blood")
    args = parser.parse_args()
    apply()
    res = ROOT / "results" / "m4" / args.name
    frames = [pd.read_csv(p) for p in sorted(res.glob("metrics_*.csv"))]
    data = pd.concat(frames)
    complete = data.groupby("method").size()
    data = data[data["method"].isin(complete[complete == 11].index)]
    order = data[data["scheme"] == "donor"].groupby("method")[METRIC].mean().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 0.62 * len(order) + 1.8))
    for y, method in enumerate(order):
        for scheme, color, offset in (("donor", ACCENT_2, 0.14), ("dataset", ACCENT, -0.14)):
            vals = data[(data["method"] == method) & (data["scheme"] == scheme)][METRIC]
            ax.scatter(vals, np.full(len(vals), y + offset), s=26, color=color, alpha=0.55,
                       edgecolor=PAPER, linewidth=0.6, zorder=2)  # fmt: skip
            ax.vlines(vals.mean(), y + offset - 0.17, y + offset + 0.17, color=color, linewidth=2.5,
                      zorder=3)  # fmt: skip
        ax.axhline(y + 0.5, color=GRID, linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(order)), [NAMES.get(m, m) for m in order])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("kNN macro F1 over 12 shared lineages, held-out cells")
    seen = "held-out people, labs seen in training (5 splits)"
    unseen = "a whole held-out lab (6 folds)"
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=ACCENT_2, label=seen),
        plt.Line2D([], [], marker="o", linestyle="", color=ACCENT, label=unseen),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2)
    means = data.groupby(["method", "scheme"])[METRIC].mean().unstack()
    lower = int((means["dataset"] < means["donor"]).sum())
    title(
        ax,
        f"A whole unseen lab lowers the score for {lower} of {len(means)} methods",
        "Six blood datasets; bar = mean over folds. Folds differ in which cells are tested, so "
        "the gap is a rough comparison.",
    )
    figure_caption(
        fig,
        "Labels are 12 lineages shared by all labs. Pretrained models saw every one of these cells "
        "in training, so for them the held-out lab is not truly unseen.",
        y=-0.13,
    )
    out = ROOT / "assets" / f"m4_lab_holdout_{args.name}.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
