"""M5 figure: the transformer learned its training task, but its cell embedding is weak.

Usage:
    python scripts/m5_figures.py --name pbmc

Left: masked-gene accuracy on validation donors per epoch, one line per donor split, against the
best guessing baseline (each gene's most common level). Right: kNN macro F1 on test donors for
every method, one dot per split. Reads results/m1, m2, m3, m5 and m6.

Writes:
    assets/m5_transformer_<name>.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellmap.plotting import ACCENT, GRID, INK, MUTED, PAPER, apply, figure_caption, title

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
NAMES = {
    "pca_harmony": "PCA + Harmony",
    "pca": "HVG + PCA",
    "scvi": "scvi-tools",
    "vae": "VAE (hand-written)",
    "tf_sapiens": "TranscriptFormer",
    "census_scvi": "Census scVI",
    "pca_10": "PCA, 10 dims",
    "linear_ae": "Linear autoencoder",
    "geneformer_ft": "Geneformer (fine-tuned)",
    "transformer": "Transformer (hand-written)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    apply()
    res = ROOT / "results"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.55}
    )
    for seed in range(5):
        h = pd.read_csv(res / "m5" / args.name / f"history_seed{seed}.csv")
        epochs = np.arange(1, len(h) + 1)
        gain = 100 * (h["val_accuracy"] - h["val_gene_majority"])
        ax1.plot(epochs, gain, color=ACCENT, linewidth=1.6, alpha=0.85)
    ax1.axhline(0, color=INK, linewidth=1, linestyle=(0, (3, 3)))
    ax1.text(30, 0.4, "guessing each gene's usual level", ha="right", va="bottom", fontsize=9,
             color=MUTED)  # fmt: skip
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("masked-gene accuracy above guessing (points)")
    ax1.set_ylim(bottom=-1)
    title(ax1, "It learned the training task", "Validation donors, one line per donor split")

    frames = [
        pd.read_csv(res / m / args.name / "metrics_by_split.csv")
        for m in ("m1", "m2", "m3", "m5", "m6")
    ]
    data = pd.concat(frames)
    order = data.groupby("method")["knn_macro_f1_fine"].mean().sort_values().index.tolist()
    for y, method in enumerate(order):
        vals = data.loc[data["method"] == method, "knn_macro_f1_fine"]
        color = ACCENT if method == "transformer" else INK
        ax2.hlines(y, vals.min(), vals.max(), color=GRID, linewidth=2, zorder=1)
        ax2.scatter(vals, [y] * len(vals), s=34, color=color, edgecolor=PAPER, linewidth=1,
                    zorder=2)  # fmt: skip
    ax2.set_yticks(range(len(order)), [NAMES.get(m, m) for m in order])
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("kNN macro F1, 19 fine cell types, held-out donors")
    title(ax2, "but its cell embedding is near the bottom", "One dot per donor split")
    figure_caption(
        fig,
        "Transformer trained for 30 epochs on a Colab T4 GPU, on training donors only. "
        "Every method is scored by the same code on the same five donor splits.",
        y=-0.03,
    )
    out = ROOT / "assets" / f"m5_transformer_{args.name}.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
