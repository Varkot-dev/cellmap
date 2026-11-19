"""Figure: the VAE's code at three KL weights, from no constraint to collapse.

Usage:
    python scripts/m2_figures.py --name pbmc

Reads data/embeddings/<name>/vae_kl_<w>.npy and results/m2/<name>/kl_sweep.csv (both written by
m2_kl_sweep.py). Each panel projects the validation donors' latent means onto their first two
principal components, drawn on one shared scale, so a collapsed code shows up as a dot.

Writes:
    assets/m2_kl_sweep_<name>.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellmap.benchmark import donor_split
from cellmap.plotting import PAPER, apply, figure_caption, title

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
WEIGHTS = (0.0, 1.0, 100.0)
HEADS = {
    0.0: "Weight 0: plain autoencoder",
    1.0: "Weight 1: the standard VAE",
    100.0: "Weight 100: collapse",
}
# Ten coarse types, fixed order and colors (most common first).
COLORS = ["#3C5A78", "#C2703D", "#4E9A8E", "#B8963E", "#8E5A94", "#6E8B3D", "#C4586A", "#5B7FBF",
          "#8A6F5A", "#2F8F5B"]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    apply()

    obs = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad", backed="r").obs
    val = donor_split(obs, 0) == "val"
    coarse = obs["cell_type_coarse"].astype(str).to_numpy()[val]
    order = pd.Series(coarse).value_counts().index.tolist()
    sweep = pd.read_csv(ROOT / "results" / "m2" / args.name / "kl_sweep.csv").set_index("kl_weight")

    projected = {}
    for w in WEIGHTS:
        z = np.load(ROOT / "data" / "embeddings" / args.name / f"vae_kl_{w:g}.npy")[val]
        z = z - z.mean(axis=0)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        projected[w] = z @ vt[:2].T
    lim = max(np.abs(p).max() for p in projected.values()) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), gridspec_kw={"wspace": 0.18})
    shuffle = np.random.default_rng(0).permutation(val.sum())
    for ax, w in zip(axes, WEIGHTS, strict=True):
        p = projected[w][shuffle]
        colors = np.array([COLORS[order.index(c)] for c in coarse[shuffle]])
        ax.scatter(p[:, 0], p[:, 1], s=3, c=colors, linewidths=0, alpha=0.7, rasterized=True)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        row = sweep.loc[w]
        title(
            ax,
            HEADS[w],
            f"KL {row.val_kl:.1f}, {int(row.latent_dims_in_use)} of 10 dims used, "
            f"macro F1 {row.val_knn_macro_f1_fine:.2f}",
        )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS[i], markersize=6, label=name)
        for i, name in enumerate(order)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
               facecolor=PAPER)  # fmt: skip
    figure_caption(
        fig,
        f"Validation donors (split seed 0, {val.sum():,} cells), first two principal components "
        "of each latent code, all panels on the same scale. Macro F1 is kNN on 19 fine types.",
        y=-0.08,
    )
    out = ROOT / "assets" / f"m2_kl_sweep_{args.name}.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
