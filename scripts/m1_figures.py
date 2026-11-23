"""Figures: what Harmony changes relative to plain PCA.

Usage:
    python scripts/m1_figures.py --name pbmc

Reads results/m1/<name>/metrics_by_split.csv and the split-seed-0 embeddings under
data/embeddings/<name>/ (both written by m1_baselines.py).

Writes:
    assets/m1_tradeoff_<name>.png   scIB trade-off and kNN scores, one line per donor split
    assets/m1_batch_umap_<name>.png the same cells before and after Harmony, by chemistry
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
import umap

from cellmap.plotting import (
    ACCENT,
    ACCENT_2,
    GRID,
    INK,
    METHOD_COLORS,
    METHOD_NAMES,
    PAPER,
    apply,
    figure_caption,
    title,
)

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def tradeoff_figure(metrics: pd.DataFrame, out: Path) -> None:
    wide = metrics.pivot(index="split_seed", columns="method")
    n = len(wide)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    for seed in wide.index:
        xs = [wide["scib_batch_mean"][m][seed] for m in ("pca", "pca_harmony")]
        ys = [wide["scib_bio_mean"][m][seed] for m in ("pca", "pca_harmony")]
        ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1.6})  # fmt: skip
    for method in ("pca", "pca_harmony"):
        ax.scatter(wide["scib_batch_mean"][method], wide["scib_bio_mean"][method], s=52,
                   color=METHOD_COLORS[method], edgecolor=PAPER, linewidth=1.2, zorder=3,
                   label=METHOD_NAMES[method])  # fmt: skip
    ax.set_xlabel("batch correction score (higher = donors better mixed)")
    ax.set_ylabel("biological conservation score")
    ax.legend(loc="lower right")
    title(
        ax,
        "Harmony raised both scIB scores",
        f"Test donors only; one linked pair per donor split (n = {n})",
    )

    for ax, metric, label, head in (
        (axes[1], "knn_accuracy_fine", "kNN accuracy, 19 fine cell types",
         "but lowered fine-label kNN accuracy"),
        (axes[2], "knn_macro_f1_fine", "kNN macro F1, 19 fine cell types",
         "with no clear change in macro F1"),
    ):  # fmt: skip
        for seed in wide.index:
            a, b = wide[metric]["pca"][seed], wide[metric]["pca_harmony"][seed]
            ax.plot([0, 1], [a, b], color=GRID, linewidth=1.6, zorder=1)
            ax.scatter([0], [a], s=52, color=INK, edgecolor=PAPER, linewidth=1.2, zorder=3)
            ax.scatter([1], [b], s=52, color=ACCENT, edgecolor=PAPER, linewidth=1.2, zorder=3)
        diff = wide[metric]["pca_harmony"] - wide[metric]["pca"]
        ax.set_xticks([0, 1], [METHOD_NAMES["pca"], METHOD_NAMES["pca_harmony"]])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(label)
        title(
            ax, head, f"Mean change {diff.mean():+.3f}; lower in {(diff < 0).sum()} of {n} splits"
        )

    figure_caption(
        fig,
        "Each donor split holds out 4 different donors. Gene selection and PCA are fitted on the "
        "13 training donors. Harmony sees every cell's expression and donor, never its cell type.",
        y=-0.04,
    )
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def umap_coords(emb: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(emb)
    np.save(cache, coords)
    return coords


def batch_umap_figure(name: str, out: Path) -> None:
    emb_dir = ROOT / "data" / "embeddings" / name
    obs = ad.read_h5ad(ROOT / "data" / "processed" / f"{name}.h5ad", backed="r").obs
    is_v3 = obs["assay"].astype(str).str.contains("v3").to_numpy()
    order = np.random.default_rng(0).permutation(len(obs))  # so neither group is drawn on top

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"wspace": 0.08})
    heads = {
        "pca": "Before: chemistry splits the map",
        "pca_harmony": "After Harmony: chemistries overlap",
    }
    for ax, method in zip(axes, ("pca", "pca_harmony"), strict=True):
        coords = umap_coords(np.load(emb_dir / f"{method}.npy"), emb_dir / f"umap_{method}.npy")
        colors = np.where(is_v3, ACCENT, ACCENT_2)
        ax.scatter(coords[order, 0], coords[order, 1], s=1.2, c=colors[order], linewidths=0,
                   alpha=0.55, rasterized=True)  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        title(
            ax, heads[method], f"UMAP of the {METHOD_NAMES[method]} embedding, {len(obs):,} cells"
        )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=ACCENT_2, markersize=7,
                   label=f"10x 3' v2 ({(~is_v3).sum():,} cells, 19 donors)"),
        plt.Line2D([], [], marker="o", linestyle="", color=ACCENT, markersize=7,
                   label=f"10x 3' v3 ({is_v3.sum():,} cells, 2 donors)"),
    ]  # fmt: skip
    axes[1].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, -0.12), ncol=2)
    figure_caption(
        fig,
        "Same cells in both panels, split seed 0. Harmony was told the donor, not the chemistry. "
        "UMAP is a picture, not a measurement: the scores are in results/m1/.",
        y=0.0,
    )
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    apply()
    metrics = pd.read_csv(ROOT / "results" / "m1" / args.name / "metrics_by_split.csv")
    assets = ROOT / "assets"
    tradeoff_figure(metrics, assets / f"m1_tradeoff_{args.name}.png")
    batch_umap_figure(args.name, assets / f"m1_batch_umap_{args.name}.png")


if __name__ == "__main__":
    main()
