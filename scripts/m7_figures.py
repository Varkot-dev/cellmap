"""Figures: over-correction, shared confusions, and marker-gene directions.

Usage:
    python scripts/m7_figures.py --name pbmc

Reads results/m7/<name>/ (written by m7_analyses.py) and, for the no-correction reference,
results/m1/<name>/metrics_by_split.csv (PCA, split seed 0).

Writes:
    assets/m7_overcorrection_<name>.png
    assets/m7_confusions_<name>.png
    assets/m7_markers_<name>.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellmap.atlas import METHODS
from cellmap.plotting import ACCENT, ACCENT_2, INK, MUTED, PAPER, apply, figure_caption, title

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
SHORT = {
    "naive thymus-derived CD4-positive, alpha-beta T cell": "naive CD4 T",
    "central memory CD4-positive, alpha-beta T cell": "central memory CD4 T",
    "effector memory CD8-positive, alpha-beta T cell": "effector memory CD8 T",
    "naive thymus-derived CD8-positive, alpha-beta T cell": "naive CD8 T",
    "classical monocyte": "classical monocyte",
    "non-classical monocyte": "non-classical monocyte",
    "naive B cell": "naive B",
    "memory B cell": "memory B",
    "transitional stage B cell": "transitional B",
    "gamma-delta T cell": "gamma-delta T",
    "mucosal invariant T cell": "MAIT",
    "conventional dendritic cell": "conventional DC",
    "plasmacytoid dendritic cell, human": "plasmacytoid DC",
    "natural killer cell": "NK",
    "CD16-negative, CD56-bright natural killer cell, human": "CD56-bright NK",
    "regulatory T cell": "regulatory T",
}


def short(name: str) -> str:
    return SHORT.get(name, name)


def overcorrection(res: Path, m1: pd.DataFrame, out: Path) -> None:
    sweep = pd.read_csv(res / "harmony_strength.csv")
    pca = m1[(m1["method"] == "pca") & (m1["split_seed"] == 0)].iloc[0]
    x = np.arange(len(sweep))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"wspace": 0.35})
    for col, label, color in (("scib_batch_mean", "scIB batch mixing", ACCENT_2),
                              ("scib_bio_mean", "scIB biology", INK),
                              ("knn_macro_f1_fine", "kNN macro F1, 19 types", ACCENT)):  # fmt: skip
        ax1.plot(x, sweep[col], marker="o", color=color, linewidth=2, label=label)
        ax1.scatter([-1], [pca[col]], marker="s", s=46, color=color, zorder=3)
    ax1.axvline(-0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax1.set_xticks([-1, *x], ["PCA\n(none)", *[f"{t:g}" for t in sweep["theta"]]])
    ax1.set_xlabel("Harmony strength (theta)")
    ax1.legend(loc="lower left", fontsize=9)
    batch_range = sweep["scib_batch_mean"].max() - sweep["scib_batch_mean"].min()
    f1_drop = sweep["knn_macro_f1_fine"].iloc[0] - sweep["knn_macro_f1_fine"].iloc[-1]
    title(
        ax1,
        f"Batch mixing moves {batch_range:.3f}; fine-type F1 falls {f1_drop:.3f}",
        f"From theta {sweep['theta'].iloc[0]:g} to {sweep['theta'].iloc[-1]:g}, split seed 0, "
        "test donors",
    )

    pairs = [c for c in sweep.columns if c.startswith("f1::")]
    for i, col in enumerate(pairs):
        name = col.removeprefix("f1::")
        ax2.plot(x, sweep[col], marker="o", linewidth=1.6, label=short(name),
                 color=[INK, ACCENT, ACCENT_2, MUTED][i // 2], alpha=1.0 if i % 2 == 0 else 0.55,
                 linestyle="-" if i % 2 == 0 else "--")  # fmt: skip
    ax2.set_xticks(x, [f"{t:g}" for t in sweep["theta"]])
    ax2.set_xlabel("Harmony strength (theta)")
    ax2.set_ylabel("per-type kNN F1")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    title(ax2, "What it does to closely related types", "Pairs that share a lineage")
    figure_caption(fig, "Squares: plain PCA, no Harmony. Theta is Harmony's diversity penalty; "
                   "at 0 Harmony still corrects within clusters.", y=-0.04)  # fmt: skip
    fig.savefig(out)
    plt.close(fig)


def confusions(res: Path, out: Path) -> pd.DataFrame:
    conf = pd.read_csv(res / "confusion_pairs.csv")
    methods = [m for m in METHODS if m in set(conf["method"])]
    table = conf.pivot_table(index=["true", "called"], columns="method", values="share_of_true",
                             fill_value=0.0)[methods]  # fmt: skip
    keep = table[(table >= 0.05).all(axis=1)].copy()
    keep["min"] = keep.min(axis=1)
    n_shared = len(keep)
    keep = keep.sort_values("min", ascending=False).drop(columns="min").head(8)
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(keep) + 2.2))
    image = ax.imshow(keep.to_numpy(), cmap="Oranges", vmin=0, vmax=max(0.6, keep.to_numpy().max()),
                      aspect="auto")  # fmt: skip
    ax.set_yticks(range(len(keep)), [f"{short(t)}  called  {short(c)}" for t, c in keep.index])
    ax.set_xticks(range(len(methods)), [METHODS[m] for m in methods], rotation=35, ha="right")
    for i in range(len(keep)):
        for j in range(len(methods)):
            v = keep.iloc[i, j]
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8.5,
                    color=PAPER if v > 0.35 else INK)  # fmt: skip
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="share of the true type")
    sub = (f"The {len(keep)} largest of {n_shared} mistakes made by all methods "
           "(at least 5% of the true type each), split seed 0")  # fmt: skip
    title(ax, "Every method confuses the same neighbours", sub)
    fig.savefig(out)
    plt.close(fig)
    return keep


def markers(res: Path, out: Path) -> None:
    corr = pd.read_csv(res / "marker_correlations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"wspace": 0.25})
    for ax, method, head in ((axes[0], "pca", "PCA: first 10 components"),
                             (axes[1], "vae", "VAE: all 10 latent dimensions")):  # fmt: skip
        grid = corr[corr["method"] == method].pivot(index="gene", columns="dimension",
                                                     values="spearman")  # fmt: skip
        grid = grid.loc[[g for g in corr["gene"].unique() if g in grid.index]]
        image = ax.imshow(grid.to_numpy(), cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
        ax.set_yticks(range(len(grid)), grid.index, fontsize=9)
        ax.set_xticks(range(grid.shape[1]), [str(d + 1) for d in grid.columns])
        ax.set_xlabel("dimension")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        title(ax, head, "Spearman correlation with each marker gene, all cells")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    apply()
    res = ROOT / "results" / "m7" / args.name
    m1 = pd.read_csv(ROOT / "results" / "m1" / args.name / "metrics_by_split.csv")
    assets = ROOT / "assets"
    overcorrection(res, m1, assets / f"m7_overcorrection_{args.name}.png")
    keep = confusions(res, assets / f"m7_confusions_{args.name}.png")
    print(keep.round(2).to_string())
    markers(res, assets / f"m7_markers_{args.name}.png")
    print("wrote 3 figures")


if __name__ == "__main__":
    main()
