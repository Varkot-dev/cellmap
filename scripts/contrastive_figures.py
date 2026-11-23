"""Figures for the contrastive transformer and the model audit.

Usage:
    python scripts/contrastive_figures.py --name pbmc

Writes:
    assets/contrastive_pbmc.png  fine macro F1 on each donor split: the masked-gene and the
                                 contrastive transformer against the two baselines
    assets/audit_pbmc.png        left: a gene's knockout effect against its expression;
                                 right: each donor's HLA-DRB5 carrier score from monocyte
                                 embeddings, by cohort and chemistry
    assets/audit_sae_pbmc.png    left: every live SAE feature, how much it follows cell type
                                 against how much it follows batch; right: what removing the
                                 top batch features, or random ones, does to iLISI and F1
Reads results/contrastive/, results/m1/ to results/m5/ and results/audit/.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cellmap.plotting import ACCENT, ACCENT_2, GRID, INK, MUTED, apply, figure_caption, title  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
F1 = "knn_macro_f1_fine"
LINES = {
    "pca": ("HVG + PCA", INK, "-"),
    "pca_harmony": ("PCA + Harmony", ACCENT_2, "-"),
    "transformer_contrastive": ("Transformer, contrastive", ACCENT, "-"),
    "transformer": ("Transformer, masked-gene", MUTED, "--"),
}


def contrastive(name: str) -> None:
    res = ROOT / "results"
    frames = [pd.read_csv(res / "contrastive" / name / "metrics_by_split.csv"),
              pd.read_csv(res / "m5" / name / "metrics_by_split.csv")]  # fmt: skip
    frames += [pd.read_csv(res / m / name / "metrics_by_split.csv") for m in ("m1", "m2", "m3")]
    f1 = pd.concat(frames).pivot_table(index="split_seed", columns="method", values=F1)
    report = json.loads((res / "contrastive" / name / "hypotheses.json").read_text())
    gap_mgm = report["C1"]["mean_gap"]
    gap_pca = f1["pca"].mean() - f1["transformer_contrastive"].mean()

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(f1))
    for method, (label, color, style) in LINES.items():
        ax.plot(x, f1[method], style, color=color, linewidth=1.6, marker="o", markersize=5,
                label=label)  # fmt: skip
    ax.set_xticks(x, [f"split {s}" for s in f1.index])
    ax.set_ylabel("kNN macro F1, fine types (held-out donors)")
    ax.set_ylim(0.2, 0.85)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    title(ax, f"A new objective lifts the transformer by {gap_mgm:.2f}; PCA still leads "
          f"by {gap_pca:.2f}",
          "Same architecture, genes and splits; only the training objective changed")  # fmt: skip
    caption = (
        "On split 3 no training donor was sequenced with 10x v3 chemistry and a test donor was; "
        "every method drops there. Contrastive beats masked-gene on 5 of 5 splits and passes "
        "the comparison rule against neither baseline on any metric."
    )
    figure_caption(fig, caption, y=-0.04)
    fig.savefig(ROOT / "assets" / f"contrastive_{name}.png")
    plt.close(fig)


def audit(name: str) -> None:
    res = ROOT / "results" / "audit" / name
    knock = pd.read_csv(res / "knockout.csv")
    knock = knock[knock["model"] == "contrastive"]
    scores = pd.read_csv(res / "donor_scores.csv")
    scores = scores[(scores["knockout"] == "none") & (scores["cell_type"] == "classical monocyte")]
    carriers = pd.read_csv(res / "carriers.csv").set_index("donor")
    probe = pd.read_csv(res / "genotype_probe.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.35})
    marker = knock["marker_rank"] <= 20
    ax1.scatter(knock.loc[~marker, "mean_expression"], knock.loc[~marker, "mean_effect"], s=9,
                color=MUTED, alpha=0.5, label="other candidate genes", linewidths=0)  # fmt: skip
    ax1.scatter(knock.loc[marker, "mean_expression"], knock.loc[marker, "mean_effect"], s=14,
                color=ACCENT, label="a top-20 marker of the type", linewidths=0)  # fmt: skip
    ax1.set_xlabel("mean log expression of the gene in the cell type")
    ax1.set_ylabel("knockout effect, 1 - cos (mean over cells)")
    ax1.legend(loc="upper left")
    rho = knock.groupby("cell_type")[["mean_effect", "mean_expression"]].corr(method="spearman")
    rho = rho.xs("mean_effect", level=1)["mean_expression"].mean()
    title(ax1, "The model leans on what is abundant, not what is specific",
          f"Contrastive transformer, 14 types; Spearman with expression {rho:.2f}")  # fmt: skip

    order = {"pca": 0, "contrastive": 1}
    for method, xpos in order.items():
        s = scores[scores["method"] == method].set_index("donor")
        auc = probe[(probe["method"] == method) & (probe["cell_type"] == "classical monocyte")
                    & (probe["knockout"] == "none")]["auroc"].iloc[0]  # fmt: skip
        for i, (donor, row) in enumerate(s.iterrows()):
            c2 = carriers.loc[donor, "cohort"] == "C2"
            color = ACCENT if row["carrier"] else INK
            jitter = (i % 7 - 3) * 0.03
            ax2.scatter(xpos + jitter, row["score"], s=42, color=color if c2 else "none",
                        edgecolors=color, linewidths=1.3, zorder=3)  # fmt: skip
            if "v3" in carriers.loc[donor, "assay"]:
                ax2.annotate(f"{donor} (v3)", (xpos + jitter, row["score"]), fontsize=8,
                             xytext=(8, -3), textcoords="offset points", color=MUTED)  # fmt: skip
        ax2.text(xpos, 1.06, f"AUROC {auc:.2f}", ha="center", fontsize=9.5, color=INK)
    ax2.set_xticks(list(order.values()), ["HVG + PCA", "Transformer, contrastive"])
    ax2.set_xlim(-0.6, 1.9)
    ax2.set_ylim(-0.05, 1.12)
    ax2.set_ylabel("predicted probability of carrying HLA-DRB5 (donor held out)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=ACCENT, label="carrier"),
        plt.Line2D([], [], marker="o", ls="", color=INK, label="non-carrier"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, label="filled: C2 cohort"),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            color=MUTED,
            label="open: other cohorts",
        ),  # fmt: skip
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=8.5)
    title(ax2, "The transformer's 'genotype' signal is mostly cohort and chemistry",
          "Classical monocytes, 21 donors; a perfect probe puts every carrier above every "
          "non-carrier")  # fmt: skip
    figure_caption(fig, "Left: every common gene knocked out of the input in turn, split 0 test "
                   "cells. Right: knocking out HLA-DRB5 moves the transformer's AUROC from "
                   "0.756 to 0.733 and PCA's from 0.933 to 0.911; no control gene moves either.",
                   y=-0.04)  # fmt: skip
    fig.savefig(ROOT / "assets" / f"audit_{name}.png")
    plt.close(fig)


def sae(name: str) -> None:
    res = ROOT / "results" / "audit" / name
    feats = pd.read_csv(res / "sae_features.csv")
    live = feats[feats["live"]]
    inter = pd.read_csv(res / "sae_intervention.csv").set_index("removal")
    removed = set(map(int, inter.loc["batch features", "features"].split()))
    typed = live["type_feature_for"].fillna("") != ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.35})
    rest = live[~typed & ~live["feature"].isin(removed)]
    ax1.scatter(rest["cell_type_eta2"], rest["batch_score"], s=12, color=MUTED, alpha=0.6,
                linewidths=0, label="other live features")  # fmt: skip
    t = live[typed]
    ax1.scatter(t["cell_type_eta2"], t["batch_score"], s=16, color=ACCENT_2, linewidths=0,
                label="cell-type features")  # fmt: skip
    b = live[live["feature"].isin(removed)]
    ax1.scatter(b["cell_type_eta2"], b["batch_score"], s=26, color=ACCENT, linewidths=0,
                label="the 10 removed (highest batch score)")  # fmt: skip
    ax1.set_xlabel("share of the feature's variance explained by cell type")
    ax1.set_ylabel("share explained by batch, within cell types")
    ax1.set_ylim(0, max(0.2, live["batch_score"].max() * 1.1))
    ax1.legend(loc="upper right")
    title(ax1, "Features follow cell type more than batch; none is a batch feature",
          f"{len(live)} live features of 1,024; batch never explains more than "
          f"{live['batch_score'].max():.2f} of one")  # fmt: skip

    base = inter.loc["none"]
    d = inter.drop(index="none")
    dx = d["scib_ilisi"] - base["scib_ilisi"]
    dy = d["knn_macro_f1_fine"] - base["knn_macro_f1_fine"]
    rnd = d.index.str.startswith("random")
    ax2.scatter(dx[rnd], dy[rnd], s=34, color=MUTED, label="10 random features (10 draws)")
    ax2.scatter(dx[~rnd], dy[~rnd], s=60, color=ACCENT, label="the 10 batch features")
    ax2.axvline(0.03, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)))
    ax2.text(0.031, 0.0, "predicted iLISI gain", fontsize=8.5, color=MUTED, va="bottom")
    ax2.axhline(0, color=GRID, linewidth=0.8)
    ax2.axvline(0, color=GRID, linewidth=0.8)
    ax2.set_xlim(-0.01, 0.04)
    ax2.set_xlabel("change in donor mixing (iLISI), test donors")
    ax2.set_ylabel("change in fine macro F1")
    ax2.legend(loc="lower right")
    title(ax2, "Removing them mixes donors no more than random removals",
          "Split 0; the unedited embedding scores F1 "
          f"{base['knn_macro_f1_fine']:.3f}, iLISI {base['scib_ilisi']:.3f}")  # fmt: skip
    figure_caption(fig, "A sparse autoencoder (1,024 features, 16.5 active per cell, 0.963 of "
                   "the variance explained on validation cells) on the contrastive "
                   "transformer's embedding. Batch: C2 cohort, other 10x v2, 10x v3.",
                   y=-0.04)  # fmt: skip
    fig.savefig(ROOT / "assets" / f"audit_sae_{name}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    apply()
    contrastive(args.name)
    audit(args.name)
    sae(args.name)


if __name__ == "__main__":
    main()
