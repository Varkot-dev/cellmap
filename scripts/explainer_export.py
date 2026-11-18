"""Build the scroll-driven explainer, the project's front page, as one self-contained file.

Usage:
    python scripts/explainer_export.py --name pbmc

Uses the same 3D data as the atlas (cellmap.atlas.build_payload) plus the results, read from
results/m1 to results/m6. Every number shown on the page comes from those files, including the
headline verdict, which applies the comparison rule fixed in docs/PLAN.md: a method is better
only if it beats both baselines in at least 4 of the 5 donor splits.

Writes:
    site/index.html
    data/embeddings/<name>/explainer_fragment.html  (same page without the document wrapper)
"""

import argparse
from pathlib import Path

import pandas as pd

from cellmap.atlas import build_payload, build_results, write_page

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("pca", "pca_harmony")
MEASURES = ("knn_macro_f1_fine", "scib_total")


def beats_both_baselines(results: list[dict], method: str, measure: str) -> bool:
    """True if `method` is higher than both baselines in at least 4 of 5 splits on `measure`."""
    by_key = {r["key"]: r for r in results}
    mine = by_key[method][f"{measure}_splits"]
    for base in BASELINES:
        theirs = by_key[base][f"{measure}_splits"]
        if sum(a > b for a, b in zip(mine, theirs, strict=True)) < 4:
            return False
    return True


M4_LABELS = {
    "pca": "HVG + PCA",
    "pca_harmony": "PCA + Harmony",
    "vae": "VAE (hand-written)",
    "transformer": "Transformer",
    "census_scvi": "Census scVI",
    "geneformer_ft": "Geneformer (fine-tuned)",
    "tf_sapiens": "TranscriptFormer",
}


def build_m4(root: Path) -> dict | None:
    """Held-out-lab results from results/m4, in the same shape as the main results."""
    files = sorted((root / "results" / "m4" / "m4_blood").glob("metrics_*.csv"))
    if not files:
        return None
    data = pd.concat(pd.read_csv(f) for f in files)
    lab = data[data["scheme"] == "dataset"]
    rows = []
    for method, group in lab.groupby("method"):
        if len(group) != lab["fold"].nunique():
            continue
        group = group.sort_values("fold")
        vals = [round(float(v), 3) for v in group["knn_macro_f1_lineage"]]
        rows.append(
            {
                "key": method,
                "label": M4_LABELS.get(method, method),
                "knn_macro_f1_lineage": round(float(group["knn_macro_f1_lineage"].mean()), 3),
                "knn_macro_f1_lineage_splits": vals,
            }
        )
    summary = pd.read_json(root / "results" / "m4" / "m4_blood" / "summary.json", typ="series")
    return {
        "results": rows,
        "n_labs": int(lab["fold"].nunique()),
        "cells": int(summary["final"]["cells"]),
        "donors": int(summary["final"]["donors"]),
        "lineages": int(summary["final"]["lineages"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    data = build_payload(ROOT, args.name)
    results = build_results(ROOT, args.name)
    others = [r["key"] for r in results if r["key"] not in BASELINES and r["n_splits"] == 5]
    winners = [m for m in others if any(beats_both_baselines(results, m, x) for x in MEASURES)]
    data["results"] = results
    data["verdict_total"] = len(others)
    data["verdict_wins"] = len(winners)
    data["m4"] = build_m4(ROOT)
    if data["m4"]:
        for key in ("n_labs", "cells", "donors", "lineages"):
            data[f"m4_{key}"] = data["m4"][key]
    size = write_page(
        ROOT / "site" / "explainer_template.html",
        data,
        [
            ROOT / "site" / "index.html",
            ROOT / "data" / "embeddings" / args.name / "explainer_fragment.html",
        ],
        "/*__EXPLAINER_DATA__*/null",
    )
    print(f"wrote site/index.html ({size / 1e6:.1f} MB); {len(winners)} of {len(others)} "
          f"methods beat both baselines: {winners}")  # fmt: skip


if __name__ == "__main__":
    main()
