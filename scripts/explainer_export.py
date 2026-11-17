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
