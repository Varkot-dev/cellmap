"""Build the interactive 3D atlas as one self-contained HTML file.

Usage:
    python scripts/atlas_export.py --name pbmc

For every method with a saved split-seed-0 embedding under data/embeddings/<name>/, computes a
3D UMAP (cached), then writes site/atlas.html: site/template.html with all data inlined.

What is inlined, per cell: 3D coordinates per method (16-bit), cell type (fine and coarse),
donor, chemistry, split, and the expression of a small panel of marker genes (8-bit).
"""

import argparse
from pathlib import Path

from cellmap.atlas import build_payload, write_page

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    data = build_payload(ROOT, args.name)
    size = write_page(
        ROOT / "site" / "template.html",
        data,
        [
            ROOT / "site" / "atlas.html",
            ROOT / "data" / "embeddings" / args.name / "atlas_fragment.html",
        ],
        "/*__ATLAS_DATA__*/null",
    )
    print(f"wrote site/atlas.html ({size / 1e6:.1f} MB), {len(data['methods'])} methods, "
          f"{len(data['genes'])} genes, {data['n']:,} cells")  # fmt: skip


if __name__ == "__main__":
    main()
