"""Figure: how the donor-level split divides cells and cell types.

Usage:
    python scripts/m0_figures.py --name pbmc

Reads results/m0/<name>/donors.csv and cell_types.csv (written by m0_prepare_data.py) and
writes assets/m0_split_<name>.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from cellmap.plotting import GRID, PAPER, SPLIT_COLORS, apply, figure_caption, title
from cellmap.splits import SPLITS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    results = ROOT / "results" / "m0" / args.name
    donors = pd.read_csv(results / "donors.csv", index_col=0)
    types = pd.read_csv(results / "cell_types.csv", index_col=0)

    apply()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 6.5), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.85}
    )

    # Left: cells per donor, grouped by split.
    order = pd.concat([donors[donors["split"] == s].sort_values("cells") for s in reversed(SPLITS)])
    labels = [
        f"{d}  (v3)" if "v3" in a else d for d, a in zip(order.index, order["assay"], strict=True)
    ]
    ax1.barh(labels, order["cells"], color=order["split"].map(SPLIT_COLORS), height=0.72)
    ax1.set_xlabel("cells after quality control")
    title(ax1, "Every donor belongs to exactly one split", "Cells per donor after quality control")
    handles = []
    for s in SPLITS:
        sub = donors[donors["split"] == s]
        label = f"{s}: {len(sub)} donors, {sub['cells'].sum():,} cells"
        handles.append(Patch(color=SPLIT_COLORS[s], label=label))
    ax1.legend(handles=handles, loc="center right")

    # Right: share of each split made up by each cell type.
    share = 100 * types[list(SPLITS)] / types[list(SPLITS)].sum()
    share = share.iloc[::-1]
    y = range(len(share))
    ax2.hlines(y, share.min(axis=1), share.max(axis=1), color=GRID, linewidth=2, zorder=1)
    for s in SPLITS:
        ax2.scatter(
            share[s],
            y,
            s=46,
            color=SPLIT_COLORS[s],
            edgecolor=PAPER,
            linewidth=1.2,
            label=s,
            zorder=2,
        )
    ax2.set_yticks(
        list(y), [t.replace(", alpha-beta", "").replace(", human", "") for t in share.index]
    )
    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_xticks([0, 1, 3, 10, 30, 60], ["0", "1", "3", "10", "30", "60"])
    ax2.set_xlim(-0.1, 70)
    ax2.set_xlabel("percent of the split's cells (log scale above 1%)")
    title(
        ax2,
        "Cell-type make-up differs between splits",
        "Share of each split's cells, one dot per split",
    )
    ax2.tick_params(axis="y", length=0)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, -0.11), ncol=3, handletextpad=0.2)

    figure_caption(
        fig,
        "(v3) marks the two donors run on 10x 3' v3 chemistry; all others are 10x 3' v2. "
        "Both v3 donors fell in train for split seed 0.",
    )
    out = ROOT / "assets" / f"m0_split_{args.name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
