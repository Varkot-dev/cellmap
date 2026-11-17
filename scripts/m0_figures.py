"""M0 figure: how the donor-level split divides cells and cell types.

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

from cellmap.plotting import INK_SECONDARY, SPLIT_COLORS, set_style
from cellmap.splits import SPLITS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    results = ROOT / "results" / "m0" / args.name
    donors = pd.read_csv(results / "donors.csv", index_col=0)
    types = pd.read_csv(results / "cell_types.csv", index_col=0)

    set_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.75}
    )

    # Left: cells per donor, grouped by split.
    order = pd.concat([donors[donors["split"] == s].sort_values("cells") for s in reversed(SPLITS)])
    labels = [
        f"{d}  (v3)" if "v3" in a else d for d, a in zip(order.index, order["assay"], strict=True)
    ]
    ax1.barh(labels, order["cells"], color=order["split"].map(SPLIT_COLORS), height=0.72)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("cells after quality control")
    ax1.set_title("Every donor belongs to exactly one split")
    handles = []
    for s in SPLITS:
        sub = donors[donors["split"] == s]
        label = f"{s}: {len(sub)} donors, {sub['cells'].sum():,} cells"
        handles.append(Patch(color=SPLIT_COLORS[s], label=label))
    ax1.legend(handles=handles, loc="center right")
    ax1.text(
        0,
        -0.16,
        "(v3) marks the two donors run on 10x 3' v3; all others are 10x 3' v2.",
        transform=ax1.transAxes,
        color=INK_SECONDARY,
        fontsize=9,
    )

    # Right: share of each split made up by each cell type.
    share = 100 * types[list(SPLITS)] / types[list(SPLITS)].sum()
    share = share.iloc[::-1]
    y = range(len(share))
    ax2.hlines(y, share.min(axis=1), share.max(axis=1), color="#c3c2b7", linewidth=1.5, zorder=1)
    for s in SPLITS:
        ax2.scatter(
            share[s],
            y,
            s=46,
            color=SPLIT_COLORS[s],
            edgecolor="#fcfcfb",
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
    ax2.set_title("Cell-type make-up differs between splits")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, -0.11), ncol=3, handletextpad=0.2)

    out = ROOT / "assets" / f"m0_split_{args.name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
