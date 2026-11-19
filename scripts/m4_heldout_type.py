"""M4: what each method does with a cell type it never saw.

Usage:
    python scripts/m4_heldout_type.py --name m4_blood

For each held-out lineage: take donor split seed 0, remove that lineage's cells from the
training cells, fit each of the project's methods on what is left, embed every cell, and ask
two questions about the held-out lineage's TEST cells:

    self_neighbor_rate   do they still group together? (share of their 15 nearest test
                         neighbours that are the same lineage)
    assigned_as          which known lineage does a kNN classifier, trained without them,
                         call them?

For reference the same self_neighbor_rate is computed on the normal benchmark embedding of
split seed 0, where the lineage was seen in training. The pretrained models are not included:
they were trained on all of these cells, so nothing is held out from them.

Writes:
    results/m4/<name>/heldout_types.csv
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellmap.benchmark import donor_split, fit_and_embed
from cellmap.novelty import nearest_known_type, self_neighbor_rate
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = ("B cell", "dendritic cell")
METHODS = ("pca", "pca_harmony", "vae")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="m4_blood")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    lineage = adata.obs["lineage"].astype(str).to_numpy()
    split = donor_split(adata.obs, 0)
    test = split == "test"
    emb_dir = ROOT / "data" / "embeddings" / args.name

    rows = []
    for target in HELD_OUT:
        is_target = lineage == target
        fit_split = split.copy()
        fit_split[(split == "train") & is_target] = "excluded"
        reference = fit_split == "train"
        for method in METHODS:
            t0 = time.perf_counter()
            emb = fit_and_embed(method, adata, fit_split, seed=0, device="cpu")
            seconds = time.perf_counter() - t0
            seen = np.load(emb_dir / f"{method}_donor_seed0.npy")
            assigned = nearest_known_type(emb, lineage, reference, test & is_target)
            row = {
                "held_out": target,
                "method": method,
                "n_test_cells": int((test & is_target).sum()),
                "share_of_test_cells": float(is_target[test].mean()),
                "self_neighbor_rate_unseen": self_neighbor_rate(emb[test], lineage[test], target),
                "self_neighbor_rate_seen": self_neighbor_rate(seen[test], lineage[test], target),
                "assigned_as_top": assigned.index[0],
                "assigned_as_top_share": float(assigned.iloc[0]),
                "assigned_as": json.dumps(assigned.round(4).to_dict()),
                "seconds": seconds,
                "commit": git_commit(ROOT),
            }
            rows.append(row)
            print(f"{target} | {method}: unseen {row['self_neighbor_rate_unseen']:.3f} "
                  f"seen {row['self_neighbor_rate_seen']:.3f} -> {row['assigned_as_top']} "
                  f"({row['assigned_as_top_share']:.2f})", flush=True)  # fmt: skip
            out = ROOT / "results" / "m4" / args.name
            pd.DataFrame(rows).to_csv(out / "heldout_types.csv", index=False)


if __name__ == "__main__":
    main()
