"""What each PBMC donor split actually tests: chemistry, size and rare-type counts per split.

An independent review found the five PBMC donor splits are not interchangeable replicates. Two of
the 21 donors are the only 10x 3' v3 samples, and where they land changes what a split tests.
This script records, for every split seed, the test set's donors, chemistry, cell count, median
genes per cell and the count of each cell type, and the share of v3 cells in training. Per-split
results elsewhere should be read against this table.

Writes results/verify/pbmc_split_composition.csv and results/verify/pbmc_split_types.csv.
"""

from pathlib import Path

import anndata as ad
import pandas as pd

from cellmap.benchmark import donor_split
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verify"


def main() -> None:
    obs = ad.read_h5ad(ROOT / "data" / "processed" / "pbmc.h5ad", backed="r").obs
    v3 = obs["assay"].astype(str).str.contains("v3")
    rows, types = [], []
    for seed in range(5):
        split = pd.Series(donor_split(obs, seed), index=obs.index)
        test, train = split == "test", split == "train"
        rows.append(
            {
                "split_seed": seed,
                "test_donors": " ".join(sorted(obs.loc[test, "donor_id"].astype(str).unique())),
                "test_v3_donors": " ".join(
                    sorted(obs.loc[test & v3, "donor_id"].astype(str).unique())
                ),
                "n_test_cells": int(test.sum()),
                "test_share_v3": float(v3[test].mean()),
                "train_share_v3": float(v3[train].mean()),
                "test_median_genes": float(obs.loc[test, "n_genes"].median()),
                "train_median_genes": float(obs.loc[train, "n_genes"].median()),
                "test_platelets": int((obs.loc[test, "cell_type"] == "platelet").sum()),
                "test_types_under_20_cells": int(
                    (obs.loc[test, "cell_type"].value_counts().between(1, 19)).sum()
                ),
                "git_commit": git_commit(ROOT),
            }
        )
        counts = obs.loc[test, "cell_type"].astype(str).value_counts()
        types += [{"split_seed": seed, "cell_type": t, "n_test": int(n)} for t, n in counts.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "pbmc_split_composition.csv", index=False)
    pd.DataFrame(types).to_csv(OUT / "pbmc_split_types.csv", index=False)
    print(table.drop(columns=["git_commit"]).to_string())


if __name__ == "__main__":
    main()
