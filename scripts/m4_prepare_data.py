"""M4: download, quality-control and label six blood datasets for the multi-dataset benchmark.

Usage:
    python scripts/m4_prepare_data.py --config configs/m4_blood.yaml

Splits are not made here: the benchmark makes them (donor splits across all datasets, and
leave-one-dataset-out), so every scheme starts from the same processed cells.

Writes:
    data/raw/<name>.h5ad                    raw download (not committed)
    data/processed/<name>.h5ad              processed data (not committed)
    results/m4/<name>/summary.json          counts at every step
    results/m4/<name>/lineage_by_dataset.csv
    results/m4/<name>/donors.csv
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import pandas as pd

from cellmap import census, qc
from cellmap.config import load_config
from cellmap.labels import to_lineage
from cellmap.preprocess import normalize_log1p

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    name = cfg["name"]

    raw_path = ROOT / "data" / "raw" / f"{name}.h5ad"
    if raw_path.exists():
        adata = ad.read_h5ad(raw_path)
    else:
        adata = census.download_many(cfg)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(raw_path, compression="gzip")

    summary: dict = {
        "config": cfg,
        "versions": {p: version(p) for p in ("anndata", "scanpy", "cellxgene-census")},
        "genes_measured_in_all_datasets": int(adata.uns["genes_measured_in_all_datasets"]),
        "downloaded": {
            "cells": adata.n_obs,
            "donors": int(adata.obs["donor_id"].nunique()),
            "cells_per_dataset": adata.obs["dataset"].value_counts().to_dict(),
        },
    }

    qc.add_qc_metrics(adata)
    keep, report = qc.cell_filter_report(
        adata.obs, cfg["qc"]["min_counts"], cfg["qc"]["min_genes"], cfg["qc"]["max_pct_mt"]
    )
    adata = adata[keep].copy()
    summary["cell_qc"] = report

    adata.obs["lineage"] = to_lineage(adata.obs["cell_type"])
    unplaced = adata.obs["lineage"].isna()
    summary["dropped_unplaceable_label"] = (
        adata.obs.loc[unplaced, "cell_type"].astype(str).value_counts().to_dict()
    )
    adata = adata[~unplaced.to_numpy()].copy()
    rare = qc.rare_cell_types(adata.obs, cfg["qc"]["min_cells_per_type"], label="lineage")
    summary["dropped_rare_lineages"] = {t: int((adata.obs["lineage"] == t).sum()) for t in rare}
    adata = adata[~adata.obs["lineage"].isin(rare).to_numpy()].copy()
    adata.obs["lineage"] = adata.obs["lineage"].astype("category")

    # Light gene filter on ALL cells: "detected in at least 10 cells" uses no labels, and the
    # benchmark reselects the 2,000 informative genes on training donors only for every split.
    detected = qc.genes_expressed_in(
        adata, adata.obs_names.notna(), cfg["qc"]["min_cells_per_gene"]
    )
    summary["gene_filter"] = {"genes_before": adata.n_vars, "genes_after": int(detected.sum())}
    adata = adata[:, detected].copy()
    normalize_log1p(adata, cfg["normalize"]["target_sum"])

    obs = adata.obs
    table = pd.crosstab(obs["lineage"], obs["dataset"])
    table["total"] = table.sum(axis=1)
    donors = obs.groupby("donor_id", observed=True).agg(
        dataset=("dataset", "first"), assay=("assay", "first"), cells=("dataset", "size"),
        median_counts=("total_counts", "median"),
    )  # fmt: skip
    summary["final"] = {
        "cells": adata.n_obs,
        "genes": adata.n_vars,
        "donors": int(obs["donor_id"].nunique()),
        "lineages": int(obs["lineage"].nunique()),
        "per_dataset": {
            ds: {
                "cells": int((obs["dataset"] == ds).sum()),
                "donors": int(obs.loc[obs["dataset"] == ds, "donor_id"].nunique()),
                "assays": sorted(obs.loc[obs["dataset"] == ds, "assay"].astype(str).unique()),
            }
            for ds in sorted(obs["dataset"].unique())
        },
    }
    out = ROOT / "results" / "m4" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    table.to_csv(out / "lineage_by_dataset.csv")
    donors.sort_values(["dataset", "cells"]).to_csv(out / "donors.csv")
    processed = ROOT / "data" / "processed" / f"{name}.h5ad"
    adata.write_h5ad(processed, compression="gzip")

    print(json.dumps({k: summary[k] for k in ("downloaded", "cell_qc", "gene_filter", "final")},
                     indent=1, default=int))  # fmt: skip
    print(table.to_string())


if __name__ == "__main__":
    main()
