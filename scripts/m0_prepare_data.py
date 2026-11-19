"""Download, quality-control, split by donor and normalize one dataset.

Usage:
    python scripts/m0_prepare_data.py --config configs/pbmc.yaml

Writes:
    data/raw/<dataset_id>.h5ad        raw download, cached (not committed)
    data/processed/<name>.h5ad        processed data (not committed)
    results/m0/<name>/summary.json    every number reported for this milestone
    results/m0/<name>/donors.csv      one row per donor
    results/m0/<name>/cell_types.csv  cells per cell type and split
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellmap import census, qc
from cellmap.config import load_config
from cellmap.labels import to_coarse
from cellmap.preprocess import normalize_log1p
from cellmap.splits import SPLITS, assert_no_donor_overlap, split_donors

ROOT = Path(__file__).resolve().parents[1]


def load_raw(cfg: dict) -> ad.AnnData:
    path = ROOT / "data" / "raw" / f"{cfg['census']['dataset_id']}.h5ad"
    if path.exists():
        return ad.read_h5ad(path)
    adata = census.download(cfg["census"])
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression="gzip")
    return adata


def subsample_per_donor(adata: ad.AnnData, n: int, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    keep = []
    for idx in adata.obs.groupby("donor_id", observed=True).indices.values():
        keep.extend(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return adata[np.sort(keep)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    name, seed = cfg["name"], cfg["split"]["seed"]

    adata = load_raw(cfg)
    summary: dict = {
        "config": cfg,
        "versions": {p: version(p) for p in ("anndata", "scanpy", "numpy", "cellxgene-census")},
        "raw": {"cells": adata.n_obs, "genes_detected": adata.n_vars},
    }

    if cfg.get("subsample_per_donor"):
        adata = subsample_per_donor(adata, cfg["subsample_per_donor"], seed)
        summary["after_subsample"] = {"cells": adata.n_obs}

    # 1. Cell QC with fixed thresholds.
    qc.add_qc_metrics(adata)
    keep, report = qc.cell_filter_report(
        adata.obs, cfg["qc"]["min_counts"], cfg["qc"]["min_genes"], cfg["qc"]["max_pct_mt"]
    )
    adata = adata[keep].copy()
    summary["cell_qc"] = report

    # 2. Drop cell types too rare to evaluate.
    rare = qc.rare_cell_types(adata.obs, cfg["qc"]["min_cells_per_type"])
    is_rare = adata.obs["cell_type"].astype(str).isin(rare).to_numpy()
    summary["rare_cell_types_dropped"] = {
        t: int((adata.obs["cell_type"].astype(str) == t).sum()) for t in rare
    }
    adata = adata[~is_rare].copy()
    adata.obs["cell_type"] = adata.obs["cell_type"].cat.remove_unused_categories()
    adata.obs["cell_type_coarse"] = to_coarse(adata.obs["cell_type"])

    # 3. Donor-level split.
    assignment = split_donors(
        adata.obs["donor_id"], cfg["split"]["val_fraction"], cfg["split"]["test_fraction"], seed
    )
    adata.obs["split"] = pd.Categorical(
        adata.obs["donor_id"].astype(str).map(assignment), categories=SPLITS
    )
    assert_no_donor_overlap(adata.obs)

    # 4. Gene filter, fitted on training donors only.
    train = (adata.obs["split"] == "train").to_numpy()
    keep_genes = qc.genes_expressed_in(adata, train, cfg["qc"]["min_cells_per_gene"])
    summary["gene_filter"] = {"genes_before": adata.n_vars, "genes_after": int(keep_genes.sum())}
    adata = adata[:, keep_genes].copy()

    # 5. Normalize (raw counts stay in layers["counts"]).
    normalize_log1p(adata, cfg["normalize"]["target_sum"])

    # Tables and summary.
    obs = adata.obs
    donors = (
        obs.groupby("donor_id", observed=True)
        .agg(
            split=("split", "first"),
            assay=("assay", "first"),
            cells=("split", "size"),
            cell_types=("cell_type", "nunique"),
            median_counts=("total_counts", "median"),
            median_genes=("n_genes", "median"),
        )
        .sort_values(["split", "cells"], ascending=[True, False])
    )
    cell_types = pd.crosstab(obs["cell_type"], obs["split"]).reindex(columns=SPLITS)
    cell_types.insert(
        0, "coarse", obs.groupby("cell_type", observed=True)["cell_type_coarse"].first()
    )
    cell_types["total"] = cell_types[list(SPLITS)].sum(axis=1)
    cell_types = cell_types.sort_values("total", ascending=False)

    donor_sets = {s: set(obs.loc[obs["split"] == s, "donor_id"].astype(str)) for s in SPLITS}
    summary["final"] = {
        "cells": adata.n_obs,
        "genes": adata.n_vars,
        "donors": int(obs["donor_id"].nunique()),
        "cell_types_fine": int(obs["cell_type"].nunique()),
        "cell_types_coarse": int(obs["cell_type_coarse"].nunique()),
    }
    summary["splits"] = {
        s: {
            "donors": len(donor_sets[s]),
            "cells": int((obs["split"] == s).sum()),
            "cell_types_fine": int(obs.loc[obs["split"] == s, "cell_type"].nunique()),
            "assays": obs.loc[obs["split"] == s]
            .groupby("assay", observed=True)["donor_id"]
            .nunique()
            .to_dict(),
            "donor_ids": sorted(donor_sets[s]),
        }
        for s in SPLITS
    }
    summary["donors_shared_between_splits"] = {
        f"{a}&{b}": len(donor_sets[a] & donor_sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    summary["cell_types_missing_from_a_split"] = {
        s: cell_types.index[cell_types[s] == 0].tolist() for s in SPLITS
    }

    out = ROOT / "results" / "m0" / name
    out.mkdir(parents=True, exist_ok=True)
    donors.to_csv(out / "donors.csv")
    cell_types.to_csv(out / "cell_types.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")

    processed = ROOT / "data" / "processed" / f"{name}.h5ad"
    processed.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed, compression="gzip")

    print(json.dumps({k: summary[k] for k in ("raw", "cell_qc", "gene_filter", "final")}, indent=2))
    print(donors.to_string())
    print(cell_types.to_string())
    print("donors shared between splits:", summary["donors_shared_between_splits"])


if __name__ == "__main__":
    main()
