"""Quality control: remove low-quality cells, rare cell types and unexpressed genes."""

import anndata as ad
import numpy as np
import pandas as pd


def add_qc_metrics(adata: ad.AnnData, gene_symbols: str = "feature_name") -> None:
    """Add per-cell QC columns to `adata.obs` in place, computed from raw counts in X.

    total_counts: number of UMIs in the cell.
    n_genes: number of genes with at least one UMI.
    pct_mt: percent of UMIs from mitochondrial genes (symbols starting with "MT-").
    """
    x = adata.X
    is_mt = adata.var[gene_symbols].astype(str).str.upper().str.startswith("MT-").to_numpy()
    total = np.asarray(x.sum(axis=1)).ravel()
    mt = np.asarray(x[:, is_mt].sum(axis=1)).ravel()
    adata.obs["total_counts"] = total
    adata.obs["n_genes"] = np.asarray((x > 0).sum(axis=1)).ravel()
    adata.obs["pct_mt"] = 100.0 * mt / np.maximum(total, 1)


def cell_filter_report(
    obs: pd.DataFrame, min_counts: int, min_genes: int, max_pct_mt: float
) -> tuple[np.ndarray, dict[str, int]]:
    """Return a keep-mask over cells and how many cells each threshold fails.

    The thresholds are fixed numbers from the config. Nothing is estimated from the data, so
    applying them before the donor split cannot leak information from test donors.
    """
    low_counts = (obs["total_counts"] < min_counts).to_numpy()
    low_genes = (obs["n_genes"] < min_genes).to_numpy()
    high_mt = (obs["pct_mt"] > max_pct_mt).to_numpy()
    keep = ~(low_counts | low_genes | high_mt)
    report = {
        "cells_before": int(len(obs)),
        "fail_min_counts": int(low_counts.sum()),
        "fail_min_genes": int(low_genes.sum()),
        "fail_max_pct_mt": int(high_mt.sum()),
        "cells_after": int(keep.sum()),
    }
    return keep, report


def rare_cell_types(obs: pd.DataFrame, min_cells: int, label: str = "cell_type") -> list[str]:
    """Cell types with fewer than `min_cells` cells, sorted by name."""
    counts = obs[label].astype(str).value_counts()
    return sorted(counts[counts < min_cells].index.tolist())


def genes_expressed_in(adata: ad.AnnData, cell_mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Mask of genes detected in at least `min_cells` of the cells selected by `cell_mask`.

    Pass the training-donor mask: which genes to keep is a choice fitted to data, so it must
    not look at validation or test donors.
    """
    n_cells = np.asarray((adata.X[cell_mask] > 0).sum(axis=0)).ravel()
    return n_cells >= min_cells
