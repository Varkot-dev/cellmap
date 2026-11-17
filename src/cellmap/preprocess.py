"""Normalization of raw counts."""

import anndata as ad
import numpy as np
import scipy.sparse as sp


def normalize_log1p(adata: ad.AnnData, target_sum: float = 1e4) -> None:
    """Library-size normalize and log-transform in place, keeping raw counts.

    Raw counts are copied to `adata.layers["counts"]` (the count-based models need them).
    Then each cell is scaled so its counts sum to `target_sum`, and X becomes log(1 + x).

    Every cell is transformed using only its own total, and `target_sum` is a fixed constant,
    so this step is identical whether it is run before or after the donor split.
    """
    counts = sp.csr_matrix(adata.X, dtype=np.float32)
    adata.layers["counts"] = counts.copy()
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = target_sum / np.maximum(totals, 1.0)
    x = sp.diags(scale.astype(np.float32)) @ counts
    x.data = np.log1p(x.data)
    adata.X = sp.csr_matrix(x, dtype=np.float32)
