"""Classical embedding baselines: highly variable genes, PCA, and PCA plus Harmony.

Everything that is estimated from data and can be frozen (which genes to keep, per-gene mean
and standard deviation, the principal axes) is fitted on training cells only and then applied
unchanged to validation and test cells. Harmony is the one exception, and its docstring says why.
"""

import inspect

import anndata as ad
import harmonypy
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def select_hvg(
    adata: ad.AnnData,
    train_mask: np.ndarray,
    n_top_genes: int = 2000,
    batch_key: str | None = "donor_id",
) -> np.ndarray:
    """Boolean mask over genes: the `n_top_genes` most variable genes among training cells.

    Only the rows selected by `train_mask` are ever passed to scanpy, so validation and test
    donors cannot influence which genes every later method gets to see.

    Flavor: scanpy's "seurat" (dispersion-based), which expects the log1p-normalized values in
    X. Genes are binned by mean expression and ranked by how far their variance-to-mean ratio
    lies above other genes of similar mean, so highly expressed genes do not win just by being
    highly expressed. The "seurat_v3" flavor works on raw counts but needs the optional
    scikit-misc package, which this project does not depend on. The flavor is passed explicitly
    because scanpy has announced that its default will change in 2.0.

    With `batch_key` set (default: donor), the ranking is done inside each training donor
    separately and genes are ordered by the number of donors in which they rank in the top
    `n_top_genes`, ties broken by mean normalized dispersion. A gene that varies between donors
    but not between cells of one donor (a sex-linked gene, for example) is then not selected;
    a gene that separates cell types inside every donor is. This needs enough cells per donor
    for a stable variance estimate. Pass `batch_key=None` to pool all training cells instead.
    """
    train_mask = _as_cell_mask(train_mask, adata.n_obs)
    if not 0 < n_top_genes <= adata.n_vars:
        raise ValueError(f"n_top_genes={n_top_genes} must be in 1..{adata.n_vars}")

    obs = pd.DataFrame(index=adata.obs_names[train_mask])
    if batch_key is not None:
        obs["batch"] = adata.obs[batch_key].to_numpy()[train_mask].astype(str)
    train = ad.AnnData(X=adata.X[train_mask], obs=obs, var=pd.DataFrame(index=adata.var_names))

    stats = _hvg_stats(train, n_top_genes, batched=batch_key is not None)
    return stats["highly_variable"].to_numpy(dtype=bool)


def _hvg_stats(train: ad.AnnData, n_top_genes: int, batched: bool) -> pd.DataFrame:
    """scanpy's "seurat" HVG table, on scanpy 1.13+ and on older versions.

    scanpy 1.13 renamed the arguments (`batch_key` became `batch`, `inplace=False` became
    `out=True`) and needs Python 3.13. Older scanpy still runs on Python 3.11 and 3.12, which is
    what Colab and the CI runner use, so both spellings are supported. The algorithm is the same.
    """
    if "batch" in inspect.signature(sc.pp.highly_variable_genes).parameters:
        return sc.pp.highly_variable_genes(
            train,
            flavor="seurat",
            n_top_genes=n_top_genes,
            batch="obs.batch" if batched else None,
            out=True,
        )
    return sc.pp.highly_variable_genes(
        train,
        flavor="seurat",
        n_top_genes=n_top_genes,
        batch_key="batch" if batched else None,
        inplace=False,
    )


def pca_embed(
    adata: ad.AnnData,
    gene_mask: np.ndarray,
    train_mask: np.ndarray,
    n_comps: int = 50,
    max_value: float = 10.0,
) -> np.ndarray:
    """Project every cell onto principal axes fitted on training cells (n_cells x n_comps).

    Steps, each fitted on the training rows only and then applied to all rows:
      1. z-score each selected gene with the training mean and standard deviation, so that a
         gene's weight in the PCA does not depend on how highly it is expressed. Genes that are
         constant in training get a standard deviation of 1 and so stay at zero.
      2. clip to [-max_value, max_value]. A gene that is silent in almost every cell has a tiny
         standard deviation, so the few cells expressing it get enormous z-scores and one such
         gene could own a principal component. Clipping bounds any single value's influence.
      3. subtract the training mean again (clipping moves it slightly off zero), then take the
         top eigenvectors of the training covariance matrix as the axes.

    The eigendecomposition is exact, so there is no random seed: the genes x genes covariance
    is small (2000 x 2000) however many cells there are. Each axis is defined only up to sign;
    the sign is fixed so that the largest-magnitude loading is positive.

    Memory: densifies the selected genes for all cells as float32 (45,000 x 2,000 is 360 MB).
    """
    train_mask = _as_cell_mask(train_mask, adata.n_obs)
    gene_mask = np.asarray(gene_mask, dtype=bool)
    if gene_mask.shape != (adata.n_vars,):
        raise ValueError(f"gene_mask has shape {gene_mask.shape}, expected ({adata.n_vars},)")
    n_train, n_genes = int(train_mask.sum()), int(gene_mask.sum())
    if not 0 < n_comps <= min(n_train - 1, n_genes):
        raise ValueError(
            f"n_comps={n_comps} must be in 1..{min(n_train - 1, n_genes)} "
            f"({n_train} training cells, {n_genes} genes)"
        )

    x = adata.X[:, gene_mask]
    z = x.toarray() if sp.issparse(x) else np.array(x)
    z = z.astype(np.float32, copy=False)

    mean = z[train_mask].mean(axis=0, dtype=np.float64)
    std = z[train_mask].std(axis=0, dtype=np.float64, ddof=1)
    std[std == 0] = 1.0
    z -= mean.astype(np.float32)
    z /= std.astype(np.float32)
    np.clip(z, -max_value, max_value, out=z)
    z -= z[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)

    axes = _principal_axes(z[train_mask], n_comps)
    return z @ axes.T.astype(np.float32)


def harmony_correct(pcs: np.ndarray, batch: np.ndarray, seed: int = 0) -> np.ndarray:
    """Harmony-corrected copy of `pcs`, same shape and row order, with `batch` as the batch.

    Harmony soft-clusters the cells in PCA space while penalizing clusters dominated by one
    batch, then within each cluster shifts every batch's cells toward the cluster's shared
    centroid, and repeats until the clusters stop changing.

    Harmony is transductive. It has no fit-on-train / apply-to-test mode: the correction for a
    cell is computed from the clustering of all the cells passed in, so validation and test
    cells take part in fitting it. It uses their expression (through `pcs`) and their batch
    label, never their cell-type label. Results for this method must be reported with that
    caveat next to them; the PCA it starts from is still fitted on training cells only.

    Library defaults are used for everything except the seed (diversity penalty theta=2, soft
    clustering width sigma=0.1, min(n_cells/30, 100) clusters, at most 10 outer iterations).
    """
    pcs = np.asarray(pcs)
    batch = np.asarray(batch).astype(str)
    if pcs.ndim != 2 or batch.shape != (pcs.shape[0],):
        raise ValueError(f"pcs {pcs.shape} and batch {batch.shape} do not describe the same cells")
    if pcs.shape[0] == pcs.shape[1]:
        # harmonypy guesses the orientation of its input from its shape; a square matrix is
        # ambiguous and would be read as components x cells.
        raise ValueError("pcs is square, so harmonypy cannot tell cells from components")
    if len(np.unique(batch)) < 2:
        raise ValueError("Harmony needs at least two batches")

    result = harmonypy.run_harmony(
        pcs, pd.DataFrame({"batch": batch}), "batch", random_state=seed, verbose=False
    )
    corrected = np.asarray(result.Z_corr)
    if corrected.shape != pcs.shape:
        raise RuntimeError(f"harmonypy returned shape {corrected.shape}, expected {pcs.shape}")
    return corrected.astype(np.float32)


def _as_cell_mask(mask: np.ndarray, n_cells: int) -> np.ndarray:
    """Validate a boolean mask over cells. An integer index array here would be a silent bug."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (n_cells,):
        raise ValueError(
            f"expected a boolean mask of shape ({n_cells},), got {mask.dtype} {mask.shape}"
        )
    if not mask.any():
        raise ValueError("mask selects no cells")
    return mask


def _principal_axes(centered: np.ndarray, n_comps: int, chunk: int = 4096) -> np.ndarray:
    """Top `n_comps` unit eigenvectors (as rows) of the covariance of column-centered data.

    The covariance is accumulated in float64 a block of rows at a time, which avoids both a
    float64 copy of the whole matrix and the precision lost by summing squares in float32.
    """
    n, d = centered.shape
    cov = np.zeros((d, d), dtype=np.float64)
    for start in range(0, n, chunk):
        block = centered[start : start + chunk].astype(np.float64)
        cov += block.T @ block
    cov /= n - 1

    _, vectors = np.linalg.eigh(cov)  # eigenvalues ascending
    axes = vectors[:, ::-1][:, :n_comps].T.copy()
    largest = np.abs(axes).argmax(axis=1)
    axes *= np.sign(axes[np.arange(n_comps), largest])[:, None]
    return axes
