"""Harmony without seeing the test cells: fit on training donors, map new donors onto it.

`cellmap.baselines.harmony_correct` is transductive: it corrects all cells together, so the
held-out donors' cells shape the correction (through their expression and donor label, never
their cell type). Every other method in this project is fitted on training donors only. This
module removes that asymmetry, following the Symphony algorithm (Kang et al., Nature
Communications 2021), implemented here from the paper on top of harmonypy:

1. Reference: run Harmony on the training donors only. Keep its corrected coordinates, its soft
   cluster assignments R, and two summaries per cluster k: the amount of reference in it,
   N_k = sum_i R_ik, and the sum of corrected reference coordinates, C_k = sum_i R_ik z_i.
2. Query: every other cell is already in the reference PCA space (PCA is fitted on training
   donors). Each query cell is softly assigned to the reference clusters by cosine distance to
   the cluster centroids, with the same softness as Harmony (sigma = 0.1):
       R_qk proportional to exp( -2 (1 - cos(z_q, Y_k)) / sigma ).
3. Correction: within each cluster, a ridge regression of the query coordinates on
   [intercept, query batch one-hot], weighted by R_qk, where the intercept also counts the
   reference (N_k added to its weight, C_k to its target). The batch terms are what gets
   removed; the intercept stays. So a query batch is pulled toward where the reference has
   that cluster, and the reference itself never moves.

Differences from the paper, stated plainly: the ridge penalty on batch terms is fixed at 1
(Symphony's default); the cluster centroids are recomputed from the corrected reference
coordinates as Symphony does; there is no separate reference-compression file format.
"""

import harmonypy
import numpy as np
import pandas as pd


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def harmony_reference_map(
    pcs: np.ndarray,
    batch: np.ndarray,
    train_mask: np.ndarray,
    seed: int = 0,
    theta: float = 2.0,
    sigma: float = 0.1,
    ridge: float = 1.0,
) -> np.ndarray:
    """Harmony-corrected coordinates for all cells, fitted on the training cells only.

    Training cells get the coordinates of a Harmony run on training cells alone. Every other
    cell is mapped onto that frozen reference (see the module docstring); it cannot change
    the reference or any training cell's coordinates. Same shape and row order as `pcs`.
    """
    pcs = np.asarray(pcs, dtype=np.float64)
    batch = np.asarray(batch).astype(str)
    train_mask = np.asarray(train_mask, dtype=bool)
    if pcs.ndim != 2 or batch.shape != (pcs.shape[0],) or train_mask.shape != batch.shape:
        raise ValueError("pcs, batch and train_mask must describe the same cells")
    if len(np.unique(batch[train_mask])) < 2:
        raise ValueError("the reference needs at least two batches")

    reference = harmonypy.run_harmony(
        pcs[train_mask],
        pd.DataFrame({"batch": batch[train_mask]}),
        "batch",
        theta=theta,
        sigma=sigma,
        random_state=seed,
        verbose=False,
        ncores=1,
    )
    z_ref = np.asarray(reference.Z_corr, dtype=np.float64)  # cells x dims
    r_ref = np.asarray(reference.R, dtype=np.float64)  # cells x clusters
    n_k = r_ref.sum(axis=0)  # reference weight per cluster
    c_k = r_ref.T @ z_ref  # clusters x dims
    centroids = _unit_rows(r_ref.T @ _unit_rows(z_ref))  # clusters x dims

    out = np.empty_like(pcs)
    out[train_mask] = z_ref
    query = ~train_mask
    if not query.any():
        return out.astype(np.float32)

    z_q = pcs[query]
    dist = 2.0 * (1.0 - _unit_rows(z_q) @ centroids.T)
    logits = -dist / sigma
    logits -= logits.max(axis=1, keepdims=True)
    r_q = np.exp(logits)
    r_q /= r_q.sum(axis=1, keepdims=True)

    q_batches, codes = np.unique(batch[query], return_inverse=True)
    design = np.zeros((len(z_q), 1 + len(q_batches)))
    design[:, 0] = 1.0
    design[np.arange(len(z_q)), 1 + codes] = 1.0
    penalty = np.diag([0.0] + [ridge] * len(q_batches))

    corrected = z_q.copy()
    for k in range(r_q.shape[1]):
        weights = r_q[:, k][:, None]
        lhs = design.T @ (weights * design) + penalty
        lhs[0, 0] += n_k[k]
        rhs = design.T @ (weights * z_q)
        rhs[0] += c_k[k]
        coef = np.linalg.solve(lhs, rhs)
        coef[0] = 0.0  # keep the cluster's position; remove only the batch terms
        corrected -= weights * (design @ coef)
    out[query] = corrected
    return out.astype(np.float32)
