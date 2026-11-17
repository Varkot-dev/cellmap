"""Measure what a method does with a cell type it never saw in training."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def self_neighbor_rate(emb: np.ndarray, labels: np.ndarray, target: str, k: int = 15) -> float:
    """Mean fraction of a `target` cell's k nearest neighbours (among the given cells) that are
    also `target`.

    High means the unseen type still forms its own group; near the type's share of the cells
    means it is scattered among the others. Compute it on test cells only.
    """
    labels = np.asarray(labels).astype(str)
    is_target = labels == target
    if is_target.sum() < 2:
        raise ValueError(f"need at least 2 {target!r} cells")
    k = min(k, len(labels) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb[is_target])
    neighbours = idx[:, 1:]  # drop the cell itself
    return float((labels[neighbours] == target).mean())


def nearest_known_type(
    emb: np.ndarray, labels: np.ndarray, reference: np.ndarray, query: np.ndarray, k: int = 15
) -> pd.Series:
    """For the `query` cells, the share of each training label they are assigned by kNN.

    `reference` marks the cells whose labels the classifier may use (training cells, which by
    construction lack the held-out type). The result says where a method "puts" the new type.
    """
    labels = np.asarray(labels).astype(str)
    nn = NearestNeighbors(n_neighbors=k).fit(emb[reference])
    _, idx = nn.kneighbors(emb[query])
    ref_labels = labels[reference][idx]
    votes = pd.DataFrame(ref_labels).apply(lambda row: row.value_counts().idxmax(), axis=1)
    return votes.value_counts(normalize=True)
