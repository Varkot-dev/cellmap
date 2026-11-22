"""Model audit: which genes does an embedding rely on, and does it encode the donor?

Two tools, both model-agnostic: they take an `embed_fn` that maps a (cells x genes) count
matrix to an embedding, so the same code audits the transformer, PCA or anything else.

1. Gene knockout. Set one gene's count to zero in the input, embed the cells again, and
   measure how far each cell moves:

       effect(cell, gene) = 1 - cos(h, h with the gene knocked out)

   0 means the gene made no difference; 1 means the new vector is orthogonal to the old one.
   This says what the MODEL uses. It is not a prediction of what happens to a real cell whose
   gene is switched off: no biology is simulated.

2. A donor-level probe. Fit a logistic regression from cell embeddings to a label that
   belongs to the donor (here: whether the donor's cells express HLA-DRB5 at all), holding
   out one donor at a time. A donor's score is the mean predicted probability over its cells,
   and the AUROC over donors says how well the embedding separates the two groups of people.

Worked example of the knockout effect: h = (1, 0) and, after the knockout, (0.8, 0.6). Both
have length 1, cos = 0.8, so the effect is 0.2. If the knockout gave (0, 1), cos = 0 and the
effect is 1.
"""

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EmbedFn = Callable[[sp.csr_matrix], np.ndarray]


def knockout(counts: sp.spmatrix, genes: int | list[int] | np.ndarray) -> sp.csr_matrix:
    """A copy of `counts` with the given gene columns set to zero (and removed from storage)."""
    out = sp.csr_matrix(counts, copy=True)
    drop = np.isin(out.indices, np.atleast_1d(genes))
    out.data[drop] = 0
    out.eliminate_zeros()
    return out


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cos between matching rows of `a` and `b`. A zero row against anything gives 1."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.where(norms > 0, norms, 1.0)
    return 1.0 - np.where(norms > 0, cos, 0.0)


def knockout_effects(
    embed_fn: EmbedFn,
    counts: sp.spmatrix,
    genes: np.ndarray,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """effect[i, j] = 1 - cos(h_j, h_j with gene genes[i] knocked out), shape (genes, cells).

    Only the cells that express a gene are embedded again; for the others the input is
    unchanged, so their effect is exactly 0. `base` is the embedding of `counts` if already
    computed.
    """
    csr = sp.csr_matrix(counts)
    base = embed_fn(csr) if base is None else base
    csc = csr.tocsc()
    effects = np.zeros((len(genes), csr.shape[0]))
    for i, gene in enumerate(genes):
        rows = csc.indices[csc.indptr[gene] : csc.indptr[gene + 1]]
        rows = rows[csc.data[csc.indptr[gene] : csc.indptr[gene + 1]] > 0]
        if len(rows) == 0:
            continue
        moved = embed_fn(knockout(csr[rows], gene))
        effects[i, rows] = cosine_distance(base[rows], moved)
    return effects


def overlap_at_k(ranking: list | np.ndarray, reference: list | np.ndarray, k: int = 20) -> float:
    """Share of the first k items of `ranking` that are among the first k of `reference`."""
    if min(len(ranking), len(reference)) < k:
        raise ValueError(f"both rankings need at least {k} items")
    return len(set(list(ranking)[:k]) & set(list(reference)[:k])) / k


def matched_controls(
    detection: np.ndarray,
    target: int,
    allowed: np.ndarray,
    n_pool: int = 50,
    n_draw: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """`n_draw` control genes drawn at random from the `n_pool` allowed genes whose detection
    rate is closest to the target gene's (the target itself is never drawn)."""
    allowed = np.asarray(allowed, dtype=bool).copy()
    allowed[target] = False
    candidates = np.flatnonzero(allowed)
    if len(candidates) < n_pool:
        raise ValueError("fewer allowed genes than the pool size")
    gap = np.abs(detection[candidates] - detection[target])
    pool = candidates[np.argsort(gap, kind="stable")[:n_pool]]
    return np.sort(np.random.default_rng(seed).choice(pool, size=n_draw, replace=False))


def donor_probe(
    emb: np.ndarray, donors: np.ndarray, label_by_donor: dict[str, int], c: float = 1.0
) -> dict:
    """Leave-one-donor-out logistic regression from cell embeddings to a donor-level label.

    For each donor: standardise on the other donors' cells, fit, and average the predicted
    probability of label 1 over the held-out donor's cells. Returns the AUROC over donors and
    each donor's score. Balanced class weights, so the larger group does not win by size.
    """
    donors = np.asarray(donors).astype(str)
    names = sorted(set(donors))
    y_cell = np.array([label_by_donor[d] for d in donors])
    scores = {}
    for held in names:
        fit = donors != held
        if len(set(y_cell[fit])) < 2:
            raise ValueError("each fold needs both labels among the training donors")
        scaler = StandardScaler().fit(emb[fit])
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(emb[fit]), y_cell[fit])
        scores[held] = float(clf.predict_proba(scaler.transform(emb[~fit]))[:, 1].mean())
    y = [label_by_donor[d] for d in names]
    return {"auroc": float(roc_auc_score(y, [scores[d] for d in names])), "scores": scores}
