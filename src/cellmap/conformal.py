"""Conformal prediction sets for cell-type labels: a classifier that says how sure it is.

A classifier usually returns one label per cell. Split conformal prediction instead returns a
set of labels per cell, with a guarantee: if the calibration cells and the test cells are
exchangeable (any reordering of them is equally likely, as with independent draws from the same
population), then

    P( true label is in the set ) >= 1 - alpha

for a new test cell, whatever the classifier, however wrong it is. A bad classifier gets large
sets, not broken coverage. The guarantee is about the average over test cells (marginal), not
about every cell type separately.

The recipe, for a classifier that outputs class probabilities p(y | x):

1. Pick a nonconformity score s(x, y): large when label y looks wrong for cell x.
2. Compute s(x_i, y_i) on n calibration cells with known labels (cells the classifier was not
   fitted on).
3. q = the ceil((n + 1)(1 - alpha))-th smallest calibration score. The "+ 1" accounts for the
   test cell itself; with too few calibration cells q is infinite and every label is included.
4. For a test cell, the set is every label y with s(x, y) <= q.

Scores implemented (see Angelopoulos and Bates, "A gentle introduction to conformal
prediction", 2021):

- LAC ("least ambiguous classifier", Sadinle, Lei and Wasserman 2019): s(x, y) = 1 - p(y | x).
  Smallest sets on average. Sets can be empty when no label is probable enough, which is
  itself a signal ("this looks like nothing I know").
- APS ("adaptive prediction sets", Romano, Sesia and Candes 2020): s(x, y) = total probability
  of the labels ranked above y, plus U * p(y | x) with U uniform on [0, 1]. Adds labels in order
  of probability until the mass reaches q, so uncertain cells get larger sets. The random U
  makes coverage exact rather than conservative.

Class-conditional ("Mondrian") conformal computes q separately from the calibration cells of
each class, which guarantees coverage within every class rather than only on average. It needs
enough calibration cells per class.

Grouped cells (donors, labs). Cells from one donor or lab are not exchangeable with cells from
another, so pooling calibration cells and taking the usual quantile does not guarantee coverage
for a cell from a NEW group: with few groups it under-covers. Hierarchical conformal prediction
(Lee, Barber and Willett, "Distribution-free inference with hierarchical data", arXiv
2306.06342; see also Dunn, Wasserman and Ramdas, JASA 2023) treats groups as the exchangeable
units: each of the K calibration groups gets total weight 1/(K + 1), split equally over its
cells, and the new group gets weight 1/(K + 1) at +infinity. The (1 - alpha) quantile of that
weighted distribution covers a new cell from a new group with probability at least 1 - alpha,
if groups are exchangeable. It is finite only when K / (K + 1) >= 1 - alpha: at least 9 groups
for alpha = 0.1 and 4 for alpha = 0.2. (Weights as stated in the literature review in
docs/M9_LITERATURE.md; checked here by simulation in tests/test_conformal.py, not against the
paper's text, which could not be opened from this environment.)

What this module does not do: it does not decide which cells count as exchangeable with the
test cells. That choice, random cells versus held-out donors versus held-out labs, is the
experiment in scripts/m9_conformal.py.
"""

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ScoreFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


# ---------- classifiers that output probabilities over a fixed list of classes ----------


def fit_linear_probe(emb: np.ndarray, labels: np.ndarray, seed: int = 0):
    """Multinomial logistic regression on the standardized embedding (the "linear probe")."""
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    )
    model.fit(emb, labels)
    return model


def probe_proba(model, emb: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Probabilities in the column order of `classes`; classes the model never saw get 0."""
    raw = model.predict_proba(emb)
    out = np.zeros((len(emb), len(classes)))
    index = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model.classes_):
        out[:, index[c]] = raw[:, j]
    return out


def knn_proba(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    query: np.ndarray,
    classes: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """Fraction of each class among the k nearest training cells (Euclidean)."""
    nn = NearestNeighbors(n_neighbors=k).fit(train_emb)
    idx = nn.kneighbors(query, return_distance=False)
    index = {c: i for i, c in enumerate(classes)}
    codes = np.array([index[c] for c in train_labels])
    out = np.zeros((len(query), len(classes)))
    for j in range(k):
        np.add.at(out, (np.arange(len(query)), codes[idx[:, j]]), 1.0)
    return out / k


# ---------- nonconformity scores: one value per (cell, candidate label) ----------


def lac_scores(probs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """s(x, y) = 1 - p(y | x), for every cell and every label."""
    return 1.0 - probs


def aps_scores(probs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomized APS: mass of labels ranked strictly above y, plus U * p(y | x).

    One U per cell, shared across that cell's labels, so the set {y : s(x, y) <= q} is the
    top labels in probability order (a prefix), as in Romano et al. 2020. Ties in probability
    are broken by label order.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    above = np.cumsum(sorted_p, axis=1) - sorted_p  # mass strictly before each position
    u = rng.uniform(size=(n, 1))
    sorted_scores = above + u * sorted_p
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


SCORES: dict[str, ScoreFn] = {"lac": lac_scores, "aps": aps_scores}


# ---------- calibration and sets ----------


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; infinity if that exceeds n."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if rank > n else float(scores[rank - 1])


def hierarchical_quantile(cal_scores: np.ndarray, groups: np.ndarray, alpha: float) -> float:
    """Group-weighted quantile for a new cell from a new group (see the module docstring)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.asarray(cal_scores, dtype=float)
    _, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    k = len(counts)
    if k / (k + 1) < 1 - alpha - 1e-12:
        return float("inf")
    weights = 1.0 / ((k + 1) * counts[inverse])
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(scores[order][np.searchsorted(cumulative, 1 - alpha - 1e-12)])


def true_label_scores(scores: np.ndarray, label_idx: np.ndarray) -> np.ndarray:
    """Pick s(x_i, y_i) out of an all-labels score matrix."""
    return scores[np.arange(len(label_idx)), label_idx]


def calibrate(
    cal_scores_all: np.ndarray,
    cal_label_idx: np.ndarray,
    alpha: float,
    mondrian: bool = False,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Per-label thresholds q_y. Marginal conformal: one q for every label.

    Mondrian: q_y from the calibration cells whose true label is y; a label with no
    calibration cells gets q = infinity (always included), which is the honest answer.
    groups: use the hierarchical (group-weighted) quantile instead; marginal only.
    """
    n_labels = cal_scores_all.shape[1]
    true = true_label_scores(cal_scores_all, cal_label_idx)
    if groups is not None:
        if mondrian:
            raise ValueError("hierarchical calibration is implemented for marginal coverage only")
        return np.full(n_labels, hierarchical_quantile(true, groups, alpha))
    if not mondrian:
        return np.full(n_labels, conformal_quantile(true, alpha))
    return np.array(
        [
            conformal_quantile(true[cal_label_idx == y], alpha)
            if (cal_label_idx == y).any()
            else float("inf")
            for y in range(n_labels)
        ]
    )


def prediction_sets(test_scores_all: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Boolean (cells x labels): label y is in the set if s(x, y) <= q_y."""
    return test_scores_all <= thresholds[None, :]


def set_metrics(sets: np.ndarray, label_idx: np.ndarray, min_class_cells: int = 20) -> dict:
    """Coverage, set size and per-class coverage of prediction sets.

    `label_idx` may contain -1 for cells whose true label is not among the labels (a type the
    classifier never saw); they count as not covered and are reported separately.
    """
    known = label_idx >= 0
    covered = np.zeros(len(label_idx), dtype=bool)
    covered[known] = sets[np.flatnonzero(known), label_idx[known]]
    size = sets.sum(axis=1)
    per_class = {}
    for y in np.unique(label_idx[known]):
        mask = label_idx == y
        if mask.sum() >= min_class_cells:
            per_class[int(y)] = float(covered[mask].mean())
    return {
        "coverage": float(covered[known].mean()) if known.any() else float("nan"),
        "mean_size": float(size[known].mean()) if known.any() else float("nan"),
        "singleton_share": float((size[known] == 1).mean()) if known.any() else float("nan"),
        "empty_share": float((size[known] == 0).mean()) if known.any() else float("nan"),
        "worst_class_coverage": min(per_class.values()) if per_class else float("nan"),
        "per_class_coverage": per_class,
    }
