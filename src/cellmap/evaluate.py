"""Evaluation harness: score an embedding, whatever method produced it.

Every function takes a plain numpy embedding (cells x dimensions) and per-cell arrays (cell-type
labels, donor ids, split names). Nothing here knows how the embedding was made, so PCA, a VAE
and a pretrained transformer are all scored by exactly the same code.

Three views of quality, as in the plan (section 4.3):

- `knn_classification`: can held-out donors' cells be labelled from training donors' cells?
- `label_efficiency`: how fast does that degrade when only a fraction of labels is available?
- `scib_scores`: scIB-style metrics (Luecken et al., Nature Methods 2022), in two groups that
  pull against each other: biological conservation and batch correction.
"""

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Metric names accepted by `scib_scores`, by group. The group decides which mean a metric enters.
BIO_METRICS = ("nmi", "ari", "silhouette_label", "clisi")
BATCH_METRICS = ("silhouette_batch", "graph_connectivity", "ilisi", "kbet")

# kBET is opt-in, not default: scib-metrics' version is an approximation of the scIB one and
# silently skips cell types that are small or come from one batch (see `scib_scores`). Keeping
# the default set fixed matters because the group means, and therefore the aggregate, are only
# comparable between runs that used the same metrics.
DEFAULT_METRICS = (
    "nmi",
    "ari",
    "silhouette_label",
    "clisi",
    "silhouette_batch",
    "graph_connectivity",
    "ilisi",
)

# Luecken et al. 2022: overall score = 0.6 * biological conservation + 0.4 * batch correction.
BIO_WEIGHT = 0.6
BATCH_WEIGHT = 0.4


def _as_str(values: Sequence | np.ndarray | pd.Series) -> np.ndarray:
    """Per-cell values as a plain numpy array of strings (categoricals and ints included)."""
    return np.asarray(pd.Series(np.asarray(values)).astype(str))


def _check_lengths(emb: np.ndarray, **per_cell: np.ndarray) -> None:
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D (cells x dimensions), got shape {emb.shape}")
    for name, values in per_cell.items():
        if len(values) != emb.shape[0]:
            raise ValueError(f"{name} has {len(values)} entries but emb has {emb.shape[0]} cells")


def _knn_predict(
    train_emb: np.ndarray, train_codes: np.ndarray, test_emb: np.ndarray, k: int, n_classes: int
) -> np.ndarray:
    """Majority vote among the k nearest training cells (euclidean); returns class codes.

    Ties in the vote go to the tied class that owns the nearest neighbor. That is the choice a
    1-nearest-neighbor rule would make among the tied classes, and unlike "first class in
    alphabetical order" it does not depend on how the classes happen to be named.
    """
    index = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean").fit(train_emb)
    neighbor_idx = index.kneighbors(test_emb, return_distance=False)  # sorted nearest first
    neighbor_codes = train_codes[neighbor_idx]  # (n_test, k)

    rows = np.arange(len(test_emb))[:, None]
    votes = np.zeros((len(test_emb), n_classes), dtype=np.int64)
    np.add.at(votes, (rows, neighbor_codes), 1)
    is_winner = votes == votes.max(axis=1, keepdims=True)
    # For each test cell: the first (nearest) neighbor whose class is among the winners.
    first_winning_neighbor = np.argmax(is_winner[rows, neighbor_codes], axis=1)
    return neighbor_codes[rows[:, 0], first_winning_neighbor]


def _classification_report(
    true_codes: np.ndarray, pred_codes: np.ndarray, class_names: list[str], train_classes: set[str]
) -> dict:
    """Accuracy, per-class F1, macro F1 and confusion matrix from integer class codes."""
    n = len(class_names)
    confusion = np.zeros((n, n), dtype=np.int64)
    np.add.at(confusion, (true_codes, pred_codes), 1)  # rows: true class, columns: predicted

    true_positive = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)  # test cells that truly belong to each class
    predicted = confusion.sum(axis=0)  # test cells assigned to each class
    # F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (support + predicted).
    denominator = (support + predicted).astype(float)
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(n), where=denominator > 0)
    # A class with no true test cells has no recall, so its F1 is undefined, not zero.
    f1[support == 0] = np.nan

    return {
        "accuracy": float(true_positive.sum() / len(true_codes)),
        "macro_f1": float(np.nanmean(f1)),
        "per_class_f1": {name: float(v) for name, v in zip(class_names, f1, strict=True)},
        "confusion_matrix": confusion,
        "labels": class_names,
        "classes_absent_from_test": [
            c for c, s in zip(class_names, support, strict=True) if s == 0
        ],
        "classes_absent_from_train": [c for c in class_names if c not in train_classes],
    }


def _fit_and_score(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    k: int,
) -> dict:
    """Classify test cells from training cells and score the predictions."""
    if len(train_labels) == 0 or len(test_labels) == 0:
        raise ValueError(
            f"need train and test cells, got {len(train_labels)} and {len(test_labels)}"
        )
    class_names = sorted(set(train_labels) | set(test_labels))
    code_of = {name: i for i, name in enumerate(class_names)}
    train_codes = np.array([code_of[x] for x in train_labels])
    true_codes = np.array([code_of[x] for x in test_labels])

    k_used = min(k, len(train_labels))  # cannot ask for more neighbors than there are cells
    pred_codes = _knn_predict(train_emb, train_codes, test_emb, k_used, len(class_names))
    report = _classification_report(true_codes, pred_codes, class_names, set(train_labels))
    report.update({"k": k_used, "n_train": len(train_labels), "n_test": len(test_labels)})
    return report


def knn_classification(
    emb: np.ndarray,
    labels: Sequence | np.ndarray | pd.Series,
    split: Sequence | np.ndarray | pd.Series,
    k: int = 15,
    train: str = "train",
    test: str = "test",
) -> dict:
    """k-nearest-neighbor cell-type classification of held-out donors on a frozen embedding.

    Each cell in split `test` gets the majority label of its `k` nearest cells (euclidean) in
    split `train`. kNN has no trainable parameters, so the score reflects the geometry of the
    embedding itself and not the capacity of a classifier stacked on top of it.

    Returned keys:

    - accuracy: fraction of test cells labelled correctly. Dominated by the common cell types.
    - macro_f1: unweighted mean of per-class F1, so a rare type counts as much as a common one.
    - per_class_f1: {class: F1}. NaN for a class with no test cells.
    - confusion_matrix: integer array, rows = true class, columns = predicted class.
    - labels: the row/column order, the sorted union of train and test classes.
    - classes_absent_from_test, classes_absent_from_train: see below.
    - k, n_train, n_test: what was actually used (k is lowered to n_train if n_train < k).

    Which classes enter macro F1. With donor-level splits a rare cell type can be missing from
    one side, so this has to be a stated rule:

    - Present in test, absent from train: INCLUDED with F1 = 0. The classifier can never
      predict the class, every one of its cells is mislabelled, and that failure is real.
    - Present in train, absent from test: EXCLUDED, per-class F1 reported as NaN. With no true
      cells, recall is 0/0; counting the class as 0 would lower the score because of who
      happened to be in the test donors, not because of the embedding. Test cells wrongly
      assigned to such a class are still counted, as lost recall of their true class.

    So macro F1 is the mean over the classes that have at least one test cell. Note this differs
    from `sklearn.metrics.f1_score(average="macro")` with default arguments, which averages over
    the union of true and predicted classes and so scores a predicted-but-absent class as 0.

    Vote ties go to the tied class that owns the nearest neighbor.
    """
    emb = np.asarray(emb)
    labels, split = _as_str(labels), _as_str(split)
    _check_lengths(emb, labels=labels, split=split)
    is_train, is_test = split == train, split == test
    return _fit_and_score(emb[is_train], labels[is_train], emb[is_test], labels[is_test], k)


def label_efficiency(
    emb: np.ndarray,
    labels: Sequence | np.ndarray | pd.Series,
    split: Sequence | np.ndarray | pd.Series,
    fractions: Sequence[float] = (0.01, 0.1, 1.0),
    k: int = 15,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    train: str = "train",
    test: str = "test",
) -> pd.DataFrame:
    """kNN accuracy and macro F1 when only a fraction of the training cells are labelled.

    For each fraction and seed, round(fraction * n_train) training cells (at least one) are
    drawn uniformly at random without replacement, and only those serve as the kNN reference.
    The test cells are always all cells of the `test` split. The embedding is not refitted: this
    measures how many labels are needed to USE a given embedding, which is the question for an
    embedding learned without labels.

    The draw is deliberately NOT stratified by cell type. Stratifying needs every cell's label
    in order to decide which cells to label, which defeats the premise; a uniform draw is what
    "annotate a random 1% of cells" gives in practice. The consequence is that a rare type can
    receive few or no labelled cells at a low fraction. It then scores F1 = 0 and stays in
    macro F1 (the rule in `knn_classification`), so macro F1 falls faster than accuracy.

    One row per (fraction, seed), with columns fraction, seed, n_labeled, n_classes_labeled, k,
    accuracy, macro_f1. At fraction 1.0 every seed gives the same row, since there is nothing to
    draw; the spread across seeds measures label-sampling noise only, not donor-to-donor noise.
    """
    emb = np.asarray(emb)
    labels, split = _as_str(labels), _as_str(split)
    _check_lengths(emb, labels=labels, split=split)
    train_idx = np.flatnonzero(split == train)
    is_test = split == test
    test_emb, test_labels = emb[is_test], labels[is_test]

    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"fractions must be in (0, 1], got {fraction}")
        n_labeled = max(1, round(fraction * len(train_idx)))
        for seed in seeds:
            chosen = np.random.default_rng(seed).choice(train_idx, size=n_labeled, replace=False)
            report = _fit_and_score(emb[chosen], labels[chosen], test_emb, test_labels, k)
            rows.append(
                {
                    "fraction": float(fraction),
                    "seed": int(seed),
                    "n_labeled": n_labeled,
                    "n_classes_labeled": len(set(labels[chosen])),
                    "k": report["k"],
                    "accuracy": report["accuracy"],
                    "macro_f1": report["macro_f1"],
                }
            )
    return pd.DataFrame(rows)


def _exact_neighbors(emb: np.ndarray, n_neighbors: int):
    """Exact euclidean k nearest neighbors (each cell's own index included) for scib-metrics.

    scib-metrics' own helper uses approximate search (pynndescent). Exact search is used here
    because it is deterministic and, at tens of thousands of cells in ~50 dimensions, takes
    seconds; it removes one source of run-to-run noise from the comparison between methods.
    """
    from scib_metrics.nearest_neighbors import NeighborsResults

    index = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="euclidean")
    distances, indices = index.fit(emb).kneighbors(emb)
    return NeighborsResults(indices=indices, distances=distances)


def scib_scores(
    emb: np.ndarray,
    labels: Sequence | np.ndarray | pd.Series,
    batch: Sequence | np.ndarray | pd.Series,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_neighbors: int = 15,
    lisi_neighbors: int = 90,
    kbet_neighbors: int = 50,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict:
    """scIB-style metrics for one embedding, computed with the `scib-metrics` package.

    The metrics come in two groups that are reported separately because they pull against each
    other: an embedding can mix batches perfectly by collapsing all cells to a point (losing
    the biology), or keep cell types apart perfectly by also keeping donors apart.

    All metrics are in [0, 1] with higher = better. "rescaled" below means the raw quantity was
    mapped to that range the way scIB does; "raw" means it is already in that range.

    Biological conservation (do cells of one type sit together, apart from other types?):

    - nmi: normalized mutual information between Leiden clusters and labels. 0..1, raw.
    - ari: adjusted Rand index between the same clusters and labels. About 0 for a random
      clustering (can dip slightly below 0), 1 for identical; raw, not rescaled.
    - silhouette_label: mean silhouette width of cells w.r.t. their cell type, euclidean.
      Raw range -1..1, rescaled as (asw + 1) / 2.
    - clisi: cell-type LISI, the effective number of cell types in a cell's neighborhood
      (median over cells). Raw range 1..n_types, rescaled as (n_types - x) / (n_types - 1).

    Batch correction (are donors mixed within each cell type?):

    - silhouette_batch: silhouette width w.r.t. batch, computed inside each cell type.
      Raw range -1..1 per cell; rescaled as 1 - |s|, averaged per cell type, then across types.
    - graph_connectivity: for each cell type, the fraction of its cells in the largest
      connected component of the kNN graph restricted to that type, averaged over types.
      0..1, raw. scib-metrics 0.6.2 uses STRONGLY connected components of the directed kNN
      graph (i -> j when j is among i's neighbors), which is stricter than components of a
      symmetrized graph, so expect lower values than tools that symmetrize first.
    - ilisi: batch LISI, the effective number of batches in a cell's neighborhood (median).
      Raw range 1..n_batches, rescaled as (x - 1) / (n_batches - 1).
    - kbet (opt-in, not in the default set): kBET acceptance rate per cell type, the fraction
      of cells whose neighborhood batch composition passes a chi-square test against the cell
      type's overall composition, averaged over types. 0..1, raw. scib-metrics approximates
      the scIB procedure (diffusion-map neighbors, no adaptive neighborhood size) and skips
      types with fewer than 10 cells or a single batch.

    How they are computed:

    - Neighbors: exact euclidean kNN, computed once with the largest k needed and truncated.
      `n_neighbors` (15) feeds clustering and graph connectivity, `lisi_neighbors` (90, i.e.
      perplexity 30) feeds the LISI metrics, `kbet_neighbors` (50) feeds kBET. These are the
      values scib-metrics' own Benchmarker uses.
    - Clustering for NMI/ARI: Leiden on the UMAP-style connectivity graph of the 15-NN graph,
      at resolutions 0.2, 0.4, ..., 2.0. The resolution with the highest NMI is kept and ARI is
      reported at that same resolution (the scIB convention). Picking the resolution with the
      labels in hand makes NMI an optimistic "best achievable clustering" number; that is the
      standard definition, and it is the same for every method compared.
    - `seed` seeds Leiden. Nothing else here is random.

    Aggregates: `bio_mean` and `batch_mean` are unweighted means of the metrics computed in each
    group, and `total` = 0.6 * bio_mean + 0.4 * batch_mean. Two caveats. First, the means only
    compare between runs with the same `metrics`. Second, Luecken et al. min-max scale each
    metric ACROSS the methods being compared before averaging; that needs all methods at once,
    so it is not done here. Do it on a table of these outputs if you want the paper's ranking.

    Returns {"bio": {...}, "batch": {...}, "bio_mean", "batch_mean", "total", "seconds": {...},
    "n_cells", "n_labels", "n_batches"}. `seconds` holds wall time per step, for cost reporting.
    """
    import scib_metrics

    unknown = sorted(set(metrics) - set(BIO_METRICS) - set(BATCH_METRICS))
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; choose from {BIO_METRICS + BATCH_METRICS}")
    wanted = set(metrics)
    if ("nmi" in wanted) != ("ari" in wanted):
        raise ValueError("nmi and ari come from the same clustering; request both or neither")
    if not wanted & set(BIO_METRICS) or not wanted & set(BATCH_METRICS):
        raise ValueError("need at least one metric from each group to form the aggregate")

    emb = np.ascontiguousarray(emb, dtype=np.float32)
    labels, batch = _as_str(labels), _as_str(batch)
    _check_lengths(emb, labels=labels, batch=batch)
    n_cells = emb.shape[0]

    seconds: dict[str, float] = {}
    bio: dict[str, float] = {}
    batch_scores: dict[str, float] = {}

    def timed(name: str, fn):
        start = time.perf_counter()
        out = fn()
        seconds[name] = time.perf_counter() - start
        return out

    k_needed = [n_neighbors] if wanted & {"nmi", "ari", "graph_connectivity"} else []
    k_needed += [lisi_neighbors] if wanted & {"clisi", "ilisi"} else []
    k_needed += [kbet_neighbors] if "kbet" in wanted else []
    if k_needed:
        k_max = min(max(k_needed), n_cells)
        neighbors = timed("neighbors", lambda: _exact_neighbors(emb, k_max))

        def truncated(k: int):
            return neighbors.subset_neighbors(min(k, k_max))

    if "nmi" in wanted:
        clustering = timed(
            "nmi_ari",
            lambda: scib_metrics.nmi_ari_cluster_labels_leiden(
                truncated(n_neighbors), labels, optimize_resolution=True, n_jobs=n_jobs, seed=seed
            ),
        )
        bio["nmi"], bio["ari"] = float(clustering["nmi"]), float(clustering["ari"])
    if "silhouette_label" in wanted:
        bio["silhouette_label"] = float(
            timed(
                "silhouette_label",
                lambda: scib_metrics.silhouette_label(emb, labels, rescale=True),
            )
        )
    if "clisi" in wanted:
        bio["clisi"] = float(
            timed(
                "clisi",
                lambda: scib_metrics.clisi_knn(truncated(lisi_neighbors), labels, scale=True),
            )
        )

    if "silhouette_batch" in wanted:
        batch_scores["silhouette_batch"] = float(
            timed(
                "silhouette_batch",
                lambda: scib_metrics.silhouette_batch(emb, labels, batch, rescale=True),
            )
        )
    if "graph_connectivity" in wanted:
        batch_scores["graph_connectivity"] = float(
            timed(
                "graph_connectivity",
                lambda: scib_metrics.graph_connectivity(truncated(n_neighbors), labels),
            )
        )
    if "ilisi" in wanted:
        batch_scores["ilisi"] = float(
            timed(
                "ilisi",
                lambda: scib_metrics.ilisi_knn(truncated(lisi_neighbors), batch, scale=True),
            )
        )
    if "kbet" in wanted:
        batch_scores["kbet"] = float(
            timed(
                "kbet",
                lambda: scib_metrics.kbet_per_label(truncated(kbet_neighbors), batch, labels),
            )
        )

    bio_mean = float(np.mean(list(bio.values())))
    batch_mean = float(np.mean(list(batch_scores.values())))
    return {
        "bio": bio,
        "batch": batch_scores,
        "bio_mean": bio_mean,
        "batch_mean": batch_mean,
        "total": BIO_WEIGHT * bio_mean + BATCH_WEIGHT * batch_mean,
        "seconds": seconds,
        "n_cells": n_cells,
        "n_labels": len(set(labels)),
        "n_batches": len(set(batch)),
    }
