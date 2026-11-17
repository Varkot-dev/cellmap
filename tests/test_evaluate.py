import numpy as np
import pandas as pd
import pytest

from cellmap.evaluate import (
    BATCH_WEIGHT,
    BIO_WEIGHT,
    DEFAULT_METRICS,
    knn_classification,
    label_efficiency,
    scib_scores,
)


def _column(values: list[float]) -> np.ndarray:
    """A one-dimensional embedding: one cell per value."""
    return np.array(values, dtype=float)[:, None]


def _blobs(
    centers: np.ndarray, n_per_center: int, seed: int, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blobs around `centers`; returns (points, index of the center of each point)."""
    rng = np.random.default_rng(seed)
    which = np.repeat(np.arange(len(centers)), n_per_center)
    return centers[which] + scale * rng.normal(size=(len(which), centers.shape[1])), which


# --- knn_classification -----------------------------------------------------------------------


def test_knn_hand_computed_example():
    # Training cells on a line: A near 0, B near 10, C near 20 (only two C cells). k = 3.
    train = [0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 20.0, 20.1]
    train_labels = ["A", "A", "A", "B", "B", "B", "C", "C"]
    # Test cell -> its 3 nearest training cells -> majority vote:
    #   0.05 (true A): A A A -> A  correct
    #   9.9  (true B): B B B -> B  correct
    #   10.3 (true A): B B B -> B  wrong
    #   20.05 (true C): C C B -> C  correct
    #   0.15 (true B): A A A -> A  wrong
    #   19.0 (true C): C C B -> C  correct
    test = [0.05, 9.9, 10.3, 20.05, 0.15, 19.0]
    test_labels = ["A", "B", "A", "C", "B", "C"]
    # A validation cell in the middle of the A cluster but labelled C: it must be ignored.
    emb = _column(train + test + [0.05])
    labels = train_labels + test_labels + ["C"]
    split = ["train"] * 8 + ["test"] * 6 + ["val"]

    r = knn_classification(emb, labels, split, k=3)

    assert r["labels"] == ["A", "B", "C"]
    # Rows = true class, columns = predicted class.
    np.testing.assert_array_equal(r["confusion_matrix"], [[1, 1, 0], [1, 1, 0], [0, 0, 2]])
    assert r["accuracy"] == pytest.approx(4 / 6)
    # A: TP 1, FP 1, FN 1 -> F1 = 2 / (2 + 1 + 1) = 0.5. B the same. C: all correct -> 1.
    assert r["per_class_f1"] == pytest.approx({"A": 0.5, "B": 0.5, "C": 1.0})
    assert r["macro_f1"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert (r["k"], r["n_train"], r["n_test"]) == (3, 8, 6)
    assert r["classes_absent_from_test"] == [] and r["classes_absent_from_train"] == []


def test_knn_vote_tie_goes_to_the_class_of_the_nearest_neighbor():
    # k = 2 with one A and one B neighbor is always a 1-1 tie. "Z" sorts after "A", so a rule
    # based on label order would answer "A" for both test cells.
    emb = _column([0.0, 1.0, 0.4, 0.6])
    r = knn_classification(emb, ["A", "Z", "A", "Z"], ["train", "train", "test", "test"], k=2)
    np.testing.assert_array_equal(r["confusion_matrix"], [[1, 0], [0, 1]])


def test_knn_classes_missing_from_train_or_test():
    # C exists only in test (can never be predicted); D exists only in train.
    train = [0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 30.0, 30.1, 30.2]
    train_labels = ["A"] * 3 + ["B"] * 3 + ["D"] * 3
    #   0.05 (true A) -> A   10.05 (true B) -> B   29 (true C) -> D
    #   9 (true C) -> B      31 (true B) -> D
    test = [0.05, 10.05, 29.0, 9.0, 31.0]
    test_labels = ["A", "B", "C", "C", "B"]
    emb = _column(train + test)
    split = ["train"] * 9 + ["test"] * 5

    r = knn_classification(emb, train_labels + test_labels, split, k=3)

    assert r["labels"] == ["A", "B", "C", "D"]
    assert r["classes_absent_from_train"] == ["C"]
    assert r["classes_absent_from_test"] == ["D"]
    np.testing.assert_array_equal(
        r["confusion_matrix"], [[1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0]]
    )
    assert r["accuracy"] == pytest.approx(2 / 5)
    f1 = r["per_class_f1"]
    assert f1["A"] == pytest.approx(1.0)
    assert f1["B"] == pytest.approx(0.5)  # TP 1, FP 1 (a C cell), FN 1 (sent to D)
    assert f1["C"] == 0.0  # in test, never predictable: counted as a failure
    assert np.isnan(f1["D"])  # no test cells: undefined, left out of the mean
    assert r["macro_f1"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_knn_separated_clusters_are_perfect_and_shuffled_labels_are_near_chance():
    centers = 20.0 * np.eye(4)
    emb, cell_type = _blobs(centers, n_per_center=200, seed=0)
    rng = np.random.default_rng(1)
    split = rng.choice(["train", "test"], size=len(emb))

    # Integer labels and a pandas categorical split are accepted too.
    real = knn_classification(emb, cell_type, pd.Categorical(split))
    assert real["accuracy"] == 1.0 and real["macro_f1"] == 1.0

    shuffled = knn_classification(emb, rng.permutation(cell_type), split)
    # Four balanced classes: chance is 0.25.
    assert 0.15 < shuffled["accuracy"] < 0.35
    assert shuffled["macro_f1"] < 0.35


def test_knn_rejects_mismatched_lengths_and_empty_splits():
    emb = _column([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="labels has 2 entries"):
        knn_classification(emb, ["A", "B"], ["train", "train", "test"])
    with pytest.raises(ValueError, match="need train and test cells"):
        knn_classification(emb, ["A", "B", "A"], ["train", "train", "val"])


# --- label_efficiency -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def overlapping_blobs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three overlapping cell types, 400 training and 200 test cells."""
    emb, cell_type = _blobs(2.5 * np.eye(3), n_per_center=200, seed=0)
    split = np.where(np.random.default_rng(1).random(len(emb)) < 2 / 3, "train", "test")
    return emb, cell_type, split


def test_label_efficiency_shape_and_number_of_labelled_cells(overlapping_blobs):
    emb, cell_type, split = overlapping_blobs
    n_train = int((split == "train").sum())

    df = label_efficiency(emb, cell_type, split, fractions=(0.05, 0.5, 1.0), seeds=(0, 1, 2), k=5)

    assert list(df.columns) == [
        "fraction",
        "seed",
        "n_labeled",
        "n_classes_labeled",
        "k",
        "accuracy",
        "macro_f1",
    ]
    assert len(df) == 3 * 3
    assert df[["fraction", "seed"]].drop_duplicates().shape[0] == 9
    per_fraction = df.groupby("fraction")["n_labeled"].unique().map(list).to_dict()
    assert per_fraction == {
        0.05: [round(0.05 * n_train)],
        0.5: [round(0.5 * n_train)],
        1.0: [n_train],
    }
    assert per_fraction[0.05] < per_fraction[0.5] < per_fraction[1.0]
    assert df["accuracy"].between(0, 1).all() and df["macro_f1"].between(0, 1).all()


def test_label_efficiency_full_fraction_matches_knn_classification(overlapping_blobs):
    emb, cell_type, split = overlapping_blobs
    df = label_efficiency(emb, cell_type, split, fractions=(1.0,), seeds=(0, 7), k=5)
    full = knn_classification(emb, cell_type, split, k=5)
    # Nothing is drawn at fraction 1.0, so every seed reproduces the full result.
    assert df["accuracy"].tolist() == pytest.approx([full["accuracy"]] * 2)
    assert df["macro_f1"].tolist() == pytest.approx([full["macro_f1"]] * 2)


def test_label_efficiency_is_seeded_and_fewer_labels_do_not_help(overlapping_blobs):
    emb, cell_type, split = overlapping_blobs
    kwargs = {"fractions": (0.03, 1.0), "seeds": (0, 1, 2, 3, 4), "k": 5}
    df = label_efficiency(emb, cell_type, split, **kwargs)
    pd.testing.assert_frame_equal(df, label_efficiency(emb, cell_type, split, **kwargs))

    low = df[df["fraction"] == 0.03]
    assert low["accuracy"].nunique() > 1  # different seeds label different cells
    mean_accuracy = df.groupby("fraction")["accuracy"].mean()
    assert mean_accuracy[0.03] < mean_accuracy[1.0]


def test_label_efficiency_lowers_k_when_fewer_cells_than_k_are_labelled(overlapping_blobs):
    emb, cell_type, split = overlapping_blobs
    df = label_efficiency(emb, cell_type, split, fractions=(0.01,), seeds=(0,), k=15)
    assert df.loc[0, "n_labeled"] < 15
    assert df.loc[0, "k"] == df.loc[0, "n_labeled"]


def test_label_efficiency_rejects_bad_fraction(overlapping_blobs):
    emb, cell_type, split = overlapping_blobs
    with pytest.raises(ValueError, match="fractions"):
        label_efficiency(emb, cell_type, split, fractions=(0.0,))


# --- scib_scores ------------------------------------------------------------------------------

N_TYPES, N_BATCHES, CELLS_PER_GROUP = 3, 3, 60


@pytest.fixture(scope="module")
def scib_results() -> dict[str, dict]:
    """scIB scores for two embeddings of the same 540 cells (3 cell types x 3 batches).

    "mixed": cell types far apart, batches indistinguishable. What integration should give.
    "batch_split": the same, plus a large per-batch offset, so every cell type breaks into one
    island per batch. What an uncorrected batch effect looks like.
    """
    rng = np.random.default_rng(0)
    n = N_TYPES * N_BATCHES * CELLS_PER_GROUP
    cell_type = np.repeat(np.arange(N_TYPES), N_BATCHES * CELLS_PER_GROUP)
    batch = np.tile(np.repeat(np.arange(N_BATCHES), CELLS_PER_GROUP), N_TYPES)
    type_centers = np.zeros((N_TYPES, 8))
    type_centers[np.arange(N_TYPES), np.arange(N_TYPES)] = 60.0
    batch_offsets = np.zeros((N_BATCHES, 8))
    batch_offsets[np.arange(N_BATCHES), 3 + np.arange(N_BATCHES)] = 15.0

    mixed = type_centers[cell_type] + rng.normal(size=(n, 8))
    batch_split = mixed + batch_offsets[batch]
    labels = np.array(["B cell", "T cell", "monocyte"])[cell_type]
    donors = np.array(["d1", "d2", "d3"])[batch]
    return {
        "mixed": scib_scores(mixed, labels, donors),
        "batch_split": scib_scores(batch_split, labels, donors),
    }


def test_scib_output_structure_and_aggregate(scib_results):
    r = scib_results["mixed"]
    assert set(r["bio"]) == {"nmi", "ari", "silhouette_label", "clisi"}
    assert set(r["batch"]) == {"silhouette_batch", "graph_connectivity", "ilisi"}
    assert set(r["bio"]) | set(r["batch"]) == set(DEFAULT_METRICS)
    assert all(isinstance(v, float) for v in {**r["bio"], **r["batch"]}.values())
    assert r["bio_mean"] == pytest.approx(np.mean(list(r["bio"].values())))
    assert r["batch_mean"] == pytest.approx(np.mean(list(r["batch"].values())))
    assert (BIO_WEIGHT, BATCH_WEIGHT) == (0.6, 0.4)
    assert r["total"] == pytest.approx(0.6 * r["bio_mean"] + 0.4 * r["batch_mean"])
    assert (r["n_cells"], r["n_labels"], r["n_batches"]) == (540, 3, 3)
    assert set(r["seconds"]) == {
        "neighbors",
        "nmi_ari",
        "silhouette_label",
        "clisi",
        "silhouette_batch",
        "graph_connectivity",
        "ilisi",
    }


def test_scib_separated_types_with_mixed_batches_score_high_on_both_groups(scib_results):
    r = scib_results["mixed"]
    assert r["bio"]["nmi"] > 0.95
    assert r["bio"]["ari"] > 0.95
    assert r["bio"]["silhouette_label"] > 0.9
    assert r["bio"]["clisi"] > 0.95
    assert r["batch"]["silhouette_batch"] > 0.9
    # Not exactly 1: scib-metrics counts strongly connected components of the directed kNN
    # graph, and a few cells on the edge of a blob are in nobody else's 15 nearest neighbors.
    assert r["batch"]["graph_connectivity"] > 0.95
    assert r["batch"]["ilisi"] > 0.7
    assert r["bio_mean"] > 0.9 and r["batch_mean"] > 0.85


def test_scib_batch_separated_embedding_scores_clearly_lower_on_batch_metrics(scib_results):
    mixed, split = scib_results["mixed"], scib_results["batch_split"]
    assert split["batch"]["silhouette_batch"] < mixed["batch"]["silhouette_batch"] - 0.4
    # Each cell type is three disconnected islands, one per batch: largest component = 1/3.
    assert split["batch"]["graph_connectivity"] == pytest.approx(1 / 3)
    assert split["batch"]["ilisi"] < 0.05
    assert split["batch_mean"] < mixed["batch_mean"] - 0.5
    assert split["total"] < mixed["total"]
    # The cell types are still far apart, so a cell's neighbors are still its own type.
    assert split["bio"]["clisi"] > 0.95


def test_scib_rejects_bad_metric_selections():
    emb = np.zeros((4, 2))
    labels, batch = ["a", "a", "b", "b"], ["x", "y", "x", "y"]
    with pytest.raises(ValueError, match="unknown metrics"):
        scib_scores(emb, labels, batch, metrics=("nmi", "ari", "ilisi", "pcr"))
    with pytest.raises(ValueError, match="same clustering"):
        scib_scores(emb, labels, batch, metrics=("nmi", "ilisi"))
    with pytest.raises(ValueError, match="each group"):
        scib_scores(emb, labels, batch, metrics=("nmi", "ari"))
