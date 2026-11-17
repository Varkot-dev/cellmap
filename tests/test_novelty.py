import numpy as np
import pytest

from cellmap.novelty import nearest_known_type, self_neighbor_rate


def two_blobs(separated: bool):
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (50, 2))
    b = rng.normal(10 if separated else 0, 1, (50, 2))
    return np.vstack([a, b]), np.array(["A"] * 50 + ["B"] * 50)


def test_separate_group_scores_near_one():
    emb, labels = two_blobs(separated=True)
    assert self_neighbor_rate(emb, labels, "B", k=10) > 0.95


def test_scattered_group_scores_near_its_share():
    emb, labels = two_blobs(separated=False)
    assert 0.3 < self_neighbor_rate(emb, labels, "B", k=10) < 0.7


def test_self_neighbor_rate_needs_two_cells():
    with pytest.raises(ValueError):
        self_neighbor_rate(np.zeros((3, 2)), np.array(["A", "A", "B"]), "B")


def test_nearest_known_type_reports_where_new_cells_land():
    # Reference has only types A (near 0) and C (near 10); query cells sit near 10.
    rng = np.random.default_rng(1)
    emb = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(10, 1, (30, 2)),
                     rng.normal(10, 1, (20, 2))])  # fmt: skip
    labels = np.array(["A"] * 30 + ["C"] * 30 + ["B"] * 20)
    reference = labels != "B"
    shares = nearest_known_type(emb, labels, reference, ~reference, k=5)
    assert shares.idxmax() == "C" and shares["C"] == 1.0
