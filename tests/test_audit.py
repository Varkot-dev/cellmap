import numpy as np
import pytest
import scipy.sparse as sp

from cellmap.audit import (
    cosine_distance,
    donor_probe,
    knockout,
    knockout_effects,
    matched_controls,
    overlap_at_k,
)


def test_knockout_zeroes_only_the_named_genes_and_copies():
    counts = sp.csr_matrix(np.array([[1, 2, 0], [0, 3, 4]]))
    out = knockout(counts, [1])
    assert out.toarray().tolist() == [[1, 0, 0], [0, 0, 4]]
    assert counts.toarray()[0, 1] == 2  # the input is untouched
    assert out.nnz == 2  # zeros are not stored, so the tokenizer never sees the gene


def test_cosine_distance_worked_example():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.8, 0.6], [0.0, 1.0], [1.0, 0.0]])
    assert cosine_distance(a, b) == pytest.approx([0.2, 1.0, 1.0])


def test_knockout_effects_only_move_cells_that_express_the_gene():
    # embedding = the raw counts, so knocking out gene g moves exactly the cells with g > 0
    counts = sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 0.0]]))
    calls = []

    def embed_fn(x):
        calls.append(x.shape[0])
        return x.toarray()

    seen = []
    effects = knockout_effects(
        embed_fn, counts, np.array([0, 2]), on_moved=lambda i, rows, m: seen.append((i, list(rows)))
    )
    assert effects.shape == (2, 3)
    assert effects[0, 0] == pytest.approx(1 - 1 / np.sqrt(2))
    assert effects[0, 1] == 0.0
    assert effects[0, 2] == pytest.approx(1.0)  # the cell's only gene is gone
    assert effects[1].tolist() == pytest.approx([0.0, 1 - 1 / np.sqrt(2), 0.0])
    assert calls == [3, 2, 1]  # base, then only the expressing cells
    assert seen == [(0, [0, 2]), (1, [1])]


def test_overlap_at_k():
    assert overlap_at_k(list("abcd"), list("cdxy"), k=2) == 0.0
    assert overlap_at_k(list("abcd"), list("bdxy"), k=2) == 0.5
    assert overlap_at_k(list("abcd"), list("bayx"), k=2) == 1.0
    with pytest.raises(ValueError):
        overlap_at_k(list("ab"), list("abc"), k=3)


def test_matched_controls_draw_from_the_closest_allowed_genes():
    detection = np.array([0.5, 0.49, 0.51, 0.9, 0.1, 0.48])
    allowed = np.array([True, True, True, True, True, False])
    out = matched_controls(detection, target=0, allowed=allowed, n_pool=2, n_draw=2)
    assert out.tolist() == [1, 2]  # 5 is not allowed, 0 is the target
    with pytest.raises(ValueError):
        matched_controls(detection, 0, allowed, n_pool=10)


def test_donor_probe_separates_a_planted_donor_signal():
    rng = np.random.default_rng(0)
    donors = np.repeat([f"d{i}" for i in range(8)], 30)
    label = {f"d{i}": int(i % 2) for i in range(8)}
    y = np.array([label[d] for d in donors])
    planted = rng.normal(size=(len(donors), 5))
    planted[:, 0] += 2.0 * y
    assert donor_probe(planted, donors, label)["auroc"] == 1.0
    noise = rng.normal(size=(len(donors), 5))
    result = donor_probe(noise, donors, label)
    assert set(result["scores"]) == set(label)
    assert 0.0 <= result["auroc"] <= 1.0
