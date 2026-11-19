import numpy as np
import pytest

from cellmap.reference_mapping import harmony_reference_map


def two_types_four_batches(seed: int = 0):
    """Two cell types far apart; four batches, each adding its own shift to every cell."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((2, 10))
    centers[1, 0] = 12.0
    shifts = {b: rng.normal(0, 1.5, 10) for b in "abcd"}
    rows, types, batches = [], [], []
    for b in "abcd":
        for t in (0, 1):
            rows.append(centers[t] + shifts[b] + rng.normal(0, 0.5, (150, 10)))
            types += [t] * 150
            batches += [b] * 150
    return np.vstack(rows), np.array(types), np.array(batches)


def test_shape_and_training_cells_match_a_reference_only_run():
    x, _, batch = two_types_four_batches()
    train = np.isin(batch, ["a", "b", "c"])
    out = harmony_reference_map(x, batch, train)
    assert out.shape == x.shape and out.dtype == np.float32
    again = harmony_reference_map(x[train], batch[train], np.ones(train.sum(), bool))
    np.testing.assert_allclose(out[train], again, atol=1e-5)


def test_query_cells_cannot_move_the_reference():
    x, _, batch = two_types_four_batches()
    train = np.isin(batch, ["a", "b", "c"])
    scrambled = x.copy()
    scrambled[~train] = np.random.default_rng(1).normal(0, 50, scrambled[~train].shape)
    np.testing.assert_allclose(
        harmony_reference_map(x, batch, train)[train],
        harmony_reference_map(scrambled, batch, train)[train],
        atol=1e-6,
    )


def test_unseen_batch_is_mapped_next_to_its_cell_type():
    x, types, batch = two_types_four_batches()
    train = np.isin(batch, ["a", "b", "c"])
    out = harmony_reference_map(x, batch, train)
    query = ~train

    def mean_gap(emb):
        # distance between a query type's centre and the same type's reference centre
        return np.mean(
            [
                np.linalg.norm(
                    emb[query & (types == t)].mean(0) - emb[train & (types == t)].mean(0)
                )
                for t in (0, 1)
            ]
        )

    assert mean_gap(out) < 0.5 * mean_gap(x)
    # and the two types stay far apart in the query
    gap_types = np.linalg.norm(
        out[query & (types == 0)].mean(0) - out[query & (types == 1)].mean(0)
    )
    assert gap_types > 8.0


def test_deterministic_and_validates_input():
    x, _, batch = two_types_four_batches()
    train = np.isin(batch, ["a", "b", "c"])
    np.testing.assert_array_equal(
        harmony_reference_map(x, batch, train), harmony_reference_map(x, batch, train)
    )
    with pytest.raises(ValueError):
        harmony_reference_map(x, batch, np.isin(batch, ["a"]))
    with pytest.raises(ValueError):
        harmony_reference_map(x, batch[:-1], train)
