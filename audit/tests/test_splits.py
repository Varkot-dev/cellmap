import numpy as np
import pandas as pd
import pytest

from leakaudit import (
    GroupLeakError,
    assert_no_group_overlap,
    group_kfold,
    group_split,
    random_split,
    split_report,
)


def _groups(n_groups=21, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(20, 200, size=n_groups)
    return np.repeat([f"d{i}" for i in range(n_groups)], sizes)


def test_group_split_keeps_groups_whole_and_counts_groups():
    g = _groups()
    split = group_split(g, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert_no_group_overlap(g, split, required=("train", "val", "test"))
    per = pd.DataFrame({"g": g, "s": split}).groupby("s")["g"].nunique()
    assert per["test"] == 4 and per["val"] == 4 and per["train"] == 13  # round(0.2 * 21) = 4


def test_group_split_without_validation():
    split = group_split(_groups(), val_fraction=0.0, test_fraction=0.25, seed=0)
    assert set(split) == {"train", "test"}


def test_group_split_is_deterministic_and_order_invariant():
    g = _groups()
    a = group_split(g, seed=7)
    assert np.array_equal(a, group_split(g, seed=7))
    perm = np.random.default_rng(0).permutation(len(g))
    b = group_split(g[perm], seed=7)
    assert np.array_equal(a[perm], b)
    assert not np.array_equal(a, group_split(g, seed=8))


def test_overlap_assertion_catches_row_level_split():
    g = _groups()
    leaky = random_split(len(g), val_fraction=0.2, test_fraction=0.2, seed=0)
    with pytest.raises(GroupLeakError, match="more than one split"):
        assert_no_group_overlap(g, leaky)


def test_overlap_assertion_catches_single_row_and_empty_split():
    g = np.array(["a", "a", "b", "b", "c"])
    with pytest.raises(GroupLeakError, match="1 of 3 groups"):
        assert_no_group_overlap(g, np.array(["train", "test", "train", "train", "test"]))
    with pytest.raises(GroupLeakError, match="required splits"):
        assert_no_group_overlap(g, np.array(["train"] * 5), required=("train", "test"))
    assert issubclass(GroupLeakError, AssertionError)


def test_group_kfold_whole_groups_and_balanced_rows():
    g = _groups(30)
    folds = group_kfold(g, n_splits=5, seed=1)
    assert_no_group_overlap(g, folds)
    counts = np.bincount(folds, minlength=5)
    assert counts.min() > 0
    assert counts.max() - counts.min() <= 200  # within one largest group
    assert np.array_equal(folds, group_kfold(g, n_splits=5, seed=1))


def test_split_report_counts_and_missing_classes():
    g = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 2)
    s = np.array(["train"] * 8 + ["test"] * 2)
    y = np.array(["x", "x", "y", "y", "x", "z", "z", "y", "x", "x"])
    rep = split_report(g, s, y)
    assert list(rep.per_split.index) == ["train", "test"]
    assert rep.per_split.loc["train", "n_rows"] == 8
    assert rep.per_split.loc["test", "n_groups"] == 1
    assert rep.class_counts.loc["x", "test"] == 2
    assert rep.missing_classes["test"] == ["y", "z"]
    assert rep.missing_classes["train"] == []
    assert np.isclose(rep.max_class_gap, 1.0 - 3 / 8)  # x: 3/8 of train, all of test
    text = rep.summary()
    assert "test has no rows of 2 classes" in text and chr(0x2014) not in text  # no em dash


def test_split_report_without_labels():
    rep = split_report(["a", "b", "c"], ["train", "train", "test"])
    assert rep.class_counts.empty and np.isnan(rep.max_class_gap)
    assert "test: 1 rows" in rep.summary()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"test_fraction": 0.0}, "test_fraction"),
        ({"test_fraction": 1.0}, "test_fraction"),
        ({"val_fraction": -0.1}, "val_fraction"),
        ({"val_fraction": 0.5, "test_fraction": 0.5}, "too few"),
    ],
)
def test_group_split_rejects_bad_fractions(kwargs, match):
    with pytest.raises(ValueError, match=match):
        group_split(["a", "b", "c"], **kwargs)


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="missing values"):
        group_split(np.array(["a", None, "b"], dtype=object))
    with pytest.raises(ValueError, match="expected 3"):
        assert_no_group_overlap(["a", "b", "c"], ["train", "test"])
    with pytest.raises(ValueError, match="more than the 2 groups"):
        group_kfold(["a", "b"], n_splits=3)
    with pytest.raises(ValueError, match="n_splits"):
        group_kfold(["a", "b"], n_splits=1)
    with pytest.raises(TypeError, match="column name"):
        group_split("donor_id")
    with pytest.raises(ValueError, match="one-dimensional"):
        group_split(np.zeros((3, 2)))
