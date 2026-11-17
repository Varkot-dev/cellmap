import pandas as pd
import pytest

from cellmap.splits import assert_no_donor_overlap, split_donors

DONORS = [f"d{i}" for i in range(21)]


def test_every_donor_assigned_once_with_expected_sizes():
    a = split_donors(DONORS, val_fraction=0.2, test_fraction=0.2, seed=0)
    assert set(a) == set(DONORS)
    sizes = pd.Series(a).value_counts().to_dict()
    assert sizes == {"train": 13, "val": 4, "test": 4}


def test_deterministic_and_independent_of_input_order():
    a = split_donors(DONORS, 0.2, 0.2, seed=3)
    b = split_donors(list(reversed(DONORS)) + DONORS, 0.2, 0.2, seed=3)
    assert a == b


def test_different_seeds_give_different_splits():
    assert split_donors(DONORS, 0.2, 0.2, seed=0) != split_donors(DONORS, 0.2, 0.2, seed=1)


def test_too_few_donors_raises():
    with pytest.raises(ValueError):
        split_donors(["a", "b"], 0.2, 0.2, seed=0)


def test_overlap_check_passes_on_donor_split(toy_adata):
    a = split_donors(toy_adata.obs["donor_id"], 0.2, 0.2, seed=0)
    toy_adata.obs["split"] = toy_adata.obs["donor_id"].map(a)
    assert_no_donor_overlap(toy_adata.obs)


def test_overlap_check_catches_a_cell_level_split(toy_adata):
    # Assigning cells (not donors) to splits puts the same donor on both sides.
    toy_adata.obs["split"] = ["train", "val", "test"] * 20
    with pytest.raises(AssertionError, match="more than one split"):
        assert_no_donor_overlap(toy_adata.obs)


def test_overlap_check_catches_an_empty_split(toy_adata):
    toy_adata.obs["split"] = "train"
    with pytest.raises(AssertionError, match="empty splits"):
        assert_no_donor_overlap(toy_adata.obs)
