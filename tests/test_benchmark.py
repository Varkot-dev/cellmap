import numpy as np
import pandas as pd
import pytest

from cellmap.benchmark import dataset_out_split


def obs_three_datasets() -> pd.DataFrame:
    rows = []
    for ds, n_donors in (("a", 6), ("b", 6), ("c", 4)):
        for d in range(n_donors):
            rows += [{"dataset": ds, "donor_id": f"{ds}{d}"}] * 5
    return pd.DataFrame(rows)


def test_held_out_dataset_is_all_test_and_nothing_else_is():
    obs = obs_three_datasets()
    split = dataset_out_split(obs, "c")
    assert (split[obs["dataset"] == "c"] == "test").all()
    assert (split[obs["dataset"] != "c"] != "test").all()


def test_train_and_val_share_no_donor():
    obs = obs_three_datasets()
    split = dataset_out_split(obs, "a", val_fraction=0.25, seed=3)
    per_donor = pd.Series(split).groupby(obs["donor_id"]).nunique()
    assert (per_donor == 1).all()
    assert set(split) == {"train", "val", "test"}


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        dataset_out_split(obs_three_datasets(), "z")


def test_deterministic():
    obs = obs_three_datasets()
    assert np.array_equal(dataset_out_split(obs, "b", seed=1), dataset_out_split(obs, "b", seed=1))
