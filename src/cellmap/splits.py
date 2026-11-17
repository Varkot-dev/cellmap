"""Donor-level train / validation / test splits."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def split_donors(
    donors: Iterable[str], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    """Assign each donor to exactly one of train / val / test.

    Whole donors are assigned, never individual cells: cells from one donor share genetics,
    sample handling and sequencing run, so splitting them across train and test would let a
    model score well by recognising the donor instead of the biology.

    The result depends only on the set of donor ids and the seed, not on input order.
    """
    unique = sorted(set(map(str, donors)))
    n = len(unique)
    n_val = max(1, round(val_fraction * n))
    n_test = max(1, round(test_fraction * n))
    if n - n_val - n_test < 1:
        raise ValueError(f"{n} donors is too few for val={n_val} and test={n_test}")

    order = np.random.default_rng(seed).permutation(n)
    shuffled = [unique[i] for i in order]
    assignment = {d: "test" for d in shuffled[:n_test]}
    assignment.update({d: "val" for d in shuffled[n_test : n_test + n_val]})
    assignment.update({d: "train" for d in shuffled[n_test + n_val :]})
    return assignment


def assert_no_donor_overlap(
    obs: pd.DataFrame, donor_key: str = "donor_id", split_key: str = "split"
) -> None:
    """Raise if any donor has cells in more than one split, or a split is empty."""
    splits_per_donor = obs.groupby(donor_key, observed=True)[split_key].nunique()
    leaked = splits_per_donor[splits_per_donor > 1]
    if len(leaked) > 0:
        raise AssertionError(f"donors in more than one split: {sorted(leaked.index)}")
    missing = set(SPLITS) - set(obs[split_key].astype(str).unique())
    if missing:
        raise AssertionError(f"empty splits: {sorted(missing)}")
