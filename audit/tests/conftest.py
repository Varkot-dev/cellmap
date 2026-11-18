"""Synthetic datasets shared by the tests. Everything is small so the suite runs in seconds."""

import numpy as np
import pytest


def make_group_leak(n_groups: int = 40, per_group: int = 25, n_features: int = 8, seed: int = 0):
    """Planted leakage: each group has its own feature offset and mostly one label.

    Features carry no information about the label except through the group, so a model can
    only score above chance by recognising groups it has seen in training.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, 3, size=(n_groups, n_features))
    group_label = rng.permutation(np.arange(n_groups) % 2)  # half the groups lean to class 1
    groups = np.repeat(np.arange(n_groups), per_group)
    X = offsets[groups] + rng.normal(0, 0.5, size=(len(groups), n_features))
    flip = rng.random(len(groups)) < 0.1
    y = np.where(flip, 1 - group_label[groups], group_label[groups])
    return X, y, np.array([f"g{i}" for i in groups])


def make_no_leak(n_groups: int = 40, per_group: int = 25, n_features: int = 8, seed: int = 0):
    """No leakage: the label depends on the features the same way in every group."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(size=(len(groups), n_features))
    y = (X[:, 0] + X[:, 1] + rng.normal(0, 0.5, len(groups)) > 0).astype(int)
    return X, y, np.array([f"g{i}" for i in groups])


@pytest.fixture
def group_leak_data():
    return make_group_leak()


@pytest.fixture
def no_leak_data():
    return make_no_leak()
