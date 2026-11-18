import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from leakaudit import shortcut_probe

BLOCKS = {"used": [0, 1, 2, 3, 4], "ignored": [5, 6, 7, 8, 9]}


@pytest.fixture
def probe_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 10))
    y = (X[:, :5].sum(axis=1) > 0).astype(int)
    return X, y


def uses_first_block(X):
    """A model that by construction reads only columns 0 to 4."""
    return (np.asarray(X)[:, :5].sum(axis=1) > 0).astype(int)


def test_ignored_block_has_zero_drop_and_used_block_large(probe_data):
    X, y = probe_data
    res = shortcut_probe(uses_first_block, X, y, BLOCKS, accuracy_score, n_repeats=3)
    table = res.table.set_index("probe")
    assert res.baseline == 1.0
    assert table.loc["ignored shuffled", "drop_mean"] == 0.0
    assert table.loc["used shuffled", "drop_mean"] > 0.4  # shuffled rows: chance accuracy
    assert res.n_uncovered_columns == 0
    assert "The model does not need: ignored shuffled" in res.summary()


def test_fitted_estimator_learns_to_ignore_noise_block(probe_data):
    X, y = probe_data
    model = LogisticRegression().fit(X[:400], y[:400])
    res = shortcut_probe(model, X[400:], y[400:], BLOCKS, accuracy_score, constant="mean")
    table = res.table.set_index("probe")
    assert abs(table.loc["ignored shuffled", "drop_mean"]) < 0.05
    assert table.loc["used shuffled", "drop_mean"] > 0.3
    assert table.loc["used constant", "drop_mean"] > 0.3
    assert set(res.table["kind"]) == {"shuffle", "constant"}


def test_dataframe_blocks_by_name_and_custom_probe(probe_data):
    X, y = probe_data
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(10)])
    blocks = {"used": [f"c{i}" for i in range(5)], "ignored": [f"c{i}" for i in range(5, 10)]}

    def sign_only(x, rng):
        return np.sign(x)

    res = shortcut_probe(
        uses_first_block, df, y, blocks, accuracy_score, constant=0.0, custom={"signs": sign_only}
    )
    table = res.table.set_index("probe")
    assert table.loc["ignored shuffled", "drop_mean"] == 0.0
    assert table.loc["ignored constant", "drop_mean"] == 0.0
    assert table.loc["used constant", "score_mean"] < 0.6  # all zeros: one class predicted
    assert "signs" in table.index


def test_probe_is_deterministic_and_block_order_independent(probe_data):
    X, y = probe_data
    a = shortcut_probe(uses_first_block, X, y, BLOCKS, accuracy_score, seed=4)
    b = shortcut_probe(
        uses_first_block, X, y, dict(reversed(BLOCKS.items())), accuracy_score, seed=4
    )
    a_t, b_t = a.table.set_index("probe"), b.table.set_index("probe").loc[a.table["probe"]]
    pd.testing.assert_frame_equal(a_t, b_t)
    c = shortcut_probe(uses_first_block, X, y, BLOCKS, accuracy_score, seed=5)
    assert c.table.loc[0, "score_mean"] != a.table.loc[0, "score_mean"]


def test_uncovered_columns_and_masks(probe_data):
    X, y = probe_data
    mask = np.zeros(10, dtype=bool)
    mask[:5] = True
    res = shortcut_probe(uses_first_block, X, y, {"used": mask}, accuracy_score, n_repeats=1)
    assert res.n_uncovered_columns == 5 and np.isnan(res.table.loc[0, "drop_std"])
    res = shortcut_probe(uses_first_block, X, y, {"last": slice(5, None)}, accuracy_score)
    assert res.table.loc[0, "n_columns"] == 5


@pytest.mark.parametrize(
    "blocks, kwargs, error, match",
    [
        ({"a": [10]}, {}, ValueError, "out of range"),
        ({"a": []}, {}, ValueError, "no columns"),
        ({"a": ["x"]}, {}, ValueError, "DataFrame"),
        ({"a": np.ones(3, dtype=bool)}, {}, ValueError, "boolean mask"),
        ({}, {}, ValueError, "at least one block"),
        (BLOCKS, {"n_repeats": 0}, ValueError, "n_repeats"),
        (BLOCKS, {"constant": "median"}, ValueError, "constant"),
        (BLOCKS, {"prediction": "nope"}, ValueError, "prediction"),
    ],
)
def test_probe_rejects_bad_arguments(probe_data, blocks, kwargs, error, match):
    X, y = probe_data
    with pytest.raises(error, match=match):
        shortcut_probe(uses_first_block, X, y, blocks, accuracy_score, **kwargs)


def test_probe_rejects_bad_inputs(probe_data):
    X, y = probe_data
    with pytest.raises(ValueError, match="expected"):
        shortcut_probe(uses_first_block, X, y[:-1], BLOCKS, accuracy_score)
    with pytest.raises(ValueError, match="2-D"):
        shortcut_probe(uses_first_block, X[:, 0], y, BLOCKS, accuracy_score)
    with pytest.raises(TypeError, match="model must"):
        shortcut_probe(object(), X, y, BLOCKS, accuracy_score)
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(10)])
    with pytest.raises(KeyError, match="not in X"):
        shortcut_probe(uses_first_block, df, y, {"a": ["zz"]}, accuracy_score)
