import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier

from leakaudit import measure_inflation, measure_preprocessing_leak, random_split

SEEDS = (0, 1, 2, 3, 4)


def test_planted_group_leak_shows_clear_inflation(group_leak_data):
    X, y, g = group_leak_data
    res = measure_inflation(X, y, g, KNeighborsClassifier(5), accuracy_score, seeds=SEEDS)
    assert res.leaky_mean > 0.8  # near-copies of each test row's group are in training
    assert abs(res.honest_mean - 0.5) < 0.15  # unseen groups: the label cannot be read off
    assert res.inflation > 0.25
    assert res.inflation > 3 * res.inflation_se
    random_rows = res.scores[res.scores["scheme"] == "random row split"]
    group_rows = res.scores[res.scores["scheme"] == "group split"]
    assert (random_rows["frac_test_rows_group_in_train"] == 1.0).all()
    assert (group_rows["frac_test_rows_group_in_train"] == 0.0).all()
    assert "clear effect" in res.summary()


def test_no_leak_dataset_shows_near_zero_inflation(no_leak_data):
    X, y, g = no_leak_data
    res = measure_inflation(X, y, g, LogisticRegression(), accuracy_score, seeds=SEEDS)
    assert res.honest_mean > 0.8
    assert abs(res.inflation) < 0.03
    assert "no clear leak" in res.summary()


def test_inflation_is_deterministic(group_leak_data):
    X, y, g = group_leak_data
    a = measure_inflation(X, y, g, KNeighborsClassifier(5), accuracy_score, seeds=(0, 1))
    b = measure_inflation(X, y, g, KNeighborsClassifier(5), accuracy_score, seeds=(0, 1))
    pd.testing.assert_frame_equal(a.scores, b.scores)
    c = measure_inflation(X, y, g, KNeighborsClassifier(5), accuracy_score, seeds=(2, 3))
    assert not a.scores["score"].equals(c.scores["score"])


def test_callable_model_and_row_index_trick(group_leak_data):
    X, y, g = group_leak_data
    seen = []

    def fit_predict(train_idx, y_train, test_idx):
        seen.append(len(train_idx))
        model = KNeighborsClassifier(5).fit(X[train_idx], y_train)
        return model.predict(X[test_idx])

    res = measure_inflation(np.arange(len(y)), y, g, fit_predict, accuracy_score, seeds=(0,))
    est = measure_inflation(X, y, g, KNeighborsClassifier(5), accuracy_score, seeds=(0,))
    assert np.allclose(res.scores["score"], est.scores["score"])
    assert len(seen) == 2
    assert np.isnan(res.inflation_se) and "one seed" in res.summary()


def test_estimator_passed_in_is_not_fitted(group_leak_data):
    X, y, g = group_leak_data
    model = LogisticRegression()
    measure_inflation(X, y, g, model, accuracy_score, seeds=(0,))
    assert not hasattr(model, "coef_")


def test_loss_metric_with_probabilities(group_leak_data):
    X, y, g = group_leak_data
    res = measure_inflation(
        X,
        y,
        g,
        KNeighborsClassifier(15),
        lambda t, p: log_loss(t, p, labels=[0, 1]),
        seeds=(0, 1, 2),
        prediction="predict_proba",
        greater_is_better=False,
        metric_name="log loss",
    )
    assert res.inflation > 0  # the random split's loss is lower, so it looks better
    assert "log loss" in res.summary()


def test_validation_rows_are_set_aside(group_leak_data):
    X, y, g = group_leak_data
    res = measure_inflation(
        X, y, g, KNeighborsClassifier(5), accuracy_score, seeds=(0,), val_fraction=0.2
    )
    row = res.scores.iloc[0]
    assert row["n_train"] == len(y) - 2 * round(0.2 * len(y))


def test_random_split_sizes_and_determinism():
    s = random_split(100, val_fraction=0.2, test_fraction=0.3, seed=0)
    assert (s == "test").sum() == 30 and (s == "val").sum() == 20
    assert np.array_equal(s, random_split(100, val_fraction=0.2, test_fraction=0.3, seed=0))


def test_preprocessing_leak_feature_selection_on_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2000))  # pure noise: no feature predicts the label
    y = rng.permutation(np.arange(120) % 2)
    res = measure_preprocessing_leak(
        X,
        y,
        SelectKBest(f_classif, k=20),
        LogisticRegression(),
        accuracy_score,
        seeds=SEEDS,
        test_fraction=0.3,
    )
    assert res.leaky_mean > 0.75  # features chosen using the test labels look predictive
    assert abs(res.honest_mean - 0.5) < 0.15
    assert res.inflation > 0.2 and res.paired
    assert res.summary_frame().loc[0, "inflation"] == pytest.approx(res.inflation)


def test_preprocessing_leak_near_zero_for_label_free_scaling(no_leak_data):
    from sklearn.preprocessing import StandardScaler

    X, y, g = no_leak_data
    res = measure_preprocessing_leak(
        X, y, StandardScaler(), LogisticRegression(), accuracy_score, groups=g, seeds=SEEDS
    )
    assert abs(res.inflation) < 0.02


def test_anndata_input_with_obs_column_names(group_leak_data):
    ad = pytest.importorskip("anndata")
    X, y, g = group_leak_data
    obs = pd.DataFrame({"label": y, "donor": g}, index=[f"c{i}" for i in range(len(y))])
    adata = ad.AnnData(X=X, obs=obs)

    def fit_predict(a_train, y_train, a_test):
        return KNeighborsClassifier(5).fit(a_train.X, y_train).predict(a_test.X)

    res = measure_inflation(adata, "label", "donor", fit_predict, accuracy_score, seeds=(0, 1))
    ref = measure_inflation(X, y, g, KNeighborsClassifier(5), accuracy_score, seeds=(0, 1))
    assert np.allclose(res.scores["score"], ref.scores["score"])


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"seeds": ()}, ValueError, "seeds is empty"),
        ({"seeds": (1, 1)}, ValueError, "distinct"),
        ({"prediction": "transform"}, ValueError, "prediction must be"),
        ({"test_fraction": 0}, ValueError, "test_fraction"),
    ],
)
def test_inflation_rejects_bad_arguments(group_leak_data, kwargs, error, match):
    X, y, g = group_leak_data
    with pytest.raises(error, match=match):
        measure_inflation(X, y, g, KNeighborsClassifier(5), accuracy_score, **kwargs)


def test_inflation_rejects_bad_inputs(group_leak_data):
    X, y, g = group_leak_data
    with pytest.raises(ValueError, match="expected"):
        measure_inflation(X, y[:-1], g, KNeighborsClassifier(5), accuracy_score)
    with pytest.raises(TypeError, match="model must"):
        measure_inflation(X, y, g, object(), accuracy_score)
    with pytest.raises(TypeError, match="preprocess must"):
        measure_preprocessing_leak(X, y, object(), KNeighborsClassifier(5), accuracy_score)
    with pytest.raises(ValueError, match="metric returned"):
        measure_inflation(X, y, g, KNeighborsClassifier(5), lambda t, p: float("nan"), seeds=(0,))
