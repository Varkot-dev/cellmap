"""Measure how much a leaky evaluation overstates a score, instead of asserting that it does.

Two leaks are measured, each by scoring the same model twice and taking the difference:

- `measure_inflation`: random row split (leaky when rows of one group are related) versus a
  group split (honest), over several seeds.
- `measure_preprocessing_leak`: preprocessing (feature selection, scaling) fitted on all rows
  versus on training rows only, on the same splits.

A model can be a scikit-learn style estimator (anything with `fit` and `predict`; it is
cloned for every split, so the object you pass is never fitted) or a plain function
`fit_predict(X_train, y_train, X_test) -> predictions`. The function form covers models that
are not estimators, and pipelines that need to see the whole dataset: pass
`X = np.arange(n_rows)` and the function receives row indices and can look up anything itself.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from leakaudit._utils import (
    as_1d,
    as_str_groups,
    check_fraction,
    check_same_length,
    check_seeds,
    n_rows,
    take_rows,
)
from leakaudit.splits import assert_no_group_overlap, group_split

Metric = Callable[[Any, Any], float]
PREDICTION_METHODS = ("predict", "predict_proba", "decision_function")


def random_split(
    n: int, *, val_fraction: float = 0.2, test_fraction: float = 0.2, seed: int = 0
) -> np.ndarray:
    """Per-row split names assigned to ROWS at random: the leaky split, for comparison.

    round(test_fraction * n) rows are test and round(val_fraction * n) are validation, chosen
    from one permutation of the rows (test first). This is the same procedure as
    `random_cell_split` in cellmap's `scripts/leakage_splits.py`.
    """
    if int(n) != n or n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n}")
    check_fraction(val_fraction, "val_fraction", allow_zero=True)
    check_fraction(test_fraction, "test_fraction", allow_zero=False)
    n_test, n_val = round(test_fraction * n), round(val_fraction * n)
    if n_test < 1 or n - n_test - n_val < 1:
        raise ValueError(f"{n} rows is too few for {n_test} test and {n_val} validation rows")
    order = np.random.default_rng(seed).permutation(int(n))
    split = np.full(int(n), "train", dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test : n_test + n_val]] = "val"
    return split


def _check_model(model: Any, prediction: str) -> None:
    if prediction not in PREDICTION_METHODS:
        raise ValueError(f"prediction must be one of {PREDICTION_METHODS}, got {prediction!r}")
    is_estimator = hasattr(model, "fit") and hasattr(model, prediction)
    if not is_estimator and not callable(model):
        raise TypeError(
            f"model must have fit() and {prediction}(), or be a function "
            "fit_predict(X_train, y_train, X_test)"
        )


def _fresh(estimator: Any) -> Any:
    """An unfitted copy: sklearn.base.clone when it works, otherwise a deep copy."""
    try:
        from sklearn.base import clone

        return clone(estimator)
    except Exception:  # not a scikit-learn estimator; a deep copy is the best we can do
        return copy.deepcopy(estimator)


def _reduce(pred: Any, prediction: str) -> Any:
    pred = np.asarray(pred) if not isinstance(pred, pd.Series | pd.DataFrame) else pred
    if prediction == "predict_proba" and getattr(pred, "ndim", 1) == 2 and pred.shape[1] == 2:
        return np.asarray(pred)[:, 1]  # binary: positive-class probability, as roc_auc_score wants
    return pred


def _fit_predict(
    model: Any, x_train: Any, y_train: np.ndarray, x_test: Any, prediction: str
) -> Any:
    if hasattr(model, "fit") and hasattr(model, prediction):
        est = _fresh(model)
        est.fit(x_train, y_train)
        return _reduce(getattr(est, prediction)(x_test), prediction)
    return _reduce(model(x_train, y_train, x_test), prediction)


def _score(metric: Metric, y_true: np.ndarray, y_pred: Any) -> float:
    value = float(metric(y_true, y_pred))
    if not np.isfinite(value):
        raise ValueError(f"metric returned {value}")
    return value


@dataclass(frozen=True)
class InflationResult:
    """The same model scored under a leaky and an honest evaluation, over several seeds.

    Attributes:
        scores: one row per (scheme, seed) with the score and split sizes.
        leaky_name, honest_name: the two values of `scores["scheme"]`.
        metric_name: name of the metric, for the summary.
        greater_is_better: if False (a loss), inflation is honest minus leaky.
        paired: True when both schemes used the same rows for each seed (preprocessing check),
            so the standard error comes from per-seed differences.
    """

    scores: pd.DataFrame
    leaky_name: str
    honest_name: str
    metric_name: str
    greater_is_better: bool
    paired: bool

    def _values(self, scheme: str) -> np.ndarray:
        frame = self.scores[self.scores["scheme"] == scheme].sort_values("seed")
        return frame["score"].to_numpy(dtype=float)

    @property
    def n_seeds(self) -> int:
        return len(self._values(self.honest_name))

    @property
    def leaky_mean(self) -> float:
        return float(self._values(self.leaky_name).mean())

    @property
    def honest_mean(self) -> float:
        return float(self._values(self.honest_name).mean())

    @property
    def leaky_std(self) -> float:
        v = self._values(self.leaky_name)
        return float(v.std(ddof=1)) if len(v) > 1 else float("nan")

    @property
    def honest_std(self) -> float:
        v = self._values(self.honest_name)
        return float(v.std(ddof=1)) if len(v) > 1 else float("nan")

    @property
    def inflation(self) -> float:
        """How much the leaky evaluation overstates performance (positive = overstated)."""
        diff = self.leaky_mean - self.honest_mean
        return diff if self.greater_is_better else -diff

    @property
    def inflation_se(self) -> float:
        """Standard error of `inflation` across seeds (NaN with one seed).

        Unpaired (random vs group split): the two splits for one seed are unrelated, so
        sqrt(sd_leaky^2 / n + sd_honest^2 / n). Paired (preprocessing check): sd of the
        per-seed differences / sqrt(n).
        """
        n = self.n_seeds
        if n < 2:
            return float("nan")
        if self.paired:
            diff = self._values(self.leaky_name) - self._values(self.honest_name)
            return float(diff.std(ddof=1) / np.sqrt(n))
        return float(np.sqrt(self.leaky_std**2 / n + self.honest_std**2 / n))

    def summary_frame(self) -> pd.DataFrame:
        """One-row table: means, standard deviations, inflation and its standard error."""
        return pd.DataFrame(
            [
                {
                    "metric": self.metric_name,
                    "n_seeds": self.n_seeds,
                    "leaky_scheme": self.leaky_name,
                    "leaky_mean": self.leaky_mean,
                    "leaky_std": self.leaky_std,
                    "honest_scheme": self.honest_name,
                    "honest_mean": self.honest_mean,
                    "honest_std": self.honest_std,
                    "inflation": self.inflation,
                    "inflation_se": self.inflation_se,
                    "paired": self.paired,
                }
            ]
        )

    def summary(self) -> str:
        """Plain-language description of the result."""
        n, m = self.n_seeds, self.metric_name

        def sd(x: float) -> str:
            return "" if np.isnan(x) else f" (sd {x:.3f} over {n} seeds)"

        lines = [
            f"{self.leaky_name}: mean {m} {self.leaky_mean:.3f}{sd(self.leaky_std)}.",
            f"{self.honest_name}: mean {m} {self.honest_mean:.3f}{sd(self.honest_std)}.",
        ]
        infl, se = self.inflation, self.inflation_se
        word = "overstates" if infl >= 0 else "understates"
        se_text = "" if np.isnan(se) else f" (standard error {se:.3f})"
        lines.append(f"The leaky evaluation {word} {m} by {abs(infl):.3f}{se_text}.")
        if np.isnan(se):
            lines.append("With one seed there is no estimate of how much this varies.")
        elif abs(infl) < 2 * se:
            lines.append(
                "That is within two standard errors of zero: no clear leak at this many seeds."
            )
        else:
            lines.append("That is more than two standard errors from zero: a clear effect.")
        if not self.paired and n > 1 and self.honest_std > 2 * self.leaky_std:
            lines.append(
                f"The honest scores also vary more across seeds (sd {self.honest_std:.3f} versus "
                f"{self.leaky_std:.3f}), so the leaky evaluation hides real uncertainty too."
            )
        return "\n".join(lines)


def _metric_name(metric: Metric, metric_name: str | None) -> str:
    return metric_name or getattr(metric, "__name__", "score")


def measure_inflation(
    X: Any,
    y: Any,
    groups: Any,
    model: Any,
    metric: Metric,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    val_fraction: float = 0.0,
    test_fraction: float = 0.2,
    prediction: str = "predict",
    greater_is_better: bool = True,
    metric_name: str | None = None,
) -> InflationResult:
    """Score one model under a random row split and a group split; return the difference.

    Why: when rows of one group (a patient's cells or visits, a lab's samples, a scaffold's
    molecules) are related, a random split puts near-copies of test rows in training, and
    the score rewards recognising the group. Holding out whole groups removes that. Running
    both on the same model and data turns "random splits leak" from a warning into a number
    for this dataset: inflation = mean(random) - mean(group), with its spread over seeds.

    For each seed, a random row split and a group split are made with that seed (validation
    rows, if any, are set aside and not used), the model is fitted on the training rows and
    scored on the test rows. The fractions mean rows for the random split and groups for the
    group split, so the two test sets can differ in size; `scores` records both sizes.

    The model's own randomness (for example a random forest's random_state) is left as you set
    it, so differences come from the splits alone.

    Args:
        X: features: array, sparse matrix, DataFrame, AnnData, or row indices (see module doc).
        y: one label (or target) per row; with an AnnData X, an obs column name also works.
        groups: one group id per row; with an AnnData X, an obs column name also works.
        model: estimator with fit/predict, or a function fit_predict(X_train, y_train, X_test).
        metric: metric(y_true, y_pred) -> float, for example sklearn.metrics.f1_score.
        seeds: seeds to repeat over; at least two for a spread.
        val_fraction: fraction set aside as validation in both splits (0 = none).
        test_fraction: fraction held out for test (rows or groups, see above).
        prediction: "predict", "predict_proba" or "decision_function" (estimators only).
            For binary predict_proba, the positive-class column is passed to the metric.
        greater_is_better: set False for losses, so a positive inflation still means "the
            leaky evaluation looks better than it should".
        metric_name: name used in the summary (defaults to the metric function's name).
    """
    n = n_rows(X)
    yy = as_1d(y, "y", X)
    g = as_str_groups(groups, data=X)
    check_same_length(n, y=yy, groups=g)
    seed_list = check_seeds(seeds)
    _check_model(model, prediction)

    rows = []
    for seed in seed_list:
        schemes = {
            "random row split": random_split(
                n, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
            ),
            "group split": group_split(
                g, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
            ),
        }
        assert_no_group_overlap(g, schemes["group split"], required=("train", "test"))
        for scheme, split in schemes.items():
            train, test = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
            pred = _fit_predict(
                model, take_rows(X, train), yy[train], take_rows(X, test), prediction
            )
            train_groups = set(g[train])
            rows.append(
                {
                    "scheme": scheme,
                    "seed": seed,
                    "score": _score(metric, yy[test], pred),
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_test_groups": len(set(g[test])),
                    "frac_test_rows_group_in_train": float(
                        np.mean([x in train_groups for x in g[test]])
                    ),
                }
            )
    return InflationResult(
        scores=pd.DataFrame(rows),
        leaky_name="random row split",
        honest_name="group split",
        metric_name=_metric_name(metric, metric_name),
        greater_is_better=greater_is_better,
        paired=False,
    )


def measure_preprocessing_leak(
    X: Any,
    y: Any,
    preprocess: Any,
    model: Any,
    metric: Metric,
    *,
    groups: Any = None,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    val_fraction: float = 0.0,
    test_fraction: float = 0.2,
    prediction: str = "predict",
    greater_is_better: bool = True,
    metric_name: str | None = None,
) -> InflationResult:
    """Score a model with preprocessing fitted on all rows versus on training rows only.

    Why: a step that learns from data (choosing the genes most associated with the label,
    scaling with a mean and variance, imputing) is part of the model. Fitting it on all rows
    lets the test rows shape the features before the model is ever tested; feature selection
    using the labels is the classic case, and with many features it can make pure noise look
    predictive. This check runs both versions on identical splits, so the difference is due
    to the preprocessing alone.

    Leaky arm: preprocess.fit(X_all, y_all), transform all rows, then fit the model on the
    training rows and score on the test rows. Honest arm: fit the preprocessing on the
    training rows only and apply it, unchanged, to the test rows.

    Args:
        X: features (array, sparse matrix or DataFrame).
        y: one label per row.
        preprocess: a scikit-learn transformer (fit(X, y) and transform(X)), for example
            SelectKBest or StandardScaler, or a Pipeline of them. It is cloned for every fit.
        model: estimator with fit/predict, or a function fit_predict(X_train, y_train, X_test).
        metric: metric(y_true, y_pred) -> float.
        groups: if given, splits are group splits (recommended, so the only leak measured is
            the preprocessing one); if None, random row splits.
        seeds, val_fraction, test_fraction, prediction, greater_is_better, metric_name: as
            in `measure_inflation`.
    """
    n = n_rows(X)
    yy = as_1d(y, "y", X)
    g = None if groups is None else as_str_groups(groups, data=X)
    check_same_length(n, y=yy, groups=g)
    seed_list = check_seeds(seeds)
    _check_model(model, prediction)
    if not (hasattr(preprocess, "fit") and hasattr(preprocess, "transform")):
        raise TypeError("preprocess must have fit() and transform()")

    leaky_name = "preprocessing fitted on all rows"
    honest_name = "preprocessing fitted on training rows"
    rows = []
    for seed in seed_list:
        if g is None:
            split = random_split(
                n, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
            )
        else:
            split = group_split(
                g, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
            )
            assert_no_group_overlap(g, split, required=("train", "test"))
        train, test = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
        x_train, x_test = take_rows(X, train), take_rows(X, test)

        x_all = _fresh(preprocess).fit(X, yy).transform(X)
        leaky = _fit_predict(
            model, take_rows(x_all, train), yy[train], take_rows(x_all, test), prediction
        )
        fitted = _fresh(preprocess).fit(x_train, yy[train])
        honest = _fit_predict(
            model, fitted.transform(x_train), yy[train], fitted.transform(x_test), prediction
        )
        for scheme, pred in ((leaky_name, leaky), (honest_name, honest)):
            rows.append(
                {
                    "scheme": scheme,
                    "seed": seed,
                    "score": _score(metric, yy[test], pred),
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
    return InflationResult(
        scores=pd.DataFrame(rows),
        leaky_name=leaky_name,
        honest_name=honest_name,
        metric_name=_metric_name(metric, metric_name),
        greater_is_better=greater_is_better,
        paired=True,
    )
