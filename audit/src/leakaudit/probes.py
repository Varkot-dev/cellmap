"""Shortcut probes: which parts of the input does a fitted model actually use?

A model can score well for the wrong reason: a drug-target model that only looks at the
ligand, a patient classifier that reads the hospital from batch-specific features, a cell
model that only checks which genes are on and ignores how much. Scores alone cannot tell
these apart. The probe can: keep the fitted model fixed, break one named block of the input
at inference time, and see how much the metric drops. A block whose damage costs nothing is a
block the model does not need.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from leakaudit._utils import as_1d, check_same_length
from leakaudit.inflation import PREDICTION_METHODS, Metric, _reduce, _score

BlockSpec = Sequence[int] | Sequence[str] | slice | np.ndarray
CustomProbe = Callable[[Any, np.random.Generator], Any]


def _resolve_block(name: str, spec: Any, X: Any) -> np.ndarray:
    """Integer column positions for one block."""
    n_cols = X.shape[1]
    if isinstance(spec, slice):
        cols = np.arange(n_cols)[spec]
    else:
        arr = np.asarray(spec)
        if arr.size == 0:
            raise ValueError(f"block {name!r} has no columns")
        if arr.dtype == bool:
            if arr.shape != (n_cols,):
                raise ValueError(f"block {name!r}: boolean mask must have {n_cols} entries")
            cols = np.flatnonzero(arr)
        elif arr.dtype.kind in "iu":
            cols = arr.astype(np.int64)
            if cols.size and (cols.min() < -n_cols or cols.max() >= n_cols):
                raise ValueError(f"block {name!r}: column index out of range 0..{n_cols - 1}")
            cols = np.where(cols < 0, cols + n_cols, cols)
        else:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"block {name!r}: column names need X to be a DataFrame")
            missing = [c for c in arr.tolist() if c not in X.columns]
            if missing:
                raise KeyError(f"block {name!r}: columns not in X: {missing[:5]}")
            cols = X.columns.get_indexer(arr.tolist())
    cols = np.unique(cols)
    if cols.size == 0:
        raise ValueError(f"block {name!r} has no columns")
    return cols


def _predictor(model: Any, prediction: str) -> Callable[[Any], Any]:
    if prediction not in PREDICTION_METHODS:
        raise ValueError(f"prediction must be one of {PREDICTION_METHODS}, got {prediction!r}")
    if hasattr(model, prediction):
        method = getattr(model, prediction)
        return lambda x: _reduce(method(x), prediction)
    if callable(model):
        return lambda x: _reduce(model(x), prediction)
    raise TypeError(f"model must have {prediction}() or be a function predict(X)")


def _shuffled(X: Any, cols: np.ndarray, perm: np.ndarray) -> Any:
    """Copy of X where the block's columns are moved together to other rows."""
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        for c in cols:  # column by column keeps each column's dtype
            out.iloc[:, c] = X.iloc[perm, c].to_numpy()
        return out
    out = np.array(X, copy=True)
    out[:, cols] = X[perm][:, cols]
    return out


def _constant(X: Any, cols: np.ndarray, value: Any) -> Any:
    """Copy of X where every column of the block holds one value for all rows."""
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        for c in cols:
            col = X.iloc[:, c]
            if value != "mean":
                fill = value
            elif pd.api.types.is_numeric_dtype(col):
                fill = col.mean()
            else:
                fill = col.mode().iloc[0]  # non-numeric: the most common value
            out.iloc[:, c] = fill
        return out
    out = np.array(X, copy=True, dtype=float if value == "mean" else None)
    out[:, cols] = X[:, cols].mean(axis=0) if value == "mean" else value
    return out


@dataclass(frozen=True)
class ProbeResult:
    """Metric drop per damaged input block.

    Attributes:
        baseline: the metric on the undamaged evaluation set.
        table: one row per probe with columns probe, kind ("shuffle", "constant", "custom"),
            block, n_columns, score_mean, score_std, drop_mean, drop_std, n_repeats.
            drop = baseline - damaged score (sign flipped for losses), so a large positive
            drop means the model relies on that block.
        n_uncovered_columns: columns that are in no block (never probed).
        metric_name: name of the metric, for the summary.
    """

    baseline: float
    table: pd.DataFrame
    n_uncovered_columns: int
    metric_name: str

    def summary(self, tolerance: float = 0.02) -> str:
        """Plain-language result; drops of at most `tolerance` either way are called near zero."""
        m = self.metric_name
        lines = [f"Undamaged {m}: {self.baseline:.3f}."]
        for _, row in self.table.sort_values("drop_mean", ascending=False).iterrows():
            spread = "" if np.isnan(row["drop_std"]) else f" (sd {row['drop_std']:.3f})"
            verdict = "near zero" if abs(row["drop_mean"]) <= tolerance else ""
            lines.append(
                f"{row['probe']}: {m} {row['score_mean']:.3f}, drop {row['drop_mean']:.3f}"
                f"{spread}{', ' + verdict if verdict else ''}."
            )
        near_zero = self.table.loc[self.table["drop_mean"].abs() <= tolerance, "probe"].tolist()
        if near_zero:
            lines.append(
                "The model does not need: " + ", ".join(near_zero) + ". (Not needed is not "
                "the same as unused: a block that duplicates another can be dropped at no cost.)"
            )
        if self.n_uncovered_columns:
            lines.append(f"{self.n_uncovered_columns} columns are in no block and were not probed.")
        return "\n".join(lines)


def shortcut_probe(
    model: Any,
    X: Any,
    y: Any,
    blocks: Mapping[str, BlockSpec],
    metric: Metric,
    *,
    n_repeats: int = 5,
    seed: int = 0,
    constant: Any = None,
    custom: Mapping[str, CustomProbe] | None = None,
    prediction: str = "predict",
    greater_is_better: bool = True,
    metric_name: str | None = None,
) -> ProbeResult:
    """Damage each named input block at inference time and report the metric drop per block.

    Why: if a model that is meant to read both a protein and a ligand scores the same after
    the protein columns are shuffled across rows, it was never using the protein, whatever
    its benchmark score says. This is the shuffle probe used in Bindwell, "The Bitter Lesson
    for Biology" (2026), and in Fang et al. (2026): keep the trained model fixed and break one
    input at a time. It generalizes scikit-learn's permutation importance from single columns
    to named blocks of columns, which is the unit a biologist reasons about ("the protein",
    "the assay text", "which genes are on" versus "how much").

    Probes, each scored on the same evaluation rows:

    - shuffle (always): the block's columns are moved together to a random other row, so each
      row keeps a realistic block, but from another sample. Repeated `n_repeats` times.
    - constant (if `constant` is set): every column of the block is set to one value: "mean"
      for the evaluation-set column mean (most common value for non-numeric DataFrame
      columns), or a number such as 0. Removes the block's information without moving it.
    - custom (optional): named functions f(X, rng) -> damaged X for perturbations that are
      not column blocks, such as "set every nonzero count to 1". Repeated `n_repeats` times.

    Caveat: shuffling one block breaks its correlation with the others, so a drop means the
    model uses the block or is confused by combinations it never saw. Run it on held-out rows
    (for example a group split's test set), not on training rows.

    Args:
        model: a fitted estimator (with `prediction` method) or a function predict(X).
        X: evaluation features, a 2-D numpy array or a DataFrame.
        y: true labels for the evaluation rows.
        blocks: block name -> columns, as integer positions, column names (DataFrame),
            a boolean mask, or a slice. Blocks may overlap.
        metric: metric(y_true, y_pred) -> float.
        n_repeats: shuffles per block (at least 1).
        seed: base seed. Each block's shuffles are seeded from (seed, block name, repeat), so a
            block's result does not depend on which other blocks are probed.
        constant: None (skip), "mean", or a value.
        custom: optional extra whole-input perturbations.
        prediction: "predict", "predict_proba" or "decision_function".
        greater_is_better: set False for losses.
        metric_name: name used in the summary.
    """
    if not isinstance(X, pd.DataFrame):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
    yy = as_1d(y, "y")
    check_same_length(X.shape[0], y=yy)
    if not blocks and not custom:
        raise ValueError("give at least one block (or one custom probe)")
    if int(n_repeats) != n_repeats or n_repeats < 1:
        raise ValueError(f"n_repeats must be an integer >= 1, got {n_repeats}")
    if constant is not None and isinstance(constant, str) and constant != "mean":
        raise ValueError(f"constant must be None, 'mean' or a value, got {constant!r}")
    predict = _predictor(model, prediction)
    sign = 1.0 if greater_is_better else -1.0
    name = metric_name or getattr(metric, "__name__", "score")

    baseline = _score(metric, yy, predict(X))
    resolved = {b: _resolve_block(b, spec, X) for b, spec in blocks.items()}
    n_rows = X.shape[0]

    def rng_for(label: str, repeat: int) -> np.random.Generator:
        return np.random.default_rng([int(seed), zlib.crc32(label.encode()), int(repeat)])

    rows = []

    def add(probe: str, kind: str, block: str, n_cols: int, scores: list[float]) -> None:
        s = np.asarray(scores, dtype=float)
        drops = sign * (baseline - s)
        rows.append(
            {
                "probe": probe,
                "kind": kind,
                "block": block,
                "n_columns": n_cols,
                "score_mean": float(s.mean()),
                "score_std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
                "drop_mean": float(drops.mean()),
                "drop_std": float(drops.std(ddof=1)) if len(s) > 1 else float("nan"),
                "n_repeats": len(s),
            }
        )

    for block, cols in resolved.items():
        scores = []
        for r in range(int(n_repeats)):
            perm = rng_for(block, r).permutation(n_rows)
            scores.append(_score(metric, yy, predict(_shuffled(X, cols, perm))))
        add(f"{block} shuffled", "shuffle", block, len(cols), scores)
        if constant is not None:
            score = _score(metric, yy, predict(_constant(X, cols, constant)))
            add(f"{block} constant", "constant", block, len(cols), [score])

    for probe, damage in (custom or {}).items():
        scores = []
        for r in range(int(n_repeats)):
            damaged = damage(X.copy(), rng_for("custom:" + probe, r))
            scores.append(_score(metric, yy, predict(damaged)))
        add(probe, "custom", "", X.shape[1], scores)

    covered = np.unique(np.concatenate(list(resolved.values()))) if resolved else np.array([])
    return ProbeResult(
        baseline=baseline,
        table=pd.DataFrame(rows),
        n_uncovered_columns=int(X.shape[1] - len(covered)),
        metric_name=name,
    )
