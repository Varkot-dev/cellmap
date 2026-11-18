"""Small helpers shared by the three checks: input validation and row indexing.

Nothing here imports a single-cell library. AnnData objects are recognised by duck typing
(they have `.obs` and `.n_obs`), so the kit works with them when anndata is installed and
does not need it otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def is_anndata(x: Any) -> bool:
    """True for objects that look like an AnnData (rows are `.obs`, count is `.n_obs`)."""
    return hasattr(x, "obs") and hasattr(x, "n_obs") and hasattr(x, "var")


def n_rows(x: Any) -> int:
    """Number of rows (samples) in an array, sparse matrix, DataFrame or AnnData."""
    if is_anndata(x):
        return int(x.n_obs)
    shape = getattr(x, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return len(x)


def take_rows(x: Any, idx: np.ndarray) -> Any:
    """Rows `idx` (integer positions) of x, keeping its type where possible."""
    if isinstance(x, pd.DataFrame | pd.Series):
        return x.iloc[idx]
    if isinstance(x, list | tuple):
        return np.asarray(x)[idx]
    return x[idx]


def as_1d(values: Any, name: str, data: Any = None) -> np.ndarray:
    """A per-row vector (labels, groups, split names) as a 1-D numpy array, with checks.

    If `values` is a string and `data` is an AnnData, the column `data.obs[values]` is used,
    so `groups="donor_id"` works directly on single-cell data.
    """
    if isinstance(values, str):
        if data is not None and is_anndata(data):
            if values not in data.obs:
                raise KeyError(f"{name}={values!r} is not a column of obs")
            values = data.obs[values]
        else:
            raise TypeError(
                f"{name} must be an array with one entry per row; a column name is only "
                "accepted when X is an AnnData"
            )
    arr = np.asarray(values)  # keeps numeric dtypes; categoricals become their values
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError(f"{name} is empty")
    if pd.isna(arr).any():
        raise ValueError(f"{name} contains missing values ({int(pd.isna(arr).sum())} rows)")
    return arr


def as_str_groups(values: Any, name: str = "groups", data: Any = None) -> np.ndarray:
    """Group ids as strings, so that 1 and "1" are the same group and sorting is defined."""
    return as_1d(values, name, data).astype(str).astype(object)


def check_same_length(n: int, **arrays: np.ndarray) -> None:
    """Raise if any of the named per-row arrays does not have `n` entries."""
    for name, arr in arrays.items():
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} entries, expected {n} (one per row)")


def check_fraction(value: float, name: str, allow_zero: bool) -> None:
    """Raise unless 0 < value < 1 (or 0 <= value < 1 when allow_zero)."""
    low_ok = value >= 0 if allow_zero else value > 0
    if not (low_ok and value < 1):
        bound = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{name} must be in {bound}, got {value}")


def check_seeds(seeds: Any) -> list[int]:
    """Seeds as a non-empty list of distinct ints."""
    out = [int(s) for s in seeds]
    if not out:
        raise ValueError("seeds is empty")
    if len(set(out)) != len(out):
        raise ValueError(f"seeds must be distinct, got {out}")
    return out
