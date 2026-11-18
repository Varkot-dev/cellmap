"""Group-aware splits that keep every group (patient, donor, lab, scaffold, paper) on one side.

Why this matters: rows from one group share things the label does not cause (a patient's
genetics, a lab's protocol, a chemical scaffold, a paper's assay). If a group has rows in both
training and test, a model can score well by recognising the group instead of learning what
the label means, and the test score then overstates how it will do on new groups. The fix is
to assign whole groups, check that it was done, and look at what the split did to the classes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from leakaudit._utils import as_1d, as_str_groups, check_fraction, check_same_length

SPLIT_NAMES = ("train", "val", "test")


class GroupLeakError(AssertionError):
    """A group has rows in more than one split, or a required split is empty."""


def group_split(
    groups: Any,
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    data: Any = None,
) -> np.ndarray:
    """Per-row split names ("train", "val", "test"), assigning whole groups only.

    The fractions are fractions of GROUPS, not of rows: with 21 donors and test_fraction=0.2,
    round(0.2 * 21) = 4 donors are test, whatever their cell counts. Test gets at least one
    group; validation gets at least one unless val_fraction is 0; at least one group must be
    left for training.

    Group ids are compared as strings and sorted before shuffling, so the result depends only
    on the set of groups and the seed, not on row order. This is the same procedure as
    `cellmap.splits.split_donors`, so for the same donors and seed it gives the same split.

    Args:
        groups: one group id per row, or an obs column name when `data` is an AnnData.
        val_fraction: fraction of groups for validation, in [0, 1).
        test_fraction: fraction of groups for test, in (0, 1).
        seed: seed for the shuffle of groups.
        data: optional AnnData, only used to look up `groups` by column name.
    """
    g = as_str_groups(groups, data=data)
    check_fraction(val_fraction, "val_fraction", allow_zero=True)
    check_fraction(test_fraction, "test_fraction", allow_zero=False)
    unique = sorted(set(g))
    n = len(unique)
    n_test = max(1, round(test_fraction * n))
    n_val = 0 if val_fraction == 0 else max(1, round(val_fraction * n))
    if n - n_val - n_test < 1:
        raise ValueError(
            f"{n} groups is too few for {n_val} validation and {n_test} test groups "
            "plus at least one training group"
        )
    order = np.random.default_rng(seed).permutation(n)
    shuffled = [unique[i] for i in order]
    assignment = {x: "test" for x in shuffled[:n_test]}
    assignment.update({x: "val" for x in shuffled[n_test : n_test + n_val]})
    assignment.update({x: "train" for x in shuffled[n_test + n_val :]})
    return pd.Series(g).map(assignment).to_numpy(dtype=object)


def group_kfold(groups: Any, n_splits: int = 5, *, seed: int = 0, data: Any = None) -> np.ndarray:
    """Per-row fold index (0 to n_splits - 1), assigning whole groups, folds balanced by rows.

    Groups are shuffled with `seed`, then taken largest first (ties in random order) and each
    is put in the fold that currently has the fewest rows. This keeps fold sizes close even
    when groups differ a lot in size, which a plain round-robin over groups does not. Use
    `folds == k` as the test mask of fold k. For folds that are also balanced by class, use
    `sklearn.model_selection.StratifiedGroupKFold` and check it with `assert_no_group_overlap`.

    Args:
        groups: one group id per row, or an obs column name when `data` is an AnnData.
        n_splits: number of folds, at least 2 and at most the number of groups.
        seed: seed for the shuffle that breaks ties between equal-sized groups.
        data: optional AnnData, only used to look up `groups` by column name.
    """
    g = as_str_groups(groups, data=data)
    if int(n_splits) != n_splits or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits}")
    sizes = pd.Series(g).value_counts().sort_index()
    if n_splits > len(sizes):
        raise ValueError(f"n_splits={n_splits} is more than the {len(sizes)} groups")
    order = np.random.default_rng(seed).permutation(len(sizes))
    shuffled = sizes.iloc[order]
    by_size = shuffled.sort_values(ascending=False, kind="stable")
    fold_rows = np.zeros(int(n_splits), dtype=np.int64)
    assignment: dict[str, int] = {}
    for name, size in by_size.items():
        k = int(np.argmin(fold_rows))
        assignment[str(name)] = k
        fold_rows[k] += int(size)
    return pd.Series(g).map(assignment).to_numpy(dtype=np.int64)


def assert_no_group_overlap(
    groups: Any,
    split: Any,
    *,
    required: Iterable[Any] | None = None,
    data: Any = None,
) -> None:
    """Raise GroupLeakError if any group has rows in more than one split.

    Works for any split labels: names ("train", "val", "test"), fold numbers, or booleans.
    Call it right after making a split and again right before scoring, so a split made by
    other code (a colleague's notebook, a benchmark's own files) is checked too.

    Args:
        groups: one group id per row, or an obs column name when `data` is an AnnData.
        split: one split label per row.
        required: split labels that must each have at least one row (for example
            ("train", "test")); an empty required split also raises.
        data: optional AnnData, only used to look up `groups` by column name.
    """
    g = as_str_groups(groups, data=data)
    s = as_1d(split, "split")
    check_same_length(len(g), split=s)
    per_group = pd.DataFrame({"g": g, "s": s}).groupby("g")["s"].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked) > 0:
        names = sorted(leaked.index)
        shown = ", ".join(names[:10]) + (" ..." if len(names) > 10 else "")
        raise GroupLeakError(
            f"{len(names)} of {len(per_group)} groups have rows in more than one split: {shown}"
        )
    if required is not None:
        present = set(s.tolist())
        missing = [r for r in required if r not in present]
        if missing:
            raise GroupLeakError(f"required splits with no rows: {missing}")


@dataclass(frozen=True)
class SplitReport:
    """What a split did: rows, groups and classes per split.

    Attributes:
        per_split: one row per split with n_rows, frac_rows, n_groups, frac_groups and, when
            labels were given, n_classes.
        class_counts: classes x splits table of row counts (empty without labels).
        class_fractions: the same, as the fraction of each split's rows (columns sum to 1).
        missing_classes: for each split, the classes present elsewhere but absent from it.
        max_class_gap: the largest difference, over classes, between the highest and lowest
            share a class has in any split (0 = every split has the same class mix).
    """

    per_split: pd.DataFrame
    class_counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    class_fractions: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_classes: dict[str, list[str]] = field(default_factory=dict)
    max_class_gap: float = float("nan")

    def summary(self) -> str:
        """A few plain sentences describing the split."""
        lines = []
        for name, row in self.per_split.iterrows():
            line = (
                f"{name}: {int(row['n_rows'])} rows ({row['frac_rows']:.0%}), "
                f"{int(row['n_groups'])} groups"
            )
            if "n_classes" in row:
                line += f", {int(row['n_classes'])} classes"
            lines.append(line)
        for name, missing in self.missing_classes.items():
            if missing:
                shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
                lines.append(f"{name} has no rows of {len(missing)} classes: {shown}")
        if not np.isnan(self.max_class_gap):
            lines.append(
                f"Largest gap in a class's share between splits: {self.max_class_gap:.1%} "
                "(0% means every split has the same class mix)."
            )
        return "\n".join(lines)


def _split_order(values: Sequence[Any]) -> list[Any]:
    names = list(dict.fromkeys(values))
    if set(names) <= set(SPLIT_NAMES):
        return [s for s in SPLIT_NAMES if s in names]
    return sorted(names, key=str)


def split_report(groups: Any, split: Any, labels: Any = None, *, data: Any = None) -> SplitReport:
    """Report how balanced a split is: rows, groups and classes in each split.

    A group split is honest but can be lopsided: one large patient can make the test set
    twice the intended size, and a rare class can end up only in training, so the test score
    says nothing about it. This report makes those effects visible before anything is scored.

    Args:
        groups: one group id per row, or an obs column name when `data` is an AnnData.
        split: one split label per row.
        labels: optional class label per row (or obs column name with `data`).
        data: optional AnnData, only used to look up columns by name.
    """
    g = as_str_groups(groups, data=data)
    s = as_1d(split, "split")
    check_same_length(len(g), split=s)
    frame = pd.DataFrame({"g": g, "s": s})
    if labels is not None:
        y = as_1d(labels, "labels", data).astype(str)
        check_same_length(len(g), labels=y)
        frame["y"] = y
    order = _split_order(frame["s"].tolist())
    grouped = frame.groupby("s")
    per_split = pd.DataFrame(
        {
            "n_rows": grouped.size(),
            "n_groups": grouped["g"].nunique(),
        }
    ).reindex(order)
    per_split["frac_rows"] = per_split["n_rows"] / len(frame)
    per_split["frac_groups"] = per_split["n_groups"] / frame["g"].nunique()
    per_split.index.name = "split"
    if labels is None:
        return SplitReport(per_split=per_split[["n_rows", "frac_rows", "n_groups", "frac_groups"]])

    per_split["n_classes"] = grouped["y"].nunique().reindex(order)
    counts = pd.crosstab(frame["y"], frame["s"]).reindex(columns=order, fill_value=0)
    counts.index.name, counts.columns.name = "class", "split"
    fractions = counts / counts.sum(axis=0)
    missing = {str(c): sorted(counts.index[counts[c] == 0].tolist()) for c in counts.columns}
    gap = float((fractions.max(axis=1) - fractions.min(axis=1)).max())
    return SplitReport(
        per_split=per_split[["n_rows", "frac_rows", "n_groups", "frac_groups", "n_classes"]],
        class_counts=counts,
        class_fractions=fractions,
        missing_classes=missing,
        max_class_gap=gap,
    )
