"""Score any embedding the same way, so every method in the comparison is treated alike."""

import numpy as np
import pandas as pd

from cellmap.evaluate import knn_classification, scib_scores
from cellmap.splits import split_donors


def donor_split(
    obs: pd.DataFrame, seed: int, val_fraction: float = 0.2, test_fraction: float = 0.2
) -> np.ndarray:
    """Per-cell split names (train / val / test) for a donor-level split with this seed.

    Repeating the benchmark over several seeds measures how much a score moves when different
    donors are held out, which is the main source of uncertainty with only 21 donors.
    """
    assignment = split_donors(obs["donor_id"], val_fraction, test_fraction, seed)
    return obs["donor_id"].astype(str).map(assignment).to_numpy()


LEVELS_M1 = {"fine": "cell_type", "coarse": "cell_type_coarse"}


def dataset_out_split(
    obs: pd.DataFrame, held_out: str, val_fraction: float = 0.15, seed: int = 0
) -> np.ndarray:
    """Per-cell split names for "leave one dataset out".

    Every cell of the `held_out` dataset is test. The other datasets' donors are divided into
    train and val, by whole donor. This is a harder test than a donor split: the test cells
    come from a lab, protocol and often a chemistry that training never saw.
    """
    in_test = (obs["dataset"].astype(str) == held_out).to_numpy()
    if not in_test.any():
        raise ValueError(f"no cells from dataset {held_out!r}")
    rest = obs.loc[~in_test, "donor_id"].astype(str)
    donors = np.sort(rest.unique())
    rng = np.random.default_rng(seed)
    n_val = max(1, round(val_fraction * len(donors)))
    val_donors = set(rng.choice(donors, size=n_val, replace=False))
    split = np.where(obs["donor_id"].astype(str).isin(val_donors), "val", "train")
    split[in_test] = "test"
    return split


def score_embedding(
    emb: np.ndarray,
    obs: pd.DataFrame,
    split: np.ndarray,
    scib_seed: int = 0,
    levels: dict[str, str] | None = None,
    scib_label: str = "cell_type",
) -> dict[str, float]:
    """All headline metrics for one embedding and one donor split, as a flat dictionary.

    kNN: classifier fitted on training donors, scored on test donors, at each label level in
    `levels` (name to obs column; default the M1 fine and coarse levels). scIB: computed on
    the test donors' cells only, with `scib_label` as the cell type, so every number in the
    row describes cells from donors the method was not fitted on (Harmony excepted, see
    `cellmap.baselines.harmony_correct`).
    """
    out: dict[str, float] = {}
    for level, column in (levels or LEVELS_M1).items():
        knn = knn_classification(emb, obs[column], split)
        out[f"knn_accuracy_{level}"] = knn["accuracy"]
        out[f"knn_macro_f1_{level}"] = knn["macro_f1"]

    test = split == "test"
    scib = scib_scores(emb[test], obs[scib_label][test], obs["donor_id"][test], seed=scib_seed)
    for group in ("bio", "batch"):
        for name, value in scib[group].items():
            out[f"scib_{name}"] = value
    out["scib_bio_mean"] = scib["bio_mean"]
    out["scib_batch_mean"] = scib["batch_mean"]
    out["scib_total"] = scib["total"]
    return out
