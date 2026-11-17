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


def score_embedding(
    emb: np.ndarray, obs: pd.DataFrame, split: np.ndarray, scib_seed: int = 0
) -> dict[str, float]:
    """All headline metrics for one embedding and one donor split, as a flat dictionary.

    kNN: classifier fitted on training donors, scored on test donors, at the fine and the
    coarse label level. scIB: computed on the test donors' cells only, so every number in the
    row describes cells from donors the method was not fitted on (Harmony excepted, see
    `cellmap.baselines.harmony_correct`).
    """
    out: dict[str, float] = {}
    for level, column in (("fine", "cell_type"), ("coarse", "cell_type_coarse")):
        knn = knn_classification(emb, obs[column], split)
        out[f"knn_accuracy_{level}"] = knn["accuracy"]
        out[f"knn_macro_f1_{level}"] = knn["macro_f1"]

    test = split == "test"
    scib = scib_scores(emb[test], obs["cell_type"][test], obs["donor_id"][test], seed=scib_seed)
    for group in ("bio", "batch"):
        for name, value in scib[group].items():
            out[f"scib_{name}"] = value
    out["scib_bio_mean"] = scib["bio_mean"]
    out["scib_batch_mean"] = scib["batch_mean"]
    out["scib_total"] = scib["total"]
    return out
