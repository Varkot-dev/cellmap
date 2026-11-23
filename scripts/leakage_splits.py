"""How much does a leaky split inflate the score? Same method, three ways of splitting.

Usage:
    python scripts/leakage_splits.py

HVG + PCA + kNN, scored with cells split three ways:
    random cells   cells assigned to train / test at random, so the same person appears on both
                   sides (the leaky way)
    held-out people  whole donors held out (the project's standard)
    held-out lab     a whole dataset held out (six-lab data only)

The gap between "random cells" and the other two is how much a leaky split would have
overstated the result. This is the single-cell version of the train-test leakage reported
for protein-ligand benchmarks (Graber et al., Nature Machine Intelligence 2025).

Writes:
    results/leakage/splits.csv
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellmap.baselines import pca_embed, select_hvg
from cellmap.benchmark import dataset_out_split, donor_split
from cellmap.evaluate import knn_classification
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"pbmc": "cell_type", "m4_blood": "lineage"}


def random_cell_split(n: int, seed: int, val: float = 0.2, test: float = 0.2) -> np.ndarray:
    """Assign cells (not donors) at random: the leaky split."""
    order = np.random.default_rng(seed).permutation(n)
    split = np.full(n, "train", dtype=object)
    n_test, n_val = round(test * n), round(val * n)
    split[order[:n_test]] = "test"
    split[order[n_test : n_test + n_val]] = "val"
    return split


def score(adata: ad.AnnData, split: np.ndarray, label: str) -> dict:
    train = split == "train"
    genes = select_hvg(adata, train, n_top_genes=2000)
    emb = pca_embed(adata, genes, train, n_comps=50)
    knn = knn_classification(emb, adata.obs[label], split)
    return {"knn_accuracy": knn["accuracy"], "knn_macro_f1": knn["macro_f1"]}


def main() -> None:
    rows = []
    for name, label in DATASETS.items():
        adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{name}.h5ad")
        schemes = [
            ("random cells", f"seed{s}", random_cell_split(adata.n_obs, s)) for s in range(5)
        ]
        schemes += [("held-out people", f"seed{s}", donor_split(adata.obs, s)) for s in range(5)]
        if "dataset" in adata.obs:
            for ds in sorted(adata.obs["dataset"].astype(str).unique()):
                schemes.append(("held-out lab", ds, dataset_out_split(adata.obs, ds)))
        for scheme, fold, split in schemes:
            rows.append({"dataset": name, "label": label, "scheme": scheme, "fold": fold,
                         **score(adata, split, label), "commit": git_commit(ROOT)})  # fmt: skip
            print(rows[-1]["dataset"], scheme, fold, round(rows[-1]["knn_macro_f1"], 3), flush=True)
    out = ROOT / "results" / "leakage"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "splits.csv", index=False)
    print(frame.groupby(["dataset", "scheme"])[["knn_accuracy", "knn_macro_f1"]].mean().round(3))


if __name__ == "__main__":
    main()
