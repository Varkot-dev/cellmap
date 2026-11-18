"""leakaudit on real single-cell data: reproduce cellmap's random-cell versus donor split result.

Usage (from the cellmap repository, with cellmap and leakaudit installed):
    OMP_NUM_THREADS=2 python audit/examples/cellmap_pbmc.py

Method, exactly as in cellmap's scripts/leakage_splits.py: highly variable genes (2,000) and
PCA (50 components) fitted on the training cells, then kNN (k = 15) from training cells to test
cells; metric macro F1 over the cell types present in the test cells. Five seeds, 20% test and
20% validation (validation cells are set aside, as in the original script).

leakaudit does the splitting, the overlap assertion and the inflation summary. The model is
passed as a fit_predict function that receives row indices, so the HVG and PCA steps can use the
whole AnnData while being fitted on the training rows only.

Writes to audit/examples/output/:
    cellmap_pbmc_scores.csv     one row per scheme and seed, with the reference values
    cellmap_pbmc_summary.csv    the inflation summary
    cellmap_pbmc_split_report.csv  rows, donors and cell types per split (donor split, seed 0)
    cellmap_pbmc_summary.txt    the plain-language summaries
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cellmap.baselines import pca_embed, select_hvg  # noqa: E402
from cellmap.evaluate import _knn_predict  # noqa: E402  (the project's own tie rule)
from cellmap.splits import split_donors  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from leakaudit import group_split, measure_inflation, random_split, split_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
SEEDS = (0, 1, 2, 3, 4)
SCHEME_NAMES = {"random row split": "random cells", "group split": "held-out people"}


def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over the classes that have test cells (cellmap's convention)."""
    return float(
        f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)
    )


def load_original_random_split():
    """`random_cell_split` from cellmap's scripts/leakage_splits.py, to check ours matches."""
    spec = importlib.util.spec_from_file_location(
        "leakage_splits", ROOT / "scripts/leakage_splits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imports cellmap.benchmark, which loads torch
    if "torch" in sys.modules:  # not used here, but keep its CPU use modest
        sys.modules["torch"].set_num_threads(2)
    return module.random_cell_split


def main() -> None:
    start = time.time()
    adata = ad.read_h5ad(ROOT / "data" / "processed" / "pbmc.h5ad")
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    donors = adata.obs["donor_id"].astype(str).to_numpy()
    n = adata.n_obs
    print(f"{n} cells, {len(set(donors))} donors, {len(set(labels))} cell types", flush=True)

    # 1. The kit's splits are the same splits cellmap used (cellmap.benchmark.donor_split is
    #    split_donors mapped to cells).
    original_random = load_original_random_split()
    for seed in SEEDS:
        same_group = np.array_equal(
            group_split(donors, val_fraction=0.2, test_fraction=0.2, seed=seed),
            pd.Series(donors).map(split_donors(donors, 0.2, 0.2, seed)).to_numpy(),
        )
        same_random = np.array_equal(
            random_split(n, val_fraction=0.2, test_fraction=0.2, seed=seed),
            original_random(n, seed),
        )
        if not (same_group and same_random):
            raise RuntimeError(f"seed {seed}: splits differ from cellmap's")
    print("splits identical to cellmap's for all seeds", flush=True)

    # 2. The model: HVG + PCA fitted on training rows, then kNN. Predictions are kept so that
    #    accuracy can be computed too without rerunning.
    kept: list[tuple[np.ndarray, np.ndarray]] = []

    def fit_predict(train_idx: np.ndarray, y_train: np.ndarray, test_idx: np.ndarray):
        train = np.zeros(n, dtype=bool)
        train[train_idx] = True
        genes = select_hvg(adata, train, n_top_genes=2000)
        emb = pca_embed(adata, genes, train, n_comps=50)
        classes = np.array(sorted(set(y_train)))
        codes = np.searchsorted(classes, y_train)
        k = min(15, len(train_idx))
        pred = classes[_knn_predict(emb[train_idx], codes, emb[test_idx], k, len(classes))]
        kept.append((test_idx, pred))
        print(f"  fitted on {len(train_idx)} cells, {time.time() - start:.0f}s", flush=True)
        return pred

    result = measure_inflation(
        np.arange(n),
        labels,
        donors,
        fit_predict,
        macro_f1_present,
        seeds=SEEDS,
        val_fraction=0.2,
        test_fraction=0.2,
        metric_name="kNN macro F1",
    )

    # 3. Compare with the saved cellmap result.
    scores = result.scores.copy()
    scores["knn_accuracy"] = [float(np.mean(labels[t] == p)) for t, p in kept]
    ref = pd.read_csv(ROOT / "results" / "leakage" / "splits.csv")
    ref = ref[ref["dataset"] == "pbmc"].set_index(["scheme", "fold"])
    pairs = zip(scores["scheme"], scores["seed"], strict=True)
    keys = [(SCHEME_NAMES[s], f"seed{seed}") for s, seed in pairs]
    scores["reference_knn_macro_f1"] = [ref.loc[k, "knn_macro_f1"] for k in keys]
    scores["reference_knn_accuracy"] = [ref.loc[k, "knn_accuracy"] for k in keys]
    scores["diff_macro_f1"] = scores["score"] - scores["reference_knn_macro_f1"]
    scores["diff_accuracy"] = scores["knn_accuracy"] - scores["reference_knn_accuracy"]
    scores = scores.rename(columns={"score": "knn_macro_f1"})

    summary = result.summary_frame()
    ref_means = ref.groupby(level="scheme")["knn_macro_f1"].mean()
    summary["reference_leaky_mean"] = ref_means["random cells"]
    summary["reference_honest_mean"] = ref_means["held-out people"]
    summary["reference_inflation"] = ref_means["random cells"] - ref_means["held-out people"]
    summary["max_abs_diff_per_seed"] = float(scores["diff_macro_f1"].abs().max())

    report = split_report(
        donors, group_split(donors, val_fraction=0.2, test_fraction=0.2, seed=0), labels
    )

    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / "cellmap_pbmc_scores.csv", index=False)
    summary.to_csv(OUT / "cellmap_pbmc_summary.csv", index=False)
    report.per_split.to_csv(OUT / "cellmap_pbmc_split_report.csv")
    matches = summary.loc[0, "max_abs_diff_per_seed"] < 1e-9
    text = "\n\n".join(
        [
            result.summary(),
            "Donor split, seed 0:\n" + report.summary(),
            f"Largest per-seed difference from results/leakage/splits.csv: "
            f"{summary.loc[0, 'max_abs_diff_per_seed']:.2e} "
            f"({'matches' if matches else 'does NOT match'}). Runtime {time.time() - start:.0f}s.",
        ]
    )
    (OUT / "cellmap_pbmc_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
