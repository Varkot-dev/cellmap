"""M7: the biology analyses from section 4.4 of the plan, on the PBMC data, split seed 0.

Usage:
    python scripts/m7_analyses.py --name pbmc

1. Shared confusions. For every method with a saved split-seed-0 embedding, the kNN confusion
   between the 19 fine cell types on test donors. Pairs that every method confuses are a
   property of the data (or of the labels), not of any one method.
2. Over-correction. Harmony at increasing strength (theta). Batch mixing should rise; at some
   point real, closely related cell types start to merge, which shows up as falling fine-label
   macro F1 and per-type F1 of close pairs.
3. Marker genes. Spearman correlation between each of 20 known marker genes and each latent
   dimension of the VAE and each of the first 10 PCA components: do learned directions line up
   with known biology?

Writes:
    results/m7/<name>/confusion_pairs.csv     per method, the share of each type called each other
    results/m7/<name>/harmony_strength.csv    one row per theta
    results/m7/<name>/marker_correlations.csv one row per (method, dimension, gene)
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cellmap.atlas import MARKER_GENES, METHODS
from cellmap.baselines import harmony_correct
from cellmap.evaluate import knn_classification, scib_scores
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]
THETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CLOSE_PAIRS = (
    ("naive thymus-derived CD4-positive, alpha-beta T cell",
     "central memory CD4-positive, alpha-beta T cell"),
    ("classical monocyte", "non-classical monocyte"),
    ("naive B cell", "memory B cell"),
    ("gamma-delta T cell", "mucosal invariant T cell"),
)  # fmt: skip


def confusions(emb_dir: Path, obs: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    rows = []
    for method, label in METHODS.items():
        path = emb_dir / f"{method}.npy"
        if not path.exists():
            continue
        knn = knn_classification(np.load(path), obs["cell_type"], split)
        cm = np.asarray(knn["confusion_matrix"], dtype=float)
        totals = cm.sum(axis=1, keepdims=True)
        share = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
        for i, true in enumerate(knn["labels"]):
            for j, called in enumerate(knn["labels"]):
                if i != j and cm[i, j] > 0:
                    rows.append({"method": method, "label": label, "true": true, "called": called,
                                 "n": int(cm[i, j]), "share_of_true": float(share[i, j]),
                                 "n_true": int(totals[i, 0])})  # fmt: skip
    return pd.DataFrame(rows)


def harmony_strength(emb_dir: Path, obs: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    pcs = np.load(emb_dir / "pca.npy")
    donors = obs["donor_id"].astype(str).to_numpy()
    test = split == "test"
    rows = []
    for theta in THETAS:
        emb = harmony_correct(pcs, donors, seed=0, theta=theta)
        knn = knn_classification(emb, obs["cell_type"], split)
        scib = scib_scores(emb[test], obs["cell_type"][test], obs["donor_id"][test], seed=0)
        row = {"theta": theta, "knn_accuracy_fine": knn["accuracy"],
               "knn_macro_f1_fine": knn["macro_f1"], "scib_bio_mean": scib["bio_mean"],
               "scib_batch_mean": scib["batch_mean"], "scib_total": scib["total"],
               "scib_ilisi": scib["batch"].get("ilisi")}  # fmt: skip
        for a, b in CLOSE_PAIRS:
            row[f"f1::{a}"] = knn["per_class_f1"].get(a)
            row[f"f1::{b}"] = knn["per_class_f1"].get(b)
        rows.append(row)
        print(f"theta {theta}: f1={row['knn_macro_f1_fine']:.3f} bio={row['scib_bio_mean']:.3f} "
              f"batch={row['scib_batch_mean']:.3f}", flush=True)  # fmt: skip
    return pd.DataFrame(rows)


def marker_correlations(adata: ad.AnnData, emb_dir: Path) -> pd.DataFrame:
    symbols = adata.var["feature_name"].astype(str).to_numpy()
    rows = []
    for method, n_dims in (("vae", 10), ("pca", 10)):
        emb = np.load(emb_dir / f"{method}.npy")[:, :n_dims]
        for gene in MARKER_GENES:
            idx = np.flatnonzero(symbols == gene)
            if len(idx) == 0:
                continue
            x = np.asarray(adata.X[:, idx[0]].todense()).ravel()
            for d in range(n_dims):
                rho = spearmanr(x, emb[:, d]).statistic
                rows.append({"method": method, "dimension": d, "gene": gene, "spearman": rho})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    emb_dir = ROOT / "data" / "embeddings" / args.name
    split = np.load(emb_dir / "split_seed0.npy", allow_pickle=True)
    out = ROOT / "results" / "m7" / args.name
    out.mkdir(parents=True, exist_ok=True)
    commit = git_commit(ROOT)

    conf = confusions(emb_dir, obs, split)
    conf["commit"] = commit
    conf.to_csv(out / "confusion_pairs.csv", index=False)
    print(f"confusions: {conf['method'].nunique()} methods", flush=True)

    markers = marker_correlations(adata, emb_dir)
    markers["commit"] = commit
    markers.to_csv(out / "marker_correlations.csv", index=False)
    print("marker correlations done", flush=True)

    strength = harmony_strength(emb_dir, obs, split)
    strength["commit"] = commit
    strength.to_csv(out / "harmony_strength.csv", index=False)


if __name__ == "__main__":
    main()
