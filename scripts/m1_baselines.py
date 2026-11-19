"""M1: classical baselines (HVG + PCA, PCA + Harmony) scored by the evaluation harness.

Usage:
    python scripts/m1_baselines.py --name pbmc

Reads data/processed/<name>.h5ad. For each of several donor splits it selects genes and fits
PCA on the training donors, runs Harmony, and scores both embeddings on the test donors.

Writes:
    results/m1/<name>/metrics_by_split.csv   one row per (method, split seed)
    results/m1/<name>/summary.csv            mean, sd, min, max over split seeds
    results/m1/<name>/label_efficiency.csv   split seed 0 only
    results/m1/<name>/confusion_<method>.csv fine labels, split seed 0
    results/m1/<name>/per_class_f1.csv       fine labels, split seed 0
    results/m1/<name>/run.json               settings, timings, package versions
    data/embeddings/<name>/<method>.npy      split seed 0 embeddings (not committed)
"""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellmap.baselines import harmony_correct, pca_embed, select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.evaluate import knn_classification, label_efficiency

ROOT = Path(__file__).resolve().parents[1]
N_TOP_GENES = 2000
N_COMPS = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    out = ROOT / "results" / "m1" / args.name
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / "data" / "embeddings" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows, timings = [], []
    for seed in args.split_seeds:
        split = donor_split(obs, seed)
        train = split == "train"

        t0 = time.perf_counter()
        genes = select_hvg(adata, train, n_top_genes=min(N_TOP_GENES, adata.n_vars))
        t_hvg = time.perf_counter() - t0
        t0 = time.perf_counter()
        pcs = pca_embed(adata, genes, train, n_comps=N_COMPS)
        t_pca = time.perf_counter() - t0
        t0 = time.perf_counter()
        harmony = harmony_correct(pcs, obs["donor_id"].to_numpy(), seed=seed)
        t_harmony = time.perf_counter() - t0
        timings.append({"split_seed": seed, "hvg_s": t_hvg, "pca_s": t_pca, "harmony_s": t_harmony})

        embeddings = {"pca": pcs, "pca_harmony": harmony}
        for method, emb in embeddings.items():
            scores = score_embedding(emb, obs, split)
            rows.append(
                {
                    "method": method,
                    "split_seed": seed,
                    "n_train_cells": int(train.sum()),
                    "n_test_cells": int((split == "test").sum()),
                    **scores,
                }
            )
            print(
                f"seed {seed} {method}: "
                + ", ".join(
                    f"{k}={scores[k]:.3f}"
                    for k in ("knn_macro_f1_fine", "scib_bio_mean", "scib_batch_mean")
                )
            )

        if seed == args.split_seeds[0]:
            eff, per_class = [], {}
            for method, emb in embeddings.items():
                np.save(emb_dir / f"{method}.npy", emb)
                knn = knn_classification(emb, obs["cell_type"], split)
                pd.DataFrame(
                    knn["confusion_matrix"], index=knn["labels"], columns=knn["labels"]
                ).to_csv(out / f"confusion_{method}.csv")
                per_class[method] = knn["per_class_f1"]
                frame = label_efficiency(emb, obs["cell_type"], split)
                frame.insert(0, "method", method)
                eff.append(frame)
            pd.concat(eff).to_csv(out / "label_efficiency.csv", index=False)
            per_class_frame = pd.DataFrame(per_class)
            test_counts = obs["cell_type"][split == "test"].astype(str).value_counts()
            per_class_frame.insert(0, "n_test_cells", test_counts.reindex(per_class_frame.index))
            per_class_frame.sort_values("n_test_cells", ascending=False).to_csv(
                out / "per_class_f1.csv", index_label="cell_type"
            )
            np.save(emb_dir / "split_seed0.npy", split)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_by_split.csv", index=False)
    value_columns = [c for c in metrics.columns if c.startswith(("knn_", "scib_"))]
    summary = (
        metrics.melt(id_vars=["method", "split_seed"], value_vars=value_columns, var_name="metric")
        .groupby(["metric", "method"])["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)

    # Paired difference per split seed: the same donors are held out for both methods.
    wide = metrics.pivot(index="split_seed", columns="method")
    diff = pd.DataFrame({m: wide[m]["pca_harmony"] - wide[m]["pca"] for m in value_columns})
    paired = pd.DataFrame(
        {
            "mean_diff": diff.mean(),
            "min_diff": diff.min(),
            "max_diff": diff.max(),
            "splits_harmony_higher": (diff > 0).sum(),
            "n_splits": len(diff),
        }
    )
    paired.to_csv(out / "paired_harmony_minus_pca.csv", index_label="metric")

    run = {
        "name": args.name,
        "split_seeds": args.split_seeds,
        "n_top_genes": N_TOP_GENES,
        "n_comps": N_COMPS,
        "timings": timings,
        "versions": {
            p: version(p) for p in ("scanpy", "harmonypy", "scib-metrics", "scikit-learn", "numpy")
        },
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(summary.to_string())
    print(paired.to_string())


if __name__ == "__main__":
    main()
