"""Embeddings from pretrained models, scored like every other method.

Usage:
    python scripts/m6_pretrained.py --name pbmc

Nothing is trained here. The embeddings are the ones CZ CELLxGENE publishes for Census cells
(see cellmap.census_embeddings.PRETRAINED). They are downloaded for this project's cells and
scored on the same five donor splits with the same harness. The donor split still matters for
them: the kNN classifier is fitted on training donors' labels and scored on test donors.

Read every score with the overlap notes written to run.json: these models were trained on
Census data, and one of them was fine-tuned with cell-type labels.

Writes:
    results/m6/<name>/metrics_by_split.csv  one row per (method, split seed)
    results/m6/<name>/run.json              what each model saw in training, timings, commit
    data/embeddings/<name>/<method>.npy     embeddings for all cells (not committed)
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellmap.benchmark import donor_split, score_embedding
from cellmap.census_embeddings import PRETRAINED, census_joinids, fetch_embedding
from cellmap.config import load_config
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / f"{args.name}.yaml")["census"]
    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad", backed="r")
    obs = adata.obs
    out = ROOT / "results" / "m6" / args.name
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / "data" / "embeddings" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Processed cells are named "cell<i>", where i is the row in the raw download, and the raw
    # download is ordered by soma_joinid. Check that claim before trusting it.
    census_obs = census_joinids(cfg)
    raw_rows = np.array([int(name.removeprefix("cell")) for name in obs.index])
    matched = census_obs.iloc[raw_rows]
    same_donor = (
        matched["donor_id"].astype(str).to_numpy() == obs["donor_id"].astype(str).to_numpy()
    )
    same_type = (
        matched["cell_type"].astype(str).to_numpy() == obs["cell_type"].astype(str).to_numpy()
    )
    if not (same_donor.all() and same_type.all()):
        raise RuntimeError("could not match processed cells to Census soma_joinids")
    joinids = matched["soma_joinid"].to_numpy()

    embeddings, fetch_seconds = {}, {}
    for method, meta in PRETRAINED.items():
        cache = emb_dir / f"{method}.npy"
        t0 = time.perf_counter()
        if cache.exists():
            embeddings[method] = np.load(cache)
        else:
            embeddings[method] = fetch_embedding(cfg, meta["id"], joinids)
            np.save(cache, embeddings[method])
        fetch_seconds[method] = time.perf_counter() - t0
        print(f"{method}: {embeddings[method].shape}, {fetch_seconds[method]:.0f}s", flush=True)

    rows = []
    for seed in args.split_seeds:
        split = donor_split(obs, seed)
        for method, emb in embeddings.items():
            scores = score_embedding(emb, obs, split)
            rows.append({"method": method, "split_seed": seed, **scores})
            print(
                f"seed {seed} {method}: acc={scores['knn_accuracy_fine']:.3f} "
                f"f1={scores['knn_macro_f1_fine']:.3f} bio={scores['scib_bio_mean']:.3f} "
                f"batch={scores['scib_batch_mean']:.3f}",
                flush=True,
            )
        pd.DataFrame(rows).to_csv(out / "metrics_by_split.csv", index=False)

    run = {
        "name": args.name,
        "commit": git_commit(ROOT),
        "census_version": cfg["version"],
        "n_cells": int(len(obs)),
        "models": PRETRAINED,
        "fetch_seconds": fetch_seconds,
        "versions": versions("cellxgene-census", "tiledbsoma", "scib-metrics"),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
