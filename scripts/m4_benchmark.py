"""Every method on six blood datasets, under two split schemes.

Usage:
    python scripts/m4_benchmark.py --name m4_blood                        # CPU methods
    python scripts/m4_benchmark.py --name m4_blood --methods transformer --device cuda

Schemes:
    donor     donor splits pooled across all datasets (split seeds 0 to 4)
    dataset   leave one dataset out: train on five labs, test on the sixth (six folds)

Labels are the 12 shared lineages (see cellmap.labels.LINEAGE); fine labels differ between labs.
The donor is the batch for Harmony, for the VAE decoder and for the scIB batch metrics.

Writes one file per method, so a GPU machine can add its method without touching the others:
    results/m4/<name>/metrics_<method>.csv  one row per (scheme, fold)
    results/m4/<name>/run_<method>.json     settings, timings, commit
    data/embeddings/<name>/<method>_<scheme>_<fold>.npy  embeddings (not committed)
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellmap.benchmark import dataset_out_split, donor_split, fit_and_embed, score_embedding
from cellmap.census_embeddings import PRETRAINED, fetch_embedding
from cellmap.config import load_config
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {"lineage": "lineage"}
TRAINED = ("pca", "pca_harmony", "vae", "transformer")


def folds(obs: pd.DataFrame, schemes: list[str]) -> list[tuple[str, str, np.ndarray]]:
    """(scheme, fold name, per-cell split) for every fold of the requested schemes."""
    out = []
    if "donor" in schemes:
        out += [("donor", f"seed{s}", donor_split(obs, s)) for s in range(5)]
    if "dataset" in schemes:
        for ds in sorted(obs["dataset"].astype(str).unique()):
            out.append(("dataset", ds, dataset_out_split(obs, ds)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="m4_blood")
    default_methods = ["pca", "pca_harmony", "vae", *PRETRAINED]
    parser.add_argument("--methods", nargs="+", default=default_methods)
    parser.add_argument("--schemes", nargs="+", default=["donor", "dataset"])
    parser.add_argument("--folds", nargs="+", help="run only these folds (e.g. seed0 hao)")
    parser.add_argument(
        "--resume", action="store_true", help="keep rows already saved and skip their folds"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace saved results instead of refusing"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    out = ROOT / "results" / "m4" / args.name
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / "data" / "embeddings" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)
    all_folds = folds(obs, args.schemes)
    if args.folds:
        all_folds = [f for f in all_folds if f[1] in args.folds]

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite contradict each other")
    clash = [m for m in args.methods if (out / f"metrics_{m}.csv").exists()]
    if clash and not (args.resume or args.overwrite):
        # Without this, a run limited to one fold would replace a complete file with one row.
        parser.error(f"results exist for {clash}; pass --resume to add to them or --overwrite")

    for method in args.methods:
        rows, timings = [], []
        saved = out / f"metrics_{method}.csv"
        if args.resume and saved.exists():
            rows = pd.read_csv(saved).to_dict("records")
        done = {(r["scheme"], r["fold"]) for r in rows}
        run_path = out / f"run_{method}.json"
        if args.resume and run_path.exists():
            timings = json.loads(run_path.read_text())["timings"]
        pretrained = None
        if method in PRETRAINED:
            cache = emb_dir / f"{method}.npy"
            if not cache.exists():
                cfg = load_config(ROOT / "configs" / f"{args.name}.yaml")["census"]
                ids = obs["soma_joinid"].to_numpy()
                np.save(cache, fetch_embedding(cfg, PRETRAINED[method]["id"], ids))
            pretrained = np.load(cache)
        elif method not in TRAINED:
            raise ValueError(f"unknown method {method!r}")

        for scheme, fold, split in all_folds:
            if (scheme, fold) in done:
                continue
            seed = int(fold.removeprefix("seed")) if scheme == "donor" else 0
            t0 = time.perf_counter()
            emb = pretrained if pretrained is not None else fit_and_embed(
                method, adata, split, seed, args.device
            )  # fmt: skip
            seconds = time.perf_counter() - t0
            if pretrained is None:
                np.save(emb_dir / f"{method}_{scheme}_{fold}.npy", emb)
            scores = score_embedding(emb, obs, split, levels=LEVELS, scib_label="lineage")
            rows.append({"method": method, "scheme": scheme, "fold": fold,
                         "n_test_cells": int((split == "test").sum()), **scores})  # fmt: skip
            timings.append({"scheme": scheme, "fold": fold, "seconds": seconds})
            summary = " ".join(
                f"{key}={scores[col]:.3f}"
                for key, col in (("acc", "knn_accuracy_lineage"), ("f1", "knn_macro_f1_lineage"),
                                 ("bio", "scib_bio_mean"), ("batch", "scib_batch_mean"))
            )  # fmt: skip
            print(f"{method} {scheme} {fold}: {summary} ({seconds:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(saved, index=False)
            # Rewritten after every fold, so an interrupted run keeps everything it finished.
            run = {
                "method": method,
                "commit": git_commit(ROOT),
                "device": args.device,
                "torch_threads": args.threads,
                "timings": timings,
                "versions": versions("torch", "scanpy", "scib-metrics"),
            }
            run_path.write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
