"""M3: parity check of the hand-written VAE against scvi-tools.

Usage:
    python scripts/m3_scvi_parity.py --name pbmc

Trains the scvi-tools model with settings matched to cellmap.vae (see
cellmap.scvi_baseline.scvi_settings) on the same donor splits, genes and seeds as
scripts/m2_vae.py, and scores it with the same harness. Compare the rows written here with the
"vae" rows of results/m2/<name>/metrics_by_split.csv.

Writes:
    results/m3/<name>/metrics_by_split.csv  one row per split seed, method "scvi"
    results/m3/<name>/run.json              settings, timings, commit
    data/embeddings/<name>/scvi.npy         split seed 0 embedding (not committed)
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.provenance import git_commit, versions
from cellmap.scvi_baseline import scvi_embed

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    out = ROOT / "results" / "m3" / args.name
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / "data" / "embeddings" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows, runs = [], []
    for seed in args.split_seeds:
        split = donor_split(obs, seed)
        train = split == "train"
        genes = select_hvg(adata, train, n_top_genes=min(2000, adata.n_vars))
        emb, info = scvi_embed(adata, genes, train, max_epochs=args.epochs, seed=seed)
        scores = score_embedding(emb, obs, split)
        rows.append({"method": "scvi", "split_seed": seed, **scores})
        runs.append(
            {
                "split_seed": seed,
                "train_seconds": info["train_seconds"],
                "final_train_recon": info["history"]["train_recon"][-1],
                "final_train_kl": info["history"]["train_kl"][-1],
                "settings": info["settings"],
            }
        )
        print(
            f"seed {seed} scvi: acc={scores['knn_accuracy_fine']:.3f} "
            f"f1={scores['knn_macro_f1_fine']:.3f} bio={scores['scib_bio_mean']:.3f} "
            f"batch={scores['scib_batch_mean']:.3f}",
            flush=True,
        )
        if seed == args.split_seeds[0]:
            np.save(emb_dir / "scvi.npy", emb)
        pd.DataFrame(rows).to_csv(out / "metrics_by_split.csv", index=False)

    run = {
        "name": args.name,
        "commit": git_commit(ROOT),
        "torch_threads": args.threads,
        "timing_note": "other jobs shared the machine during this run; times are upper bounds",
        "runs": runs,
        "versions": versions("scvi-tools", "torch", "scanpy", "scib-metrics"),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
