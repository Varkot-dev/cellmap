"""M5: the hand-written transformer, trained with masked gene modeling and scored like the rest.

Usage:
    python scripts/m5_transformer.py --name pbmc --device cuda     # on a GPU
    python scripts/m5_transformer.py --name pbmc --threads 4       # on CPU, slow

For each donor split: select genes on the training donors, train on the training donors only,
embed every cell, and score on the test donors with the same harness as every other method.
The donor is never an input, so this model has no batch correction mechanism.

Writes:
    results/m5/<name>/metrics_by_split.csv  one row per split seed, method "transformer"
    results/m5/<name>/history_seed<k>.csv   masked-token loss and accuracy per epoch
    results/m5/<name>/run.json              settings, timings, parameter count, commit
    data/embeddings/<name>/transformer.npy  split seed 0 embedding (not committed)
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellmap import transformer
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    out = ROOT / "results" / "m5" / args.name
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / "data" / "embeddings" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows, runs = [], []
    for seed in args.split_seeds:
        split = donor_split(obs, seed)
        train, val = split == "train", split == "val"
        genes = select_hvg(adata, train, n_top_genes=min(args.n_top_genes, adata.n_vars))
        counts = adata.layers["counts"][:, genes].tocsr()

        t0 = time.perf_counter()
        model, history = transformer.train_transformer(
            counts[train],
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
            counts_val=counts[val],
        )
        seconds = time.perf_counter() - t0
        pd.DataFrame(history).to_csv(out / f"history_seed{seed}.csv", index_label="epoch")
        emb = transformer.embed(model, counts, device=args.device)

        scores = score_embedding(emb, obs, split)
        rows.append({"method": "transformer", "split_seed": seed, **scores})
        runs.append(
            {
                "split_seed": seed,
                "train_seconds": seconds,
                "n_parameters": transformer.count_parameters(model),
                "final_val_accuracy": history["val_accuracy"][-1],
                "val_gene_majority_baseline": history["val_gene_majority"][-1],
            }
        )
        print(
            f"seed {seed} transformer: acc={scores['knn_accuracy_fine']:.3f} "
            f"f1={scores['knn_macro_f1_fine']:.3f} bio={scores['scib_bio_mean']:.3f} "
            f"batch={scores['scib_batch_mean']:.3f} ({seconds:.0f}s)",
            flush=True,
        )
        if seed == args.split_seeds[0]:
            np.save(emb_dir / "transformer.npy", emb)
        pd.DataFrame(rows).to_csv(out / "metrics_by_split.csv", index=False)

    run = {
        "name": args.name,
        "commit": git_commit(ROOT),
        "device": args.device,
        "torch_threads": args.threads,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_top_genes": args.n_top_genes,
        "settings": "cellmap.transformer.train_transformer defaults except those listed here",
        "runs": runs,
        "versions": versions("torch", "scanpy", "scib-metrics"),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
