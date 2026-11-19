"""M2: the hand-written VAE and the linear autoencoder, scored like the baselines.

Usage:
    python scripts/m2_vae.py --name pbmc

For each donor split: select genes on the training donors, train the negative binomial VAE and
the linear autoencoder on the training donors only, embed every cell, and score on the test
donors with the same harness as M1. A 10-component PCA is scored too, because the VAE and the
linear autoencoder have 10 latent dimensions while the M1 PCA baseline has 50.

Writes:
    results/m2/<name>/metrics_by_split.csv  one row per (method, split seed)
    results/m2/<name>/history_seed<k>.csv   VAE loss terms per epoch
    results/m2/<name>/run.json              settings, timings, parameter counts, commit
    data/embeddings/<name>/<method>.npy     split seed 0 embeddings (not committed)
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellmap import linear_ae, vae
from cellmap.baselines import pca_embed, select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]
N_TOP_GENES = 2000
N_LATENT = 10
VAE_EPOCHS = 100
LINEAR_AE_EPOCHS = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--vae-epochs", type=int, default=VAE_EPOCHS)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    donors = obs["donor_id"].astype(str).to_numpy()
    out = ROOT / "results" / "m2" / args.name
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / "data" / "embeddings" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows, timings = [], []
    n_params = {}
    for seed in args.split_seeds:
        split = donor_split(obs, seed)
        train, val = split == "train", split == "val"
        genes = select_hvg(adata, train, n_top_genes=min(N_TOP_GENES, adata.n_vars))
        counts = adata.layers["counts"][:, genes].tocsr()
        lognorm = adata.X[:, genes].tocsr()

        train_donors = sorted(set(donors[train]))
        codes = vae.batch_codes(donors, train_donors)  # held-out donors get -1

        t0 = time.perf_counter()
        model, history = vae.train_vae(
            counts[train],
            codes[train],
            n_batches=len(train_donors),
            n_latent=N_LATENT,
            n_epochs=args.vae_epochs,
            seed=seed,
            counts_val=counts[val],
            batch_val=codes[val],
        )
        t_vae = time.perf_counter() - t0
        emb_vae = vae.embed(model, counts)
        pd.DataFrame(history).to_csv(out / f"history_seed{seed}.csv", index_label="epoch")
        n_params["vae"] = sum(p.numel() for p in model.parameters())

        t0 = time.perf_counter()
        lae, _ = linear_ae.train_linear_ae(
            lognorm[train], n_latent=N_LATENT, n_epochs=LINEAR_AE_EPOCHS, seed=seed
        )
        t_lae = time.perf_counter() - t0
        emb_lae = linear_ae.embed(lae, lognorm)
        n_params["linear_ae"] = sum(p.numel() for p in lae.parameters())

        t0 = time.perf_counter()
        emb_pca10 = pca_embed(adata, genes, train, n_comps=N_LATENT)
        t_pca = time.perf_counter() - t0

        timings.append(
            {"split_seed": seed, "vae_s": t_vae, "linear_ae_s": t_lae, "pca_10_s": t_pca}
        )
        embeddings = {"vae": emb_vae, "linear_ae": emb_lae, "pca_10": emb_pca10}
        for method, emb in embeddings.items():
            scores = score_embedding(emb, obs, split)
            rows.append({"method": method, "split_seed": seed, **scores})
            print(
                f"seed {seed} {method}: acc={scores['knn_accuracy_fine']:.3f} "
                f"f1={scores['knn_macro_f1_fine']:.3f} bio={scores['scib_bio_mean']:.3f} "
                f"batch={scores['scib_batch_mean']:.3f}",
                flush=True,
            )
            if seed == args.split_seeds[0]:
                np.save(emb_dir / f"{method}.npy", emb)
        pd.DataFrame(rows).to_csv(out / "metrics_by_split.csv", index=False)

    run = {
        "name": args.name,
        "commit": git_commit(ROOT),
        "split_seeds": args.split_seeds,
        "n_top_genes": N_TOP_GENES,
        "n_latent": N_LATENT,
        "vae_epochs": args.vae_epochs,
        "linear_ae_epochs": LINEAR_AE_EPOCHS,
        "vae_settings": "cellmap.vae.train_vae defaults except n_latent, n_epochs, seed",
        "torch_threads": args.threads,
        "timing_note": "other jobs shared the machine during this run; times are upper bounds",
        "n_parameters": n_params,
        "timings": timings,
        "versions": versions("torch", "scanpy", "scib-metrics", "numpy"),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
