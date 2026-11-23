"""Sweep the KL weight of the hand-written VAE to show both failure modes.

Usage:
    python scripts/m2_kl_sweep.py --name pbmc

With weight 0 the model is a plain autoencoder: nothing ties the codes to the prior, so the KL
term is large. With a very large weight the codes collapse onto the prior and every cell gets
the same code. Everything here uses split seed 0 and is scored on VALIDATION donors, because
choosing a KL weight is model selection and the test donors must not be used for that.

Writes:
    results/m2/<name>/kl_sweep.csv            one row per KL weight
    data/embeddings/<name>/vae_kl_<w>.npy     embeddings per weight (not committed)
"""

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellmap import vae
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split
from cellmap.evaluate import knn_classification
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [0.0, 0.1, 1.0, 10.0, 100.0]
N_EPOCHS = 60
ACTIVE_VARIANCE = 0.01  # a latent dimension counts as in use if its mean varies this much


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    donors = obs["donor_id"].astype(str).to_numpy()
    split = donor_split(obs, 0)
    train, val = split == "train", split == "val"
    genes = select_hvg(adata, train, n_top_genes=min(2000, adata.n_vars))
    counts = adata.layers["counts"][:, genes].tocsr()
    train_donors = sorted(set(donors[train]))
    codes = vae.batch_codes(donors, train_donors)

    out = ROOT / "results" / "m2" / args.name
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / "data" / "embeddings" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for weight in WEIGHTS:
        t0 = time.perf_counter()
        model, _ = vae.train_vae(
            counts[train],
            codes[train],
            n_batches=len(train_donors),
            kl_weight=weight,
            n_epochs=args.epochs,
            seed=0,
        )
        seconds = time.perf_counter() - t0
        emb = vae.embed(model, counts)
        np.save(emb_dir / f"vae_kl_{weight:g}.npy", emb)
        train_recon, train_kl = vae.elbo_terms(model, counts[train], codes[train])
        val_recon, val_kl = vae.elbo_terms(model, counts[val], codes[val])
        variance = emb[val].var(axis=0)
        knn = knn_classification(emb, obs["cell_type"], split, test="val")
        rows.append(
            {
                "kl_weight": weight,
                "n_epochs": args.epochs,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_recon": val_recon,
                "val_kl": val_kl,
                "latent_dims_in_use": int((variance > ACTIVE_VARIANCE).sum()),
                "max_latent_variance": float(variance.max()),
                "val_knn_accuracy_fine": knn["accuracy"],
                "val_knn_macro_f1_fine": knn["macro_f1"],
                "seconds": seconds,
                "commit": git_commit(ROOT),
            }
        )
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out / "kl_sweep.csv", index=False)


if __name__ == "__main__":
    main()
