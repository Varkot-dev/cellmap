"""Which parts of a cell's input do the learned models actually use? A perturbation probe.

Usage:
    python scripts/probe_inputs.py --name pbmc

Same idea as the shuffle probes in Bindwell's "The Bitter Lesson for Biology" (2026) and
Fang et al. (2026): keep a trained model fixed, damage one part of every cell's input at
inference time, and see how much cell-type information survives in the embedding.

Models: the hand-written VAE (retrained here on split seed 0, as in M2) and the hand-written
transformer (the weights trained on a GPU in M5, results/m5/<name>/model_seed0.pt).

Probes, applied to every cell's counts over the 2,000 selected genes:
    none             the real input
    presence only    every nonzero count set to 1: which genes are on, not how much
    levels shuffled  each cell's nonzero counts shuffled among its own expressed genes
    genes relabelled the gene columns permuted: right amounts, wrong gene identities

Score: kNN on test donors, references from training donors, all embedded from the same probed
input. If a probe barely changes the score, the model does not need that information.

Writes:
    results/probe/<name>/probe_inputs.csv
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from cellmap import transformer, vae
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split
from cellmap.evaluate import knn_classification
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]


def presence_only(x: sp.csr_matrix) -> sp.csr_matrix:
    out = x.copy()
    out.data = np.ones_like(out.data)
    return out


def levels_shuffled(x: sp.csr_matrix, seed: int = 0) -> sp.csr_matrix:
    out = x.copy()
    rng = np.random.default_rng(seed)
    for i in range(out.shape[0]):
        start, end = out.indptr[i], out.indptr[i + 1]
        out.data[start:end] = rng.permutation(out.data[start:end])
    return out


def genes_relabelled(x: sp.csr_matrix, seed: int = 0) -> sp.csr_matrix:
    perm = np.random.default_rng(seed).permutation(x.shape[1])
    return x[:, perm].tocsr()


PROBES = {
    "none": lambda x: x,
    "presence only": presence_only,
    "levels shuffled": levels_shuffled,
    "genes relabelled": genes_relabelled,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    split = donor_split(obs, 0)
    train = split == "train"
    genes = select_hvg(adata, train, n_top_genes=2000)
    counts = adata.layers["counts"][:, genes].tocsr()
    donors = obs["donor_id"].astype(str).to_numpy()

    known = sorted(set(donors[train]))
    codes = vae.batch_codes(donors, known)
    vae_model, _ = vae.train_vae(counts[train], codes[train], n_batches=len(known), seed=0)

    saved = torch.load(ROOT / "results" / "m5" / args.name / "model_seed0.pt", weights_only=True)
    if int(genes.sum()) != saved["n_genes"]:
        raise RuntimeError("gene selection does not match the saved transformer")
    tf_model = transformer.CellTransformer(saved["n_genes"])
    tf_model.load_state_dict(saved["state_dict"])

    models = {
        "VAE (hand-written)": lambda x: vae.embed(vae_model, x),
        "Transformer (hand-written)": lambda x: transformer.embed(tf_model, x),
    }
    rows = []
    for probe, damage in PROBES.items():
        probed = damage(counts)
        for model, embed in models.items():
            emb = embed(probed)
            knn = knn_classification(emb, obs["cell_type"], split)
            coarse = knn_classification(emb, obs["cell_type_coarse"], split)
            rows.append(
                {
                    "model": model,
                    "probe": probe,
                    "knn_accuracy_fine": knn["accuracy"],
                    "knn_macro_f1_fine": knn["macro_f1"],
                    "knn_macro_f1_coarse": coarse["macro_f1"],
                    "commit": git_commit(ROOT),
                }
            )
            print(f"{model} | {probe}: acc={knn['accuracy']:.3f} f1={knn['macro_f1']:.3f}",
                  flush=True)  # fmt: skip
    out = ROOT / "results" / "probe" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "probe_inputs.csv", index=False)


if __name__ == "__main__":
    main()
