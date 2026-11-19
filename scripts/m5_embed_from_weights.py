"""Rebuild the transformer embedding from saved weights, without retraining.

Usage:
    python scripts/m5_embed_from_weights.py --name pbmc

The transformer is trained on a GPU (notebooks/m5_transformer_gpu.ipynb), which saves the
first split's weights to results/m5/<name>/model_seed<k>.pt. This script loads them, selects
the same genes (selection is deterministic given the split), and writes
data/embeddings/<name>/transformer.npy for the atlas. It does not change any score.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from cellmap import transformer
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    args = parser.parse_args()

    out = ROOT / "results" / "m5" / args.name
    saved = torch.load(next(out.glob("model_seed*.pt")), weights_only=True)
    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    split = donor_split(adata.obs, saved["split_seed"])
    genes = select_hvg(adata, split == "train", n_top_genes=min(args.n_top_genes, adata.n_vars))
    if int(genes.sum()) != saved["n_genes"]:
        raise RuntimeError(f"selected {genes.sum()} genes, the saved model has {saved['n_genes']}")

    model = transformer.CellTransformer(saved["n_genes"])
    model.load_state_dict(saved["state_dict"])
    counts = adata.layers["counts"][:, genes].tocsr()
    emb = transformer.embed(model, counts)
    path = ROOT / "data" / "embeddings" / args.name / "transformer.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    print(f"wrote {path} {emb.shape}")


if __name__ == "__main__":
    main()
