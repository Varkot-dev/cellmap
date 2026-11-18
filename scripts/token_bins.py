"""How the transformer's two level encodings spread training tokens over the 10 level bins.

The rank encoding (M5) bins each expressed gene by its rank within the cell; the value encoding
(M7) bins depth-normalised values with cut points fitted on training cells. This records the
share of training tokens in each bin under both, and the entropy in bits (log2 10 = 3.32 is the
most a 10-bin code can carry), on PBMC split seed 0 with the 2,000 genes the models use. It is
the saved source for the share of tokens in bin 0 (57 percent; first quoted as 61 percent from an unsaved check), which an independent
review found had no file under results/.

Writes results/m7/<name>/token_bins.csv.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellmap import transformer
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]
N_BINS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    train = donor_split(adata.obs, 0) == "train"
    genes = select_hvg(adata, train, n_top_genes=2000)
    counts = adata.layers["counts"][:, genes].tocsr()[train]

    encodings = {
        "rank": transformer.tokenize(counts, N_BINS),
        "value": transformer.tokenize(counts, N_BINS, transformer.fit_value_edges(counts, N_BINS)),
    }
    rows = []
    for name, tokens in encodings.items():
        share = np.bincount(tokens.bins, minlength=N_BINS) / len(tokens.bins)
        nonzero = share[share > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
        for b, s in enumerate(share):
            rows.append({"encoding": name, "bin": b, "share_of_tokens": float(s),
                         "entropy_bits": entropy, "n_tokens": len(tokens.bins),
                         "git_commit": git_commit(ROOT)})  # fmt: skip
    out = ROOT / "results" / "m7" / args.name
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "token_bins.csv", index=False)
    print(table.pivot(index="bin", columns="encoding", values="share_of_tokens").round(3))
    print(table.groupby("encoding")["entropy_bits"].first().round(2))


if __name__ == "__main__":
    main()
