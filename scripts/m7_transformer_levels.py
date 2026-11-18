"""M7: does keeping expression levels fix the hand-written transformer?

Usage:
    python scripts/m7_transformer_levels.py --name pbmc

The input probe (scripts/probe_inputs.py) showed the M5 transformer scores the same with only
"which genes are on" as with its real input: its per-cell rank binning put about 57 percent of
tokens in bin 0, so the levels carried almost no information. This trains the same model with
the depth-normalized "value" encoding instead (cellmap.transformer.fit_value_edges), on split
seed 0 with every other setting unchanged, then scores it and reruns the probe.

Compare with the "rank" model of M5 on the same split: results/m5/<name>/metrics_by_split.csv
(split seed 0) and results/probe/<name>/probe_inputs.csv.

Writes:
    results/m7/<name>/transformer_levels.csv   scores and probe results
    results/m7/<name>/transformer_levels_history.csv
"""

import argparse
import time
from pathlib import Path

import anndata as ad
import pandas as pd
import torch

from cellmap import transformer
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.evaluate import knn_classification
from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    import probe_inputs  # same folder; reuse the exact probe definitions

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    split = donor_split(obs, 0)
    train, val = split == "train", split == "val"
    genes = select_hvg(adata, train, n_top_genes=2000)
    counts = adata.layers["counts"][:, genes].tocsr()

    t0 = time.perf_counter()
    model, history = transformer.train_transformer(
        counts[train], seed=0, device=args.device, counts_val=counts[val], level_encoding="value"
    )
    seconds = time.perf_counter() - t0
    out = ROOT / "results" / "m7" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "transformer_levels_history.csv", index_label="epoch")

    emb = transformer.embed(model, counts, device=args.device)
    full = score_embedding(emb, obs, split)
    rows = [{"encoding": "value", "probe": "none (full scoring)", **full}]
    for probe, damage in probe_inputs.PROBES.items():
        probed = transformer.embed(model, damage(counts), device=args.device)
        knn = knn_classification(probed, obs["cell_type"], split)
        rows.append({"encoding": "value", "probe": probe, "knn_accuracy_fine": knn["accuracy"],
                     "knn_macro_f1_fine": knn["macro_f1"]})  # fmt: skip
        print(f"value | {probe}: f1={knn['macro_f1']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    frame["train_seconds"] = seconds
    frame["final_val_accuracy"] = history["val_accuracy"][-1]
    frame["val_gene_majority"] = history["val_gene_majority"][-1]
    frame["commit"] = git_commit(ROOT)
    frame.to_csv(out / "transformer_levels.csv", index=False)
    print(frame[["probe", "knn_accuracy_fine", "knn_macro_f1_fine"]].round(3).to_string())


if __name__ == "__main__":
    main()
