"""The same transformer, trained with a contrastive cell-level objective, scored like the rest.

Usage:
    python scripts/transformer_contrastive.py --name pbmc --device cuda       # on a GPU
    python scripts/transformer_contrastive.py --name pbmc --device cuda --resume

Why: the masked-gene transformer learned its own task (and, with value bins, learned it well)
but its cell embedding stayed weak. This trains the identical architecture with SimCLR
(cellmap.contrastive) instead, on the same five PBMC donor splits, 2,000 genes, rank level
encoding and 30 epochs, and scores it with the same harness. Two objectives:

    contrastive      SimCLR loss only
    contrastive_mgm  SimCLR loss plus the masked-gene loss of the original transformer

Hypotheses are fixed in docs/PLAN.md (2026-11-19, contrastive transformer) before any result.

For split seed 0 the input probes of scripts/probe_inputs.py are also run (presence only,
levels shuffled, genes relabelled), to see whether the new objective makes the model use
expression levels.

Writes results/contrastive/<name>/:
    metrics_by_split.csv       one row per objective and split seed
    probes.csv                 split seed 0, both objectives
    history_<objective>_seed<k>.csv
    model_<objective>_seed0.pt weights, so the embedding can be rebuilt without a GPU
    run.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import pandas as pd
import torch

from cellmap import transformer
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.contrastive import train_contrastive
from cellmap.evaluate import knn_classification
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_inputs  # noqa: E402  (same folder; reuse the exact probe definitions)

OBJECTIVES = {"contrastive": 0.0, "contrastive_mgm": 1.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--resume", action="store_true", help="skip runs already saved")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    out = ROOT / "results" / "contrastive" / args.name
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, probes_path = out / "metrics_by_split.csv", out / "probes.csv"
    if (metrics_path.exists() or probes_path.exists()) and not args.resume:
        parser.error("results exist; pass --resume to add to them")
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    probes = pd.read_csv(probes_path).to_dict("records") if probes_path.exists() else []
    done = {(r["method"], int(r["split_seed"])) for r in rows}

    for seed in args.split_seeds:
        split = donor_split(obs, seed)
        train, val = split == "train", split == "val"
        genes = select_hvg(adata, train, n_top_genes=min(args.n_top_genes, adata.n_vars))
        counts = adata.layers["counts"][:, genes].tocsr()
        for objective in args.objectives:
            method = f"transformer_{objective}"
            if (method, seed) in done:
                continue
            t0 = time.perf_counter()
            model, history = train_contrastive(
                counts[train],
                mgm_weight=OBJECTIVES[objective],
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                device=args.device,
                counts_val=counts[val],
            )
            seconds = time.perf_counter() - t0
            pd.DataFrame(history).to_csv(
                out / f"history_{objective}_seed{seed}.csv", index_label="epoch"
            )
            emb = transformer.embed(model, counts, device=args.device)
            scores = score_embedding(emb, obs, split)
            rows.append(
                {
                    "method": method,
                    "split_seed": seed,
                    **scores,
                    "train_seconds": seconds,
                    "final_val_partner_accuracy": history["val_partner_accuracy"][-1],
                    "commit": git_commit(ROOT),
                }
            )
            print(
                f"seed {seed} {objective}: acc={scores['knn_accuracy_fine']:.3f} "
                f"f1={scores['knn_macro_f1_fine']:.3f} bio={scores['scib_bio_mean']:.3f} "
                f"batch={scores['scib_batch_mean']:.3f} ({seconds:.0f}s)",
                flush=True,
            )
            if seed == 0:
                for probe, damage in probe_inputs.PROBES.items():
                    probed = transformer.embed(model, damage(counts), device=args.device)
                    knn = knn_classification(probed, obs["cell_type"], split)
                    probes.append(
                        {
                            "method": method,
                            "probe": probe,
                            "knn_accuracy_fine": knn["accuracy"],
                            "knn_macro_f1_fine": knn["macro_f1"],
                        }
                    )
                pd.DataFrame(probes).to_csv(probes_path, index=False)
                weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(
                    {"n_genes": int(counts.shape[1]), "split_seed": seed, "state_dict": weights},
                    out / f"model_{objective}_seed0.pt",
                )
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

    run = {
        "name": args.name,
        "commit": git_commit(ROOT),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_top_genes": args.n_top_genes,
        "objectives": OBJECTIVES,
        "settings": "cellmap.contrastive.train_contrastive defaults except those listed here",
        "versions": versions("torch", "scanpy", "scib-metrics"),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
