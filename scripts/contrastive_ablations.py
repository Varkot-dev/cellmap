"""Contrastive transformer ablations: one change per run, to learn why it works and where not.

Usage:
    python scripts/contrastive_ablations.py --name pbmc --device cuda
    python scripts/contrastive_ablations.py --name pbmc --variants base tau_0.5 --resume

Design and predictions are fixed in docs/PLAN.md (2026-11-19, contrastive ablations) before
any ablation result. Every run uses split seed 0, the `contrastive` objective and the settings
of scripts/transformer_contrastive.py, except the one change its variant names:

    base         nothing; a rerun, to measure run-to-run noise against the main run
    tau_0.05     temperature 0.05 (base 0.1)
    tau_0.2      temperature 0.2
    tau_0.5      temperature 0.5
    keep_strong  each view keeps 30 to 60 percent of a cell's tokens (base 50 to 90)
    keep_weak    each view keeps 80 to 100 percent
    no_jitter    no level nudging in the views (base: 20 percent of tokens move one bin)
    same_donor   every mini-batch holds cells of one donor, so donor identity cannot tell a
                 cell's partner from its negatives
    debiased     debiased loss (Chuang et al. 2020) with tau+ = 0.1, which corrects for
                 negatives that share the anchor's type

Each run is scored twice with the usual harness: on the [CLS] state h (the embedding used
everywhere) and on the projection z = head(h). Alignment and uniformity on validation cells
come from the last epoch.

Writes results/contrastive_ablations/<name>/:
    metrics.csv                 one row per variant and embedding (h or z)
    history_<variant>.csv       per-epoch losses, partner accuracy, alignment, uniformity
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

from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.contrastive import project, train_contrastive
from cellmap.provenance import git_commit, versions
from cellmap.transformer import embed

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED = 0
VARIANTS: dict[str, dict] = {
    "base": {},
    "tau_0.05": {"temperature": 0.05},
    "tau_0.2": {"temperature": 0.2},
    "tau_0.5": {"temperature": 0.5},
    "keep_strong": {"keep_range": (0.3, 0.6)},
    "keep_weak": {"keep_range": (0.8, 1.0)},
    "no_jitter": {"jitter": 0.0},
    "same_donor": {"batch_groups": "donor"},
    "debiased": {"loss": "debiased", "tau_plus": 0.1},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--resume", action="store_true", help="skip variants already saved")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    out = ROOT / "results" / "contrastive_ablations" / args.name
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.csv"
    if metrics_path.exists() and not args.resume:
        parser.error("results exist; pass --resume to add to them")
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    done = {r["variant"] for r in rows if r["embedding"] == "z"}

    split = donor_split(obs, SPLIT_SEED)
    train, val = split == "train", split == "val"
    genes = select_hvg(adata, train, n_top_genes=min(args.n_top_genes, adata.n_vars))
    counts = adata.layers["counts"][:, genes].tocsr()
    donors = obs["donor_id"].astype(str).to_numpy()

    for variant in args.variants:
        if variant in done:
            continue
        settings = dict(VARIANTS[variant])
        if settings.get("batch_groups") == "donor":
            settings["batch_groups"] = donors[train]
        t0 = time.perf_counter()
        model, history, head = train_contrastive(
            counts[train],
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            seed=SPLIT_SEED,
            device=args.device,
            counts_val=counts[val],
            return_head=True,
            **settings,
        )
        seconds = time.perf_counter() - t0
        pd.DataFrame(history).to_csv(out / f"history_{variant}.csv", index_label="epoch")
        embeddings = {
            "h": embed(model, counts, device=args.device),
            "z": project(model, head, counts, device=args.device),
        }
        for kind, emb in embeddings.items():
            scores = score_embedding(emb, obs, split)
            rows.append(
                {
                    "variant": variant,
                    "embedding": kind,
                    **scores,
                    "train_seconds": seconds,
                    "final_val_partner_accuracy": history["val_partner_accuracy"][-1],
                    "final_val_alignment": history["val_alignment"][-1],
                    "final_val_uniformity": history["val_uniformity"][-1],
                    "commit": git_commit(ROOT),
                }
            )
            print(
                f"{variant} {kind}: acc={scores['knn_accuracy_fine']:.3f} "
                f"f1={scores['knn_macro_f1_fine']:.3f} ilisi={scores['scib_ilisi']:.3f} "
                f"bio={scores['scib_bio_mean']:.3f} ({seconds:.0f}s)",
                flush=True,
            )
        pd.DataFrame(rows).to_csv(metrics_path, index=False)

    run = {
        "name": args.name,
        "split_seed": SPLIT_SEED,
        "commit": git_commit(ROOT),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_top_genes": args.n_top_genes,
        "variants": {k: {n: str(v) for n, v in s.items()} for k, s in VARIANTS.items()},
        "settings": "cellmap.contrastive.train_contrastive defaults except those listed here",
        "versions": versions("torch", "scanpy", "scib-metrics"),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
