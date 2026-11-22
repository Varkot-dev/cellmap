"""Audit part 1: which genes does each transformer rely on, cell type by cell type?

Usage:
    python scripts/audit_knockout.py --name pbmc

Design fixed in docs/PLAN.md (2026-11-22, model audit) before any result. For each fine type
with at least 30 test cells on split seed 0, up to 100 test cells are sampled. Every gene
detected in at least 20 percent of them is knocked out in turn (its count set to zero in the
model's input) and the cells are embedded again. A gene's effect on the type is the mean of
1 - cos(h, h after the knockout); its flip rate is the share of the cells whose kNN label
(15 neighbours among the training cells, as in the benchmark) changes.

Two rankings of the same candidates are compared with the knockout ranking:
    markers     Wilcoxon one-versus-rest on training cells (scanpy rank_genes_groups)
    expression  mean log expression in the sampled cells

Writes results/audit/<name>/:
    knockout.csv   one row per model, type and candidate gene
    overlap.csv    one row per model and type: overlap at 20 of each ranking with the markers
    run.json
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import KNeighborsClassifier

from cellmap import transformer
from cellmap.audit import knockout_effects, overlap_at_k
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED, MIN_CELLS, PER_TYPE, MIN_DETECTION, MIN_CANDIDATES, TOP = 0, 30, 100, 0.2, 40, 20
MODELS = {
    "contrastive": "results/contrastive/{name}/model_contrastive_seed0.pt",
    "contrastive_mgm": "results/contrastive/{name}/model_contrastive_mgm_seed0.pt",
    "masked_gene": "results/m5/{name}/model_seed0.pt",
}
STRESS = {"FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "IER2", "IER3", "DUSP1", "HSPA1A",
          "HSPA1B", "ZFP36", "KLF6", "NR4A1"}  # fmt: skip


def gene_group(name: str) -> str:
    """A coarse label for genes that would point at something other than cell identity."""
    if name.startswith("HLA-"):
        return "HLA"
    if name.startswith(("RPL", "RPS")):
        return "ribosomal"
    if name.startswith("MT-"):
        return "mitochondrial"
    if name in STRESS:
        return "stress (immediate early)"
    if name in {"HBB", "HBA1", "HBA2"}:
        return "haemoglobin"
    return "other"


def load(path: Path, n_genes: int) -> transformer.CellTransformer:
    saved = torch.load(path, weights_only=True)
    if saved["n_genes"] != n_genes:
        raise RuntimeError(f"{path} was trained on {saved['n_genes']} genes, not {n_genes}")
    model = transformer.CellTransformer(saved["n_genes"])
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def marker_ranks(logx, labels: np.ndarray, types: list[str], names: np.ndarray) -> pd.DataFrame:
    """Wilcoxon score of every gene for every type (one versus the rest), training cells."""
    train = ad.AnnData(logx, obs=pd.DataFrame({"t": pd.Categorical(labels)}))
    train.var_names = names
    train.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(train, "t", groups=types, reference="rest", method="wilcoxon",
                            n_genes=len(names))  # fmt: skip
    out = {}
    for t in types:
        df = sc.get.rank_genes_groups_df(train, group=t)
        out[t] = df.set_index("names")["scores"]
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    split = donor_split(obs, SPLIT_SEED)
    train, test = split == "train", split == "test"
    genes = select_hvg(adata, train, n_top_genes=2000)
    counts = adata.layers["counts"][:, genes].tocsr()
    logx = adata.X[:, genes].tocsr()
    names = np.asarray(adata.var["feature_name"]).astype(str)[genes]
    labels = obs["cell_type"].astype(str).to_numpy()

    rng = np.random.default_rng(SPLIT_SEED)
    sizes = pd.Series(labels[test]).value_counts()
    types = sorted(sizes[sizes >= MIN_CELLS].index)
    sampled, candidates = {}, {}
    for t in types:
        rows = np.flatnonzero(test & (labels == t))
        rows = np.sort(rng.choice(rows, size=min(PER_TYPE, len(rows)), replace=False))
        detection = np.asarray((counts[rows] > 0).mean(axis=0)).ravel()
        cand = np.flatnonzero(detection >= MIN_DETECTION)
        if len(cand) >= MIN_CANDIDATES:
            sampled[t], candidates[t] = rows, cand
    types = sorted(sampled)
    print(f"{len(types)} types; candidates per type: "
          f"{[len(candidates[t]) for t in types]}", flush=True)  # fmt: skip
    markers = marker_ranks(logx[train], labels[train], types, names)

    rows_out, overlaps = [], []
    for model_name in args.models:
        model = load(ROOT / MODELS[model_name].format(name=args.name), len(names))

        def embed_fn(x, model=model):
            return transformer.embed(model, x)

        t0 = time.perf_counter()
        emb_train = embed_fn(counts[train])
        knn = KNeighborsClassifier(n_neighbors=15).fit(emb_train, labels[train])
        for t in types:
            rows, cand = sampled[t], candidates[t]
            base = embed_fn(counts[rows])
            before = knn.predict(base)
            flips = np.zeros(len(cand))

            def count_flips(i, moved_rows, moved, before=before, flips=flips, knn=knn):
                flips[i] = (knn.predict(moved) != before[moved_rows]).sum()

            effects = knockout_effects(embed_fn, counts[rows], cand, base=base,
                                       on_moved=count_flips)  # fmt: skip
            mean_effect = effects.mean(axis=1)
            detection = np.asarray((counts[rows][:, cand] > 0).mean(axis=0)).ravel()
            expression = np.asarray(logx[rows][:, cand].mean(axis=0)).ravel()
            de_score = markers.loc[names[cand], t].to_numpy()
            frame = pd.DataFrame({
                "model": model_name, "cell_type": t, "gene": names[cand], "gene_index": cand,
                "group": [gene_group(g) for g in names[cand]], "detection": detection,
                "mean_expression": expression, "mean_effect": mean_effect,
                "flip_rate": flips / len(rows), "marker_score": de_score,
            })  # fmt: skip
            for col, key in (("knockout_rank", "mean_effect"), ("marker_rank", "marker_score"),
                             ("expression_rank", "mean_expression")):  # fmt: skip
                frame[col] = frame[key].rank(ascending=False, method="first").astype(int)
            rows_out.append(frame)
            order = {c: frame.sort_values(c)["gene"].tolist()
                     for c in ("knockout_rank", "marker_rank", "expression_rank")}  # fmt: skip
            overlaps.append({
                "model": model_name, "cell_type": t, "n_cells": len(rows),
                "n_candidates": len(cand),
                "overlap_knockout_markers": overlap_at_k(order["knockout_rank"],
                                                         order["marker_rank"], TOP),
                "overlap_expression_markers": overlap_at_k(order["expression_rank"],
                                                           order["marker_rank"], TOP),
                "top_knockout": " ".join(order["knockout_rank"][:TOP]),
                "top_markers": " ".join(order["marker_rank"][:TOP]),
                "top_expression": " ".join(order["expression_rank"][:TOP]),
            })  # fmt: skip
            o = overlaps[-1]
            print(f"{model_name} | {t}: knockout {o['overlap_knockout_markers']:.2f} "
                  f"expression {o['overlap_expression_markers']:.2f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)  # fmt: skip

    out = ROOT / "results" / "audit" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows_out).to_csv(out / "knockout.csv", index=False)
    pd.DataFrame(overlaps).to_csv(out / "overlap.csv", index=False)
    run = {
        "name": args.name, "split_seed": SPLIT_SEED, "commit": git_commit(ROOT),
        "models": {m: MODELS[m].format(name=args.name) for m in args.models},
        "min_test_cells": MIN_CELLS, "cells_per_type": PER_TYPE,
        "min_detection": MIN_DETECTION, "min_candidates": MIN_CANDIDATES, "top": TOP,
        "knn_neighbours": 15, "versions": versions("torch", "scanpy", "scikit-learn"),
    }  # fmt: skip
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
