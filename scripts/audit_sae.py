"""Audit part 3: a sparse autoencoder on the contrastive transformer's embedding.

Usage:
    python scripts/audit_sae.py --name pbmc

Design fixed in docs/PLAN.md (2026-11-23, model audit part 3) before any result. Every cell is
embedded with the contrastive split-0 model; the embeddings are centred and scaled on training
cells; an SAE with 1,024 features is fitted on training donors for each L1 weight in a fixed
grid and one is chosen on validation donors. Features are then labelled (cell-type features,
batch features, top genes), and the 10 strongest batch features are removed from every cell's
embedding to see whether donors mix better without losing cell types.

Writes results/audit/<name>/:
    sae_grid.csv          per L1 weight: explained variance and mean active features (val)
    sae_features.csv      per feature: frequency, type label, batch score, AUROC, top genes
    sae_intervention.csv  harness scores of the embedding before and after each removal
    sae.pt                the chosen SAE's weights and the centring constants
    sae_run.json
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from cellmap import transformer
from cellmap.baselines import select_hvg
from cellmap.benchmark import donor_split, score_embedding
from cellmap.provenance import git_commit, versions
from cellmap.sae import (
    activations,
    eta_squared,
    explained_variance,
    reconstruct,
    remove_features,
    train_sae,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED, N_FEATURES, EPOCHS, MAX_L0, K_REMOVE, N_CONTROLS = 0, 1024, 200, 20, 10, 10
GRID = [0.001, 0.003, 0.01, 0.03, 0.1]
TYPE_RATIO, TYPE_ACTIVE, MIN_GROUP_CELLS, N_TOP_GENES = 3.0, 0.2, 50, 8


def batch_group(donor: str, assay: str) -> str:
    if "v3" in assay:
        return "10x v3"
    return "C2 (10x v2)" if donor.startswith("C2") else "other 10x v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    split = donor_split(obs, SPLIT_SEED)
    train, val = split == "train", split == "val"
    genes = select_hvg(adata, train, n_top_genes=2000)
    counts = adata.layers["counts"][:, genes].tocsr()
    logx = adata.X[:, genes].tocsr()
    names = np.asarray(adata.var["feature_name"]).astype(str)[genes]
    donors = obs["donor_id"].astype(str).to_numpy()
    fine = obs["cell_type"].astype(str).to_numpy()
    coarse = obs["cell_type_coarse"].astype(str).to_numpy()
    groups = np.array(
        [batch_group(d, a) for d, a in zip(donors, obs["assay"].astype(str), strict=True)]
    )

    saved = torch.load(ROOT / "results" / "contrastive" / args.name / "model_contrastive_seed0.pt",
                       weights_only=True)  # fmt: skip
    model = transformer.CellTransformer(saved["n_genes"])
    model.load_state_dict(saved["state_dict"])
    h = transformer.embed(model, counts)
    mu = h[train].mean(axis=0)
    scale = float(np.sqrt(((h[train] - mu) ** 2).sum(axis=1).mean()))
    x = ((h - mu) / scale).astype(np.float32)

    grid, models = [], {}
    for l1 in GRID:
        sae = train_sae(x[train], n_features=N_FEATURES, l1=l1, n_epochs=EPOCHS, seed=SPLIT_SEED)
        ev = explained_variance(x[val], reconstruct(sae, x[val]))
        l0 = float((activations(sae, x[val]) > 0).sum(axis=1).mean())
        grid.append({"l1": l1, "val_explained_variance": ev, "val_mean_l0": l0})
        models[l1] = sae
        print(f"l1 {l1}: explained variance {ev:.3f}, L0 {l0:.1f}", flush=True)
    grid = pd.DataFrame(grid)
    ok = grid[grid["val_mean_l0"] <= MAX_L0]
    chosen = (ok.sort_values("val_explained_variance").iloc[-1] if len(ok)
              else grid.sort_values("val_mean_l0").iloc[0])  # fmt: skip
    l1 = float(chosen["l1"])
    sae = models[l1]
    grid["chosen"] = grid["l1"] == l1

    acts = activations(sae, x)
    a_train = acts[train]
    live = (a_train > 0).any(axis=0)

    # cell-type features: mean activation per coarse type, training cells
    types = sorted(set(coarse[train]))
    means = np.stack([a_train[coarse[train] == t].mean(axis=0) for t in types])
    active = np.stack([(a_train[coarse[train] == t] > 0).mean(axis=0) for t in types])
    type_label = []
    for j in range(N_FEATURES):
        best = int(np.argmax(means[:, j]))
        others = np.delete(means[:, j], best)
        is_type = (live[j] and means[best, j] >= TYPE_RATIO * others.max()
                   and active[best, j] >= TYPE_ACTIVE)  # fmt: skip
        type_label.append(types[best] if is_type else "")

    # batch score: eta squared on batch group within each fine type, cell-weighted mean
    g_train, f_train = groups[train], fine[train]
    usable = []
    for t in sorted(set(f_train)):
        sizes = pd.Series(g_train[f_train == t]).value_counts()
        if (sizes >= MIN_GROUP_CELLS).sum() >= 2:
            keep = (f_train == t) & np.isin(g_train, sizes[sizes >= MIN_GROUP_CELLS].index)
            usable.append(keep)
    batch_score = np.zeros(N_FEATURES)
    type_eta = np.zeros(N_FEATURES)
    for j in np.flatnonzero(live):
        parts = [(m.sum(), eta_squared(a_train[m, j], g_train[m])) for m in usable]
        batch_score[j] = sum(n * e for n, e in parts) / sum(n for n, _ in parts)
        type_eta[j] = eta_squared(a_train[:, j], f_train)

    # S3: C2 against other 10x v2 donors, training classical monocytes
    mono = (f_train == "classical monocyte") & np.isin(g_train, ["C2 (10x v2)", "other 10x v2"])
    y = (g_train[mono] == "C2 (10x v2)").astype(int)
    auroc = np.full(N_FEATURES, 0.5)
    for j in np.flatnonzero(live):
        if a_train[mono, j].std() > 0:
            auroc[j] = roc_auc_score(y, a_train[mono, j])

    # top genes: Pearson correlation of activation with log expression, training cells
    rng = np.random.default_rng(SPLIT_SEED)
    sample = rng.choice(np.flatnonzero(train), size=min(10000, int(train.sum())), replace=False)
    expr = logx[sample].toarray().astype(np.float64)
    expr = (expr - expr.mean(axis=0)) / np.where(expr.std(axis=0) > 0, expr.std(axis=0), 1.0)
    act = acts[sample].astype(np.float64)
    act = (act - act.mean(axis=0)) / np.where(act.std(axis=0) > 0, act.std(axis=0), 1.0)
    corr = act.T @ expr / len(sample)
    top_genes = [" ".join(names[np.argsort(-corr[j])[:N_TOP_GENES]]) if live[j] else ""
                 for j in range(N_FEATURES)]  # fmt: skip

    features = pd.DataFrame({
        "feature": np.arange(N_FEATURES), "live": live,
        "frequency_train": (a_train > 0).mean(axis=0),
        "mean_activation_train": a_train.mean(axis=0),
        "type_feature_for": type_label, "batch_score": batch_score, "cell_type_eta2": type_eta,
        "auroc_c2_vs_other_v2_classical_monocyte": auroc, "top_genes": top_genes,
    })  # fmt: skip

    # intervention: remove the strongest batch features, against random live features
    ranked = np.argsort(-np.where(live, batch_score, -1.0))
    removed = ranked[:K_REMOVE]
    sets = {"none": np.array([], dtype=int), "batch features": removed}
    for i in range(N_CONTROLS):
        sets[f"random {i}"] = np.random.default_rng(i).choice(np.flatnonzero(live), K_REMOVE,
                                                              replace=False)  # fmt: skip
    rows = []
    for label, idx in sets.items():
        edited = remove_features(sae, x, idx) * scale + mu if len(idx) else h
        scores = score_embedding(edited, obs, split)
        rows.append({"removal": label, "features": " ".join(map(str, idx)), **scores})
        print(f"{label}: f1 {scores['knn_macro_f1_fine']:.3f} ilisi {scores['scib_ilisi']:.3f}",
              flush=True)  # fmt: skip

    out = ROOT / "results" / "audit" / args.name
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "sae_grid.csv", index=False)
    features.to_csv(out / "sae_features.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "sae_intervention.csv", index=False)
    torch.save({"state_dict": sae.state_dict(), "mu": torch.as_tensor(mu), "scale": scale,
                "l1": l1}, out / "sae.pt")  # fmt: skip
    run = {
        "name": args.name, "split_seed": SPLIT_SEED, "commit": git_commit(ROOT),
        "n_features": N_FEATURES, "epochs": EPOCHS, "grid": GRID, "max_l0": MAX_L0,
        "chosen_l1": l1, "k_removed": K_REMOVE, "n_controls": N_CONTROLS,
        "type_ratio": TYPE_RATIO, "type_active": TYPE_ACTIVE,
        "min_group_cells": MIN_GROUP_CELLS, "removed_features": removed.tolist(),
        "versions": versions("torch", "scikit-learn"),
    }  # fmt: skip
    (out / "sae_run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
