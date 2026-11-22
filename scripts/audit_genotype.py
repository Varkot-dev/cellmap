"""Audit part 2: does the embedding encode whether a donor carries HLA-DRB5?

Usage:
    python scripts/audit_genotype.py --name pbmc

Design fixed in docs/PLAN.md (2026-11-22, model audit) before any result.

Carrier status. HLA-DRB5 is present on some versions of chromosome 6 and absent from others,
so some people have no copy. Here a donor is a carrier if the mean of 10,000 x count / total
count of HLA-DRB5 over its monocytes and B cells is at least 0.5. Nobody was genotyped: this
is the expression pattern, and the label says no more than that.

Probe. For classical monocytes (which express HLA class II) and naive CD4 T cells (which
hardly do), up to 300 cells per donor are embedded, and a logistic regression predicts the
donor's carrier status, one donor held out at a time (`cellmap.audit.donor_probe`). The score
is the AUROC over donors. Each embedding is computed under several knockouts (genes set to
zero in the input, then embedded again, then the probe refitted): none, HLA-DRB5, each of 10
expression-matched control genes, and every HLA gene among the 2,000.

Methods: the three saved transformers of split seed 0 and PCA (50 components fitted on the
training cells of split seed 0; a knocked-out gene is zero in its log-normalised input).

Writes results/audit/<name>/:
    carriers.csv        per donor: HLA-DRB5 level, carrier status, cohort, chemistry
    genotype_probe.csv  per method, cell type and knockout: AUROC (all donors, and C2 alone)
    donor_scores.csv    per method, cell type, knockout and donor: the probe's score
    genotype_run.json
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score

from cellmap import transformer
from cellmap.audit import donor_probe, knockout, matched_controls
from cellmap.baselines import _principal_axes, select_hvg
from cellmap.benchmark import donor_split
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED, PER_DONOR, MIN_CELLS, THRESHOLD, GENE = 0, 300, 20, 0.5, "HLA-DRB5"
TYPES = {
    "classical monocyte": "classical monocyte",
    "naive CD4 T cell": "naive thymus-derived CD4-positive, alpha-beta T cell",
}
MODELS = {
    "contrastive": "results/contrastive/{name}/model_contrastive_seed0.pt",
    "contrastive_mgm": "results/contrastive/{name}/model_contrastive_mgm_seed0.pt",
    "masked_gene": "results/m5/{name}/model_seed0.pt",
}


def carriers(adata: ad.AnnData) -> pd.DataFrame:
    obs = adata.obs
    names = np.asarray(adata.var["feature_name"]).astype(str)
    counts = adata.layers["counts"].tocsr()
    cells = obs["cell_type_coarse"].astype(str).isin(["monocyte", "B cell"]).to_numpy()
    total = np.asarray(counts[cells].sum(axis=1)).ravel()
    level = counts[cells][:, np.flatnonzero(names == GENE)[0]].toarray().ravel()
    frame = pd.DataFrame({"donor": obs["donor_id"].astype(str).to_numpy()[cells],
                          "level": 1e4 * level / np.maximum(total, 1)})  # fmt: skip
    per = frame.groupby("donor")["level"].mean().rename("drb5_per_10k").to_frame()
    meta = obs.assign(donor=obs["donor_id"].astype(str)).groupby("donor", observed=True)
    per["assay"] = meta["assay"].first().astype(str)
    per["cohort"] = ["C2" if d.startswith("C2") else "other" for d in per.index]
    per["carrier"] = (per["drb5_per_10k"] >= THRESHOLD).astype(int)
    return per


def pca_fit(logx: sp.csr_matrix, train: np.ndarray, n_comps: int = 50) -> Callable:
    """`cellmap.baselines.pca_embed`, split into fitting (training cells) and projecting."""
    z = logx[train].toarray().astype(np.float32)
    mean = z.mean(axis=0, dtype=np.float64)
    std = z.std(axis=0, dtype=np.float64, ddof=1)
    std[std == 0] = 1.0

    def scale(x: np.ndarray) -> np.ndarray:
        return np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)

    z = scale(z)
    center = z.mean(axis=0, dtype=np.float64).astype(np.float32)
    axes = _principal_axes(z - center, n_comps).T.astype(np.float32)
    return lambda x: (scale(x.toarray()) - center) @ axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    split = donor_split(obs, SPLIT_SEED)
    train = split == "train"
    genes = select_hvg(adata, train, n_top_genes=2000)
    counts = adata.layers["counts"][:, genes].tocsr()
    logx = adata.X[:, genes].tocsr()
    names = np.asarray(adata.var["feature_name"]).astype(str)[genes]
    donors = obs["donor_id"].astype(str).to_numpy()
    labels = obs["cell_type"].astype(str).to_numpy()

    status = carriers(adata)
    label = status["carrier"].to_dict()
    print(status.to_string(), flush=True)

    target = int(np.flatnonzero(names == GENE)[0])
    hla = np.flatnonzero(np.char.startswith(names, "HLA-"))
    mono = np.flatnonzero(labels == TYPES["classical monocyte"])
    carrier_mono = mono[np.isin(donors[mono], [d for d, c in label.items() if c])]
    detection = np.asarray((counts[carrier_mono] > 0).mean(axis=0)).ravel()
    controls = matched_controls(detection, target, ~np.char.startswith(names, "HLA-"),
                                seed=SPLIT_SEED)  # fmt: skip
    conditions = {"none": [], GENE: [target], "all HLA": list(hla)}
    conditions |= {f"control {names[g]}": [int(g)] for g in controls}

    rng = np.random.default_rng(SPLIT_SEED)
    cells = {}
    for short, full in TYPES.items():
        picked = []
        for d in sorted(label):
            rows = np.flatnonzero((labels == full) & (donors == d))
            if len(rows) >= MIN_CELLS:
                picked.append(np.sort(rng.choice(rows, min(PER_DONOR, len(rows)), replace=False)))
        cells[short] = np.concatenate(picked)
    status["n_classical_monocyte"] = pd.Series(donors[cells["classical monocyte"]]).value_counts()
    status["n_naive_cd4"] = pd.Series(donors[cells["naive CD4 T cell"]]).value_counts()

    methods: dict[str, Callable] = {}
    for model_name, path in MODELS.items():
        saved = torch.load(ROOT / path.format(name=args.name), weights_only=True)
        model = transformer.CellTransformer(saved["n_genes"])
        model.load_state_dict(saved["state_dict"])
        methods[model_name] = lambda rows, ko, m=model: transformer.embed(
            m, knockout(counts[rows], ko) if ko else counts[rows]
        )
    project = pca_fit(logx, train)
    methods["pca"] = lambda rows, ko: project(knockout(logx[rows], ko) if ko else logx[rows])

    c2 = [d for d in label if d.startswith("C2")]
    results, scores = [], []
    for method, embed_fn in methods.items():
        for short, rows in cells.items():
            for condition, ko in conditions.items():
                probe = donor_probe(embed_fn(rows, ko), donors[rows], label)
                keep = np.isin(donors[rows], c2)
                emb_c2 = embed_fn(rows[keep], ko)
                probe_c2 = donor_probe(emb_c2, donors[rows][keep], {d: label[d] for d in c2})
                results.append({"method": method, "cell_type": short, "knockout": condition,
                                "n_genes_knocked": len(ko), "auroc": probe["auroc"],
                                "auroc_c2_only": probe_c2["auroc"]})  # fmt: skip
                scores += [{"method": method, "cell_type": short, "knockout": condition,
                            "donor": d, "carrier": label[d], "score": s}
                           for d, s in probe["scores"].items()]  # fmt: skip
                print(f"{method} | {short} | {condition}: auroc {probe['auroc']:.3f} "
                      f"(C2 only {probe_c2['auroc']:.3f})", flush=True)  # fmt: skip

    out = ROOT / "results" / "audit" / args.name
    out.mkdir(parents=True, exist_ok=True)
    status.to_csv(out / "carriers.csv", index_label="donor")
    pd.DataFrame(results).to_csv(out / "genotype_probe.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "donor_scores.csv", index=False)
    check = roc_auc_score(status["carrier"], status["drb5_per_10k"])
    run = {
        "name": args.name, "split_seed": SPLIT_SEED, "commit": git_commit(ROOT),
        "gene": GENE, "carrier_threshold_per_10k": THRESHOLD, "cells_per_donor": PER_DONOR,
        "min_cells_per_donor": MIN_CELLS, "types": TYPES,
        "controls": [names[g] for g in controls], "hla_genes": [names[g] for g in hla],
        "auroc_of_the_raw_level_itself": check, "models": MODELS,
        "versions": versions("torch", "scikit-learn"),
    }  # fmt: skip
    (out / "genotype_run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
