"""M8: how much of Harmony's lead comes from seeing the test cells?

Usage:
    python scripts/m8_harmony_inductive.py

Harmony as used in M1 and M4 is transductive: it corrects all cells together, so held-out
donors take part in the correction. Here Harmony is fitted on training donors only and every
other cell is mapped onto that frozen reference (cellmap.reference_mapping, the Symphony
algorithm). Same genes, same PCA, same folds and same scoring as before, so each fold can be
compared directly with the saved PCA and transductive-Harmony rows.

Folds: PBMC, 5 donor splits (labels: 19 fine types). Six labs, 5 donor splits and 6 held-out
labs (labels: 12 lineages).

Writes:
    results/m8/<dataset>/metrics_by_split.csv   method "pca_harmony_inductive", one row per fold
    results/m8/<dataset>/paired.csv             inductive minus transductive, and minus PCA
"""

import time
from pathlib import Path

import anndata as ad
import pandas as pd

from cellmap.baselines import pca_embed, select_hvg
from cellmap.benchmark import dataset_out_split, donor_split, score_embedding
from cellmap.provenance import git_commit, versions
from cellmap.reference_mapping import harmony_reference_map

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("knn_accuracy", "knn_macro_f1", "scib_bio_mean", "scib_batch_mean", "scib_total")


def run_dataset(name: str) -> pd.DataFrame:
    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{name}.h5ad")
    obs = adata.obs
    donors = obs["donor_id"].astype(str).to_numpy()
    if name == "pbmc":
        folds = [("donor", f"seed{s}", s, donor_split(obs, s)) for s in range(5)]
        kwargs = {}
    else:
        folds = [("donor", f"seed{s}", s, donor_split(obs, s)) for s in range(5)]
        folds += [("dataset", ds, 0, dataset_out_split(obs, ds))
                  for ds in sorted(obs["dataset"].astype(str).unique())]  # fmt: skip
        kwargs = {"levels": {"lineage": "lineage"}, "scib_label": "lineage"}
    rows = []
    for scheme, fold, seed, split in folds:
        train = split == "train"
        t0 = time.perf_counter()
        genes = select_hvg(adata, train, n_top_genes=2000)
        pcs = pca_embed(adata, genes, train, n_comps=50)
        emb = harmony_reference_map(pcs, donors, train, seed=seed)
        seconds = time.perf_counter() - t0
        scores = score_embedding(emb, obs, split, **kwargs)
        rows.append({"method": "pca_harmony_inductive", "scheme": scheme, "fold": fold,
                     "split_seed": seed, "seconds": seconds, **scores,
                     "commit": git_commit(ROOT)})  # fmt: skip
        shown = {k: round(v, 3) for k, v in scores.items() if k.startswith(("knn_macro", "scib_t"))}
        print(name, scheme, fold, shown, flush=True)
    return pd.DataFrame(rows)


def previous(name: str) -> pd.DataFrame:
    """The saved PCA and transductive-Harmony rows for the same folds."""
    if name == "pbmc":
        old = pd.read_csv(ROOT / "results" / "m1" / name / "metrics_by_split.csv")
        old["scheme"], old["fold"] = "donor", "seed" + old["split_seed"].astype(str)
        return old
    res = ROOT / "results" / "m4" / name
    return pd.concat(pd.read_csv(res / f"metrics_{m}.csv") for m in ("pca", "pca_harmony"))


def paired(new: pd.DataFrame, old: pd.DataFrame, suffix: str) -> pd.DataFrame:
    both = pd.concat([new, old])
    out = []
    for scheme, group in both.groupby("scheme"):
        wide = group.pivot_table(index="fold", columns="method",
                                 values=[f"{m}_{suffix}" if m.startswith("knn") else m
                                         for m in METRICS])  # fmt: skip
        for metric in wide.columns.levels[0]:
            for base in ("pca_harmony", "pca"):
                diff = wide[metric]["pca_harmony_inductive"] - wide[metric][base]
                out.append({"scheme": scheme, "metric": metric, "versus": base,
                            "mean_diff": diff.mean(), "min_diff": diff.min(),
                            "max_diff": diff.max(), "folds_higher": int((diff > 0).sum()),
                            "n_folds": int(diff.notna().sum())})  # fmt: skip
    return pd.DataFrame(out)


def main() -> None:
    for name, suffix in (("pbmc", "fine"), ("m4_blood", "lineage")):
        new = run_dataset(name)
        out = ROOT / "results" / "m8" / name
        out.mkdir(parents=True, exist_ok=True)
        new.to_csv(out / "metrics_by_split.csv", index=False)
        table = paired(new, previous(name), suffix)
        table.to_csv(out / "paired.csv", index=False)
        print(table.round(3).to_string(index=False), flush=True)
    print(versions("harmonypy", "scanpy", "scib-metrics"))


if __name__ == "__main__":
    main()
