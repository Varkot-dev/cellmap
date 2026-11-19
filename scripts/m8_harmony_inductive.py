"""How much of Harmony's lead comes from seeing the test cells?

Usage:
    python scripts/m8_harmony_inductive.py

Harmony as used in the main benchmarks is transductive: it corrects all cells together, so held-out
donors take part in the correction. Here Harmony is fitted on training donors only and every
other cell is mapped onto that frozen reference (cellmap.reference_mapping, the Symphony
algorithm). Same genes, same PCA, same folds and same scoring as before, so each fold can be
compared directly with the saved PCA and transductive-Harmony rows.

Folds: PBMC, 5 donor splits (labels: 19 fine types). Six labs, 5 donor splits and 6 held-out
labs (labels: 12 lineages).

Writes:
    results/m8/<dataset>/metrics_by_split.csv   method "pca_harmony_inductive", one row per fold
    results/m8/<dataset>/paired.csv             inductive minus transductive, and minus PCA
    results/m8/<dataset>/share_by_fold.csv      per fold: transductive lead over PCA, and the
                                                part of it the inductive version loses, for the
                                                scIB aggregate, bio and batch scores
    results/m8/pbmc/ridge_sensitivity.csv       PBMC only: the same with the query ridge at
                                                0.1, 1 and 10 (the reference is built with
                                                harmonypy's estimated lambda, so no single
                                                query ridge matches it exactly)

Read the result as the gap between two procedures (joint Harmony on all cells against a
Symphony-style mapping onto a reference built from training cells), not as a pure measure of
what access to test cells is worth: the correction algorithm changes as well as the inputs.
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


RIDGES = (0.1, 1.0, 10.0)
SCORES = ("scib_total", "scib_bio_mean", "scib_batch_mean")


def run_dataset(name: str, ridges: tuple[float, ...] = (1.0,)) -> pd.DataFrame:
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
        for ridge in ridges:
            emb = harmony_reference_map(pcs, donors, train, seed=seed, ridge=ridge)
            seconds = time.perf_counter() - t0
            scores = score_embedding(emb, obs, split, **kwargs)
            rows.append({"method": "pca_harmony_inductive", "ridge": ridge, "scheme": scheme,
                         "fold": fold, "split_seed": seed, "seconds": seconds, **scores,
                         "commit": git_commit(ROOT)})  # fmt: skip
            shown = {k: round(v, 3) for k, v in scores.items()
                     if k.startswith(("knn_macro", "scib_t"))}  # fmt: skip
            print(name, scheme, fold, ridge, shown, flush=True)
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


def share_by_fold(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Per fold and ridge: transductive lead over PCA and the part the inductive map loses."""
    base = old.set_index(["method", "scheme", "fold"])
    out = []
    for _, row in new.iterrows():
        key = (row["scheme"], row["fold"])
        entry = {"ridge": row["ridge"], "scheme": row["scheme"], "fold": row["fold"]}
        for s in SCORES:
            lead = base.loc[("pca_harmony", *key), s] - base.loc[("pca", *key), s]
            lost = base.loc[("pca_harmony", *key), s] - row[s]
            entry |= {f"{s}_lead": lead, f"{s}_lost": lost, f"{s}_share_lost": lost / lead}
        out.append(entry)
    return pd.DataFrame(out)


def main() -> None:
    for name, suffix in (("pbmc", "fine"), ("m4_blood", "lineage")):
        new_all = run_dataset(name, RIDGES if name == "pbmc" else (1.0,))
        out = ROOT / "results" / "m8" / name
        out.mkdir(parents=True, exist_ok=True)
        shares = share_by_fold(new_all, previous(name))
        if name == "pbmc":
            shares.to_csv(out / "ridge_sensitivity.csv", index=False)
        shares[shares["ridge"] == 1.0].to_csv(out / "share_by_fold.csv", index=False)
        new = new_all[new_all["ridge"] == 1.0]
        new.to_csv(out / "metrics_by_split.csv", index=False)
        table = paired(new.drop(columns="ridge"), previous(name), suffix)
        table.to_csv(out / "paired.csv", index=False)
        print(table.round(3).to_string(index=False), flush=True)
    print(versions("harmonypy", "scanpy", "scib-metrics"))


if __name__ == "__main__":
    main()
