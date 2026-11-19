"""Apply the pre-declared comparison rule to every metric it names, for every method.

The rule (docs/PLAN.md, 2026-11-17, fixed before any deep model was scored): a method is better
than the baselines on a metric if its paired difference against HVG + PCA and against
PCA + Harmony has the same sign in at least 4 of 5 donor splits. Metrics that count: scIB
biological conservation and batch correction, and kNN accuracy and macro F1 at the fine and
coarse level.

An independent review found that the earlier headline ("no method beat both baselines") had been
checked on two measures only (fine macro F1 and scIB aggregate). This script checks every
metric the rule names, in both directions, and writes every comparison so the count of passes
can be read next to the count of comparisons made.

For the six-lab data the labels are 12 lineages, so the fine/coarse kNN metrics become the
lineage ones. The donor scheme has 5 splits and uses the same 4-of-5 threshold. The held-out-lab
scheme has 6 folds; no threshold was declared for it in advance, so it is reported with the
same 80 percent bar (5 of 6) and marked as not pre-declared.

Writes results/verify/rule_table.csv (one row per dataset, scheme, method, metric) and
results/verify/rule_summary.json.
"""

import json
from math import comb
from pathlib import Path

import pandas as pd

from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verify"
BASELINES = ("pca", "pca_harmony")
PBMC_METRICS = (
    "scib_bio_mean",
    "scib_batch_mean",
    "knn_accuracy_fine",
    "knn_macro_f1_fine",
    "knn_accuracy_coarse",
    "knn_macro_f1_coarse",
)
M4_METRICS = ("scib_bio_mean", "scib_batch_mean", "knn_accuracy_lineage", "knn_macro_f1_lineage")


def pbmc_table() -> pd.DataFrame:
    files = ["m1", "m2", "m3", "m5", "m6"]
    data = pd.concat(
        pd.read_csv(ROOT / "results" / m / "pbmc" / "metrics_by_split.csv") for m in files
    )
    data = data.rename(columns={"split_seed": "fold"})
    data["fold"] = data["fold"].astype(str)
    data["dataset"], data["scheme"] = "pbmc", "donor"
    return data


def m4_table() -> pd.DataFrame:
    files = sorted((ROOT / "results" / "m4" / "m4_blood").glob("metrics_*.csv"))
    data = pd.concat(pd.read_csv(f) for f in files)
    data["dataset"] = "m4_blood"
    return data


def compare(data: pd.DataFrame, metrics: tuple[str, ...]) -> list[dict]:
    rows = []
    for (dataset, scheme), block in data.groupby(["dataset", "scheme"]):
        wide = {m: block.pivot_table(index="fold", columns="method", values=m) for m in metrics}
        n = block["fold"].nunique()
        need = 4 if n == 5 else 5
        for metric, table in wide.items():
            for method in table.columns:
                if method in BASELINES:
                    continue
                diffs = {b: table[method] - table[b] for b in BASELINES}
                above = {b: int((d > 0).sum()) for b, d in diffs.items()}
                below = {b: int((d < 0).sum()) for b, d in diffs.items()}
                rows.append(
                    {
                        "dataset": dataset,
                        "scheme": scheme,
                        "method": method,
                        "metric": metric,
                        "n_folds": n,
                        "threshold": need,
                        "pre_declared": n == 5,
                        "mean_minus_pca": float(diffs["pca"].mean()),
                        "mean_minus_harmony": float(diffs["pca_harmony"].mean()),
                        "folds_above_pca": above["pca"],
                        "folds_above_harmony": above["pca_harmony"],
                        "folds_below_pca": below["pca"],
                        "folds_below_harmony": below["pca_harmony"],
                        "better": all(v >= need for v in above.values()),
                        "worse": all(v >= need for v in below.values()),
                    }
                )
    return rows


def main() -> None:
    rows = compare(pbmc_table(), PBMC_METRICS) + compare(m4_table(), M4_METRICS)
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "rule_table.csv", index=False)

    declared = table[table["pre_declared"]]
    passes = declared[declared["better"]]
    # Chance that one method beats ONE baseline in at least 4 of 5 splits if each split were a
    # fair coin: (C(5,4) + C(5,5)) / 2^5. Beating both is less likely, but the two baselines
    # are correlated, so this single-baseline figure is quoted as an upper reference only.
    p_one = (comb(5, 4) + comb(5, 5)) / 2**5
    summary = {
        "comparisons_pre_declared": int(len(declared)),
        "passes_pre_declared": passes[["dataset", "scheme", "method", "metric"]].to_dict("records"),
        "worse_pre_declared": int(declared["worse"].sum()),
        "comparisons_lab_scheme_not_pre_declared": int((~table["pre_declared"]).sum()),
        "passes_lab_scheme": table[(~table["pre_declared"]) & table["better"]][
            ["method", "metric"]
        ].to_dict("records"),
        "chance_one_baseline_4_of_5": p_one,
        "git_commit": git_commit(ROOT),
    }
    (OUT / "rule_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
