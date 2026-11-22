"""Check the pre-registered contrastive hypotheses (C1 and C2 in docs/PLAN.md, 2026-11-19).

Usage:
    python scripts/contrastive_evaluate.py --name pbmc

C1: `transformer_contrastive` beats the masked-gene transformer (results/m5/) on fine macro F1
in at least 4 of 5 splits.
C2: neither contrastive variant passes the comparison rule against both baselines on any
declared metric. The rule is the project's, applied by the same code as
scripts/verify_rule.py: better on a metric means a paired difference against HVG + PCA and
against PCA + Harmony with the same sign in at least 4 of 5 donor splits.

Writes results/contrastive/<name>/hypotheses.json.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_rule import BASELINES, PBMC_METRICS, compare  # noqa: E402

from cellmap.provenance import git_commit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("transformer_contrastive", "transformer_contrastive_mgm")
F1 = "knn_macro_f1_fine"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    res = ROOT / "results"
    ours = pd.read_csv(res / "contrastive" / args.name / "metrics_by_split.csv")
    masked = pd.read_csv(res / "m5" / args.name / "metrics_by_split.csv")
    base = pd.concat(pd.read_csv(res / m / args.name / "metrics_by_split.csv")
                     for m in ("m1", "m2", "m3"))  # fmt: skip
    base = base[base["method"].isin(BASELINES)]
    if sorted(ours["split_seed"].unique()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("C1 and C2 need all five split seeds")

    f1 = pd.concat([ours, masked, base]).pivot_table(index="split_seed", columns="method",
                                                     values=F1)  # fmt: skip
    gap = f1["transformer_contrastive"] - f1["transformer"]
    wins = int((gap > 0).sum())

    data = pd.concat([ours, base]).rename(columns={"split_seed": "fold"})
    data["fold"] = data["fold"].astype(str)
    data["dataset"], data["scheme"] = args.name, "donor"
    rule = pd.DataFrame(compare(data, PBMC_METRICS))
    rule = rule[rule["method"].isin(VARIANTS)]
    passes = rule[rule["better"]]

    report = {
        "C1": {
            "claim": "transformer_contrastive beats the masked-gene transformer on fine macro F1 "
            "in at least 4 of 5 splits.",
            "by_split": {
                int(s): {
                    "contrastive": round(f1.loc[s, "transformer_contrastive"], 4),
                    "masked_gene": round(f1.loc[s, "transformer"], 4),
                    "gap": round(gap[s], 4),
                }
                for s in f1.index
            },
            "mean_gap": round(gap.mean(), 4),
            "wins": f"{wins} of 5",
            "verdict": "supported" if wins >= 4 else "not supported",
        },
        "C2": {
            "claim": "Neither contrastive variant passes the comparison rule against both "
            "baselines on any declared metric.",
            "comparisons": len(rule),
            "passes": passes[["method", "metric"]].to_dict("records"),
            "worse_than_both": rule[rule["worse"]][["method", "metric"]].to_dict("records"),
            "table": rule[
                [
                    "method",
                    "metric",
                    "mean_minus_pca",
                    "mean_minus_harmony",
                    "folds_above_pca",
                    "folds_above_harmony",
                ]
            ]
            .round(4)
            .to_dict("records"),
            "verdict": "supported" if passes.empty else "not supported",
        },
        "mean_fine_macro_f1": {k: round(v, 4) for k, v in f1.mean().items()},
        "commit": git_commit(ROOT),
    }
    out = res / "contrastive" / args.name / "hypotheses.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"C1: {report['C1']['verdict']} ({report['C1']['wins']}); "
          f"C2: {report['C2']['verdict']} ({len(passes)} passes of {len(rule)})")  # fmt: skip


if __name__ == "__main__":
    main()
