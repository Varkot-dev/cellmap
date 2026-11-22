"""Check the pre-registered audit predictions (K1, K2, G1 to G4 in docs/PLAN.md).

Usage:
    python scripts/audit_evaluate.py --name pbmc

Reads results/audit/<name>/overlap.csv and knockout.csv (scripts/audit_knockout.py) and
genotype_probe.csv (scripts/audit_genotype.py). Writes results/audit/<name>/predictions.json:
for every prediction the exact test, the numbers and a verdict, plus what the predictions
did not cover: which groups of genes appear among each model's top 20 knockouts that are not
in the marker top 20.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOP = 20


def r(x: float) -> float:
    return round(float(x), 4)


def k1(overlap: pd.DataFrame) -> dict:
    o = overlap[overlap["model"] == "contrastive"]
    wins = int((o["overlap_knockout_markers"] > o["overlap_expression_markers"]).sum())
    return {
        "claim": "For the contrastive model, the knockout ranking overlaps the marker top 20 "
        "more than the expression ranking does, in at least two thirds of the types.",
        "knockout_higher_in": f"{wins} of {len(o)}",
        "ties": int((o["overlap_knockout_markers"] == o["overlap_expression_markers"]).sum()),
        "mean_overlap": {
            "knockout": r(o["overlap_knockout_markers"].mean()),
            "expression": r(o["overlap_expression_markers"].mean()),
        },  # fmt: skip
        "verdict": "supported" if wins >= 2 * len(o) / 3 else "not supported",
    }


def k2(overlap: pd.DataFrame) -> dict:
    mean = overlap.groupby("model")["overlap_knockout_markers"].mean()
    return {
        "claim": "Mean knockout overlap with the markers is higher for the contrastive model "
        "than for the masked-gene transformer.",
        "mean_overlap_by_model": {k: r(v) for k, v in mean.items()},
        "verdict": "supported" if mean["contrastive"] > mean["masked_gene"] else "not supported",
    }


def off_marker_groups(knock: pd.DataFrame) -> dict:
    top = knock[(knock["knockout_rank"] <= TOP) & (knock["marker_rank"] > TOP)]
    counts = top.groupby(["model", "group"]).size().unstack(fill_value=0)
    named = top[top["group"] != "other"]
    examples = named.groupby(["model", "group"])["gene"].agg(lambda g: sorted(set(g)))
    return {
        "counts_of_top20_knockouts_not_in_marker_top20": counts.to_dict("index"),
        "genes": {f"{m}/{g}": v for (m, g), v in examples.items()},
    }


def genotype(probe: pd.DataFrame) -> dict:
    def auc(method: str, cell_type: str, knockout: str = "none", col: str = "auroc") -> float:
        row = probe[(probe["method"] == method) & (probe["cell_type"] == cell_type)
                    & (probe["knockout"] == knockout)]  # fmt: skip
        return float(row[col].iloc[0])

    mono, tcell = "classical monocyte", "naive CD4 T cell"
    base = auc("contrastive", mono)
    m = probe[(probe["method"] == "contrastive") & (probe["cell_type"] == mono)]
    drops = (base - m.set_index("knockout")["auroc"]).drop("none")
    control_drops = drops[[k for k in drops.index if k.startswith("control ")]]
    out = {
        "G1": {
            "claim": "On classical monocytes the contrastive embedding gives AUROC >= 0.80.",
            "auroc": r(base),
            "verdict": "supported" if base >= 0.8 else "not supported",
        },
        "G2": {
            "claim": "Knocking out HLA-DRB5 lowers that AUROC by more than the largest drop "
            "among the 10 control genes.",
            "drop_hla_drb5": r(drops["HLA-DRB5"]),
            "largest_control_drop": r(control_drops.max()),
            "control_drops": {k: r(v) for k, v in control_drops.items()},
            "verdict": "supported" if drops["HLA-DRB5"] > control_drops.max() else "not supported",
        },
        "G3": {
            "claim": "On naive CD4 T cells the AUROC is lower than on classical monocytes.",
            "auroc_naive_cd4": r(auc("contrastive", tcell)),
            "auroc_classical_monocyte": r(base),
            "verdict": "supported" if auc("contrastive", tcell) < base else "not supported",
        },
        "G4": {
            "claim": "PCA also gives AUROC >= 0.80 on classical monocytes.",
            "auroc": r(auc("pca", mono)),
            "verdict": "supported" if auc("pca", mono) >= 0.8 else "not supported",
        },
    }
    table = probe.pivot_table(index=["method", "cell_type"], columns="knockout", values="auroc")
    out["auroc_table"] = {f"{a}/{b}": {k: r(v) for k, v in row.items()}
                          for (a, b), row in table.iterrows()}  # fmt: skip
    c2 = probe[probe["knockout"].isin(["none", "HLA-DRB5"])]
    cols = ["method", "cell_type", "knockout", "auroc_c2_only"]
    out["c2_cohort_only"] = {f"{a}/{b}/{c}": r(v) for a, b, c, v in c2[cols].values}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()
    res = ROOT / "results" / "audit" / args.name
    report = {}
    if (res / "overlap.csv").exists():
        overlap = pd.read_csv(res / "overlap.csv")
        report["K1"] = k1(overlap)
        report["K2"] = k2(overlap)
        report["not_predicted"] = off_marker_groups(pd.read_csv(res / "knockout.csv"))
    if (res / "genotype_probe.csv").exists():
        report |= genotype(pd.read_csv(res / "genotype_probe.csv"))
    (res / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, value in report.items():
        if isinstance(value, dict) and "verdict" in value:
            print(f"{key}: {value['verdict']}")


if __name__ == "__main__":
    main()
