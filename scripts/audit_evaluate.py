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
    rho = {}
    for model, g in knock.groupby("model"):
        per_type = g.groupby("cell_type")[["mean_effect", "mean_expression", "marker_score"]]
        corr = per_type.corr(method="spearman").xs("mean_effect", level=1)
        rho[model] = {"with_expression": r(corr["mean_expression"].mean()),
                      "with_marker_score": r(corr["marker_score"].mean())}  # fmt: skip
    sensitivity = knock.groupby("model")[["mean_effect", "flip_rate"]].mean()
    return {
        "mean_spearman_of_effect_over_types": rho,
        "mean_effect_and_flip_rate": {
            m: {k: r(v) for k, v in row.items()} for m, row in sensitivity.iterrows()
        },  # fmt: skip
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


def sae(res: Path) -> dict:
    grid = pd.read_csv(res / "sae_grid.csv")
    feats = pd.read_csv(res / "sae_features.csv")
    inter = pd.read_csv(res / "sae_intervention.csv").set_index("removal")
    chosen = grid[grid["chosen"]].iloc[0]
    live = feats[feats["live"]]
    typed = sorted(set(live["type_feature_for"].dropna()) - {""})
    auroc = live["auroc_c2_vs_other_v2_classical_monocyte"]
    strength = (auroc - 0.5).abs() + 0.5
    best = live.loc[strength.idxmax()]
    base, edit = inter.loc["none"], inter.loc["batch features"]
    controls = inter[inter.index.str.startswith("random ")]
    d_ilisi = edit["scib_ilisi"] - base["scib_ilisi"]
    d_f1 = edit["knn_macro_f1_fine"] - base["knn_macro_f1_fine"]
    control_ilisi = (controls["scib_ilisi"] - base["scib_ilisi"]).mean()
    s4 = d_ilisi >= 0.03 and d_ilisi > control_ilisi and -d_f1 < 0.069
    top_batch = live.sort_values("batch_score", ascending=False).head(10)
    return {
        "S1": {
            "claim": "The chosen SAE explains at least 0.90 of the variance on validation cells.",
            "l1": float(chosen["l1"]),
            "explained_variance": r(chosen["val_explained_variance"]),
            "mean_l0": r(chosen["val_mean_l0"]),
            "verdict": "supported" if chosen["val_explained_variance"] >= 0.9 else "not supported",
        },
        "S2": {
            "claim": "At least 6 of the 8 coarse types have a type feature.",
            "types_with_a_feature": typed,
            "n_type_features": int((live["type_feature_for"].fillna("") != "").sum()),
            "verdict": "supported" if len(typed) >= 6 else "not supported",
        },
        "S3": {
            "claim": "Some feature separates C2 from other 10x v2 donors within training "
            "classical monocytes with AUROC at least 0.80 (either direction).",
            "best_feature": int(best["feature"]),
            "best_auroc": r(best["auroc_c2_vs_other_v2_classical_monocyte"]),
            "best_top_genes": best["top_genes"],
            "verdict": "supported" if strength.max() >= 0.8 else "not supported",
        },
        "S4": {
            "claim": "Removing the 10 batch features raises test iLISI by at least 0.03, by more "
            "than the mean random removal, and lowers fine macro F1 by less than 0.069.",
            "ilisi_change": r(d_ilisi),
            "mean_random_ilisi_change": r(control_ilisi),
            "f1_change": r(d_f1),
            "mean_random_f1_change": r(
                (controls["knn_macro_f1_fine"] - base["knn_macro_f1_fine"]).mean()
            ),  # fmt: skip
            "verdict": "supported" if s4 else "not supported",
        },
        "sae_summary": {
            "live_features": int(len(live)),
            "dead_features": int((~feats["live"]).sum()),
            "top_batch_features": top_batch[
                ["feature", "batch_score", "cell_type_eta2", "frequency_train", "top_genes"]
            ]
            .round(4)
            .to_dict("records"),  # fmt: skip
        },
    }


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
    if (res / "sae_intervention.csv").exists():
        report |= sae(res)
    (res / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, value in report.items():
        if isinstance(value, dict) and "verdict" in value:
            print(f"{key}: {value['verdict']}")


if __name__ == "__main__":
    main()
