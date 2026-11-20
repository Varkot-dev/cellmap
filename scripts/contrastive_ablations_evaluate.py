"""Check the pre-registered ablation predictions (A1 to A4 in docs/PLAN.md) against the results.

Usage:
    python scripts/contrastive_ablations_evaluate.py

Reads results/contrastive_ablations/pbmc/metrics.csv (scripts/contrastive_ablations.py) and the
main run's split seed 0 in results/contrastive/pbmc/metrics_by_split.csv. Writes
results/contrastive_ablations/pbmc/predictions.json: the noise bar, every variant's change from
base, and for each prediction the exact test, the numbers it rests on and a verdict. Variants
that have not finished are listed as missing, and a prediction that needs them is "pending".

The rules, as fixed in the plan before any ablation result:
    noise rule  a difference from base counts only if it exceeds both 0.01 and twice the gap
                between base and the main run's split seed 0 in fine macro F1
    A1          h beats z on fine macro F1 in at least 7 of the 9 variants
    A2          same_donor raises iLISI over base by at least 0.03
    A3          keep_weak has lower fine macro F1 than base
    A4          tau_0.5 has the lowest fine macro F1 of the four temperatures

How the rules are applied (written after seeing only the base run, before any other variant):
A3 is a difference from base, so it counts only if it passes the noise rule. A4 compares
variants with each other; it is supported only if tau_0.5 is lowest and below each other
temperature by more than the noise bar, and "direction only" if it is lowest but within the
bar. A1 compares h and z within one run, so the noise rule (defined for differences between
runs) does not apply; the count is reported as is, with the size of each gap. A2 has its own
threshold and uses it.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "contrastive_ablations" / "pbmc"
MAIN = ROOT / "results" / "contrastive" / "pbmc" / "metrics_by_split.csv"
VARIANTS = ["base", "tau_0.05", "tau_0.2", "tau_0.5", "keep_strong", "keep_weak", "no_jitter",
            "same_donor", "debiased"]  # fmt: skip
TEMPERATURES = {"tau_0.05": 0.05, "base": 0.1, "tau_0.2": 0.2, "tau_0.5": 0.5}
F1, ILISI, ACC = "knn_macro_f1_fine", "scib_ilisi", "knn_accuracy_fine"


def r(x: float) -> float:
    return round(float(x), 4)


def noise_bar(h: pd.DataFrame, main: pd.DataFrame) -> dict:
    main_f1 = main.loc[
        (main["method"] == "transformer_contrastive") & (main["split_seed"] == 0), F1
    ].iloc[0]
    gap = abs(h.loc["base", F1] - main_f1)
    return {
        "main_run_seed0_f1": r(main_f1),
        "base_f1": r(h.loc["base", F1]),
        "rerun_gap": r(gap),
        "bar": r(max(0.01, 2 * gap)),
    }


def changes(h: pd.DataFrame, bar: float) -> dict:
    out = {}
    for v in h.index:
        if v == "base":
            continue
        d_f1 = h.loc[v, F1] - h.loc["base", F1]
        out[v] = {
            "f1": r(h.loc[v, F1]),
            "f1_change": r(d_f1),
            "f1_change_counts": bool(abs(d_f1) > bar),
            "ilisi_change": r(h.loc[v, ILISI] - h.loc["base", ILISI]),
            "bio_change": r(h.loc[v, "scib_bio_mean"] - h.loc["base", "scib_bio_mean"]),
            "accuracy_change": r(h.loc[v, ACC] - h.loc["base", ACC]),
        }
    return out


def a1(h: pd.DataFrame, z: pd.DataFrame) -> dict:
    done = [v for v in VARIANTS if v in h.index and v in z.index]
    gaps = {v: r(h.loc[v, F1] - z.loc[v, F1]) for v in done}
    wins = sum(g > 0 for g in gaps.values())
    remaining = len(VARIANTS) - len(done)
    if wins >= 7:
        verdict = "supported"
    elif wins + remaining < 7:
        verdict = "not supported"
    else:
        verdict = "pending"
    return {
        "claim": "h beats z on fine macro F1 in at least 7 of the 9 variants.",
        "h_minus_z_f1": gaps,
        "h_higher_in": f"{wins} of {len(done)} finished",
        "gaps_above_0.01": sum(g > 0.01 for g in gaps.values()),
        "verdict": verdict,
    }


def a2(h: pd.DataFrame) -> dict:
    claim = "same_donor raises iLISI over base by at least 0.03."
    if "same_donor" not in h.index:
        return {"claim": claim, "verdict": "pending"}
    change = h.loc["same_donor", ILISI] - h.loc["base", ILISI]
    return {
        "claim": claim,
        "ilisi_base": r(h.loc["base", ILISI]),
        "ilisi_same_donor": r(h.loc["same_donor", ILISI]),
        "change": r(change),
        "verdict": "supported" if change >= 0.03 else "not supported",
    }


def a3(h: pd.DataFrame, bar: float) -> dict:
    claim = "keep_weak has lower fine macro F1 than base (by more than the noise bar)."
    if "keep_weak" not in h.index:
        return {"claim": claim, "verdict": "pending"}
    change = h.loc["keep_weak", F1] - h.loc["base", F1]
    if change < -bar:
        verdict = "supported"
    elif change < 0:
        verdict = "direction only (within the noise bar)"
    else:
        verdict = "not supported"
    return {"claim": claim, "change": r(change), "bar": bar, "verdict": verdict}


def a4(h: pd.DataFrame, bar: float) -> dict:
    claim = "tau_0.5 has the lowest fine macro F1 of the four temperatures."
    missing = [v for v in TEMPERATURES if v not in h.index]
    if missing:
        return {"claim": claim, "missing": missing, "verdict": "pending"}
    f1 = {v: r(h.loc[v, F1]) for v in TEMPERATURES}
    others = [f1[v] for v in TEMPERATURES if v != "tau_0.5"]
    lowest = f1["tau_0.5"] < min(others)
    margin = min(others) - f1["tau_0.5"]
    if lowest and margin > bar:
        verdict = "supported"
    elif lowest:
        verdict = "direction only (within the noise bar)"
    else:
        verdict = "not supported"
    return {"claim": claim, "f1_by_temperature": f1, "margin": r(margin), "bar": bar,
            "verdict": verdict}  # fmt: skip


def main() -> None:
    m = pd.read_csv(RES / "metrics.csv")
    h = m[m["embedding"] == "h"].set_index("variant")
    z = m[m["embedding"] == "z"].set_index("variant")
    noise = noise_bar(h, pd.read_csv(MAIN))
    bar = noise["bar"]
    report = {
        "finished": [v for v in VARIANTS if v in z.index],
        "missing": [v for v in VARIANTS if v not in z.index],
        "noise": noise,
        "change_from_base_h": changes(h, bar),
        "alignment_uniformity": {
            v: {
                "alignment": r(h.loc[v, "final_val_alignment"]),
                "uniformity": r(h.loc[v, "final_val_uniformity"]),
                "partner_accuracy": r(h.loc[v, "final_val_partner_accuracy"]),
            }
            for v in h.index
        },  # fmt: skip
        "A1": a1(h, z),
        "A2": a2(h),
        "A3": a3(h, bar),
        "A4": a4(h, bar),
    }
    (RES / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"noise bar {bar} (rerun gap {noise['rerun_gap']}); missing: {report['missing']}")
    for key in ("A1", "A2", "A3", "A4"):
        print(f"{key}: {report[key]['verdict']}")


if __name__ == "__main__":
    main()
