"""Check the pre-registered conformal hypotheses (H1 to H7 in docs/PLAN.md) against the results.

Usage:
    python scripts/m9_evaluate.py

Reads results/m9/m4_blood/ and results/m9/pbmc/ (written by scripts/m9_conformal.py) and the
shared confusions in results/m7/pbmc/confusion_pairs.csv. Writes results/m9/hypotheses.json:
for every hypothesis, the exact test, the numbers it rests on and a verdict.

Unless a hypothesis says otherwise the primary setting is the one fixed in the plan: APS
scores, not class-conditional, alpha 0.1 (target 90 percent coverage), and both classifiers
(linear probe and kNN). "Lab folds" means the six-lab blood data with one lab held out per fold;
the coverage of a fold is the coverage on that lab.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "m9"
TARGET = 0.9


def primary(c: pd.DataFrame, alpha: float = 0.1, score: str = "aps") -> pd.DataFrame:
    return c[(~c["mondrian"]) & (c["alpha"] == alpha) & (c["score"] == score)]


def h1(c: pd.DataFrame) -> dict:
    rows = primary(c)
    rows = rows[(rows["scheme"] == "dataset") & (rows["calibration"] == "random")]
    mean = rows["coverage"].mean()
    return {
        "claim": "On held-out labs, random-cell calibration under-covers: mean coverage below "
        "0.88 at the 0.90 target.",
        "test": "Six-lab blood, lab folds, calibration=random, APS, alpha 0.1: mean coverage "
        "over 6 labs x 6 methods x 2 classifiers.",
        "mean_coverage": round(mean, 4),
        "by_classifier": rows.groupby("classifier")["coverage"].mean().round(4).to_dict(),
        "by_lab": rows.groupby("fold")["coverage"].mean().round(4).to_dict(),
        "n_rows": len(rows),
        "verdict": "supported" if mean < 0.88 else "not supported",
    }


def h2(c: pd.DataFrame) -> dict:
    rows = primary(c)
    rows = rows[(rows["scheme"] == "donor") & (rows["calibration"] == "donor")]
    mean = rows["coverage"].mean()
    return {
        "claim": "On held-out donors, donor calibration is within 0.02 of the target.",
        "test": "Six-lab blood, donor folds, calibration=donor, APS, alpha 0.1: mean coverage "
        "over 5 folds x 6 methods x 2 classifiers; range shown for information.",
        "mean_coverage": round(mean, 4),
        "min_coverage": round(rows["coverage"].min(), 4),
        "max_coverage": round(rows["coverage"].max(), 4),
        "n_rows": len(rows),
        "verdict": "supported" if abs(mean - TARGET) <= 0.02 else "not supported",
    }


def h3(c: pd.DataFrame) -> dict:
    rows = primary(c)
    rows = rows[(rows["scheme"] == "dataset") & rows["calibration"].isin(["lab", "donor"])]
    worst = rows.groupby(["method", "classifier", "calibration"])["coverage"].min().unstack()
    wins = int((worst["lab"] > worst["donor"]).sum())
    size = rows.groupby("calibration")["mean_size"].mean()
    return {
        "claim": "On held-out labs, lab calibration covers better than donor calibration "
        "(higher worst-lab coverage).",
        "test": "Six-lab blood, lab folds, APS, alpha 0.1. For each method and classifier, the "
        "lowest coverage over the 6 held-out labs, lab calibration against donor calibration. "
        "Supported only if lab is higher in every one of the 12 cases.",
        "lab_higher_in": f"{wins} of {len(worst)}",
        "mean_worst_lab_coverage": {
            "lab": round(worst["lab"].mean(), 4),
            "donor": round(worst["donor"].mean(), 4),
        },
        "mean_set_size": {k: round(v, 3) for k, v in size.items()},
        "table": {f"{m}/{k}": v for (m, k), v in worst.round(4).to_dict("index").items()},
        "verdict": "supported" if wins == len(worst) else f"partly ({wins} of {len(worst)})",
    }


def h4(sizes: pd.DataFrame, confusions: pd.DataFrame) -> dict:
    # the 18 shared confusions: pairs where every method sends at least 5 percent of the true
    # type to the same wrong type (the same rule as scripts/m7_figures.py)
    table = confusions.pivot_table(index=["true", "called"], columns="method",
                                   values="share_of_true", fill_value=0.0)  # fmt: skip
    shared = table[(table >= 0.05).all(axis=1)]
    confused = set(shared.index.get_level_values("true"))
    rows = sizes[(sizes["classifier"] == "probe") & (sizes["score"] == "aps")]
    rows = rows[rows["n_test"] >= 20]
    per_method = {}
    named = ["regulatory T cell", "transitional stage B cell"]
    for method, g in rows.groupby("method"):
        g = g.assign(rank=g["mean_size"].rank(ascending=False, method="min"))
        inside, outside = g[g["cell_type"].isin(confused)], g[~g["cell_type"].isin(confused)]
        per_method[method] = {
            "mean_size_confused_types": round(inside["mean_size"].mean(), 3),
            "mean_size_other_types": round(outside["mean_size"].mean(), 3),
            "rank_of": {t: int(g.loc[g["cell_type"] == t, "rank"].iloc[0]) for t in named},
            "n_types_ranked": len(g),
        }
    larger = sum(v["mean_size_confused_types"] > v["mean_size_other_types"]
                 for v in per_method.values())  # fmt: skip
    return {
        "claim": "On PBMC fine types, the classes with the largest mean set size include the "
        "ones in the 18 shared confusions.",
        "test": "PBMC split seed 0, linear probe, APS, donor calibration, alpha 0.1, types with "
        "at least 20 test cells. Per method, mean set size of the types that are the true type "
        "in a shared confusion against the other types. Supported if larger for every method. "
        "This operational test was written after a first look at one method's per-type table; "
        "the plan did not fix one.",
        "n_shared_confusions": len(shared),
        "confused_types": sorted(confused),
        "confused_types_larger_in": f"{larger} of {len(per_method)}",
        "note": "CD56-bright NK has 5 test cells on split seed 0, so it is excluded.",
        "per_method": per_method,
        "verdict": "supported"
        if larger == len(per_method)
        else f"partly ({larger} of {len(per_method)})",
    }


def h5(novel: pd.DataFrame) -> dict:
    aps = novel[novel["score"] == "aps"].set_index(["dataset", "method", "novel_type"])
    lac = novel[novel["score"] == "lac"].set_index(["dataset", "method", "novel_type"])
    larger = aps["novel_mean_size"] > aps["known_mean_size"]
    emptier = lac["novel_empty_share"] > lac["known_empty_share"]
    both = larger & emptier
    by_type = (
        pd.DataFrame({"aps_larger": larger, "lac_emptier": emptier})
        .groupby(level=["dataset", "novel_type"])
        .mean()
        .round(3)
    )
    return {
        "claim": "A type removed from the classifier's training labels gets larger APS sets "
        "than known cells, and more empty LAC sets.",
        "test": "Donor split seed 0, linear probe, donor calibration, alpha 0.1. One case per "
        "dataset, method and removed type; both conditions must hold.",
        "aps_larger_in": f"{int(larger.sum())} of {len(larger)}",
        "lac_emptier_in": f"{int(emptier.sum())} of {len(emptier)}",
        "both_in": f"{int(both.sum())} of {len(both)}",
        "share_by_type": {f"{d}/{t}": v for (d, t), v in by_type.to_dict("index").items()},
        "verdict": "supported" if both.all() else f"partly ({int(both.sum())} of {len(both)})",
    }


def h6(points: pd.DataFrame) -> dict:
    out = {}
    flips_total = 0
    for (dataset, scheme), g in points.groupby(["dataset", "scheme"]):
        f1 = g.groupby(["method", "classifier"])["macro_f1"].mean().unstack()
        methods = list(f1.index)
        flips = [
            f"{a} vs {b}"
            for i, a in enumerate(methods)
            for b in methods[i + 1 :]
            if np.sign(f1.loc[a, "knn"] - f1.loc[b, "knn"])
            != np.sign(f1.loc[a, "probe"] - f1.loc[b, "probe"])
        ]
        flips_total += len(flips)
        out[f"{dataset}/{scheme}"] = {
            "kendall_tau": round(kendalltau(f1["knn"], f1["probe"]).statistic, 3),
            "pairs_that_flip": flips,
            "n_pairs": len(methods) * (len(methods) - 1) // 2,
            "macro_f1": f1.round(4).to_dict("index"),
        }
    return {
        "claim": "Method rankings by kNN macro F1 and by linear-probe macro F1 differ for at "
        "least one pair of methods.",
        "test": "Macro F1 averaged over folds per method and classifier; count pairs of methods "
        "whose order differs between the two classifiers.",
        "by_setting": out,
        "verdict": "supported" if flips_total > 0 else "not supported",
    }


def h7(c: pd.DataFrame) -> dict:
    result = {}
    for score in ("aps", "lac"):
        rows = primary(c, alpha=0.2, score=score)
        rows = rows[(rows["scheme"] == "dataset") & (rows["calibration"] == "lab_hier")]
        labs_ok = rows.assign(ok=rows["coverage"] >= 0.8).groupby(["method", "classifier"])["ok"]
        result[score] = {
            "labs_at_or_above_0.80": {f"{m}/{k}": int(v) for (m, k), v in labs_ok.sum().items()},
            "min_coverage": round(rows["coverage"].min(), 4),
            "mean_set_size": round(rows["mean_size"].mean(), 3),
            "all_pass": bool((labs_ok.sum() >= 5).all()),
        }
    # the price: random calibration at the same alpha, same rows
    ref = primary(c, alpha=0.2)
    ref = ref[(ref["scheme"] == "dataset") & (ref["calibration"] == "random")]
    return {
        "claim": "On held-out labs at alpha 0.2, lab_hier reaches at least 0.80 coverage in at "
        "least 5 of 6 labs for every method.",
        "test": "Six-lab blood, lab folds, calibration=lab_hier, alpha 0.2; checked for APS "
        "(primary) and LAC, both classifiers.",
        "by_score": result,
        "random_calibration_mean_set_size_aps": round(ref["mean_size"].mean(), 3),
        "verdict": "supported" if result["aps"]["all_pass"] else "not supported",
    }


def main() -> None:
    c = pd.read_csv(RES / "m4_blood" / "conformal.csv")
    novel = pd.concat([pd.read_csv(RES / n / "novel.csv") for n in ("m4_blood", "pbmc")])
    points = pd.concat([pd.read_csv(RES / n / "point_metrics.csv") for n in ("m4_blood", "pbmc")])
    sizes = pd.read_csv(RES / "pbmc" / "size_by_class.csv")
    confusions = pd.read_csv(ROOT / "results" / "m7" / "pbmc" / "confusion_pairs.csv")
    report = {
        "H1": h1(c),
        "H2": h2(c),
        "H3": h3(c),
        "H4": h4(sizes, confusions),
        "H5": h5(novel),
        "H6": h6(points),
        "H7": h7(c),
    }
    (RES / "hypotheses.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, value in report.items():
        print(f"{key}: {value['verdict']}")


if __name__ == "__main__":
    main()
