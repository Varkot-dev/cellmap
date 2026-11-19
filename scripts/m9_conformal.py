"""M9: can a classifier on these embeddings say how sure it is, on new people and new labs?

Usage:
    python scripts/m9_conformal.py --name m4_blood     # six labs, 12 lineages, 11 folds
    python scripts/m9_conformal.py --name pbmc         # 21 donors, 19 fine types, split seed 0

Design and hypotheses are fixed in docs/PLAN.md (2026-11-19, M9) before any result. In short:
a linear probe (logistic regression) and kNN are fitted on the training cells of each saved
embedding; conformal prediction sets (cellmap.conformal) are calibrated three ways and scored
on the test cells:

- random: 20 percent of training-donor cells held out at random; the classifier is refitted on
  the other 80 percent. Calibration cells share donors with the classifier's training cells,
  which is the common shortcut and breaks exchangeability with new people.
- donor: the validation donors, used by neither the embedding nor the classifier.
- lab (lab folds only): each training lab in turn is left out, a classifier is fitted on the
  other training labs, and the left-out lab's cells are scored; the scores of all training labs
  are pooled. Test sets still use the classifier fitted on all training cells. This is a
  heuristic (cross-conformal at the lab level), not a procedure with a guarantee.
- donor_hier and lab_hier: the same calibration cells, but with the hierarchical quantile
  (cellmap.conformal.hierarchical_quantile), which treats donors or labs as the exchangeable
  units and carries a coverage guarantee for a new donor or lab when there are enough of them
  (at least 9 groups at alpha 0.1, 4 at alpha 0.2; with fewer, the set is every label).

Each calibration is crossed with score (LAC, APS), marginal or class-conditional (Mondrian),
and alpha (0.05, 0.1, 0.2). For PBMC the script also records set size per fine type and how
often two types share a set, to compare with the shared confusions in M7. A novel-type test
removes one type from the classifier's labels (not from the embedding) and records set sizes.

Writes results/m9/<name>/: conformal.csv, point_metrics.csv, novel.csv, run.json, and for PBMC
size_by_class.csv and pair_cooccurrence.csv.
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from cellmap.benchmark import dataset_out_split, donor_split
from cellmap.conformal import (
    SCORES,
    calibrate,
    fit_linear_probe,
    knn_proba,
    prediction_sets,
    probe_proba,
    set_metrics,
)
from cellmap.provenance import git_commit, versions

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.05, 0.1, 0.2)
M4_TRAINED = ("pca", "pca_harmony", "vae")
M4_PRETRAINED = ("census_scvi", "geneformer_ft", "tf_sapiens")
PBMC_METHODS = ("pca", "pca_harmony", "vae", "scvi", "linear_ae", "pca_10",
                "census_scvi", "geneformer_ft", "tf_sapiens", "transformer")  # fmt: skip
NOVEL_TYPES = {
    "m4_blood": ("B cell", "dendritic cell", "plasma cell"),
    "pbmc": (
        "plasmablast",
        "regulatory T cell",
        "CD16-negative, CD56-bright natural killer cell, human",
    ),
}


class Classifier:
    """Fit on (emb, labels); return probabilities over a fixed class list."""

    def __init__(self, kind: str, classes: np.ndarray, seed: int):
        self.kind, self.classes, self.seed = kind, classes, seed

    def fit(self, emb: np.ndarray, labels: np.ndarray) -> "Classifier":
        if self.kind == "probe":
            self.model = fit_linear_probe(emb, labels, seed=self.seed)
        else:
            self.train_emb, self.train_labels = emb, labels
        return self

    def proba(self, emb: np.ndarray) -> np.ndarray:
        if self.kind == "probe":
            return probe_proba(self.model, emb, self.classes)
        return knn_proba(self.train_emb, self.train_labels, emb, self.classes)


def label_index(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index.get(c, -1) for c in labels])


def calibration_scores(kind, emb, labels, split, groups, donors, classes, seed, rng):
    """{scheme: (classifier for test cells, calibration probabilities, calibration labels,
    calibration group ids or None)}. The classifier used for test cells is fitted on all
    training cells, except under 'random'. Group ids (donor or lab) feed the hierarchical
    variant, which treats groups rather than cells as the exchangeable units."""
    train = split == "train"
    out = {}
    # random cells from training donors
    tr_idx = np.flatnonzero(train)
    cal_mask = np.zeros(len(split), bool)
    cal_mask[rng.choice(tr_idx, size=len(tr_idx) // 5, replace=False)] = True
    fit_mask = train & ~cal_mask
    clf_random = Classifier(kind, classes, seed).fit(emb[fit_mask], labels[fit_mask])
    out["random"] = (clf_random, clf_random.proba(emb[cal_mask]), labels[cal_mask], None)
    # held-out donors (validation split)
    clf_all = Classifier(kind, classes, seed).fit(emb[train], labels[train])
    val = split == "val"
    out["donor"] = (clf_all, clf_all.proba(emb[val]), labels[val], donors[val])
    # held-out labs: only when the test set is a whole lab
    if groups is not None:
        probs, labs, ids = [], [], []
        for lab in np.unique(groups[split != "test"]):
            left = (groups == lab) & (split != "test")
            fit = train & (groups != lab)
            clf = Classifier(kind, classes, seed).fit(emb[fit], labels[fit])
            probs.append(clf.proba(emb[left]))
            labs.append(labels[left])
            ids.append(groups[left])
        out["lab"] = (clf_all, np.vstack(probs), np.concatenate(labs), np.concatenate(ids))
    return out


def evaluate_fold(emb, labels, split, groups, donors, classes, seed, fold_info, extra):
    rows, points = [], []
    test = split == "test"
    y_test = label_index(labels[test], classes)
    for kind in ("probe", "knn"):
        rng = np.random.default_rng(seed)
        cal = calibration_scores(kind, emb, labels, split, groups, donors, classes, seed, rng)
        test_probs_all = cal["donor"][0].proba(emb[test])
        pred = classes[test_probs_all.argmax(1)]
        points.append(
            {
                **fold_info,
                "classifier": kind,
                "accuracy": accuracy_score(labels[test], pred),
                "macro_f1": f1_score(
                    labels[test],
                    pred,
                    average="macro",
                    labels=np.unique(labels[test]),
                    zero_division=0,
                ),
            }
        )
        for scheme, (clf, cal_probs, cal_labels, cal_groups) in cal.items():
            test_probs = clf.proba(emb[test]) if scheme == "random" else test_probs_all
            cal_idx = label_index(cal_labels, classes)
            for score_name, score_fn in SCORES.items():
                s_cal = score_fn(cal_probs, np.random.default_rng(seed + 1))
                s_test = score_fn(test_probs, np.random.default_rng(seed + 2))
                variants = [(scheme, False, None), (scheme, True, None)]
                if cal_groups is not None:
                    variants.append((f"{scheme}_hier", False, cal_groups))
                for name, mondrian, hier in variants:
                    for alpha in ALPHAS:
                        q = calibrate(s_cal, cal_idx, alpha, mondrian=mondrian, groups=hier)
                        sets = prediction_sets(s_test, q)
                        m = set_metrics(sets, y_test)
                        rows.append(
                            {
                                **fold_info,
                                "classifier": kind,
                                "calibration": name,
                                "score": score_name,
                                "mondrian": mondrian,
                                "alpha": alpha,
                                "n_cal": len(cal_idx),
                                "n_cal_groups": (
                                    len(np.unique(cal_groups)) if cal_groups is not None else 0
                                ),
                                **{k: v for k, v in m.items() if k != "per_class_coverage"},
                            }
                        )
                        if extra is not None and name == "donor" and alpha == 0.1:
                            extra(kind, score_name, mondrian, sets, y_test)
    return rows, points


def novel_type_rows(emb, labels, split, classes, seed, fold_info):
    """Remove one type from the classifier's labels; compare its set sizes with known cells."""
    rows = []
    train, val, test = split == "train", split == "val", split == "test"
    for novel in NOVEL_TYPES[fold_info["dataset"]]:
        if not (labels[test] == novel).any():
            continue
        known_classes = classes[classes != novel]
        keep_tr, keep_val = train & (labels != novel), val & (labels != novel)
        clf = Classifier("probe", known_classes, seed).fit(emb[keep_tr], labels[keep_tr])
        cal_idx = label_index(labels[keep_val], known_classes)
        test_probs = clf.proba(emb[test])
        y_test = label_index(labels[test], known_classes)  # -1 for the novel type
        is_novel = labels[test] == novel
        for score_name, score_fn in SCORES.items():
            s_cal = score_fn(clf.proba(emb[keep_val]), np.random.default_rng(seed + 1))
            s_test = score_fn(test_probs, np.random.default_rng(seed + 2))
            sets = prediction_sets(s_test, calibrate(s_cal, cal_idx, 0.1))
            size = sets.sum(1)
            rows.append({**fold_info, "novel_type": novel, "score": score_name,
                         "n_novel": int(is_novel.sum()),
                         "novel_mean_size": float(size[is_novel].mean()),
                         "known_mean_size": float(size[~is_novel].mean()),
                         "novel_empty_share": float((size[is_novel] == 0).mean()),
                         "known_empty_share": float((size[~is_novel] == 0).mean()),
                         "novel_singleton_share": float((size[is_novel] == 1).mean()),
                         "known_coverage": float(sets[np.flatnonzero(~is_novel),
                                                      y_test[~is_novel]].mean())})  # fmt: skip
    return rows


def m4_folds(obs):
    folds = [("donor", f"seed{s}", s, donor_split(obs, s)) for s in range(5)]
    folds += [("dataset", d, 0, dataset_out_split(obs, d))
              for d in sorted(obs["dataset"].astype(str).unique())]  # fmt: skip
    return folds


def load_saved(out: Path) -> tuple[list, list, list]:
    """Rows already written by an earlier, interrupted run, with their own commit stamps."""
    frames = []
    for fname in ("conformal.csv", "point_metrics.csv", "novel.csv"):
        path = out / fname
        frames.append(pd.read_csv(path).to_dict("records") if path.exists() else [])
    return frames[0], frames[1], frames[2]


def run_m4(name: str, out: Path, resume: bool = False) -> None:
    obs = ad.read_h5ad(ROOT / "data" / "processed" / f"{name}.h5ad", backed="r").obs
    labels = obs["lineage"].astype(str).to_numpy()
    groups_all = obs["dataset"].astype(str).to_numpy()
    donors = obs["donor_id"].astype(str).to_numpy()
    classes = np.unique(labels)
    emb_dir = ROOT / "data" / "embeddings" / name
    pretrained = {m: np.load(emb_dir / f"{m}.npy") for m in M4_PRETRAINED}
    rows, points, novel = load_saved(out) if resume else ([], [], [])
    done = {(p["scheme"], p["fold"], p["method"]) for p in points}
    for scheme, fold, seed, split in m4_folds(obs):
        for method in (*M4_TRAINED, *M4_PRETRAINED):
            if (scheme, fold, method) in done:
                continue
            t0 = time.perf_counter()
            emb = pretrained[method] if method in pretrained else np.load(
                emb_dir / f"{method}_{scheme}_{fold}.npy")  # fmt: skip
            info = {"dataset": name, "scheme": scheme, "fold": fold, "method": method}
            groups = groups_all if scheme == "dataset" else None
            r, p = evaluate_fold(emb, labels, split, groups, donors, classes, seed, info, None)
            rows += r
            points += p
            if scheme == "donor" and fold == "seed0":
                novel += novel_type_rows(emb, labels, split, classes, seed, info)
            print(f"{scheme} {fold} {method} {time.perf_counter() - t0:.0f}s", flush=True)
        save(out, rows, points, novel)


def run_pbmc(name: str, out: Path) -> None:
    obs = ad.read_h5ad(ROOT / "data" / "processed" / f"{name}.h5ad", backed="r").obs
    labels = obs["cell_type"].astype(str).to_numpy()
    donors = obs["donor_id"].astype(str).to_numpy()
    classes = np.unique(labels)
    split = donor_split(obs, 0)
    saved = np.load(ROOT / "data" / "embeddings" / name / "split_seed0.npy", allow_pickle=True)
    if not np.array_equal(saved.astype(str), split):
        raise RuntimeError("saved PBMC embeddings were made with a different split")
    rows, points, novel, sizes, pairs = [], [], [], [], []
    for method in PBMC_METHODS:
        emb = np.load(ROOT / "data" / "embeddings" / name / f"{method}.npy")
        info = {"dataset": name, "scheme": "donor", "fold": "seed0", "method": method}

        def extra(kind, score_name, mondrian, sets, y_test, method=method):
            if mondrian:
                return
            size = sets.sum(1)
            for y, c in enumerate(classes):
                mask = y_test == y
                if mask.any():
                    sizes.append({"method": method, "classifier": kind, "score": score_name,
                                  "cell_type": c, "n_test": int(mask.sum()),
                                  "mean_size": float(size[mask].mean()),
                                  "coverage": float(sets[mask, y].mean())})  # fmt: skip
            co = sets.T.astype(float) @ sets.astype(float) / len(sets)
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    if co[i, j] > 0:
                        pairs.append({"method": method, "classifier": kind, "score": score_name,
                                      "type_a": classes[i], "type_b": classes[j],
                                      "share_of_cells_with_both": float(co[i, j])})  # fmt: skip

        r, p = evaluate_fold(emb, labels, split, None, donors, classes, 0, info, extra)
        rows += r
        points += p
        novel += novel_type_rows(emb, labels, split, classes, 0, info)
        print(method, flush=True)
        save(out, rows, points, novel)
    pd.DataFrame(sizes).to_csv(out / "size_by_class.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "pair_cooccurrence.csv", index=False)


def save(out: Path, rows, points, novel) -> None:
    """Write all rows; rows from an earlier run keep their commit, new rows get the current one."""
    commit = git_commit(ROOT)
    for fname, data in (("conformal.csv", rows), ("point_metrics.csv", points),
                        ("novel.csv", novel)):  # fmt: skip
        for row in data:
            if not isinstance(row.get("commit"), str):
                row["commit"] = commit
        pd.DataFrame(data).to_csv(out / fname, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="m4_blood", choices=["m4_blood", "pbmc"])
    parser.add_argument("--resume", action="store_true", help="six labs: skip finished folds")
    args = parser.parse_args()
    out = ROOT / "results" / "m9" / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.name == "pbmc":
        run_pbmc(args.name, out)
    else:
        run_m4(args.name, out, resume=args.resume)
    run = {"name": args.name, "commit": git_commit(ROOT), "alphas": ALPHAS,
           "seconds": time.perf_counter() - t0,
           "versions": versions("scikit-learn", "numpy", "scanpy")}  # fmt: skip
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")


if __name__ == "__main__":
    main()
