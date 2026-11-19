# leakaudit

A small kit for checking two things that often make machine learning on biological data look
better than it is: leakage between training and test data, and models that score well while
ignoring part of their input. It depends on numpy, pandas and scikit-learn only. AnnData
objects are accepted but anndata is not required.

Status: version 0.1.0, tested on synthetic data (43 tests) and on one real dataset (the
cellmap PBMC data, below). It has not been used yet on the other projects it is meant for.

## The three checks

### 1. Group-aware splits with a guarantee

Rows from one patient, donor, lab, scaffold or paper share things the label does not cause.
If a group has rows on both sides of a split, a model can score well by recognising the
group. These functions assign whole groups, check that no group crosses a split, and report
what the split did to the classes.

```python
from leakaudit import assert_no_group_overlap, group_kfold, group_split, split_report

split = group_split(
    patient_ids, val_fraction=0.2, test_fraction=0.2, seed=0
)  # "train"/"val"/"test"
assert_no_group_overlap(patient_ids, split, required=("train", "val", "test"))
print(split_report(patient_ids, split, labels).summary())
folds = group_kfold(patient_ids, n_splits=5, seed=0)  # fold number per row
```

Fractions are fractions of groups, not rows. `assert_no_group_overlap` accepts any split labels,
so it also checks splits made elsewhere (for example by scikit-learn's `StratifiedGroupKFold`).
`split_report` gives rows, groups and classes per split, the classes missing from each split,
and the largest gap in a class's share between splits.

### 2. How much does a leaky evaluation inflate the score?

`measure_inflation` scores the same model under a random row split and a group split, over
several seeds, and returns the difference with its standard error and a plain summary.
`measure_preprocessing_leak` does the same for preprocessing (feature selection, scaling)
fitted on all rows versus on the training rows only, on identical splits.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from leakaudit import measure_inflation, measure_preprocessing_leak

res = measure_inflation(X, y, patient_ids, KNeighborsClassifier(5), accuracy_score, seeds=range(5))
print(res.summary())  # plain language
res.summary_frame()  # one-row table; res.scores has every (scheme, seed)

pre = measure_preprocessing_leak(
    X,
    y,
    SelectKBest(f_classif, k=20),
    LogisticRegression(),
    accuracy_score,
    groups=patient_ids,
    seeds=range(5),
)
```

The model can be any estimator with `fit` and `predict` (cloned for every split) or a function
`fit_predict(X_train, y_train, X_test)`. Passing `X = np.arange(n_rows)` hands the function row
indices, which is how a pipeline that needs the whole dataset (the PBMC example) is plugged in.
Use `prediction="predict_proba"` for metrics such as ROC AUC, and `greater_is_better=False`
for losses.

On the toy data used to test these snippets (40 patients, each with its own feature offset and
mostly one label, no other signal), the random split scored accuracy 1.000 and the group split
0.481 (sd 0.176 over 5 seeds). Feature selection fitted on all rows of pure noise (120 rows,
2,000 features) scored 0.756 against 0.400 when fitted on training rows only.

### 3. Shortcut probes: which inputs does the model use?

`shortcut_probe` keeps a fitted model fixed, damages one named block of input columns at a
time on an evaluation set, and reports the metric drop per block. Shuffling moves the block's
columns together to other rows; `constant="mean"` (or a value) replaces the block with one
value; `custom` takes arbitrary perturbations such as "set every nonzero count to 1". This is
the shuffle probe used in Bindwell, "The Bitter Lesson for Biology" (2026), and in Fang et al.
(2026), generalised from single columns (scikit-learn's permutation importance) to named blocks.

```python
from leakaudit import shortcut_probe

probe = shortcut_probe(
    model,
    X_test,
    y_test,
    {"protein": protein_cols, "ligand": ligand_cols},
    accuracy_score,
    n_repeats=5,
    constant="mean",
)
print(probe.summary())
probe.table  # one row per probe: score, drop, spread over repeats
```

Read a near-zero drop as "the model does not need this block", not "the model never looks at
it": a block that duplicates another can be removed at no cost. A drop can also come from
shuffled combinations the model never saw. Run the probe on held-out rows.

## Real data: cellmap PBMC, random cells versus held-out people

`examples/cellmap_pbmc.py` reruns cellmap's `scripts/leakage_splits.py` with this kit:
44,756 blood cells from 21 donors, 19 cell types, highly variable genes and PCA fitted on
training cells, kNN (k = 15), macro F1 over the cell types present in the test cells, 5 seeds.
The kit's splits are first checked to be identical to cellmap's for every seed. Output:
`examples/output/`.

| Scheme | Mean kNN macro F1 | sd over 5 seeds |
|---|---|---|
| Random cell split (leaky) | 0.776 | 0.006 |
| Donor split (honest) | 0.664 | 0.077 |
| Inflation | 0.112 | standard error 0.034 |

Every per-seed macro F1 equals the value in `results/leakage/splits.csv` exactly (largest
difference 0.0; accuracy differs by at most 1.1e-16, floating-point rounding). One run, 62
seconds on 2 threads. Besides the inflation, the donor split's scores vary about 12 times more
across seeds than the random split's, so a random split also hides how uncertain the result is.

## Prior art and where this kit fits

Checked in November 2026 by reading package metadata and source from PyPI wheels (downloaded,
not installed) and, for papers, by web search. Documentation sites themselves were blocked
from this machine, so descriptions come from the code, docstrings and package summaries.

| Tool | What it does | What it does not do |
|---|---|---|
| scikit-learn 1.9 `GroupKFold`, `StratifiedGroupKFold`, `GroupShuffleSplit`, `LeaveOneGroupOut`, `LeavePGroupsOut` | Group-aware splitters (shuffle option on the k-fold ones) | No train/val/test helper, no overlap assertion for splits made elsewhere, no balance report, no leak measurement |
| scikit-learn `permutation_importance` | Score drop when one column is permuted | One column at a time, no named column blocks, no constant or custom damage |
| DataSAIL 1.4.0 (Joeres, Blumenthal and Kalinina, Nature Communications 16, 3337, 2025) | Builds leakage-reduced splits from similarity (clustering plus integer linear programming), for molecules, proteins, genomes and interaction pairs | Does not measure how much a given leak inflates a score, no shortcut probes; heavier dependencies (cvxpy, solvers) |
| splito 0.1.4 (datamol) | Molecular splitters: scaffold, k-means, max-dissimilarity, MOOD, LoHi and others | Chemistry-specific; no inflation measurement or probes |
| deepchecks 0.19.1 | Train/test sample mix, index and date leakage, feature-label correlation checks | I found no check for related (non-identical) rows of one group across splits in its tabular checks (not exhaustive); no random-versus-group comparison |
| skore 0.29.1 | Report checks, including a "golden feature" check and train/test time overlap | No group splits or group-leak measurement; probes single features, not blocks |
| mlleak 0.1.1 | Reports duplicates, time leakage and group ids present in train and test | Detection only; does not score models |
| featureleak 0.1.0 | Rule-based leakage checks, including entity overlap and a preprocessing-contamination rule using random stratified CV with LightGBM | Not group-aware in that rule; fixed model rather than yours |
| featureleakagelens 0.2.0, leaklens 1.3.0 | Dataset-level audits (target leakage, drift, contamination, duplicates) | No model-based inflation measurement or probes |
| leakly 0.1.2 | Label-permutation tests for whole pipelines | Different question (above chance?), not split leakage or input blocks |
| leakproof-ml 0.0.2 | Framework for leakage-aware training, with a shuffled group k-fold | A training framework, not an audit of an existing model |

What this kit adds is small and specific: the same model scored under leaky and honest
evaluation with seed spread, as one call, plus block-level shortcut probes, with no
dependencies beyond scikit-learn. The splitters themselves are not new. Package searches
were limited (PyPI's own search page was blocked), so other similar packages may exist.
Note that an R package named `leakaudit` exists on CRAN (seen in a search result, not
examined); the name is free on PyPI as of this check.

The Bindwell (2026) and Fang et al. (2026) references are given as in cellmap's notes; their
full bibliographic details were not checked here.

## Install and test

```bash
pip install -e .            # or: uv pip install -e .
pip install -e ".[dev]"     # pytest and ruff
pytest -q
ruff check . && ruff format --check .
```

Python 3.11 or newer. MIT license.
