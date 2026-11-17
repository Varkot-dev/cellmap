# Decision notes: cellmap

This document is your study guide. It follows the format of `docs/design-notes.md` in your DCMvsACM project: every section answers a question someone could ask you, states what was chosen, why, what the alternative was, and where the evidence is. The goal is that you can explain every decision in your own words, not recite it.

The rule from your DCMvsACM roadmap still applies. For anything in this repo you must be able to answer: **What did I measure? What did I find? Why is it true?**

Sections are added as milestones are built. Day by day history is in `journal/`.

---

## What is this project, in one sentence?

> "I compare simple and deep ways of turning a cell's gene counts into a short vector, on public blood data, holding out whole donors, to find out how much the deep models actually add."

---

## What carried over from DCMvsACM, and what this project does differently

You already made the central decision once: the unit of generalization is the patient, not the cell. "166,000 cells from 8 patients is still 8 patients." This project keeps that and tightens four things that the earlier repo left loose. Each one is something you could be asked about, so know them.

| In DCMvsACM | In cellmap | Why the change |
|---|---|---|
| HVG selection was in-fold in the first pipeline, but the later RunPod and Koenig scripts selected genes on all cells before splitting | Gene selection is always fitted on training donors only (`select_hvg` takes a `train_mask`, and a test proves non-training cells cannot change the result) | "Which genes are highly variable" is learned from data. You wrote that rule yourself; here it is enforced by the function signature instead of by memory |
| scVI was trained on all 166,519 cells, then its latent space was cross-validated by donor | Every learned embedding (PCA, VAE, transformer) is fitted on training donors only. Harmony is the one exception and is labelled as such everywhere | An encoder that has seen the test donors' expression has an advantage, even without labels. That is unsupervised leakage |
| `ci_95` was mean plus or minus 2 standard deviations over 5 folds | Results are reported as paired differences per donor split, with the count of splits where a method wins, and the min and max | Five folds do not support a 95 percent interval. A paired comparison on the same held-out donors is the honest statement at this sample size |
| One deterministic partition (`StratifiedGroupKFold` with no shuffle) | Five different seeded donor splits | With 21 donors, which donors are held out moves the score more than anything else. One split hides that |

One thing carried over unchanged: the figure style. `src/cellmap/plotting.py` uses the same palette as your `figstyle.py` (ink `#1A1A1A`, paper `#FBFAF7`, terracotta `#C2703D`, slate `#3C5A78`), headlines that state the finding, and the explanation in the caption.

---

# M0: data pipeline

## Why this dataset, and not the classic PBMC 3k?

The PBMC 3k tutorial data comes from one donor. A donor-level split needs several donors, so it cannot be used.

**The choice:** healthy-donor PBMCs from Schulte-Schrepping et al. (Cell, 2020), CELLxGENE dataset `5e717147-0f75-4de1-8bd2-6fda01b8d75f`, downloaded from Census release 2025-01-30 (`configs/pbmc.yaml`, `src/cellmap/census.py`).

**Why:** 45,787 cells, 21 donors, 20 cell types with standardized ontology labels, and two 10x chemistries (v2 for 19 donors, v3 for 2), which gives a real technical batch effect to correct.

**Alternatives considered:**
- Binvignat et al. 2024 controls (18 donors): rejected, median of 388 UMIs per cell, too shallow.
- Hao et al. 2021 (161,764 cells): higher quality, but only 8 donors, too few for train, val and test.
- Yoshida et al. 2022 (35 donors): includes newborns and children, which adds an age confound for a first dataset.

**Known weakness:** the two v3 donors hold 15,498 of 44,756 cells, and chemistry is confounded with donor (no donor was run on both). So "chemistry effect" and "these two donors are different people" cannot be fully separated.

**What to say:** "I needed a small dataset with enough donors to hold some out completely. The standard tutorial dataset has one donor, so I took a 21-donor healthy PBMC set from CELLxGENE Census, pinned to a release."

---

## Why split by donor? (same decision as DCMvsACM)

Cells from one donor share genetics, handling and sequencing run. If the same donor is on both sides of the split, a model can score well by recognizing the donor.

**The choice:** `split_donors` in `src/cellmap/splits.py` assigns whole donors: 13 train, 4 val, 4 test. `assert_no_donor_overlap` runs on every pipeline run and raises if any donor is in two splits.

**Evidence:** `results/m0/pbmc/summary.json`, field `donors_shared_between_splits`: 0, 0, 0. Figure: `assets/m0_split_pbmc.png`.

**Known weakness:** in DCMvsACM your leakage test checked sklearn's splitter, not your own evaluation path. Here `test_overlap_check_catches_a_cell_level_split` feeds the check a cell-level split and asserts that it fails, so the guard itself is tested.

**Q: Why not stratify the split by cell type or chemistry?**
A: With 2 v3 donors and 3 splits, chemistry cannot be balanced. And choosing a split because it looks balanced is choosing it by looking at the outcome. Instead I repeat over five seeded splits and report all of them.

---

## Which steps run before the split, and which after?

This is the single rule behind the whole pipeline order.

**The rule:** a step that uses only fixed constants or only one cell's own values is safe before the split. A step that estimates anything from many cells must run after the split, on training donors only.

| Step | Uses | Where it runs |
|---|---|---|
| Cell QC (500 UMIs, 200 genes, 15 percent mitochondrial) | fixed thresholds | before the split |
| Drop cell types under 50 cells | counts of labels | before the split (see weakness) |
| Gene filter (gene seen in at least 10 cells) | many cells | after the split, training donors only |
| Normalize to 10,000 counts, log(1 + x) | the cell's own total, a constant | either side, identical result |
| Highly variable genes, gene mean and sd, PCA axes | many cells | training donors only |

**Evidence:** `test_gene_filter_only_looks_at_selected_cells`, `test_each_cell_is_normalized_independently`, and `test_hvg_ignores_non_training_cells` and `test_pca_fit_ignores_non_training_cells` in `tests/test_baselines.py`.

**Known weakness:** dropping rare cell types uses labels from all donors. It removed one type ("double negative thymocyte", 40 cells). It decides which cells are evaluated, not how any model is fitted, so I judged it acceptable. Also, the five donor splits in M1 all reuse the gene filter fitted on split seed 0's training donors. The filter only asks "is this gene detected in 10 cells", so the effect is negligible, but it is not strictly per-split.

---

## Why normalize to a fixed 10,000 and not the median library size?

**The choice:** `normalize_log1p(adata, target_sum=1e4)` in `src/cellmap/preprocess.py`.

**Worked example:** cell A has counts (1, 3, 0), total 4. Cell B has (10, 30, 0), total 40. Scaled to 100 both become (25, 75, 0). The depth difference is gone and the proportions remain.

**Why:** the median library size is a statistic of the dataset. Computed on all donors it is a small leak, and it would change from split to split. A constant has neither problem. It also matches what you did in the Koenig scripts.

**Known weakness:** log(1 + x) after scaling distorts low counts and makes variance depend on depth. That is the reason the VAE (M2) models raw counts with a negative binomial instead. Raw counts are kept in `adata.layers["counts"]` for that purpose.

---

# M1: classical baselines and the evaluation harness

## How are highly variable genes chosen?

**The choice:** `select_hvg` in `src/cellmap/baselines.py`: scanpy's "seurat" flavor on log-normalized values, 2,000 genes, ranked inside each training donor separately and then ordered by how many donors select the gene.

**Why per donor:** a gene that differs between people but not between cells of one person is not useful for telling cell types apart. Pooled over donors, the selection picked XIST (the sex-linked gene that exposed your sex confound in DCMvsACM). Per donor, XIST is not selected and CD3E is. The two selections share only 688 of 2,000 genes, so this choice matters.

**Alternatives considered:**
- "seurat_v3" on raw counts: the better statistic by your own July 5 finding ("Fano factor and seurat_v3 are not interchangeable"). Not used yet because it needs the scikit-misc package. This is an open item, not a settled choice.
- Pooled selection: simpler, but selects donor-difference genes.

**Known weakness:** I have not measured how much the flavor changes the downstream scores. In DCMvsACM the HVG method moved AUC by about 6 points (confounded with other changes). This should be a sensitivity check before the final write-up.

---

## Why is PCA fitted on training donors only, and how?

**The choice:** `pca_embed`: z-score each gene with the training mean and sd, clip to plus or minus 10, take the top 50 eigenvectors of the training covariance, project all cells.

**Why clip:** a gene silent in almost every cell has a tiny sd, so the few cells expressing it get huge z-scores and one gene could own a component.

**Why no random seed:** the 2,000 by 2,000 covariance matrix is decomposed exactly, so the result is deterministic. `test_pca_matches_direct_svd_up_to_sign` checks it against a direct SVD.

**Cost:** about 2.6 seconds for 44,756 cells (`results/m1/pbmc/run.json`).

---

## What does Harmony do, and why is it flagged as an exception?

Harmony takes PCA coordinates and donor labels. It soft-clusters the cells while penalizing clusters dominated by one donor, then shifts each donor's cells toward the shared cluster center, and repeats.

**The exception:** Harmony is transductive. It has no "fit on train, apply to test" mode, so the test donors' cells take part in the correction through their expression and donor label. It never sees a cell-type label.

**The choice:** run it on all cells, with library defaults, and say so next to every Harmony number (`harmony_correct` docstring, figure captions, README).

**Why not avoid it:** it is the standard non-deep baseline, and one of the simple methods the anchor paper (Kedzierska et al.) compares against. Leaving it out would remove the most important simple competitor.

**What to say:** "PCA is fitted on training donors and applied to held-out donors. Harmony cannot work that way, it adjusts all the cells it is given, so its numbers are transductive and I label them. It still never sees cell types."

**Q: Is that leakage?**
A: Not label leakage. It is an advantage in the features, the same kind your DCMvsACM scVI model had when it was trained on all donors. The difference is that here it is stated, and the deep models in this project are held to the stricter standard.

---

## How is an embedding scored?

All scoring code takes a plain array and labels, and knows nothing about the method (`src/cellmap/evaluate.py`, `src/cellmap/benchmark.py`). PCA, a VAE and a transformer go through identical code.

**Three views:**
1. **kNN classification.** Label each test-donor cell by majority vote of its 15 nearest training-donor cells. Reported as accuracy and macro F1, at 19 fine types and 10 coarse types.
2. **Label efficiency.** The same, with only 1 percent or 10 percent of training labels.
3. **scIB metrics**, two groups kept separate. Biological conservation: NMI and ARI of Leiden clusters against cell types, cell-type silhouette, cLISI. Batch correction: batch silhouette, graph connectivity, iLISI. Aggregate = 0.6 times bio plus 0.4 times batch (Luecken et al. 2022).

**Why macro F1 and not only accuracy:** classical monocytes are 57 percent of the test cells in split seed 0. A method can be 91 percent accurate while failing rare types completely. Macro F1 averages the per-type F1 scores, so each type counts equally. On split seed 0 PCA has accuracy 0.914 and macro F1 0.606.

**Macro F1 convention (a decision, state it):** averaged over cell types that have at least one test cell. A type present in test but absent from training counts as F1 = 0, because every such cell is mislabelled. A type absent from test is excluded, because its recall is 0 divided by 0. This differs from sklearn's default.

**Why scIB is computed on test donors only:** so that every number in a row describes cells from donors the method was not fitted on.

**Known weaknesses, all real:**
- NMI is the best value over ten Leiden resolutions, so it is optimistic, equally for every method.
- `scib-metrics` 0.6.2 computes graph connectivity on a directed graph, which is stricter than expected (ideal synthetic blobs score 0.993, not 1.0).
- No min-max scaling across methods is applied, unlike the ranking in the scIB paper. Scores here are absolute.
- iLISI is close to 0 for an uncorrected embedding by construction, and with only three batch metrics it pulls the batch mean down.
- Batch metrics on 4 test donors are a weak test of mixing.

---

## M1 result: what did Harmony change?

Five donor splits, each holding out 4 donors. Numbers from `results/m1/pbmc/summary.csv` and `paired_harmony_minus_pca.csv`. Figure: `assets/m1_tradeoff_pbmc.png`.

| Metric | HVG + PCA (mean over 5 splits) | PCA + Harmony | Paired change | Splits where Harmony is higher |
|---|---|---|---|---|
| scIB biological conservation | 0.735 | 0.801 | +0.066 | 5 of 5 |
| scIB batch correction | 0.569 | 0.662 | +0.093 | 5 of 5 |
| scIB aggregate | 0.669 | 0.746 | +0.077 | 5 of 5 |
| kNN accuracy, fine | 0.893 | 0.856 | -0.037 | 0 of 5 |
| kNN macro F1, fine | 0.664 | 0.657 | -0.007 | 2 of 5 |
| kNN accuracy, coarse | 0.951 | 0.947 | -0.003 | 0 of 5 |
| kNN macro F1, coarse | 0.708 | 0.718 | +0.010 | 5 of 5 |

**What I found:** Harmony improved every scIB group on every split, and lowered fine-label kNN accuracy on every split. The two families of metrics disagree.

**Why that can be true (hypothesis, not proven):** scIB rewards clusters that match cell types and donors that mix. kNN accuracy at the fine level depends on small distances between close subtypes, such as central memory versus naive CD4 T cells. Pulling donors together can blur exactly those distances. That is the "batch correction goes too far" trade-off from section 4.4 of the plan, seen in a mild form.

**What this does and does not establish:**
- It is five splits of one dataset. The direction is consistent, the size is not well estimated.
- The largest accuracy drop (0.832 to 0.751) is split seed 3, one of the two splits with a v3 donor in test. Two splits is too few to say chemistry drives it.
- Macro F1 at the fine level shows no detectable difference.

**The chemistry effect is large.** `assets/m1_batch_umap_pbmc.png`: before Harmony the v2 and v3 cells form separate islands for every cell type. After Harmony they overlap. Harmony was given the donor, not the chemistry.

**Where both methods fail (split seed 0, PCA, `results/m1/pbmc/confusion_pca.csv`):**
- transitional B cells called memory B cells: 39 of 63
- conventional dendritic cells called classical monocytes: 33 of 54
- gamma-delta T cells called MAIT cells: 22 of 53
- platelets: F1 of 0 with 3 test cells

These are biologically close pairs (same lineage), which is the reassuring kind of error.

**Label efficiency (split seed 0, `label_efficiency.csv`):** with 1 percent of labels (337 cells), macro F1 is 0.286 for PCA and 0.334 for Harmony. With 10 percent, 0.482 and 0.536. With all labels, 0.606 and 0.590. Harmony is ahead when labels are scarce and behind when they are plentiful. One split only, so treat it as a lead to check, not a finding.

**What to say:** "On my first dataset Harmony beat plain PCA on the standard integration metrics in all five donor splits, but it lost about 4 points of fine-grained kNN accuracy in all five. So which method is better depends on the metric, and I report both rather than one aggregate."

**Q: Which one is the baseline the deep models have to beat?**
A: Both. A deep model that beats PCA on scIB but not Harmony has not earned its cost, and one that beats Harmony on scIB but loses kNN accuracy has the same trade-off Harmony has.

---

## What you built so far, file by file

| File | What it does | Key decision |
|---|---|---|
| `src/cellmap/census.py` | Downloads one dataset from CELLxGENE Census | Release pinned; opened by S3 location |
| `src/cellmap/qc.py` | Cell filter, rare-type filter, gene filter | Gene filter takes a training-cell mask |
| `src/cellmap/splits.py` | Donor-level split and overlap check | Whole donors only; depends on the donor set and seed, not input order |
| `src/cellmap/preprocess.py` | Normalize to 10,000, log(1 + x) | Fixed target sum; raw counts kept |
| `src/cellmap/labels.py` | Fine to coarse cell-type mapping | Raises on any unmapped type |
| `src/cellmap/baselines.py` | HVG selection, PCA, Harmony | Fitted on training donors; Harmony flagged transductive |
| `src/cellmap/evaluate.py` | kNN, label efficiency, scIB metrics | Method-agnostic; explicit macro F1 convention |
| `src/cellmap/benchmark.py` | One function that scores any embedding | Same code path for every method |
| `src/cellmap/plotting.py` | Shared figure style | Same palette as DCMvsACM `figstyle.py` |
| `scripts/m0_prepare_data.py` | Runs M0 and saves every reported number | Order of steps follows the before/after-split rule |
| `scripts/m1_baselines.py` | Runs M1 over five donor splits | Paired comparison on identical held-out donors |
