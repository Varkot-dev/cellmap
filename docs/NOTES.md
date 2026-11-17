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

# M2: the hand-written VAE

## What is the model, in plain words?

An encoder reads a cell's counts (log of 1 + count, over 2,000 genes) and outputs a mean and a variance for a 10-number code z. A decoder takes z plus the donor, and outputs, for every gene, the share of the cell's RNA it expects that gene to have. Multiply the share by the cell's actual total count and you have the expected count. The loss asks how likely the real counts are under a negative binomial with that expected count, and adds a penalty (the KL term) for codes that wander away from a standard normal.

File: `src/cellmap/vae.py`. Every piece has a hand-checked test in `tests/test_vae.py`.

---

## Why a negative binomial and not a Gaussian?

Counts are integers, mostly zero, and their variance grows faster than their mean. A Gaussian assumes constant variance and allows negative values. The negative binomial has a mean mu and an inverse dispersion theta, with variance mu + mu^2 / theta. When theta is large it becomes a Poisson; when theta is small the variance is much larger than the mean, which is what real genes look like.

**Evidence it is implemented right:** `nb_log_likelihood` matches `scipy.stats.nbinom.logpmf` (with n = theta, p = theta / (theta + mu)), sums to 1 over x, and tends to the Poisson as theta grows. All three are tests.

---

## Why does the encoder not see the donor, but the decoder does?

**The decoder gets the donor** so it can explain donor-specific differences itself. Then the code z does not have to carry them. That is the whole batch-correction mechanism of this model.

**The encoder does not get the donor** so that a cell from a donor never seen in training can still be embedded. This is what makes donor-level splits possible for a deep model.

**Unseen donors in the decoder:** only needed to compute a validation loss. They get the average of all training donors' one-hot vectors (1/13 each), meaning "decode as a typical training donor".

**This matches scvi-tools' default** (`encode_covariates=False`), checked in its source.

---

## What does the KL weight do? (the plan's "both failure modes")

Loss = reconstruction + w * KL. Sweep over w, split seed 0, scored on VALIDATION donors because picking w is model selection (`results/m2/pbmc/kl_sweep.csv`, written by `scripts/m2_kl_sweep.py`).

| KL weight w | KL per cell (val) | Reconstruction (val) | Latent dimensions in use | kNN macro F1 (val) |
|---|---|---|---|---|
| 0 | 52.3 | 310.6 | 10 of 10 | 0.591 |
| 0.1 | 22.1 | 312.9 | 10 of 10 | 0.614 |
| 1 | 7.3 | 323.6 | 10 of 10 | 0.638 |
| 10 | 1.5 | 351.5 | 4 of 10 | 0.503 |
| 100 | 0.0 | 397.5 | 0 of 10 | 0.258 |

Both failure modes appear on the real data:
- **Weight 0, plain autoencoder:** KL is 52 per cell, seven times the value at weight 1, and reconstruction is best (310.6). Nothing ties the codes to the prior.
- **Weight 100, collapse:** KL is 0.003 and no latent dimension varies across cells. Every cell gets essentially the same code, so kNN macro F1 falls to 0.258.
- **In between:** at weight 10, six of the ten dimensions are already switched off.

**What to say:** "At weight zero it is a plain autoencoder: the codes drift far from the prior and reconstruction is best. As the weight grows the model trades reconstruction for simpler codes, switches dimensions off, and at weight 100 every cell gets the same code. The standard weight of 1 gave the best validation macro F1, so the benchmark uses it."

**Known weakness:** one split, one training seed per weight, 60 epochs. The differences between weights 0.1 and 1 (macro F1 0.614 versus 0.638) are within what a single seed can move.

---

## M2 result: did the VAE beat the baselines?

Five donor splits, test donors (`results/m2/pbmc/metrics_by_split.csv`, compared with `results/m1/pbmc/`). The bar was fixed in advance: better in at least 4 of 5 splits against both baselines.

| Metric | HVG + PCA (50 dims) | PCA + Harmony | VAE (10 dims) | PCA (10 dims) | Linear AE (10 dims) |
|---|---|---|---|---|---|
| kNN accuracy, fine | 0.893 | 0.856 | 0.879 | 0.846 | 0.842 |
| kNN macro F1, fine | 0.664 | 0.657 | 0.646 | 0.525 | 0.519 |
| scIB bio | 0.735 | 0.801 | 0.737 | 0.733 | 0.730 |
| scIB batch | 0.569 | 0.662 | 0.600 | 0.527 | 0.530 |
| scIB aggregate | 0.669 | 0.746 | 0.682 | 0.651 | 0.650 |

**Against the bar:**
- versus PCA: no detectable difference on kNN or scIB bio. Higher batch score in 4 of 5 splits (+0.031), so a small, consistent batch-correction gain.
- versus Harmony: lower on scIB in 5 of 5 splits (aggregate -0.064), higher on fine kNN accuracy in 5 of 5 (+0.023). Same trade-off Harmony showed against PCA.
- versus PCA with the same 10 dimensions: higher macro F1 in 5 of 5 splits (+0.121). The VAE packs much more cell-type information into 10 numbers.

**The linear autoencoder ties 10-dimension PCA**, as theory says it should (its optimum spans the PCA subspace). That is a check that the comparison is set up correctly, not a finding.

**What to say:** "My VAE did not beat the simple baselines. It matched 50-dimension PCA using 10 dimensions and gave a small batch-correction gain, but Harmony on top of PCA was better on every integration metric. That is the result the plan said to report if it happened."

**Known weaknesses:**
- Latent size 10 follows scvi-tools defaults. A larger code was not tried, so "the VAE loses to Harmony" is for this size only.
- Training time (230 to 447 seconds per split) was measured while other jobs shared the CPU.

---

# M3: parity with scvi-tools

## Why check the hand-written VAE against a library at all?

A hand-written model can be wrong in ways that still train: a sign error in the KL term, a library size computed over the wrong genes, a likelihood that is off by a constant. Each still gives a loss that goes down. The only way to know the VAE is the model you think it is, is to compare it with an independent implementation of the same model under the same conditions.

**The choice:** `src/cellmap/scvi_baseline.py` trains scvi-tools 1.5.2 with every setting matched to `cellmap.vae` (negative binomial likelihood, observed library size, one dispersion per gene, the same layer sizes, optimizer, learning rate, batch size, KL warm-up over 10 epochs, no internal validation split), on the training donors only, then `scripts/m3_scvi_parity.py` scores it on the same five donor splits as M2.

---

## How do you know the two are the same computation?

Three levels of evidence, from strongest to weakest (all on the tiny development data):

1. **Weight transplant.** Copying a trained scvi-tools model's weights into `cellmap.vae.VAE` gives identical embeddings for all 3,087 cells (difference 0.0), in 8 of 8 seeds. Same architecture, same forward pass.
2. **Single training steps.** From the same starting weights and the same 128 cells, the reconstruction and KL terms agree to every printed digit, gradients agree to about 1 part in a million, and the weights after an Adam step agree to 1e-7. Same loss, same optimizer.
3. **Trained from scratch, 8 seeds.** kNN accuracy 0.840 for both; macro F1 0.400 versus 0.408. Same result within noise.

**The one real difference:** the hand-written model reached a reconstruction loss about 0.5 percent lower (215.9 versus 217.0 per cell), consistently. It traced to one thing: scvi-tools starts each gene's log dispersion at a random normal draw, the hand-written model at 0. Giving the hand-written model scvi-tools' start closed the gap.

**What to say:** "I checked my VAE three ways against scvi-tools: copying weights gives identical embeddings, single gradient steps match to a millionth, and trained from scratch they agree within seed noise. The one difference I found was the dispersion initialization, and swapping it in removed the gap."

---

## How do held-out donors get embedded by scvi-tools?

scvi-tools refuses to embed cells whose donor it was not set up with. But with `encode_covariates=False` the encoder never reads the donor label, so any training donor's label can stand in. The code checks this on every call: it embeds the held-out cells under a second training donor's label and raises if the embeddings differ by more than float32 rounding.

**A mistake worth naming:** the first version of that check demanded an exact zero difference. On split 3 the difference was 0.0000005, because the second pass runs on a differently sized block of cells and the matrix products round differently in the last bit. The run stopped, and I committed the three finished splits under a message that said five. The check now allows float32 rounding, the commit after it says the earlier one was partial, and the full run was repeated.

---

## M3 result on the full data

Five donor splits, test donors (`results/m3/pbmc/` against the `vae` rows of `results/m2/pbmc/`):

| Metric | Hand-written VAE | scvi-tools | Paired difference (VAE minus scvi) | Splits where VAE is higher |
|---|---|---|---|---|
| kNN accuracy, fine | 0.879 | 0.884 | -0.005 | 2 of 5 |
| kNN macro F1, fine | 0.646 | 0.646 | 0.000 | 3 of 5 |
| kNN macro F1, coarse | 0.712 | 0.710 | +0.002 | 3 of 5 |
| scIB bio | 0.737 | 0.744 | -0.008 | 2 of 5 |
| scIB batch | 0.600 | 0.599 | +0.001 | 3 of 5 |
| scIB aggregate | 0.682 | 0.686 | -0.004 | 2 of 5 |

**Parity holds.** No metric moves the same way in more than 3 of 5 splits, and every mean difference is under 0.01.

**How big is run-to-run noise?** The first scvi-tools run used one CPU thread and the rerun used four. Same code, seed and cells. On split seed 0, kNN accuracy moved 0.001 and the batch score 0.001, but scIB bio moved from 0.815 to 0.741. Changing the thread count changes floating-point rounding, which changes training slightly. The bio score rests on Leiden clustering, which amplifies small changes.

**What this means for every comparison in the project:** a difference in scIB bio smaller than about 0.07 between two methods on one split is not evidence of anything here. kNN scores and the batch score are far more stable. This is one more reason the comparison rules count splits rather than trusting one mean.

---

# M5: the hand-written transformer

## How is a cell turned into a sequence? (plan question 7)

A sentence is a sequence of words; a cell is not naturally a sequence of anything. Three choices define the answer:

**1. Which genes become tokens.** Only the genes the cell expresses (count above 0), among the 2,000 selected genes. A cell has about 60 to 300 such genes. The alternative, one token per gene including zeros, makes "this gene is off" explicit but costs far more computation, because attention scales with the square of the sequence length. Cells with more than 256 expressed genes are cut to 256: a random subset each time in training, the most expressed ones when embedding.

**2. How the expression level is encoded.** Each token is a learned vector for the gene plus a learned vector for its level. The level is one of 10 bins, computed inside each cell from the rank of the gene's value among that cell's expressed genes. So the bins do not change if a cell was sequenced twice as deeply. This is close to scGPT's approach. Alternatives:
- a continuous value through a small network: keeps exact values, but is sensitive to depth and outliers;
- Geneformer's approach: no value at all, the genes are ordered by expression and the order itself carries the level.

**3. Order.** None. Genes have no natural order, so there is no positional encoding, and a test checks that shuffling a cell's tokens does not change its embedding.

**Worked example:** a cell expresses gene 7 (count 1), gene 2 (count 5) and gene 9 (count 1). Ranks: gene 7 and gene 9 tie at the bottom, gene 2 is above both. With 10 bins and 3 genes, the two tied genes go in bin 0 and gene 2 in bin floor(10 * 2/3) = 6. Tokens: (gene 2, bin 6), (gene 7, bin 0), (gene 9, bin 0), in any order.

**Known weakness of this binning:** about 60 percent of nonzero counts in this data are exactly 1, and all of them land in bin 0, so the bins are very unequal.

---

## What is the model trained to do?

**Masked gene modeling.** In each cell, 15 percent of the tokens keep their gene identity but have their level replaced by a [MASK] vector. The model must predict the hidden level from the rest of the cell. It is the cell version of BERT's masked words.

**Why keep the gene and hide the level:** the question "how high is CD14 in this cell, given everything else it expresses" can only be answered by learning which genes go together, which is the structure that separates cell types. Hiding the gene identity instead (as Geneformer does) asks "which gene is here", a different task.

**How to tell it learned something:** two guessing baselines are reported next to the accuracy. Always guessing the most common bin, and guessing each gene's most common bin. Beating the second shows the model uses the rest of the cell, not just facts about each gene.

**A second objective for the cell embedding:** the cell's embedding is the final state of a [CLS] vector added at the front. With masked prediction alone, nothing reads that state, so it would never be trained. A second small head therefore has to predict every token's level from the [CLS] state alone, which forces it to summarize the whole cell.

**What the model does not have:** any donor input. It has no batch-correction mechanism, unlike the VAE and Harmony. That is a real limitation to state with its results.

**Size:** 4 layers, 4 attention heads, width 128, about 1 million parameters with 1,000 genes.

---

## M5 result

Trained on a Colab T4 GPU, 30 epochs per split, 6 to 11 minutes per split, on training donors only (`results/m5/pbmc/`). The notebook first rebuilt the data from Census and confirmed it matched M0 exactly.

| | kNN accuracy | kNN macro F1 | scIB bio | scIB batch | scIB aggregate |
|---|---|---|---|---|---|
| Transformer | 0.745 | 0.371 | 0.554 | 0.481 | 0.525 |
| HVG + PCA | 0.893 | 0.664 | 0.735 | 0.569 | 0.669 |
| VAE | 0.879 | 0.646 | 0.737 | 0.600 | 0.682 |

**It learned the task:** at the end of training it predicted hidden levels 4.5 to 8.6 points better than guessing each gene's usual level, on validation donors (`assets/m5_transformer_pbmc.png`).

**But the embedding is weak:** lower than PCA, Harmony and the VAE on every metric in all 5 splits.

**Training had levelled off:** validation loss moved by less than 0.01 over the last 10 epochs (the learning rate also decays to zero by design).

**Possible reasons (untested):** the masked task does not force the [CLS] summary to organise cells by type; no donor correction; most nonzero counts fall in the lowest bin; 34,000 cells is very little data for a transformer.

**What to say:** "My transformer learned to predict hidden genes better than a per-gene baseline, so the training worked, but its cell embedding was the weakest of my own models. Being good at the pretraining task did not make it good at separating cell types, which is the same gap the foundation-model papers report at much larger scale."

---

# M6: published pretrained-model embeddings

## Where do these embeddings come from, and why not run the models?

CZ CELLxGENE publishes, for each Census release, an embedding of every cell from several models. For release 2025-01-30 that includes Census scVI (50 dims), a fine-tuned Geneformer (512) and TranscriptFormer TF-Sapiens (2048). `src/cellmap/census_embeddings.py` downloads them for this project's 44,756 cells by their Census id, and `scripts/m6_pretrained.py` scores them on the same five splits.

**Why not run the models:** no GPU here, Hugging Face is blocked from this machine, and the published embeddings are exactly what the model authors produced, so there is no risk of running them wrong.

**How cells were matched:** by `soma_joinid`. The script checks that donor and cell type agree for every cell before using the match, and raises if any cell has no embedding.

---

## What did each model see in training? (state this next to every number)

| Model | Saw these cells? | Used cell-type labels? |
|---|---|---|
| Census scVI | Yes: trained on all primary Census cells | No |
| Geneformer (Census version) | Fine-tuned on Census data | **Yes**: multi-task fine-tuning that includes cell type. Not zero-shot |
| TranscriptFormer TF-Sapiens | Pretrained on 57 million human cells; overlap not checked | No |

All three had an advantage the project's own models did not: they were fitted on data that includes the test donors.

---

## M6 result

| Method | kNN acc (fine) | Macro F1 (fine) | scIB bio | scIB batch | scIB aggregate |
|---|---|---|---|---|---|
| TranscriptFormer (2048) | 0.892 | 0.629 | 0.662 | 0.442 | 0.574 |
| Geneformer, fine-tuned (512) | 0.719 | 0.367 | 0.643 | 0.484 | 0.580 |
| Census scVI (50) | 0.841 | 0.533 | 0.746 | 0.550 | 0.668 |
| HVG + PCA (50) | 0.893 | 0.664 | 0.735 | 0.569 | 0.669 |
| PCA + Harmony (50) | 0.856 | 0.657 | 0.801 | 0.662 | 0.746 |

**Against the bar, none of the three beats PCA on any metric.**
- TranscriptFormer ties PCA on kNN accuracy (higher in 2 of 5) and is lower on every scIB group in 5 of 5.
- Geneformer is lower than PCA on every metric in 5 of 5 splits, despite having been trained with cell-type labels.
- Census scVI ties PCA on the scIB aggregate and is lower on macro F1 in 5 of 5 (-0.131).

**Why can a huge pretrained model lose to PCA here? (likely reasons, not tested)**
- Its embedding was not built for this question. TranscriptFormer's 2048 numbers include information that is not cell type (for example the assay, which it was given as an input), and kNN and scIB treat every dimension equally.
- PCA is fitted on exactly these cells and these genes; the pretrained models are general.
- Census scVI corrects for assay, dataset, donor and suspension all at once across 100 million cells, which may smooth over the fine T-cell subtypes that drive macro F1 here.

**What to say:** "None of the published foundation-model embeddings beat PCA on my held-out donors, even though they had all seen these cells during training. That agrees with Kedzierska et al."

**Known weaknesses:** different embedding sizes (50 to 2048), and no attempt to reduce or fine-tune them. That is deliberate: the question is how they do used as published.

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
| `src/cellmap/vae.py` | Hand-written negative binomial VAE | Encoder blind to donor; decoder gets donor |
| `src/cellmap/linear_ae.py` | Linear autoencoder | Same scaling as PCA, to isolate what nonlinearity adds |
| `src/cellmap/scvi_baseline.py` | scvi-tools with settings matched to `vae.py` | Trained on training donors only; unseen donors handled through the donor-blind encoder |
| `src/cellmap/transformer.py` | Hand-written transformer, masked gene modeling | Expressed genes as tokens, per-cell expression bins, no positions, no donor input |
| `src/cellmap/census_embeddings.py` | Downloads published model embeddings | Raises on any missing cell; records what each model saw |
| `src/cellmap/provenance.py` | Commit hash and versions saved with results | Marks results from uncommitted code as "+dirty" |
| `scripts/atlas_export.py`, `site/` | The interactive 3D atlas | One self-contained page, works on GitHub Pages |
| `notebooks/m5_transformer_gpu.ipynb` | Trains the transformer on a Colab GPU | Rebuilds data and checks it matches M0 before training |
