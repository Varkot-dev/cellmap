# cellmap: plan

Learned representations of single cells from gene-expression data, evaluated honestly.

## 1. The question

Single-cell RNA sequencing measures, for each cell, how many RNA copies of each of roughly 20,000 to 30,000 genes were captured. A dataset is a huge, sparse table: cells by genes. The central computational task is to turn each cell's row into a short vector (an embedding) in which cells of the same type sit together regardless of which donor or lab they came from.

In the last few years large pretrained "foundation models" for cells have been proposed. Careful evaluations have found they often do not beat much simpler methods. This project asks that question directly and from scratch:

**On public data, with splits that prevent leakage, how much do deep models add over simple baselines for representing cells, and where?**

This extends work you have already done (tree-based classifiers on single-cell data with patient-level splits) into representation learning.

## 2. What exists today (verify before building)

**The key paper to anchor on.** Kedzierska, Crawford, Amini and Lu, "Zero-shot evaluation reveals limitations of single-cell foundation models" (Genome Biology, 2025; preprint titled "Assessing the limits of zero-shot foundation models in single-cell biology"). They evaluate two popular pretrained models, Geneformer and scGPT, used without fine-tuning, on cell-type clustering and batch integration across five datasets. Both were unreliable in that setting and were often outperformed by simpler methods: selecting highly variable genes (HVG), Harmony, and scVI. Code: the `microsoft/zero-shot-scfoundation` repository. Read this first; this project follows its evaluation logic.

**Models and tools.**
- **Scanpy** and **AnnData**: the standard Python stack for single-cell analysis.
- **scVI** (Lopez and coauthors, 2018), in the `scvi-tools` library: a variational autoencoder with a count-based likelihood. The standard deep baseline. You will write a small version by hand and check it against the library.
- **Harmony**: a fast, non-deep batch-correction method that works on PCA coordinates.
- **Geneformer** and **scGPT**: pretrained transformers for cells. Used here only to extract embeddings, as a comparison point.

**Evaluation standard: scIB.** Luecken and coauthors, "Benchmarking atlas-level data integration in single-cell genomics" (Nature Methods, 2022). It defines 14 metrics in two groups: removal of batch effects, and conservation of biological variation. The `scib` Python package implements them (check also for the faster `scib-metrics` package). A 2025 follow-up (scIB-E) argues the original metrics miss variation within a cell type; worth reading before finalizing metrics.

**Data.**
- **CZ CELLxGENE Census**: a standardized public collection of single-cell data with a Python API (`pip install cellxgene-census`). You can query by tissue and cell type and get an AnnData object, with donor and dataset identifiers for every cell. Pin a census version for reproducibility.
- **Tabula Sapiens**: a multi-organ human reference atlas, available through CELLxGENE.
- Start small with a standard blood dataset (PBMC) to build the pipeline, then scale.

## 3. Background (learn first, intuition and math together)

**Why counts need a special model.** The data are counts with many zeros and variance larger than the mean. A Gaussian is a poor fit. The negative binomial distribution has a mean and a separate dispersion parameter, so it can model that extra variance.

**Batch effects.** Two cells of the same type measured in different labs differ for technical reasons. A good embedding removes that difference while keeping real biological differences. These two goals pull against each other: remove too much and distinct cell types merge.

**The VAE objective.** An encoder maps a cell's counts x to a distribution over a short vector z. A decoder maps z (and the batch label) back to the parameters of a count distribution. Training maximizes:

**ELBO = E[ log p(x | z, batch) ] - KL( q(z | x) || p(z) )**

- First term: reconstruct the counts well.
- Second term: keep the code distribution close to a standard normal, which regularizes the space.
- Giving the batch label to the decoder lets the decoder explain batch differences, so the code z does not have to. That is the mechanism for batch correction.

A check you can do by hand: with the KL weight at 0 the model is a plain autoencoder; with a very large weight the codes collapse to the prior and all cells look the same. Seeing both failure modes is a useful first experiment.

**Masked modeling for the transformer.** Hide a fraction of genes in each cell and train the model to predict them from the rest, as in masked language modeling. The design question is how to represent a cell as a sequence: which genes, in what order, and how to encode expression level (binning, rank ordering or a continuous embedding). Geneformer and scGPT answer differently; read how each does it.

## 4. Design

### 4.1 Methods compared, in increasing complexity
1. Highly variable genes, then PCA
2. PCA plus Harmony
3. A linear autoencoder
4. A VAE with negative binomial likelihood, written by hand in PyTorch
5. The same model from `scvi-tools` (parity check for number 4)
6. A small transformer trained with masked gene modeling on this project's own data
7. Embeddings from one or two pretrained models, used without fine-tuning

### 4.2 Splits (the part most projects get wrong)
- Split by **donor**, never by cell. Cells from one donor are correlated; a random split leaks.
- Hold out one entire **dataset or lab** as a harder test of generalization.
- Hold out one or two **cell types** completely to see what each method does with a type it has never seen.

### 4.3 Evaluation
- scIB biological-conservation metrics (for example NMI, ARI, cell-type silhouette) and batch-correction metrics (for example batch silhouette, graph connectivity), reported separately and as the standard aggregate.
- A k-nearest-neighbor cell-type classifier on frozen embeddings, with accuracy and macro F1 on held-out donors. Macro F1 matters because rare cell types are the hard ones.
- Label efficiency: how accuracy changes with 1%, 10% and 100% of labels.
- Compute cost for each method: training time, memory, parameters.
- Uncertainty: repeat over several seeds and donor splits; report intervals, and say "no detectable difference" when intervals overlap.

### 4.4 Analyses that show understanding of the biology
- Which cell types does every method confuse, and are those types biologically close?
- Do known marker genes line up with directions in the learned space?
- What does each method do with the held-out cell type: place it near its closest relative, or scatter it?
- Where batch correction goes too far: find a pair of real cell types that an aggressive setting merges.

### 4.5 Pitfalls to design against
- Selecting highly variable genes using test donors leaks information. Select on training donors only.
- Normalization choices change results; fix one pipeline and state it.
- Cell-type labels come from the original studies and are inconsistent across datasets. Use the standardized ontology labels in CELLxGENE and report at a coarse and a fine level.
- A pretrained model may have seen the evaluation data during its pretraining. Check each model's training data and note any overlap.
- Do not claim more than the data show. If the VAE and PCA plus Harmony are indistinguishable, that is the finding.

## 5. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 (built 2026-11-17) | Data pipeline on a small blood dataset: load, quality control, normalize, donor-level splits | Counts of cells, donors and cell types reported; split verified to share no donors |
| M1 (built 2026-11-17) | Classical baselines (methods 1 and 2) and the full evaluation harness | scIB metrics and kNN accuracy produced by one script |
| M2 (built 2026-11-17) | Hand-written VAE (method 4) with unit tests for the negative binomial likelihood and the KL term | Loss components match a hand-computed example; KL weight sweep shows both failure modes |
| M3 (built 2026-11-17) | Parity with `scvi-tools` (method 5) | Metrics agree within seed-to-seed noise, or the difference is explained |
| M4 (done 2026-11-18) | Scale up through CELLxGENE Census to a multi-tissue, multi-dataset set; rerun everything | Held-out dataset and held-out cell-type results added |
| M5 (built 2026-11-17; trained on GPU 2026-11-17) | Small transformer with masked gene modeling (method 6) | Trained on a free GPU; added to the comparison |
| M6 (built 2026-11-17) | Pretrained model embeddings (method 7) | Added to the comparison with training-data overlap noted |
| M7 (analyses done 2026-11-18; public release is the owner's step) | Analyses in 4.4, interactive atlas, write-up | README complete, repo ready to make public |
| M9 (started 2026-11-19, beyond the original plan) | Can the classifier be trusted? Linear probe beside kNN; conformal prediction sets calibrated three ways (random cells, held-out donors, held-out labs) and tested on new people and new labs | Coverage, set size and class-conditional coverage reported for every method and fold; the hypotheses below marked supported or not; a figure; a guide section that teaches the method |

Stretch: semi-supervised labels (scANVI style); mapping a new dataset onto a trained reference; applying the pipeline to a tumor dataset, which connects to your earlier glioblastoma work.

## 6. Visuals

- **Hero:** an interactive 3D map of the cells (UMAP of the embedding rendered with WebGL, for example deck.gl or Plotly, hosted on GitHub Pages). Rotate it, color by cell type, donor or the expression of any gene, and switch between methods to watch the same cells rearrange.
- **Before and after batch correction:** the same cells colored by lab, uncorrected vs corrected.
- **The trade-off plot:** batch-correction score against biological-conservation score, one point per method with error bars.
- **Cost vs quality:** training time against kNN macro F1.
- **Confusion matrix** between cell types for the best method.
- **Held-out cell type:** where each method places a type it never saw.
- **KL sweep:** the embedding at three KL weights, from no structure to collapse.

## 7. Compute plan
Baselines and the VAE on a subsample run on a laptop. The full VAE and the small transformer are expected to fit a free Colab or Kaggle GPU; measure and record time and memory. Cache preprocessed data as `.h5ad` files. Keep a `--tiny` config (a few thousand cells) that runs the whole pipeline in minutes.

## 8. Stack
scanpy, anndata, cellxgene-census, scvi-tools (parity only), harmonypy, PyTorch, scib or scib-metrics, scikit-learn, umap-learn, Plotly or deck.gl, pytest.

## 9. Questions you should be able to answer
1. Why is a negative binomial likelihood a better fit than a Gaussian for these data?
2. Write the ELBO and explain each term. What happens at KL weight 0 and at a very large weight?
3. How does giving the batch label to the decoder remove batch effects from the code?
4. Why split by donor, and what would a random split have reported instead?
5. What do batch-correction metrics and biological-conservation metrics each measure, and why can a method score well on one by failing the other?
6. Why macro F1 instead of accuracy?
7. How is a cell turned into a sequence for a transformer, and what are the trade-offs between the approaches?
8. Why can a large pretrained model lose to PCA on this task?
9. What is the reparameterization trick and why is it needed?
10. If your deep model did not beat the baseline, what would you conclude and what would you try next?

## Decision log

**2026-11-17, M0**

- **Dataset for M0.** Healthy-donor PBMCs from Schulte-Schrepping et al. (Cell 2020), CELLxGENE dataset `5e717147-0f75-4de1-8bd2-6fda01b8d75f`: 45,787 cells, 21 donors, 20 cell types, two 10x chemistries. Why: the classic PBMC 3k tutorial data comes from a single donor, so a donor-level split is impossible on it. This dataset is still small, has enough donors for train / val / test, and has standardized ontology labels. A first candidate (Binvignat et al. 2024 controls) was rejected because its cells have a median of only 388 UMIs.
- **Census is used from M0, not only from M4.** It is the simplest source that carries donor ids and ontology labels for every cell. Release pinned to `2025-01-30` (long-term support). Verified against the current package: `cellxgene-census` 1.18.0, `tiledbsoma` 2.3.0.
- **Census opened by S3 location.** The config passes the public S3 URI of the pinned release directly, so a run does not depend on the Census directory web service (which was unreachable from the build machine).
- **Order of steps, chosen to prevent leakage.** (1) Cell quality control with fixed thresholds, (2) drop cell types with fewer than 50 cells, (3) split by donor, (4) gene filter fitted on training donors only, (5) per-cell normalization with a fixed target sum of 10,000, then log(1 + x). Steps that use fixed constants or only a cell's own values can run before the split; any step that estimates something from many cells runs after it, on training donors.
- **Fixed target sum (10,000) rather than the median library size.** The median is a statistic of the dataset, so it would differ between splits and would be a small leak if computed on all donors.
- **QC thresholds:** at least 500 UMIs, at least 200 genes, at most 15% mitochondrial UMIs. They remove 991 of 45,787 cells (231 low counts, 761 high mitochondrial, some both). The data had already been filtered by the original authors, so these are a light safety net.
- **Dropped cell type:** "double negative thymocyte" (40 cells after QC, below the 50-cell minimum).
- **Kept but flagged:** the generic label "lymphocyte" (170 cells) is retained at the fine level and mapped to "unresolved lymphocyte" at the coarse level. Open question for M1: exclude it from classification metrics or not.
- **Coarse labels** are a hand-written mapping of the 20 fine types to 10 lineages in `src/cellmap/labels.py`.
- **Known limits of split seed 0:** both 10x 3' v3 donors are in train, one donor holds a third of the training cells, and some types are very rare in test (3 platelets). Recorded rather than fixed by picking a nicer seed, because choosing the seed by looking at the outcome is its own form of leakage. Repeating over several donor splits (section 4.3) addresses it.
- **Python version.** Developed on 3.13; CI runs 3.12. The plan's "3.11+" still holds.

**2026-11-17, M1**

- **HVG selection:** scanpy "seurat" flavor on log-normalized values, 2,000 genes, ranked per training donor. Per-donor and pooled selection share only 688 of 2,000 genes; pooled selects XIST. Open item: compare against "seurat_v3" on raw counts (needs scikit-misc), because in the owner's DCMvsACM project the HVG method changed results.
- **PCA** is an exact eigendecomposition fitted on training donors (z-score, clip to 10, 50 components), so it has no seed.
- **Harmony is transductive** and is run on all cells with library defaults. This is stated wherever its numbers appear. Verified against `harmonypy` 2.1.0, whose API differs from 0.x.
- **scIB metrics:** `scib-metrics` 0.6.2, individual metric functions rather than `Benchmarker`, whose defaults no longer match the metrics named in section 4.3 (it uses KMeans for NMI and ARI, and a different batch silhouette). Default set: NMI, ARI, cell-type silhouette, cLISI (bio); batch silhouette, graph connectivity, iLISI (batch). kBET is opt-in. No min-max scaling across methods. Computed on test donors only.
- **Macro F1 convention:** averaged over types with at least one test cell; a type in test but not in train scores 0; a type absent from test is excluded.
- **Uncertainty:** five seeded donor splits, reported as paired differences with the count of splits won, not as a confidence interval. Five splits do not support one.
- **scanpy pinned to 1.13 or later** because `highly_variable_genes(batch=...)` replaces the deprecated `batch_key`.
- **Figure style** follows the owner's DCMvsACM `figstyle.py` (same palette and conventions), replacing the first palette used for the M0 figure.
- **Notes and journal** follow the DCMvsACM formats: `docs/NOTES.md` (question, choice, why, alternatives, evidence, weakness, what to say) and `journal/YYYY-MM-DD/README.md`.

**2026-11-17, comparison rules fixed before any deep model result exists**

Written down now, before the VAE has been scored, so the rules cannot be bent toward a result.

- **Same harness for every method:** `cellmap.benchmark.score_embedding`, unchanged, on the same five donor splits (seeds 0 to 4).
- **Metrics that count:** scIB biological conservation and batch correction (the default metric set in `cellmap.evaluate`), and kNN accuracy and macro F1 at the fine and coarse level. No metric is added or dropped after seeing a deep model's score without a dated entry here saying why.
- **What "better" means:** a paired difference against both baselines (HVG + PCA and PCA + Harmony) with the same sign in at least 4 of 5 splits. Anything less is reported as "no detectable difference".
- **Model selection uses validation donors only.** Hyperparameters, epochs and the KL weight are chosen on val donors. Test donors are scored once per configuration that is reported.
- **Every reported number carries its provenance:** the script, the config, the seed, and the commit that produced it, saved next to the number under `results/`.
- **Tool configuration is not edited to make a check pass.** A failing lint rule or test is fixed in the code, not in `pyproject.toml` or the CI file.

**2026-11-17, M2, M3, M5, M6**

- **VAE design matches scvi-tools defaults** (negative binomial, observed library size, per-gene dispersion, 1 hidden layer of 128, 10 latent dimensions, encoder blind to the donor, decoder given the donor), checked in the scvi-tools 1.5.2 source. KL warm-up over 10 epochs instead of scvi-tools' 400, applied identically to both models in M3.
- **KL weight 1** kept for the benchmark: best validation macro F1 in the sweep (0, 0.1, 1, 10, 100), chosen on validation donors only.
- **10-dimensional PCA added** as a comparison at equal size, because the VAE has 10 latent dimensions and the M1 baseline 50.
- **M3 unseen-donor check** relaxed from exact equality to float32 rounding after a run stopped on a 4.8e-7 difference. The partial results committed before the fix were replaced by a full rerun.
- **Run-to-run noise measured:** the same scvi-tools run with 1 and 4 CPU threads differed by 0.074 in scIB bio on one split and by about 0.001 in kNN and batch scores. Recorded as a limit on reading small scIB bio differences.
- **M6 uses published Census embeddings** instead of running the models: release 2025-01-30 has Census scVI, a Geneformer fine-tuned on Census with cell-type supervision, and TranscriptFormer TF-Sapiens. The plan expected zero-shot Geneformer and scGPT; the Census Geneformer is not zero-shot, and scGPT is only published for older Census releases. Both facts are reported with the results.
- **M5 transformer** written by hand: expressed genes as tokens, per-cell rank bins for the level, no positions, masked gene modeling plus a second head that trains the [CLS] embedding. Training on the full data moved to a Colab GPU notebook because one CPU run would take hours per split.
- **Interactive atlas** built as one self-contained HTML file with WebGL, so it works on GitHub Pages without a server.

**2026-11-17, CI and Python version**

- CI had been failing at the install step since scanpy was pinned to 1.13 or later: scanpy 1.13 needs Python 3.13, while CI and Colab run 3.12. Gene selection now works with scanpy 1.11 to 1.13. On the PBMC data, scanpy 1.12.6 on Python 3.12 and scanpy 1.13.0 on Python 3.13 select identical genes, for both per-donor and pooled selection.
- The Colab notebook also failed `ruff check`; its cells were rewritten to pass rather than excluding notebooks from lint.

**2026-11-17, M5 results and the guide**

- The transformer was trained on a Colab T4 GPU (torch 2.14.1, scanpy 1.13.0), 30 epochs, batch size 64, 2,000 genes, five splits. The notebook rebuilt the M0 data from Census and checked it matched before training. Results were pushed to branch `colab/m5-results` and merged.
- The first split's weights are saved (`results/m5/pbmc/model_seed0.pt`, 4 MB) so the embedding can be rebuilt without retraining; the rebuilt embedding reproduced the GPU run's kNN scores exactly.
- Added `docs/GUIDE.md` at the owner's request: one walkthrough of the whole project, kept current after every milestone (rule added to `CLAUDE.md`).

**2026-11-17, M4 design**

- **Six healthy adult blood datasets** from Census 2025-01-30, chosen for different labs and 10x chemistries: Schulte-Schrepping (3' v2 and v3, the M0 dataset), Hao (3' v3), Stephenson (3'), COMBAT (5' v1), AIDA (5' v2, 30 of 508 donors drawn at random) and OneK1K (3' v2, 30 of 981 donors). At most 500 cells per donor, so no dataset or donor dominates. Children excluded, so age does not become a second batch variable.
- **Same tissue, not several tissues** (a change from the plan's "multi-tissue"). The held-out-dataset test needs cell types that every dataset shares; across tissues most types would exist in only one dataset.
- **Shared lineage labels.** The labs label at different levels of detail ("naive B cell" versus "B cell"), so all 75 names are mapped by hand to 12 lineages in `cellmap.labels.LINEAGE`. Names that fit no single lineage ("T cell", "blood cell", "unknown") are dropped. Fine-label scores are not compared across datasets.
- **Genes measured in every dataset only**, from the Census presence matrix. A gene one lab did not measure would otherwise look like zero expression in all of its cells.
- **Two split schemes:** donor splits pooled over all datasets (seeds 0 to 4), and leave one dataset out (six folds; the other datasets' donors split into train and val).
- **scvi-tools is not rerun in M4;** M3 showed it matches the hand-written VAE.

**2026-11-17, public site:** the owner chose one scroll-driven explainer page as the front of the project (the 3D map changes as the reader scrolls from raw data to each method, ending on the results), linking to the existing atlas. Built after M4.

**2026-11-18, M4 results and extras**

- M4 finished. Seven methods on six labs, two schemes (5 donor splits, 6 held-out labs). No method passed the pre-declared rule in either scheme. The transformer was trained on a Colab T4 GPU through the one-cell notebook, which pushed after every fold; a disconnect after 9 folds was resumed without retraining anything.
- Held-out cell type test (B cells, dendritic cells) added on split seed 0 only.
- Two experiments beyond the plan, prompted by preparing for an interview: the inflation from a leaky random-cell split (`results/leakage/`), and an input perturbation probe on the VAE and transformer (`results/probe/`), which showed the transformer ignores expression amounts.
- A data-script bug (missing output folder) only appeared on a fresh clone in Colab. Rule adopted: before handing over a GPU notebook, replay its steps on a fresh clone here.

**2026-11-18, M7**

- Analyses from section 4.4 done on PBMC split seed 0: shared confusions (all nine methods), over-correction (Harmony theta 0 to 16), marker genes versus learned directions. The held-out cell type analysis was done in M4.
- Harmony's theta is now a parameter of `harmony_correct`; the default (2) reproduces every earlier result exactly (checked).
- The transformer got a second, depth-normalized level encoding (`level_encoding="value"`), because the input probe showed the original rank encoding carried almost no level information (57 percent of training tokens in bin 0, `results/m7/pbmc/token_bins.csv`; first quoted as 61 percent from an unsaved check). Tested on one split on CPU.
- README now opens with the atlas (rendered from the real page by `scripts/hero_screenshot.cjs`) and three sentences, as `CLAUDE.md` asks.
- Release checklist in `docs/RELEASE.md`: credentials scan done (none found); license, merge to main, visibility and Pages are the owner's decisions. Pages deploys only by hand.
- Not done from the plan: the cost-versus-quality figure. Timings in `results/` were measured on shared machines (CPU jobs overlapping, one GPU), so they would not support a fair comparison.

**2026-11-18, M8 (beyond the original plan)**

- Added because Harmony was the only transductive method. Reference mapping (Symphony) implemented in `src/cellmap/reference_mapping.py` instead of installing a third-party package: an agent that was told it could install one was blocked by an automatic safety check, and the algorithm is short enough to implement and test.
- Result: on 21 donors about 30 percent of Harmony's scIB lead came from seeing test cells; on six labs none did.

**2026-11-18, independent review of every claim**

- Ten agents that had not written the code reviewed the project: five compared each finding with the literature, four rechecked it (recomputed every README number, reviewed the code for leakage, compared the Symphony implementation with its source, reran from a clean clone), and one hostile reviewer tried to break the conclusions with all their reports in hand. The clean rerun reproduced every rerun result to the last digit and found no leakage in the core pipeline.
- **The comparison rule had been applied to 2 of its 6 declared metrics.** `scripts/verify_rule.py` now applies it to all of them, in both directions, for every method on both datasets. Result: 3 of 68 comparisons pass. The rule itself is unchanged.
- **No threshold had been declared for the held-out-lab scheme (6 folds).** It is reported with the same 80 percent bar (5 of 6) and marked as not pre-declared in `results/verify/rule_table.csv`.
- **The five PBMC splits are not replicates.** Split 3 holds out the v3 chemistry entirely and split 1 tests on a quarter of all cells (`results/verify/pbmc_split_composition.csv`). Kept as they are, since changing splits after seeing results would break the rule above; reported next to every per-split claim. A future dataset should stratify donor splits by assay.
- **Symphony centroids** now follow Symphony's buildReference (they had normalised each cell first). M8 rerun, with the query ridge at 0.1, 1 and 10 on PBMC because harmonypy estimates lambda for the reference.
- Wording corrected wherever it outran the data: the transformer probe, the confusion count, pretrained-model training exposure, the leakage caveats, over-correction, the gene filter, and the 61 percent bin figure (57 percent measured).

**2026-11-18, transformer level encoding result**

- Value bins made the masked-level task far easier to learn but did not improve the embedding (fine macro F1 0.348 against 0.366, one split). The encoding hypothesis is recorded as not supported; repeating it on all splits is deprioritised.

**2026-11-19, M9 design and hypotheses, fixed before any M9 result exists**

Why: every result so far uses one classifier (kNN, k = 15) and reports a single label per cell. M9 asks whether a classifier on these embeddings can say how sure it is, with a guarantee, and whether that guarantee survives new people and new labs, which is the same structure that made leakage matter.

- **Classifiers:** multinomial logistic regression on the standardized embedding (the "linear probe", scikit-learn, L2, C = 1) and kNN (k = 15, class probabilities = neighbour fractions). Both fitted on training cells only.
- **Conformal method:** split conformal at miscoverage alpha = 0.1 (target 90 percent), also 0.05 and 0.2. Primary score: APS (adaptive prediction sets, randomized); secondary: LAC (1 minus the probability of the true label). Class-conditional (Mondrian) conformal as a third variant.
- **Calibration, three ways:** (1) random cells: 20 percent of training-donor cells held out at random, classifier fitted on the rest, so calibration cells share donors with training; (2) held-out donors: the validation donors, never used to fit the embedding or the classifier; (3) held-out labs (lab folds only): each training lab is left out in turn, the classifier refitted on the others, and those scores pooled.
- **Data:** six labs (12 lineages), 5 donor folds and 6 lab folds, embeddings PCA, PCA + Harmony, VAE, Census scVI, Geneformer, TranscriptFormer; PBMC (19 fine types), split seed 0, all nine methods, for set sizes on fine types.
- **Metrics:** marginal coverage on test cells, mean set size, share of singleton sets, coverage per class (worst class with at least 20 test cells), coverage per held-out lab.
- **Hypotheses (each reported as supported or not):**
  - H1: on held-out labs, random-cell calibration under-covers (mean coverage below 0.88 at the 0.90 target).
  - H2: on held-out donors, donor calibration is within 0.02 of the target.
  - H3: on held-out labs, lab calibration covers better than donor calibration (higher worst-lab coverage).
  - H4: on PBMC fine types, the classes with the largest mean set size include the ones in the 18 shared confusions (regulatory T, CD56-bright NK, transitional B).
  - H5: when one type is removed from the classifier's training labels only, its cells get larger sets than known cells under APS, and more empty sets under LAC.
  - H6: method rankings by kNN macro F1 and by linear-probe macro F1 differ for at least one pair of methods.
- **Not claimed in advance:** which method calibrates best.

**2026-11-19, M9 changes after the literature review (before any lab-fold result)**

Seen before this entry: the first PBMC rows (split seed 0, PCA and Harmony) and one smoke-test fold (Hao held out, PCA). No other M9 result.

- **Added hierarchical calibration** (`donor_hier`, `lab_hier`; Lee, Barber and Willett 2023, and Dunn, Wasserman and Ramdas 2023): donors or labs, not cells, are the exchangeable units, which gives a coverage guarantee for a new donor or lab. It needs at least 9 groups at alpha 0.1 and 4 at alpha 0.2, so with 5 training labs it can only be used at alpha 0.2.
- **The lab calibration in the original design is a heuristic** (cross-conformal at the lab level, at best a 1 minus 2 alpha guarantee), and is reported as one.
- **PBMC donor calibration rests on 4 validation donors**, too few for a donor-level guarantee; the PBMC part is used for set sizes, not coverage claims.
- **Novel-type detection has been published** (Lopez-De-Castro et al., Bioinformatics 2025); ours is reported as a replication.
- **H7 (added):** on held-out labs at alpha 0.2, `lab_hier` reaches at least 0.80 coverage in at least 5 of 6 labs for every method.
- The full review, with which sources were read in full and which only as abstracts, is in `docs/M9_LITERATURE.md`.

**2026-11-19, contrastive transformer: design and hypotheses, fixed before any result**

Why: with value bins the masked-gene transformer learned its own task far better, yet its embedding did not improve, which points at the objective. This trains the identical architecture (same tokens, rank level encoding, 2,000 genes, 30 epochs, five PBMC donor splits) with SimCLR instead (`src/cellmap/contrastive.py`, `scripts/transformer_contrastive.py`), on a Colab GPU.

- **Objectives:** `contrastive` (SimCLR only) and `contrastive_mgm` (SimCLR plus the masked-gene loss, weight 1).
- **Fixed settings, not tuned:** temperature 0.1, projection to 64 dimensions, views keep each token with a rate drawn from 0.5 to 0.9 and move 20 percent of levels by one bin, batch 256, learning rate 1e-3. The embedding is the [CLS] state.
- **C1:** `contrastive` beats the masked-gene transformer (`results/m5/pbmc/`) on fine macro F1 in at least 4 of 5 splits.
- **C2:** neither variant passes the comparison rule against both baselines on any declared metric.
- **Reported without a prediction:** the input probes on split seed 0 (does the new objective make the model use expression levels?).

**2026-11-19, M9 results against the hypotheses**

Checked by `scripts/m9_evaluate.py` (`results/m9/hypotheses.json`). H1 supported (0.876), H2 supported (0.901), H3 partly (lab calibration has the higher worst-lab coverage in 7 of 12 embedding and classifier pairs; the test required all 12), H4 supported (10 of 10 methods), H5 partly (22 of 48 cases; empty LAC sets fail for close siblings), H6 supported, H7 supported (every lab at least 0.893).

- **H4 operational test chosen after a first look.** The plan named three types but no test. Before writing the test, one per-type table (linear probe, APS, all methods, with PCA sorted) had been seen. The test chosen compares the mean set size of every type that is the true type in a shared confusion with the other types, rather than a top-k cut, because CD56-bright NK has 5 test cells and regulatory T sits mid-table for PCA. Reported with this caveat.
- **H3 and H5 strictness** was fixed when the evaluation script was written: "every case" for supported, otherwise "partly (k of n)".

**2026-11-19, contrastive ablations: design and predictions, fixed before any ablation result**

Seen before this entry: split seed 0 of the main contrastive run (both objectives, with input probes). No other split, and no ablation.

Why: the main run asks whether the objective matters. The ablations ask why it works and where it is weak, one question per run. All on split seed 0, objective `contrastive`, every setting as in the main run except the one named (`scripts/contrastive_ablations.py`, results in `results/contrastive_ablations/pbmc/`).

| Variant | Changed | Question |
|---|---|---|
| base | nothing (a rerun) | how much does a rerun move the numbers? |
| tau_0.05, tau_0.2, tau_0.5 | temperature | how hard should the loss focus on the closest negatives? |
| keep_strong, keep_weak | keep rate 0.3 to 0.6, or 0.8 to 1.0 | how much corruption teaches the right invariance? |
| no_jitter | no level nudging | does level noise in the views matter? |
| same_donor | every batch drawn from one donor | is donor identity a shortcut for telling cells apart? |
| debiased | debiased loss (Chuang et al. 2020), tau+ = 0.1 | does correcting for same-type negatives help? |

Every run is scored twice: on the [CLS] state h (the embedding used everywhere) and on the projection z. Alignment and uniformity (Wang and Isola 2020) are measured on validation cells.

- **Noise rule:** a difference from base counts only if it exceeds both 0.01 and twice the gap between base and the main run's split seed 0 (fine macro F1 0.506).
- **A1:** h beats z on fine macro F1 in at least 7 of the 9 variants (as SimCLR found for images).
- **A2:** same_donor raises iLISI (donor mixing) over base by at least 0.03.
- **A3:** keep_weak has lower fine macro F1 than base.
- **A4:** tau_0.5 has the lowest fine macro F1 of the four temperatures.
- **Reported without a prediction:** keep_strong, no_jitter, debiased, and alignment and uniformity.

**Working principle (owner's decision, 2026-11-17): full speed, learn afterwards.** The owner chose to have all milestones built as fast as possible, with parallel agents, and to study the material at the end. This replaces "teach before you build" and the per-milestone quiz in `CLAUDE.md` for the build phase. What makes that safe: every decision is written down as it is made (this log and `docs/NOTES.md`), so the reasoning can be read and defended later. The honesty rules are unchanged: no number enters the README unless a script here produced it. Lessons and quizzes move to the end.

**2026-11-20, contrastive ablations: how the predictions are scored (written after seeing only the base run)**

Seen before this entry: the base ablation only (h fine macro F1 0.540, z 0.539, iLISI 0.244). The main run's split seed 0 gave 0.506 with the same seed, code and settings; the training losses differ from the fourth decimal in epoch 0, so the gap is GPU non-determinism, not a code difference. The noise bar is therefore max(0.01, 2 x 0.0345) = 0.069.

The plan did not say how the noise rule meets each prediction. Fixed now, in `scripts/contrastive_ablations_evaluate.py`:
- A3 is a difference from base, so it is supported only if keep_weak is below base by more than the bar; below but within the bar is reported as "direction only".
- A4 compares temperatures with each other: supported only if tau_0.5 is lowest and below each other temperature by more than the bar; lowest but within the bar is "direction only".
- A1 compares h and z inside one run, so the between-run noise rule does not apply; the count is reported as is, with each gap.
- A2 keeps its own threshold (iLISI up by at least 0.03). For scale, the rerun moved iLISI by 0.007.

Consequence, stated before the other results: with a bar of 0.069, only large effects can count on one split. Smaller differences will be reported as within noise, not as findings.

**2026-11-22, model audit: what does the transformer rely on? Design and predictions, fixed before any audit result**

Seen before this entry: the contrastive results above (split seeds 0 to 2, and three ablations), and one exploratory table of raw data, not of any model: mean HLA-DRB5 expression per donor in monocytes and B cells, where six donors (C2P05F, C2P07H, C2P10H, C2P13F, C2P16H, Five) are at 0.0 to 0.2 reads per 10,000 and the other 15 at 1.2 to 12.4. That on/off pattern is consistent with HLA-DRB5 copy-number variation between people; nobody here was genotyped, so the audit calls it "carrier status" and does not claim more. Five of the six low donors are in the C2 cohort, so cohort is a possible confound, and the design has a check for it.

Why: the same_donor ablation showed the contrastive model partly tells cells apart by donor. The audit asks which genes the model relies on, whether they are the known markers, and whether it encodes something about the person rather than the cell. All on split seed 0, CPU only, with the saved weights (`results/contrastive/pbmc/model_contrastive_seed0.pt`, `model_contrastive_mgm_seed0.pt`, and the masked-gene transformer `results/m5/pbmc/model_seed0.pt`). "Knocking out" a gene means setting its count to zero in the model's input and embedding the cell again. It says what the model uses; it predicts nothing about a real cell.

Part 1, gene knockout (`scripts/audit_knockout.py`):
- Cells: split seed 0 test donors; fine types with at least 30 test cells; up to 100 cells per type (numpy seed 0).
- Candidate genes for a type: detected in at least 20 percent of its sampled cells. Types with fewer than 40 candidates are left out.
- Effect of a gene on a type: mean over the sampled cells of 1 - cos(h, h with the gene knocked out); a cell without the gene contributes 0.
- Reference markers: Wilcoxon one-versus-rest on training cells (log of counts per 10,000, the 2,000 genes, scanpy `rank_genes_groups`), ranked among the type's candidates.
- Baseline explanation: candidates ranked by mean log expression in the sampled cells ("the model just uses whatever is highest").
- Score: overlap at 20, the share of a ranking's top 20 that is in the marker top 20.
- **K1:** for the contrastive model, the knockout ranking overlaps the markers more than the expression ranking does, in at least two thirds of the types.
- **K2:** mean knockout overlap over types is higher for the contrastive model than for the masked-gene transformer.
- Reported without a prediction: how often a knockout changes the kNN label, and which highly ranked genes are not markers (HLA, ribosomal, mitochondrial, stress genes such as FOS and JUN, haemoglobin).

Part 2, the HLA-DRB5 test (`scripts/audit_genotype.py`):
- Carrier status per donor: mean of 10,000 x count / total count of HLA-DRB5 over the donor's monocytes and B cells; non-carrier below 0.5.
- Cells: classical monocytes (high HLA class II) and, as a control, naive CD4 T cells (little class II); up to 300 cells per donor and type (numpy seed 0); all 21 donors. The carrier label was never used in training, but 13 donors' cells were, so this tests what the embedding holds, not generalisation.
- Probe: logistic regression on standardised embeddings (C 1.0, balanced classes), leave one donor out; a donor's score is the mean predicted probability over its cells; AUROC over the 21 donors.
- Interventions, each followed by embedding again and refitting the probe: none; HLA-DRB5 knocked out; each of 10 control genes knocked out on its own, drawn at random (seed 0) from the 50 non-HLA genes whose detection rate in carriers' classical monocytes is closest to HLA-DRB5's.
- **G1:** on classical monocytes the contrastive embedding gives AUROC at least 0.80.
- **G2:** knocking out HLA-DRB5 lowers that AUROC by more than the largest drop among the 10 control genes.
- **G3:** on naive CD4 T cells the AUROC is lower than on classical monocytes.
- **G4:** PCA (50 components, the project's usual baseline, the gene zeroed in its input) also gives AUROC at least 0.80 on classical monocytes: the information is in the data, not something only the transformer invents.
- Reported without a prediction: the masked-gene and contrastive-plus-masked-gene transformers; knocking out all HLA genes at once; AUROC within the C2 cohort alone (8 donors), the cohort check.

Part 3, a sparse autoencoder on the transformer's internal states, is designed after parts 1 and 2 and gets its own entry before it runs.

**2026-11-22, model audit results against the predictions** (`scripts/audit_evaluate.py`, `results/audit/pbmc/predictions.json`)

| Prediction | Verdict | Numbers |
|---|---|---|
| K1 knockout beats expression at matching markers, two thirds of types | Not supported | 6 of 14 (4 ties); overlap 0.336 vs 0.339 |
| K2 contrastive overlaps markers more than masked-gene | Supported, negligible | 0.336 vs 0.325 (3 of 280 slots) |
| G1 contrastive monocyte carrier AUROC at least 0.80 | Not supported | 0.756 |
| G2 HLA-DRB5 knockout drop larger than any control | Supported, tiny | 0.022 drop (2 of 90 donor pairs); controls 0.000 |
| G3 naive CD4 T cells lower than monocytes | Supported | 0.321 vs 0.756 |
| G4 PCA at least 0.80 | Supported | 0.933; 0.822 with every HLA gene knocked out |

Changed reading: the transformers' carrier scores sort mostly by cohort (C2 low) and chemistry (the two 10x v3 donors are the worst errors), so the probe mostly learned the confound that the design entry flagged. The donor shortcut seen in the one-donor-per-batch ablation is therefore attributed to cohort and chemistry, not to HLA-DRB5. Lesson for future pre-registrations: a causal knockout test needs a minimum effect size, not only "larger than the controls", because controls can move by exactly zero. Part 3 (sparse autoencoder) is not started; given part 1, single-unit tools on this model need care, and its design entry comes first.

Contrastive main run: split seed 3 is in (contrastive 0.301, masked-gene 0.258), so C1 holds on 4 of 4 splits and is settled whatever split 4 shows. The full write-up waits for split 4.

**2026-11-22, contrastive results against C1 and C2** (`scripts/contrastive_evaluate.py`, `results/contrastive/pbmc/hypotheses.json`)

- **C1 supported:** contrastive beats the masked-gene transformer on fine macro F1 in 5 of 5 splits; mean 0.505 against 0.371. Gaps 0.139, 0.185, 0.126, 0.043, 0.174; the smallest is split 3, the chemistry hold-out.
- **C2 supported:** 0 of 12 comparisons pass the rule; both variants are below both baselines on all six metrics in at least 4 of 5 splits.
- Not changed by the result: the ablations continue as pre-registered on split seed 0.

**2026-11-23, model audit part 3: a sparse autoencoder on the embedding. Design and predictions, fixed before any part 3 result**

Seen before this entry: parts 1 and 2 above. Their reading (the donor signal is cohort and chemistry, not HLA-DRB5) is what part 3 tests from the inside.

Why: a sparse autoencoder (SAE) breaks each 128-number embedding into a sum of a few "features" out of many, each ideally meaning one thing. If the transformer stores cohort or chemistry in a few features, removing those features should mix donors better without losing cell types. All on split seed 0, CPU, `scripts/audit_sae.py`, module `src/cellmap/sae.py`.

- **Input:** the [CLS] embedding h of the contrastive split-0 model for every cell. Centred with the training-cell mean and divided by one number so that training cells have mean squared norm 1.
- **SAE:** 1,024 features (8 times the 128 dimensions). f = ReLU(W_enc (x - b_dec) + b_enc), x_hat = W_dec f + b_dec, decoder columns kept at unit length; loss = mean squared error + lambda x sum of f. Adam, learning rate 1e-3, batch 256, 200 epochs, seed 0, training donors only.
- **Choosing lambda:** grid 0.001, 0.003, 0.01, 0.03, 0.1. Pick the highest explained variance on validation donors among settings with mean active features per cell (L0) at most 20; if none qualifies, the lowest L0. Explained variance = 1 - sum of squared errors / sum of squared deviations from the mean.
- **Labels for features** (training donors): a feature is a "type feature" for a coarse type if its mean activation there is at least 3 times its mean in every other coarse type and it is active in at least 20 percent of that type's cells. Batch groups: C2 cohort (10x v2), other 10x v2 donors, 10x v3 donors. A feature's batch score is the eta squared of its activation on batch group within each fine type with at least 50 training cells in two or more groups, averaged with cell weights. Each feature is also linked to the genes whose log expression correlates most with its activation.
- **Intervention:** subtract the contribution of the 10 features with the highest batch score (chosen on training donors) from every cell's embedding, h' = h - sum over those features of f times its decoder column (rescaled back), and score h' with the usual harness on split seed 0. Control: 10 random sets of 10 live features (seeds 0 to 9).
- **S1:** the chosen SAE explains at least 0.90 of the variance on validation cells.
- **S2:** at least 6 of the 8 coarse types have a type feature.
- **S3:** at least one feature separates C2 donors from the other 10x v2 donors within training classical monocytes with cell-level AUROC at least 0.80 (either direction).
- **S4:** removing the 10 batch features raises test iLISI by at least 0.03, by more than the mean of the random removals, and lowers fine macro F1 by less than 0.069 (the noise bar measured by the rerun).
- Reported without a prediction: dead features, the top genes of the type and batch features, and the edited embedding's other metrics.
