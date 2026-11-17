# cellmap: the guide

**Read this when you feel lost.** It walks through the whole project from zero: what the question is, the background you need, what was built, what was found, what went wrong, and what is left. It is kept up to date as the project moves; the update log is at the bottom.

How this differs from the other documents:

| Document | What it is for |
|---|---|
| `docs/GUIDE.md` (this) | The story of the whole project, start to finish, in plain words. Start here. |
| `docs/NOTES.md` | Decision by decision, in interview format: question, choice, why, alternatives, evidence, "what to say". |
| `docs/PLAN.md` | The original plan, the milestone table, and the dated decision log. |
| `journal/` | What happened each day, in order. |
| `README.md` | The public summary: results and how to reproduce them. |

**If you only have five minutes, read Part 1 and Part 5.**

---

## Part 1. The project on one page

**The question.** When you measure which genes are active in single cells, each cell becomes a list of about 20,000 numbers. To compare cells you want to squeeze that list into a short vector, an "embedding", where cells of the same type sit close together and the lab or person they came from does not matter. Many methods exist, from a 1901 technique (PCA) to large neural networks pretrained on tens of millions of cells. Recent papers claim the big networks often do not beat the simple methods. **This project asks: on real public data, with a fair test, how much do deep models actually add?**

**What makes the test fair.** Every method is trained on some people (donors) and scored on other people it has never seen. Every number in the results comes from the same scoring code. And the rule for "better" was written down before any deep model was scored.

**The answer so far, on one dataset of 44,756 blood cells from 21 people:**

1. **No method beat both simple baselines.** The two baselines are PCA, and PCA followed by a batch-correction step called Harmony.
2. **The VAE written by hand works and matches the reference library** (scvi-tools) within noise. It packs as much cell-type information into 10 numbers as PCA does into 50, but it is worse than Harmony at removing donor differences.
3. **Three large published models (Census scVI, Geneformer, TranscriptFormer) did not beat PCA**, even though all three had seen these very cells during their training.
4. **The small transformer written by hand learned its training task but produced the weakest cell embedding of the project's own models.**
5. **Different metrics disagree.** Harmony wins the integration scores and loses fine-grained cell-type accuracy. So "which method is best" depends on what you measure, and the project reports both.

**Where things stand.** Milestones M0, M1, M2, M3, M5 and M6 are done. M4 (more datasets) and M7 (biology analyses and final write-up) are left. Everything is on GitHub on the branch `claude/clever-hamilton-wj375w`, and CI passes.

---

## Part 2. The background, from scratch

Each section: the idea in words, then the math in plain text, then a small worked example.

### 2.1 What single-cell RNA sequencing measures

A cell copies genes into RNA molecules when it uses them. Single-cell RNA sequencing catches some of those molecules from each cell and counts how many came from each gene. Each caught molecule is tagged with a unique barcode (a UMI) so it is counted once.

The result is a table: one row per cell, one column per gene, each entry a count. Three properties drive almost every decision in this project:

- **It is mostly zeros.** A typical cell here has a few thousand molecules spread over 20,000 genes, so most entries are 0. In this dataset the median cell has 2,543 molecules and 954 genes with at least one.
- **Cells are caught with different efficiency.** One cell might give 1,000 molecules and its identical neighbour 4,000. That difference is technical, not biological.
- **The variance is larger than the mean.** If a gene averages 2 molecules per cell, some cells show 0 and some show 10. A bell curve fits this badly.

### 2.2 What an embedding is, and why you want one

An embedding turns each cell's 20,000 numbers into a short vector, for example 10 or 50 numbers. A good embedding puts cells of the same type close together. Then you can cluster cells, label new ones by their neighbours, or draw a map of them.

The rest of the project is the question: which way of making that short vector is best?

### 2.3 Batch effects

Two identical cells measured in different labs, on different days, or with different chemistry come out different for technical reasons. That is a **batch effect**. Here the main batch is the **donor** (the person), and inside that there is the **chemistry**: 19 donors were sequenced with the 10x 3' v2 kit, and 2 donors with the newer v3 kit.

**Why it matters:** a raw embedding can end up grouping cells by donor or chemistry instead of by type. In this dataset the effect is huge: before correction, v2 and v3 cells of the same type form separate islands (`assets/m1_batch_umap_pbmc.png`).

**The tension:** removing batch differences too aggressively also removes real differences between similar cell types. Every batch-correction method sits somewhere on that trade-off.

### 2.4 Why the data is split by donor

Train on some cells and test on others; that is standard machine learning. The trap: if cells from the same person are on both sides, a model can score well by recognising the person, not the cell type. Cells from one person share genetics, sample handling and sequencing run.

**The rule here:** whole donors go to one side. 13 donors for training, 4 for validation (choosing settings), 4 for testing. The code checks on every run that no donor appears on two sides, and a test proves the check catches a wrong split.

**Why five splits:** with only 21 donors, which 4 are held out changes the scores a lot. So every method is run on five different donor splits (seeds 0 to 4), and results are reported per split.

You made this same decision in your DCMvsACM project ("166,000 cells from 8 patients is still 8 patients").

### 2.5 Normalisation and the log

**Normalise:** scale each cell so its counts add up to 10,000. That removes the capture-efficiency difference from 2.1.

**Log:** replace each value x with log(1 + x). That stops a few very active genes from dominating, and log(1 + 0) = 0 keeps zeros as zeros.

**Worked example.** Cell A has counts (1, 3, 0), total 4. Cell B has (10, 30, 0), total 40. Scaled to a total of 100, both become (25, 75, 0). The depth difference is gone; the proportions remain.

**Why a fixed 10,000 and not "the median total":** the median is a statistic of the whole dataset. Computing it on all donors would leak a little information from test donors into training.

### 2.6 Highly variable genes (HVG)

Most genes barely vary between cells, so they carry no information about cell type. Selecting the 2,000 most variable genes removes noise and speeds everything up.

**The rule that matters:** the selection is computed on training donors only. "Which genes vary" is learned from data, so doing it on all donors would be a leak. You wrote this rule in DCMvsACM; here the function signature enforces it.

**A detail you should know:** genes are ranked inside each training donor and then by how many donors select them. Pooling all donors instead picks genes like XIST, which differs between men and women rather than between cell types.

### 2.7 PCA

PCA finds the directions along which the cells vary most, and describes each cell by its position along the top few directions.

**Math, plainly:** standardise each gene (subtract its mean, divide by its standard deviation, computed on training cells). Compute the gene-by-gene covariance matrix. Its eigenvectors are the directions; the eigenvalues say how much variation each direction captures. Keep the top 50 eigenvectors and project every cell onto them.

**Example:** if two genes always rise and fall together (CD3E and CD3D, both T-cell genes), PCA turns them into one direction, "T-cell-ness", instead of two separate numbers.

**Here:** fitted on training donors, applied to everyone. It is exact (no randomness) and takes about 3 seconds.

### 2.8 Harmony

Harmony takes the PCA coordinates and the donor of each cell. It groups cells into soft clusters while penalising clusters that are dominated by one donor, then nudges each donor's cells toward the shared centre of each cluster. It repeats until things settle.

**The catch, said plainly:** Harmony corrects all the cells it is given at once. It cannot be fitted on training donors and then applied to test donors. So test donors' expression and donor labels take part in the correction (never their cell-type labels). This is called **transductive**, and every Harmony number in the project carries that caveat.

### 2.9 From autoencoder to VAE

**An autoencoder** has two networks. The encoder squeezes a cell into a short code z; the decoder tries to rebuild the cell from z. Training makes the rebuild as accurate as possible. The code z becomes the embedding.

**A variational autoencoder (VAE)** makes two changes.

**Change 1: the code is a distribution.** The encoder outputs a mean and a variance for z, and z is sampled from that. To train through the sampling step, write z = mean + sqrt(variance) * noise, where noise is drawn from a standard normal. The randomness is now an input rather than an operation, so gradients flow through the mean and variance. That is the **reparameterisation trick**.

**Change 2: the loss has two parts.** Training maximises the ELBO:

ELBO = E[ log p(x | z, donor) ] - KL( q(z | x) || N(0, I) )

- **First term, reconstruction:** how likely the real counts x are under the decoder's prediction. Higher is better.
- **Second term, KL:** how far the code distribution is from a standard normal. It keeps the codes tidy and comparable between cells.

**The KL term in closed form**, for a code with means m and variances v:

KL = 0.5 * sum over dimensions of ( m^2 + v - log v - 1 )

**Worked example:** m = (1, 0), v = (1, 4).
- Dimension 1: 0.5 * (1 + 1 - 0 - 1) = 0.5.
- Dimension 2: 0.5 * (0 + 4 - 1.386 - 1) = 0.807.
- KL = 1.307. It is 0 only when m = 0 and v = 1, that is, when the code is exactly the prior.

**Why a negative binomial for the reconstruction:** counts are integers, mostly zero, and their variance exceeds their mean (2.1). The negative binomial has a mean mu and a dispersion theta, with variance = mu + mu^2 / theta.
- **Example:** mu = 2 and theta = 1 gives variance 6. A Poisson with mean 2 would force variance 2.
- **As theta grows,** it turns into the Poisson.

**How batch correction happens inside the VAE.** The decoder is told the donor; the encoder is not. The decoder can then explain donor-specific differences itself, so z does not need to carry them. Because the encoder never sees the donor, a cell from a brand-new donor can still be embedded.

**The KL weight w.** Loss = reconstruction + w * KL.
- **w = 0:** a plain autoencoder. Codes drift anywhere.
- **Very large w:** every code is pushed onto the prior, and every cell gets the same code. This is **posterior collapse**.

The project shows both on the real data (Part 4, M2).

### 2.10 Transformers and masked modelling, for cells

A transformer reads a sequence of tokens. Each token looks at all the others (attention) and updates itself. Language models train by hiding some words and predicting them.

**For a cell, three choices have to be made:**

1. **Tokens = the genes the cell expresses** (count above 0). A cell has roughly 60 to 300 of them among the selected genes. At most 256 are used.
2. **Each token carries the gene's identity and its expression level.** The level is binned inside the cell: genes are ranked by value, and the rank decides one of 10 bins. Ties share a bin.
3. **No order.** Genes have no natural order, so there is no position information. Shuffling a cell's tokens gives the same embedding, and a test checks this.

**Worked example.** A cell expresses gene 7 (count 1), gene 2 (count 5) and gene 9 (count 1).
- Genes 7 and 9 tie at the bottom, so both go in bin 0.
- Gene 2 has two genes below it out of three, so it goes in bin floor(10 * 2/3) = 6.
- Tokens: (gene 2, bin 6), (gene 7, bin 0), (gene 9, bin 0).
- If the cell had been sequenced twice as deeply, the counts double but the ranks do not, so the tokens are identical.

**Training task:** in each cell hide the level of 15 percent of tokens (keep the gene identity) and predict the hidden level. Answering "how active is CD14 here, given everything else this cell expresses" requires learning which genes go together.

**The embedding** is the final state of an extra [CLS] token added to every cell. A second small objective trains it to summarise the whole cell.

**What it lacks:** the donor is never an input, so this model has no batch correction at all.

### 2.11 Pretrained foundation models

Large transformers (Geneformer, scGPT, TranscriptFormer) are pretrained on tens of millions of cells, then used to embed new cells. The anchor paper for this project, Kedzierska et al. (Genome Biology, 2025), found that used without fine-tuning they often do not beat PCA, Harmony or scVI.

**How this project uses them:** CZ CELLxGENE publishes embeddings from these models for every cell in its Census, so the project downloads them rather than running the models.

**Two facts to remember:**
- All of them saw these cells during training.
- The Geneformer embedding that is published was fine-tuned using cell-type labels, so it is not zero-shot.

---

## Part 3. How methods are scored

All scoring lives in `src/cellmap/evaluate.py` and `src/cellmap/benchmark.py`. The code only ever sees an array of numbers per cell, so every method goes through exactly the same path.

### 3.1 kNN classification

For each test-donor cell, find its 15 nearest training-donor cells in the embedding and take a majority vote of their cell types. If 9 neighbours are CD4 T cells and 6 are CD8 T cells, the cell is called CD4. Then compare with the true labels.

### 3.2 Accuracy versus macro F1

**Accuracy** is the fraction of cells labelled correctly.

**Macro F1** computes an F1 score for each cell type and averages them, so every type counts equally however rare it is.

**Worked example.** The test set has 90 monocytes and 10 platelets. The classifier calls everything a monocyte.
- **Accuracy** = 90 / 100 = 0.90. Looks great.
- **Monocyte F1:** precision 0.90, recall 1.0, F1 = 2 * 0.9 * 1.0 / 1.9 = 0.947.
- **Platelet F1** = 0.
- **Macro F1** = (0.947 + 0) / 2 = 0.474. It exposes the failure.

That is why both are reported. In this data classical monocytes are about half of some test splits.

### 3.3 scIB metrics

From the integration benchmark of Luecken et al. (Nature Methods, 2022), in two groups that pull against each other:

| Group | Metrics used here | What they reward |
|---|---|---|
| Biological conservation | NMI and ARI (do clusters match cell types), cell-type silhouette, cLISI | Cell types separate cleanly |
| Batch correction | batch silhouette, graph connectivity, iLISI | Donors are mixed within each cell type |

**Aggregate** = 0.6 * bio + 0.4 * batch. Computed on test donors only.

**Caution learned the hard way:** the bio score moved by up to 0.07 between two runs of the identical model that differed only in CPU thread count. The clustering inside it amplifies tiny numerical differences. kNN scores and the batch score moved by about 0.001. So small differences in scIB bio mean nothing here.

### 3.4 The rule for "better"

Written into the plan before any deep model was scored:

- Same code, same five donor splits.
- **"Better"** means higher than both baselines (HVG + PCA, and PCA + Harmony) in at least 4 of 5 splits. Anything less is reported as "no detectable difference".
- Settings are chosen on validation donors only, never on test donors.

---

## Part 4. What was built, milestone by milestone

### M0. The data pipeline

**Dataset:** healthy-donor blood cells (PBMCs) from Schulte-Schrepping et al. (Cell, 2020), downloaded from CZ CELLxGENE Census release 2025-01-30.
- **Why this one:** the classic tutorial dataset (PBMC 3k) has one donor, so it cannot be split by donor. This one has 21.
- **Rejected candidate:** another dataset with only 388 molecules per cell.

**Steps, in this order:**
1. Drop poor cells: fewer than 500 molecules, fewer than 200 genes, or more than 15 percent mitochondrial RNA (a sign of a damaged cell). This removed 991 cells.
2. Drop cell types with fewer than 50 cells. One type was dropped.
3. Split by donor.
4. Drop genes seen in fewer than 10 training cells (training donors only).
5. Normalise and log.

**The rule behind the order:** a step that uses only fixed numbers, or only one cell's own values, is safe before the split. Anything estimated from many cells happens after it, on training donors.

**Result:** 44,756 cells, 18,002 genes, 21 donors, 19 cell types (10 broad lineages). Train 13 donors, val 4, test 4, zero donors shared.

**Known limits:**
- Both v3-chemistry donors landed in training for split seed 0.
- One donor holds a third of the training cells.
- Some types are tiny in test (3 platelets).

Figure: `assets/m0_split_pbmc.png`.

### M1. The baselines and the scoring code

Built HVG + PCA and PCA + Harmony, plus the kNN and scIB scoring code. Scored on five donor splits.

**Result:** Harmony beat PCA on every scIB group in 5 of 5 splits (aggregate +0.077), but lowered fine-label kNN accuracy in 5 of 5 (-0.037).
- **Likely reason** (a hypothesis): pulling donors together also blurs the small differences between close subtypes, such as naive and memory CD4 T cells.

Figures: `assets/m1_tradeoff_pbmc.png`, `assets/m1_batch_umap_pbmc.png`.

### M2. The hand-written VAE

Built a negative binomial VAE (`src/cellmap/vae.py`) with 10 latent dimensions, matching scvi-tools' defaults. Also built a linear autoencoder and a 10-dimensional PCA, for a comparison at equal size.

**KL sweep, on validation donors:**

| KL weight | KL per cell | Latent dimensions in use | kNN macro F1 |
|---|---|---|---|
| 0 | 52.3 | 10 of 10 | 0.591 |
| 0.1 | 22.1 | 10 of 10 | 0.614 |
| 1 | 7.3 | 10 of 10 | 0.638 |
| 10 | 1.5 | 4 of 10 | 0.503 |
| 100 | 0.0 | 0 of 10 | 0.258 |

- **Weight 0:** a plain autoencoder. Codes drift far from the prior.
- **Weight 100:** full collapse. Every cell gets the same code.
- **Weight 1** was best and is used.

Figure: `assets/m2_kl_sweep_pbmc.png`.

**Benchmark result:**
- **Versus 50-dimensional PCA:** no detectable difference on kNN, and a small batch-correction gain in 4 of 5 splits.
- **Versus Harmony:** lower on every scIB score in 5 of 5.
- **Versus 10-dimensional PCA:** much better macro F1 (+0.121, 5 of 5).
- **The linear autoencoder tied 10-dimensional PCA,** as theory predicts. That confirms the comparison is wired correctly.

### M3. Checking the VAE against scvi-tools

Trained scvi-tools with every setting matched, on the same splits.

**Evidence the two are the same model:**
- Copying scvi-tools' trained weights into the hand-written model gives identical embeddings.
- Single training steps agree to a millionth.
- On five splits, no metric differs by more than 0.01 on average, and none consistently in one direction.

**One difference found on small data:** a 0.5 percent reconstruction gap, explained by how the per-gene dispersion is initialised. Swapping in scvi-tools' initialisation removed it.

### M5. The hand-written transformer

Built in `src/cellmap/transformer.py` (about 1.1 million parameters). Trained on a Colab T4 GPU through `notebooks/m5_transformer_gpu.ipynb`, 30 epochs per split, 6 to 11 minutes per split. Before training, the notebook rebuilt the data from Census independently and checked it matched M0 exactly.

**Result:**
- **It learned its task:** on validation donors it predicted hidden expression levels 4.5 to 8.6 points better than guessing each gene's usual level.
- **But its embedding was weak:** kNN accuracy 0.745 and macro F1 0.371, against 0.893 and 0.664 for PCA. It was lower than PCA, Harmony and the VAE on every metric in all 5 splits.
- **Training had levelled off:** validation loss changed by less than 0.01 over the last 10 epochs, partly because the learning rate decays to zero by design.

**Possible reasons, untested:**
- Being good at predicting hidden genes does not force the [CLS] summary to organise cells by type.
- The model has no donor correction.
- About 60 percent of nonzero counts fall into the lowest bin, so the levels carry little detail.
- 34,000 training cells is tiny by transformer standards.

Figure: `assets/m5_transformer_pbmc.png`.

### M6. Published pretrained embeddings

Downloaded Census scVI (50 dimensions), the fine-tuned Geneformer (512) and TranscriptFormer TF-Sapiens (2048) for these exact cells, matched by Census cell id with a check on donor and type.

**Result:** none beat plain PCA on any metric by the rule.
- **TranscriptFormer** tied PCA on kNN accuracy and was lower on every scIB score.
- **Geneformer** was lower than PCA on every metric in all 5 splits, despite having been fine-tuned with cell-type labels.

### The explainer page and the interactive atlas

**The explainer, `site/index.html`, is the front page.** A scroll-driven story: the 3D map on the right changes as you read, from plain PCA, to the chemistry batch effect, to Harmony fixing it, to the held-out-people test, to each deep model, then the results charts. Every number on it is read from `results/` when the page is built (`scripts/explainer_export.py`), including the headline verdict, which applies the "4 of 5 splits against both baselines" rule automatically.

**The atlas, `site/atlas.html`,** also at https://claude.ai/artifact/Xyki85zGs4NfqMrVcKTRBA.
- A 3D map of all cells that you can rotate.
- Recolour by cell type, donor, chemistry, split or 20 marker genes.
- Switch between nine methods and watch the same cells move.

**Try:** colour by chemistry, then switch from HVG + PCA to PCA + Harmony.

---

## Part 5. All results in one table, and what they mean

Means over five donor splits, test donors (`results/m1` to `results/m6`). Sorted by scIB aggregate.

| Method (dimensions) | kNN accuracy | kNN macro F1 | scIB bio | scIB batch | scIB aggregate |
|---|---|---|---|---|---|
| PCA + Harmony (50) | 0.856 | 0.657 | 0.801 | 0.662 | 0.746 |
| scvi-tools, matched (10) | 0.884 | 0.646 | 0.744 | 0.599 | 0.686 |
| VAE, hand-written (10) | 0.879 | 0.646 | 0.737 | 0.600 | 0.682 |
| HVG + PCA (50) | 0.893 | 0.664 | 0.735 | 0.569 | 0.669 |
| Census scVI (50) | 0.841 | 0.533 | 0.746 | 0.550 | 0.668 |
| HVG + PCA (10) | 0.846 | 0.525 | 0.733 | 0.527 | 0.651 |
| Linear autoencoder (10) | 0.842 | 0.519 | 0.730 | 0.530 | 0.650 |
| Geneformer, fine-tuned (512) | 0.719 | 0.367 | 0.643 | 0.484 | 0.580 |
| TranscriptFormer (2048) | 0.892 | 0.629 | 0.662 | 0.442 | 0.574 |
| Transformer, hand-written (128) | 0.745 | 0.371 | 0.554 | 0.481 | 0.525 |

**What we learned, in order of importance:**

1. **Simple methods are hard to beat.** Nothing beat both baselines. This agrees with the anchor paper, now under a stricter test: held-out donors, and every learned model fitted on training donors only.
2. **"Best" depends on the metric.** Harmony tops integration; plain PCA tops fine-grained kNN. A single combined score would hide that.
3. **The chemistry batch effect is the biggest structure in the raw data.** Correcting it is what Harmony is good at.
4. **The hand-written VAE is correct.** It agrees with scvi-tools, so its results are not an implementation bug.
5. **Pretraining on huge data did not help here,** even with the advantage of having seen these cells.
6. **A transformer trained from scratch on 34,000 cells is not competitive.** Learning to predict hidden genes is not the same as learning a good cell map.
7. **Noise is real.** scIB bio moved 0.07 from thread count alone. Counting wins across splits is more trustworthy than comparing means.
8. **Split seed 3 is hard for everyone.** Every method dropped there: PCA kNN accuracy 0.832, Harmony 0.751, VAE 0.754, transformer 0.555. Its test donors include one of the two v3-chemistry donors, but so do split seed 1's, and seed 1 is not hard (PCA 0.903). So chemistry alone does not explain it. The reason is not known yet.

---

## Part 6. Things that went wrong and how they were fixed

Kept here because "what went wrong and how you caught it" is often the best thing to talk about.

| What happened | How it was caught | Fix |
|---|---|---|
| First dataset candidate was too shallow (388 molecules per cell) | Inspected before writing code | Chose another dataset |
| The scvi-tools run stopped on split 3: a check demanded an exactly zero difference and got 0.0000005, which is float32 rounding | The run crashed | Check now allows rounding. The partial results had been committed under a message saying five splits; the next commit says so, and the full run was repeated |
| CI had failed at install since early on: scanpy 1.13 needs Python 3.13, CI and Colab use 3.12 | Read the CI logs | Code supports scanpy 1.11 to 1.13; both versions pick identical genes |
| The Colab notebook failed lint, and would have taken the commit author from the wrong commit | Lint on the whole repo | Rewrote the cells |
| A claim in chat that the transformer was last on every metric | Rechecked against the numbers while making its figure | It is last on scIB aggregate, second to last on kNN (above Geneformer). Corrected here and in the figure title |
| The five M1 splits reuse the gene filter fitted on split seed 0's training donors | Noted while writing the notes | Effect judged negligible (a "detected in 10 cells" filter); recorded as a known weakness |

---

## Part 7. What is left

**M4. More data (in progress).** Six healthy blood datasets from different labs and chemistries, at most 500 cells per donor, labels mapped to 12 shared lineages, only genes every lab measured. Every method is rerun under two schemes, plus a held-out cell type test:
- **Donor splits across all six labs** (five seeds).
- **A held-out dataset:** a whole lab never seen in training (six folds, one per lab).
- **A held-out cell type:** a type removed from training entirely, to see where each method puts it.

This answers whether the findings above are specific to one dataset.

**M7. Analyses and write-up:**
- Which cell types every method confuses, and whether those pairs are biologically close.
- Whether marker genes line up with directions in the embeddings.
- Where batch correction goes too far.
- A cost-versus-quality figure.
- The final README, which opens with one strong visual and three sentences.
- The checklist before making the repo public. One item: `harmonypy` is GPL-licensed, which affects the license choice.

**The public site.** You chose one scroll-driven explainer page as the front of the project: the 3D map changes as the reader scrolls from raw data through each method, ending on the results, with the atlas linked. It comes after M4.

**Open checks worth doing first:**
- Does the "seurat_v3" gene-selection method change the scores? In DCMvsACM it mattered.
- Does a larger VAE code (more than 10 dimensions) change the comparison with Harmony?
- Label efficiency (how scores change with 1 percent or 10 percent of labels) for all methods, not just the baselines.

**Learning:** you chose "build at full speed, learn afterwards". The quiz questions are at the end of `docs/PLAN.md`. Part 2 of this guide plus `docs/NOTES.md` cover all ten.

---

## Part 8. Map of the repository

| Path | What it is |
|---|---|
| `configs/` | Dataset settings: Census release, filters, split fractions |
| `src/cellmap/census.py` | Downloads cells from Census |
| `src/cellmap/qc.py`, `splits.py`, `preprocess.py`, `labels.py` | The M0 pipeline pieces |
| `src/cellmap/baselines.py` | HVG selection, PCA, Harmony |
| `src/cellmap/evaluate.py`, `benchmark.py` | All scoring |
| `src/cellmap/vae.py`, `linear_ae.py` | M2 models |
| `src/cellmap/scvi_baseline.py` | M3 scvi-tools with matched settings |
| `src/cellmap/transformer.py` | M5 model |
| `src/cellmap/census_embeddings.py` | M6 downloads |
| `src/cellmap/plotting.py`, `provenance.py` | Figure style; commit stamping of results |
| `scripts/m0_...` to `scripts/m6_...` | One script per milestone step; each writes to `results/` |
| `scripts/*_figures.py`, `assets/` | Figures |
| `src/cellmap/atlas.py`, `scripts/atlas_export.py`, `scripts/explainer_export.py`, `site/` | The explainer front page and the atlas |
| `notebooks/m5_transformer_gpu.ipynb` | GPU training on Colab |
| `results/` | Every number reported anywhere, with the commit that made it |
| `tests/` | 123 tests; run `pytest -q` |
| `data/` | Downloaded and processed data, embeddings. Not in git; rebuilt by the scripts |

---

## Part 9. Glossary

- **ARI / NMI:** scores for how well clusters match known labels (1 = perfect).
- **Batch effect:** a technical difference between groups of cells (donor, lab, chemistry).
- **CLS token:** an extra token whose final state is used as the whole-cell summary.
- **Donor:** the person a sample came from. The unit of splitting.
- **ELBO:** the VAE training objective, reconstruction minus KL.
- **Embedding:** a short vector describing a cell.
- **HVG:** highly variable genes.
- **KL divergence:** how different two distributions are. In a VAE, the distance of a cell's code from the prior.
- **kNN:** k nearest neighbours; here 15.
- **Leakage:** test information reaching training, which makes scores look better than they are.
- **Macro F1:** the average of per-type F1 scores; every type counts equally.
- **Negative binomial:** a count distribution whose variance can exceed its mean.
- **PBMC:** peripheral blood mononuclear cells (blood immune cells).
- **Posterior collapse:** a VAE whose codes ignore the input, so every cell gets the same code.
- **scIB:** the integration benchmark whose metrics are used here.
- **Transductive:** a method that must see the test data to process it (Harmony).
- **UMAP:** a method for drawing high-dimensional points in 2D or 3D. A picture, not a measurement.
- **UMI:** a molecular barcode, so each RNA molecule is counted once.
- **Zero-shot:** using a pretrained model as is, without training it on the task.

---

## Update log

- **2026-11-17 (night):** Explainer page built as the front page (`site/index.html`); the atlas moved to `site/atlas.html`.
- **2026-11-17 (night):** M4 design recorded in Part 7; public site choice recorded.
- **2026-11-17:** Guide created after M0, M1, M2, M3, M5 and M6. The transformer's GPU results are included.
