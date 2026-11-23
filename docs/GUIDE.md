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

1. **Almost nothing beat both simple baselines, and nothing did on preserving biology.** The two baselines are PCA, and PCA followed by a batch-correction step called Harmony. Applied to all six metrics the rule names, 1 of 50 comparisons passed here (scvi-tools on fine kNN accuracy, with a mean still below PCA). An earlier version of this guide said "no method", from a check that covered only two metrics; an independent review caught it (Part 6).
2. **The VAE written by hand works and matches the reference library** (scvi-tools) within noise. It packs as much cell-type information into 10 numbers as PCA does into 50, but it is worse than Harmony at removing donor differences.
3. **Three large published models (Census scVI, Geneformer, TranscriptFormer) did not pass the rule against PCA.** Census scVI was trained, without labels, on Census cells that include most of these; the Geneformer was fine-tuned with labels on CELLxGENE data that may include them; for TranscriptFormer this was not checked.
4. **The small transformer written by hand learned its training task but produced the weakest cell embedding of the project's own models.**
5. **Different metrics disagree.** Harmony wins the integration scores and loses fine-grained cell-type accuracy. So "which method is best" depends on what you measure, and the project reports both.

**Where things stand.** The benchmark, the follow-up analyses, the inductive Harmony comparison and an independent review of every claim are done and merged into `main`. Conformal prediction (section 2.12) is done: the coverage promise holds for new donors and erodes for a new lab (0.876, worst lab 0.829), and lab-level hierarchical calibration restores it at the cost of sets about twice as big. The transformer retrained with a contrastive objective is done: fine macro F1 0.505 against 0.371 for the masked-gene objective, higher on 5 of 5 splits, and still below PCA (0.664) and Harmony on every declared metric (C1 and C2 supported; README, `assets/contrastive_pbmc.png`). In progress: six of nine ablations remain on Colab. The model audit (Part 4, "What does the transformer rely on?") is done: the transformer relies on genes roughly in order of how highly they are expressed, not on how specific they are, and its apparent knowledge of HLA-DRB5 carrier status is mostly a cohort difference, not the gene. A sparse autoencoder finds clean cell-type features but no batch feature, and deleting the most batch-linked features does not mix donors. Nine one-change ablations on split 0 (temperature, view strength, level noise, one donor per batch, a debiased loss, a rerun for noise) are built, with predictions A1 to A4 fixed in `docs/PLAN.md` before any ran (`scripts/contrastive_ablations.py`, `notebooks/contrastive_ablations_gpu.ipynb`). Making the repository public is the owner's step (`docs/RELEASE.md`).

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

The project shows both on the real data (Part 4, "The hand-written VAE").

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

### 2.12 Conformal prediction: a classifier that says how sure it is

**The problem.** A classifier gives one label per cell. When it is wrong, nothing tells you. A doctor or a lab annotating new data wants something better: "this cell is a monocyte" when the model is sure, "regulatory T or memory CD4 T, check by hand" when it is not, and a promise about how often the true answer is in what it says.

**The idea.** Instead of one label, return a *set* of labels, built so that the true label is in the set at least 90 percent of the time (or any rate you pick, written 1 - alpha; alpha = 0.1 means 90 percent). Easy cells get a set of one label; hard cells get two or three. The promise holds whatever the classifier is: a bad classifier gets big sets, not broken promises.

**The recipe (split conformal), in four steps.**
1. Train any classifier that outputs probabilities p(y | x) on training cells.
2. Take a separate set of n *calibration* cells with known labels. For each, compute a "strangeness" score of its true label. The simplest (LAC): s = 1 - p(true label | cell). A confident correct prediction gives s near 0; a confident wrong one gives s near 1.
3. Sort the n scores and take q = the ceil((n + 1)(1 - alpha))-th smallest. That is roughly the 90th percentile, nudged up by the "+ 1".
4. For a new cell, include every label y whose score 1 - p(y | cell) is at most q, that is, every label with probability at least 1 - q.

**Worked example 1 (the arithmetic).** n = 9 calibration cells, alpha = 0.1. Their scores, sorted: 0.02, 0.05, 0.05, 0.10, 0.12, 0.20, 0.31, 0.45, 0.70. Rank = ceil(10 x 0.9) = 9, so q = the 9th smallest = 0.70. A new cell with probabilities monocyte 0.80, dendritic 0.15, B 0.05: include labels with p >= 1 - 0.70 = 0.30, so the set is {monocyte}. Another cell with CD4 T 0.55, regulatory T 0.40, CD8 T 0.05: the set is {CD4 T, regulatory T}. The model has told you which cell it is unsure about, and between which two types.

**Why it works (the one-line proof).** If the calibration cells and the new cell are *exchangeable* (shuffling them changes nothing, as with independent draws from the same population), the new cell's score is equally likely to land in any of the n + 1 positions in the sorted list. So the chance it lands at or below position ceil((n + 1)(1 - alpha)) is at least 1 - alpha. Nothing about the classifier was used, which is why the promise holds for any classifier.

**The catch, and the whole point of the uncertainty work.** "Exchangeable" is the assumption that does all the work. Cells from the same person or lab are more alike than cells from different ones. If the calibration cells come from the same people the classifier was trained on, but the test cells come from new people or a new lab, the new cells look stranger than anything in calibration and the promise quietly breaks. This is the same structure that made random cell splits leak (Part 4, two extra experiments).

**Worked example 2 (groups break pooling).** Suppose each lab shifts every score by its own amount: lab A's cells score around 0.1, lab B's around 0.3, and a new lab could be anywhere. Pool 1,000 cells from 5 labs and take the 90th percentile: it is set by those 5 labs, but it is computed as if you had 1,000 independent draws. A simulation in `tests/test_conformal.py` shows the cost: with 10 labs, pooled calibration covers a new lab about 88 percent of the time instead of 90; with 5 labs, about 84.

**The fix: treat groups, not cells, as the units (hierarchical conformal).** Give each calibration lab the same total weight, 1/(K + 1) for K labs, split across its cells, and put the remaining 1/(K + 1) at "infinity" for the new lab you have never seen. Take the 90th percentile of that weighted list. Now the promise is about a new cell from a new lab. The price is that you need enough labs: the threshold is finite only if K/(K + 1) >= 1 - alpha, so at least 9 labs for 90 percent and 4 for 80 percent. With 5 training labs, an honest 90 percent promise for a new lab is impossible from this data, which is itself a finding.

**Worked example 3 (the group rule in numbers).** K = 5 labs, alpha = 0.2. Each lab carries weight 1/6. The new lab's 1/6 sits at infinity, so the real labs carry 5/6 = 0.83 >= 0.80, and the threshold is finite: walk up the scores from smallest to largest, adding each cell's weight, and stop where the running total first reaches 0.80. At alpha = 0.1, 0.83 < 0.90, so the threshold is infinite and the set is every label: the method refuses to promise what the data cannot support.

**Two kinds of score.** LAC (1 - p) gives the smallest sets on average but can give an empty set when no label is probable enough, which is a useful "looks like nothing I know" signal. APS adds labels in order of probability until their total reaches q, so uncertain cells get bigger sets; a random tie-breaker makes coverage exact rather than conservative.

**Per class.** The basic promise is on average over cells, so common types can be over-covered while a rare type is under-covered. Class-conditional ("Mondrian") conformal computes q separately for each type, at the cost of needing enough calibration cells of each.

**What we found** (`results/m9/hypotheses.json`, figure `assets/m9_conformal_m4_blood.png`). Calibrated on held-out donors, coverage on new donors is 0.901. Calibrated on cells from the training people, coverage on a new lab is 0.876 on average and 0.829 on the worst lab (Stephenson). Lab-level hierarchical calibration at 80 percent covers every lab at 0.89 or more, with sets of 1.76 types instead of 0.94; at 90 percent it returns every label, as worked example 3 predicts. Types involved in the shared confusions get larger sets for all 10 methods. An empty LAC set flags an unseen type when it is distinct (dendritic cells: 16 percent empty against 6 percent for known cells) but not when it is a close sibling of a known type (regulatory T, CD56-bright NK). Five of the seven pre-registered hypotheses were supported, two partly; `docs/NOTES.md` has the table.

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

**Caution learned the hard way:** the bio score moved by 0.07 (one comparison, one split) between two runs of the identical model that differed only in CPU thread count. The clustering inside it amplifies tiny numerical differences. kNN scores and the batch score moved by about 0.001. So small differences in scIB bio mean nothing here.

### 3.4 The rule for "better"

Written into the plan before any deep model was scored:

- Same code, same five donor splits.
- **"Better"** means higher than both baselines (HVG + PCA, and PCA + Harmony) in at least 4 of 5 splits. Anything less is reported as "no detectable difference".
- Settings are chosen on validation donors only, never on test donors.

---

## Part 4. What was built, step by step

### The data pipeline

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

### The baselines and the scoring code

Built HVG + PCA and PCA + Harmony, plus the kNN and scIB scoring code. Scored on five donor splits.

**Result:** Harmony beat PCA on every scIB group in 5 of 5 splits (aggregate +0.077), but lowered fine-label kNN accuracy in 5 of 5 (-0.037).
- **Likely reason** (a hypothesis): pulling donors together also blurs the small differences between close subtypes, such as naive and memory CD4 T cells.

Figures: `assets/m1_tradeoff_pbmc.png`, `assets/m1_batch_umap_pbmc.png`.

### The hand-written VAE

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

### Checking the VAE against scvi-tools

Trained scvi-tools with every setting matched, on the same splits.

**Evidence the two are the same model:**
- Copying scvi-tools' trained weights into the hand-written model gives identical embeddings.
- Single training steps agree to a millionth.
- On five splits, no metric differs by more than 0.01 on average, and none consistently in one direction.

**One difference found on small data:** a 0.5 percent reconstruction gap, explained by how the per-gene dispersion is initialised. Swapping in scvi-tools' initialisation removed it.

### The hand-written transformer

Built in `src/cellmap/transformer.py` (about 1.1 million parameters). Trained on a Colab T4 GPU through `notebooks/m5_transformer_gpu.ipynb`, 30 epochs per split, 6 to 11 minutes per split. Before training, the notebook rebuilt the data from Census independently and checked it matched the original data pipeline's output exactly.

**Result:**
- **It learned its task:** on validation donors it predicted hidden expression levels 4.5 to 8.6 points better than guessing each gene's usual level.
- **But its embedding was weak:** kNN accuracy 0.745 and macro F1 0.371, against 0.893 and 0.664 for PCA. It was lower than PCA, Harmony and the VAE on every metric in all 5 splits.
- **Training had levelled off:** validation loss changed by less than 0.01 over the last 10 epochs, partly because the learning rate decays to zero by design.

**Possible reasons, untested:**
- Being good at predicting hidden genes does not force the [CLS] summary to organise cells by type.
- The model has no donor correction.
- About 57 percent of training tokens fall into the lowest bin (`results/m7/pbmc/token_bins.csv`; 2.10 bits of the possible 3.32), so the levels carry less detail than ten even bins would.
- 34,000 training cells is tiny by transformer standards.

Figure: `assets/m5_transformer_pbmc.png`.

### Published pretrained embeddings

Downloaded Census scVI (50 dimensions), the fine-tuned Geneformer (512) and TranscriptFormer TF-Sapiens (2048) for these exact cells, matched by Census cell id with a check on donor and type.

**Result:** none beat plain PCA on any metric by the rule.
- **TranscriptFormer** tied PCA on kNN accuracy and was lower on every scIB score.
- **Geneformer** was lower than PCA on every metric in all 5 splits, despite having been fine-tuned with cell-type labels.

### Six labs, with whole labs held out

**What was built:** six healthy blood datasets from different labs and 10x chemistries, 63,100 cells from 128 donors, at most 500 cells per donor, only genes all six labs measured. Labs name cell types differently, so every label was mapped to one of 12 shared lineages (CD4 T, CD8 T, B, monocyte, and so on), and cells with labels too vague to place were dropped.

**Two tests per method:**
- **Held-out people:** test donors come from labs also seen in training (5 splits).
- **Held-out lab:** a whole lab is never seen in training (6 folds, one per lab).

**Result, kNN macro F1 over 12 lineages** (figure: `assets/m4_lab_holdout_m4_blood.png`):

| Method | Held-out people | Held-out lab |
|---|---|---|
| HVG + PCA | 0.776 | 0.765 |
| PCA + Harmony | 0.749 | 0.761 |
| VAE (hand-written) | 0.787 | 0.761 |
| Census scVI | 0.751 | 0.711 |
| Geneformer (fine-tuned) | 0.699 | 0.677 |
| TranscriptFormer | 0.730 | 0.732 |
| Transformer (hand-written) | 0.641 | 0.640 |

- **By the rule, two comparisons passed, both on batch correction for held-out people:** the hand-written VAE (above PCA and Harmony in 5 of 5 splits) and Census scVI (5 of 5 and 4 of 5). Nothing passed on biology, accuracy or macro F1, and nothing on held-out labs. (`results/verify/rule_table.csv`.)
- **The VAE improved relative to the baselines with more data:** on held-out people its macro F1 is above PCA in 3 of 5 splits and Harmony in 4 of 5, short of the rule.
- **On an unseen lab,** PCA, Harmony and the VAE come out close (about 0.76); the pretrained models stay behind; the hand-written transformer is last (0.64), as it was on one dataset.

**Held-out cell type:** B cells removed from training still form their own group in every method and are called plasma cells, their closest relatives. Dendritic cells are called monocytes, also their closest relatives, and lose some grouping, the VAE the most. One split, so a lead rather than a finding.

Details and caveats: `docs/NOTES.md`, section "Six labs instead of one".

### Two extra experiments (2026-11-18)

**How much a leaky split inflates scores** (`scripts/leakage_splits.py`, `results/leakage/splits.csv`). Same method (HVG + PCA + kNN), split three ways. On one dataset with 19 fine types, splitting cells at random gave macro F1 0.776 against 0.664 when whole people were held out: 0.11 too optimistic, mostly on rare types, and it hid how much results vary between splits. On six labs with 12 broad lineages the gap was 0.007 against held-out people (0.018 against a held-out lab). Caveat from the review: most of the 0.11 comes from splits 0 and 3, and split 3 is a chemistry hold-out, so part of it is distribution shift, not leakage; and the two datasets differ in labels, chemistry and donor count at once, so the contrast is descriptive, not proof that donor count is the cause.

**Which inputs the models use** (`scripts/probe_inputs.py`, `results/probe/pbmc/probe_inputs.csv`). Keep the trained model fixed, damage one part of every cell's input, score again. The transformer scored the same with only "which genes are on" (0.375) as with the real input (0.366), but shuffling the amounts within each cell cut it to 0.164. So it does read the amounts; the true amounts just added nothing measurable over presence alone. (An earlier version said it "ignores" them; the shuffle result contradicts that. The binning diagnosis is a hypothesis that the later re-encoding run tests.) The VAE uses both presence and amounts (0.533 with presence only, 0.614 real). Scrambling gene identities destroys both, so both do use their main input. The idea comes from Bindwell's "The Bitter Lesson for Biology" (2026) and Fang et al. (2026); the Fang et al. reference has not been verified.

### Follow-up analyses

**Every method makes the same mistakes.** 18 confusions appear in all nine methods (the figure shows the 8 largest; an earlier version said "eight", which was the figure's display limit). The largest are between close relatives (regulatory T cells called central memory CD4 T cells, transitional B cells called memory B cells, and so on). The hardest boundaries are in the data and labels, not in any model. Figure: `assets/m7_confusions_pbmc.png`.

**Over-correction, measured.** Turning Harmony's strength from 0 to 16 left batch mixing nearly unchanged (0.651 to 0.660) while fine-type F1 fell steadily from 0.601 to 0.536 (0.590 at the default, theta 2). The aggregate score peaks near the default, so the over-correction is above the default and mostly at theta 16. One split only (seed 0, where Harmony's lead is smallest). Figure: `assets/m7_overcorrection_pbmc.png`.

**Marker genes.** PCA's first component is "monocyte versus lymphocyte" (LYZ, Spearman 0.79); the VAE has one equally clean direction and spreads the rest across dimensions. Figure: `assets/m7_markers_pbmc.png`.

**The transformer fix, tested: it did not help.** Retrained with value bins (each bin 10 percent of tokens, against 57 percent in one bin before), the transformer got much better at its own training task (validation accuracy 0.780 against a 0.270 guess-the-usual-level baseline; before, 0.749 against 0.682) but its cell embedding did not improve (fine macro F1 0.348 against 0.366), and presence-only input still scored higher (0.418). The hypothesis that the encoding was the bottleneck is not supported: getting levels right is not the same as learning cell types. One split, one seed. `results/m7/pbmc/transformer_levels.csv`.

### Does Harmony need to see the test cells?

Harmony was the only method that corrected the test cells together with the training cells. To check whether that was an unfair advantage, Harmony was re-run fitted on training donors only, with held-out cells mapped onto the frozen result (the published Symphony method, implemented in `src/cellmap/reference_mapping.py`).

- **21 donors:** the fair version scored lower in 5 of 5 splits, losing 0.024 of Harmony's 0.077 integration lead over PCA on average. The share is unstable (3 to 50 percent per split, 22 to 31 percent on average across ridge settings) and mostly in the biology score. Because the algorithm changes too, read it as "part of the lead depends on seeing test cells", not an exact amount. (Corrected after the review; an earlier version said "about 30 percent came from seeing the test cells".)
- **128 donors, six labs:** on held-out people the fair version was slightly higher (4 of 5), so none of the lead depended on it. On held-out labs Harmony led PCA by only 0.006, so that case says little.
- **Everywhere:** the fair version had slightly higher kNN accuracy, but not higher macro F1. Weaker correction raises accuracy here, so this is not evidence that it is better.

Details: `docs/NOTES.md`, section "Does Harmony need to see the test cells?".

### What does the transformer rely on? (model audit)

**Why:** the one-donor-per-batch ablation showed the contrastive model partly tells cells apart by donor. The audit asks which genes the model uses, and whether one specific gene, HLA-DRB5, works as a genetic "name tag" for the donor. Predictions K1, K2 and G1 to G4 were fixed in `docs/PLAN.md` first. Everything runs on CPU from the saved split-0 weights.

**Part 1, knocking out genes** (`scripts/audit_knockout.py`). For 14 cell types, up to 100 test cells each, every common gene was set to zero in the input in turn and the cell embedded again. The genes that move a cell most were compared with the textbook markers (a Wilcoxon test on training cells) and with a plain "most highly expressed" list.
- **K1, not supported.** The knockout list matched the markers better than the "most expressed" list in only 6 of 14 types (4 ties); mean overlap 0.336 against 0.339. The model's reliance tracks expression (Spearman 0.72 on average) and barely tracks how specific a gene is to the type (0.10).
- **K2, supported by the letter only.** Contrastive 0.336 against masked-gene 0.325: three gene slots out of 280.
- **Not predicted:** the contrastive models are about four times less sensitive to losing any one gene (mean effect 0.0053 against 0.0198; label changes 2.3 against 5.5 percent of cells). That is what training on views with random genes dropped should do. Ribosomal genes fill 34 of the 186 top-20 slots that are not markers.

**Part 2, the HLA-DRB5 test** (`scripts/audit_genotype.py`). Six donors show essentially no HLA-DRB5, the other 15 do. A probe tried to tell carriers from non-carriers using monocyte embeddings, one donor held out at a time.
- **G1, not supported:** contrastive AUROC 0.756, below the 0.80 predicted.
- **G2, supported by the letter, but tiny:** removing HLA-DRB5 lowered it to 0.733 (2 of 90 donor pairs) while no control gene changed it at all. Removing every HLA gene gave the same 0.733.
- **G3, supported:** naive CD4 T cells give 0.321, no signal (values below 0.5 are a known side effect of leave-one-out when there is nothing to find).
- **G4, supported:** PCA gives 0.933, and removing every HLA gene drops it to 0.822. PCA really does read the gene; the transformers barely do.
- **What the scores show:** the transformers' per-donor scores sort mostly by cohort (C2 donors low, the others high), and the two donors sequenced with the newer kit come out worst (a carrier at 0.04, a non-carrier at 0.95). Five of the six non-carriers are C2 donors, so the probe mostly learned the cohort. Within C2 alone the contrastive AUROC is 0.667.

**Part 3, a sparse autoencoder** (`scripts/audit_sae.py`, `src/cellmap/sae.py`). An SAE rewrites each 128-number embedding as a sum of a few features out of 1,024. Chosen setting: 16.5 active features per cell on average, 0.963 of the variance explained on validation cells.
- **S1, supported:** 0.963, above the 0.90 predicted.
- **S2, supported:** all 10 coarse types have at least one cell-type feature (46 in all; the plan said 8 coarse types, there are 10, and the verdict does not depend on it).
- **S3, not supported:** no feature separates the C2 cohort from the other 10x v2 donors within classical monocytes better than AUROC 0.68 (predicted at least 0.80).
- **S4, not supported:** removing the 10 features most tied to batch changed donor mixing by +0.003 and fine macro F1 by -0.010, the same as removing 10 random features (+0.002, -0.005).
- **Not predicted:** 773 of the 1,024 features never switched on (dead), a known problem of this simple SAE. The most batch-linked features carry monocyte activation genes (KLF4, IL1B, CXCL8, DUSP1), which can differ with how samples were handled; that is a lead, not a finding.

**Reading:** the donor shortcut is not HLA-DRB5. It is more likely cohort and chemistry, the technical differences between batches of samples, and the SAE says that information is spread thinly across many features rather than stored in a few that could be deleted. Fixing it takes a training change (the one-donor-per-batch ablation raised donor mixing by 0.133), not editing the finished embedding. The transformer uses roughly the genes that are most expressed, and it shrugs off losing any single one, which is also why single-gene knockout is a weak tool for a model trained to ignore missing genes.

**Weakness:** one split; 21 donors, so one donor pair moves AUROC by 0.011; nobody was genotyped; G2's test (a larger drop than any control) was too weak, because the controls moved by exactly zero; the HLA-DRB5 table had been looked at before the predictions.

**What to say:** "I suspected the donor shortcut was a genetic marker, HLA-DRB5, so I tested it causally: knock the gene out and see whether the donor signal goes. In PCA it partly does. In the transformer it barely moves, and the remaining signal sorts by cohort and sequencing kit. So the shortcut is technical batch, not genotype, and I only know that because I checked."

### leakaudit, the reusable audit kit

`audit/` is a separate package that turns this project's honesty checks into tools for any project (your cardiomyopathy and glioblastoma work included):
- **`group_split`, `assert_no_group_overlap`, `split_report`:** split by patient, donor or lab, prove no group crosses splits, and see how unbalanced the splits are.
- **`measure_inflation`, `measure_preprocessing_leak`:** score the same model with a leaky and an honest split, or with preprocessing fitted on all data versus training data, and report the inflation with its uncertainty.
- **`shortcut_probe`:** shuffle one named input block at a time and see which inputs the model actually needs.

It reproduces this project's leakage result exactly (0.776 versus 0.664, inflation 0.112). Its README lists what existing tools already do; the new part is measuring how much a leak inflates a score, and probing named input blocks, in one small package.

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

1. **Simple methods are hard to beat.** 3 of 68 pre-declared comparisons passed, all on one metric, none on preserving biology. This agrees with the anchor paper, now under a stricter test: held-out donors, and every learned model fitted on training donors only.
2. **"Best" depends on the metric.** Harmony tops integration; plain PCA tops fine-grained kNN. A single combined score would hide that.
3. **The chemistry batch effect is the biggest structure in the raw data.** Correcting it is what Harmony is good at.
4. **The hand-written VAE is correct.** It agrees with scvi-tools, so its results are not an implementation bug.
5. **Pretraining on huge data did not help here on biology or labels,** though Census scVI beat PCA on integration scores on six labs and passed the rule on batch correction there.
6. **A transformer trained from scratch on 34,000 cells is not competitive.** Learning to predict hidden genes is not the same as learning a good cell map.
7. **Noise is real.** scIB bio moved 0.07 from thread count alone. Counting wins across splits is more trustworthy than comparing means.
8. **Split seed 3 is hard for everyone.** Every method dropped there: PCA kNN accuracy 0.832, Harmony 0.751, VAE 0.754, transformer 0.555. Its test donors include one of the two v3-chemistry donors, and so do split seed 1's, which is not hard (PCA 0.903). The difference, found in the review: in seed 1 the other v3 donor is in training, while in seed 3 no v3 cells are left in training at all, so seed 3 is a chemistry hold-out (`results/verify/pbmc_split_composition.csv`).

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
| The five donor splits used for the baselines reuse the gene filter fitted on split seed 0's training donors | Noted while writing the notes | Effect judged negligible (a "detected in 10 cells" filter); recorded as a known weakness |
| Headline "no method beat both baselines" was checked on 2 of the 6 declared metrics; on all six, 3 of 68 comparisons pass | Independent review (2026-11-18) | `scripts/verify_rule.py` applies the rule to every metric; headline rewritten from its output |
| "The transformer ignores expression amounts" was contradicted by our own shuffle probe (0.366 to 0.164) | Independent review | Reworded: it reads amounts, but the true amounts add nothing over presence |
| "Eight shared confusions": 8 was the figure's display limit; there are 18 | Independent review | Figure subtitle and text now give both numbers |
| Symphony centroids built from per-cell normalised coordinates, unlike Symphony; docstring claimed they matched | Independent review against the Symphony source | Code now matches Symphony; inductive Harmony comparison rerun |
| `m4_benchmark.py` without `--resume` replaced a full results file | Independent review | It now refuses unless `--resume` or `--overwrite` is given |
| Provenance stamps ignored untracked files, so `results/leakage/splits.csv` names a commit without its script | Independent review | Untracked files now count as dirty |
| The VAE weight-transplant check was cited but only run as a scratch script | Independent review | Now a test in `tests/test_scvi_baseline.py` |
| The five PBMC splits were treated as replicates; split 3 is in fact a chemistry hold-out and split 1 tests on a quarter of all cells | Hostile reviewer in the independent review | Recorded in `results/verify/pbmc_split_composition.csv` and read alongside every per-split claim |
| A rerun of the contrastive transformer with the same seed, code and settings gave fine macro F1 0.540 against 0.506 in the main run | The `base` ablation, which exists to measure exactly this | Not a bug: the two runs' training losses already differ in the fourth decimal in epoch 0, which is GPU arithmetic that is not bit-for-bit repeatable, and the gap grows over 30 epochs. The pre-registered noise rule turns it into a bar of 0.069: an ablation must move macro F1 by more than that to count. Lesson: one run per setting cannot separate effects smaller than this |
| The guess that HLA-DRB5 was the donor "name tag" behind the shortcut | The audit's knockout test, planned for exactly this | Removing the gene barely changed the transformer's carrier probe (0.756 to 0.733), and the scores sorted by cohort and kit. Recorded as not the main driver; the pre-registered G2 test was too weak because the controls moved by exactly zero |

---

## Part 7. What is left

The milestones in the plan are done. What remains is yours to decide, listed in `docs/RELEASE.md`:

- **Choose a license** (MIT or GPL-3.0; `harmonypy`, a dependency, is GPL-3.0).
- **Merge the work branch into `main`,** delete the two Colab branches, make the repository public, and publish the site with the manual Pages workflow.
- **Revoke the Colab GitHub tokens.**

Experiments worth doing next, if you continue:
- A tuned or larger VAE code (only 10 dimensions were tried).
- "seurat_v3" gene selection, which mattered in DCMvsACM.
- A tissue other than blood.
- The transformer: the level-encoding fix did not help on one split (Part 4, "Follow-up analyses"), so repeating it on all splits has low priority. A different training objective, or a cell-level one, is the more promising next step.

### Questions to test yourself (answer out loud, without notes)

1. Why split by donor, and how much would a random split have overstated your result? (Part 2.4; the leakage experiment.)
2. Why a negative binomial and not a Gaussian? Give the variance formula.
3. Write the ELBO. What happens at KL weight 0 and at 100, and what did your sweep show?
4. How does telling the decoder the donor remove donor effects from the code?
5. What is the reparameterization trick and why is it needed?
6. How does your transformer turn a cell into tokens, and what did the probe reveal about it?
7. Why can a pretrained model that was trained on most of these cells still lose to PCA? Which of the three was, and which is unknown?
8. Why macro F1 and not only accuracy? Work the 90 monocytes and 10 platelets example.
9. What do scIB bio and batch scores each reward, and why can a method win one and lose the other?
10. Why is Harmony "transductive", and why does that matter for a fair comparison?
11. Where does batch correction go too far? Quote your Harmony sweep.
12. Your VAE did not beat the baselines. What do you conclude, and what would you try next?

## Part 8. Map of the repository

| Path | What it is |
|---|---|
| `configs/` | Dataset settings: Census release, filters, split fractions |
| `src/cellmap/census.py` | Downloads cells from Census |
| `src/cellmap/qc.py`, `splits.py`, `preprocess.py`, `labels.py` | The data pipeline pieces |
| `src/cellmap/baselines.py` | HVG selection, PCA, Harmony |
| `src/cellmap/evaluate.py`, `benchmark.py` | All scoring |
| `src/cellmap/vae.py`, `linear_ae.py` | The hand-written VAE and the linear autoencoder |
| `src/cellmap/scvi_baseline.py` | scvi-tools with matched settings, for the check against the VAE |
| `src/cellmap/transformer.py` | The hand-written transformer |
| `src/cellmap/census_embeddings.py` | Downloads of the published pretrained embeddings |
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
- **2026-11-23:** Audit part 3 (sparse autoencoder) done: S1, S2 supported, S3, S4 not; the batch signal is spread out, not stored in a few features. Added to Part 4 and the status line; PR #3 merged into `main`.
- **2026-11-22 (later):** Contrastive main run finished (all five splits): C1 and C2 supported. README, notes, journal and two figures (`assets/contrastive_pbmc.png`, `assets/audit_pbmc.png`) added.
- **2026-11-22:** Model audit done (gene knockout, HLA-DRB5 probe): K1 and G1 not supported, K2, G2, G3, G4 supported; the donor signal is mostly cohort, not the gene. Added to Part 4, Part 6 and the status line. Contrastive split 3 in; C1 settled.
- **2026-11-20:** First ablation (a plain rerun) landed: macro F1 0.540 against 0.506 for the same seed, so the noise bar is 0.069. Added to Part 6; scoring script `scripts/contrastive_ablations_evaluate.py` written before the other variants.
- **2026-11-19 (evening):** Contrastive ablations designed, predictions fixed, code and notebook added; status line updated.
- **2026-11-19 (later):** Conformal results added to section 2.12 and the status line: five of seven hypotheses supported, two partly; new figure.
- **2026-11-19:** Section 2.12 added: conformal prediction, from intuition to the hierarchical version, with three worked examples.
- **2026-11-18 (late):** Transformer level-encoding result added (it did not help); the inductive Harmony section rewritten from the corrected rerun.
- **2026-11-18 (late):** Independent review by ten agents that had not written the code (five literature, four verification, one hostile). Corrected the headline rule count, the transformer probe reading, the confusion count, the pretrained-model training claims, the leakage caveats, the over-correction wording and the Symphony centroids. Part 6 lists each.

- **2026-11-18:** leakaudit package added to Part 4.
- **2026-11-18:** Inductive Harmony comparison (Harmony fitted without the test cells) added to Part 4.
- **2026-11-18:** Follow-up analyses added (Part 4); Part 7 rewritten as the closing checklist and self-test questions.
- **2026-11-18:** Six-lab benchmark finished: transformer results from the Colab GPU run merged; final six-lab figure.
- **2026-11-18:** Added the leaky-split and input-probe experiments to Part 4.
- **2026-11-18:** Six-lab results (all but the transformer) added to Part 4, including the held-out cell type test.
- **2026-11-17 (night):** Explainer page built as the front page (`site/index.html`); the atlas moved to `site/atlas.html`.
- **2026-11-17 (night):** Six-lab benchmark design recorded in Part 7; public site choice recorded.
- **2026-11-17:** Guide created after the data pipeline, the baselines, the hand-written VAE, the scvi-tools check, the transformer and the pretrained embeddings were done. The transformer's GPU results are included.
