# Conformal prediction for cell-type annotation under donor and lab shift: literature review

Prepared 2026-11-19 for cellmap milestone M9 (see `docs/PLAN.md`, "M9 design and hypotheses").

## How to read the citations (access log)

The session's network blocked arxiv.org, ncbi/pmc, europepmc, biorxiv, openreview, Semantic Scholar,
Crossref, OpenAlex, publisher sites and conference proceedings. I could reach three kinds of source:

- **OPENED**: I read the document itself. These were GitHub READMEs, package files and docs fetched from
  raw.githubusercontent.com.
- **SNIPPET**: I saw only search-engine results: the abstract, metadata (venue, volume, DOI) and short
  excerpts. I did not read the full paper.
- **KNOWN**: the exact formula below is the standard statement as I know it from the paper. A SNIPPET
  confirms that the paper exists and what it is about, but I did not see the theorem text in this session.
  Check the formula against the PDF before quoting it in the README.

I did not make up any citation. Venue, volume and DOI come from search metadata, and the source type is
marked on each entry. The search could not cover bioRxiv full text, so recent preprints may be missing.

---

## (a) Theory papers to cite, with the exact guarantee each gives

Notation used throughout:
- Calibration pairs (X_1,Y_1), ..., (X_n,Y_n) and a test pair (X_{n+1},Y_{n+1}).
- A score s(x,y), where larger means "y fits x worse". s_i = s(X_i,Y_i).
- Miscoverage level alpha, for example 0.1.
- q_hat = the k-th smallest of s_1..s_n, where k = ceil((n+1)(1-alpha)). If k > n, then q_hat = +infinity
  and the set is all labels.
- Prediction set C(x) = { y : s(x,y) <= q_hat }.

### A1. Split (inductive) conformal: the base guarantee
- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World*. Springer 2005; 2nd edition 2022,
  DOI 10.1007/978-3-031-06649-8. [SNIPPET: catalogue records]
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. "Distribution-free predictive inference for regression."
  JASA 113(523):1094-1111, 2018. DOI 10.1080/01621459.2017.1307116. [SNIPPET]
- Angelopoulos & Bates. "Conformal Prediction: A Gentle Introduction" (arXiv title "A gentle introduction
  to conformal prediction and distribution-free uncertainty quantification", arXiv 2107.07511).
  Foundations and Trends in ML 16(4):494-591, 2023. DOI 10.1561/2200000101. [SNIPPET]

Guarantee [KNOWN]: if the n+1 pairs are exchangeable, then
  P( Y_{n+1} in C(X_{n+1}) ) >= 1 - alpha.
If the scores are almost surely distinct, then also
  P( Y_{n+1} in C(X_{n+1}) ) <= 1 - alpha + 1/(n+1).
- The probability is taken over the calibration draw and the test draw together. The guarantee is marginal:
  it is not conditional on x, on the class, or on the donor.
- **Feasibility:** a finite threshold needs n >= ceil(1/alpha) - 1, which is 9 at alpha = 0.1 and 19 at
  alpha = 0.05.
- **Training-conditional coverage** (Vovk 2012 below; also covered in Angelopoulos & Bates): with i.i.d.
  data and no ties, the coverage conditional on the calibration set follows Beta(n+1-l, l), where
  l = floor((n+1) alpha). Its standard deviation is about sqrt(alpha(1-alpha)/(n+2)).
  - For n = 2000 cells, this is about 0.007.
  - This formula assumes cells are independent. Cells from the same donor are not, so the real spread across
    folds will be much larger (see pitfall D3).

### A2. Score functions for classification
- **LAC / threshold score.** Sadinle, Lei, Wasserman. "Least ambiguous set-valued classifiers with bounded
  error levels." JASA 114(525):223-234, 2019. DOI 10.1080/01621459.2017.1395341. [SNIPPET]
  - Score: s(x,y) = 1 - p_hat(y|x).
  - [KNOWN] If p_hat equals the true P(Y|X), these sets have the smallest expected size among all sets with
    marginal coverage 1 - alpha.
  - Sets can be empty. Coverage is uneven: easy cells are over-covered and hard cells under-covered.
- **APS.** Romano, Sesia, Candès. "Classification with valid and adaptive coverage." NeurIPS 2020
  (spotlight). https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html
  [SNIPPET; the code README msesia/arc was OPENED and confirms NeurIPS 2020 spotlight]
  - Sort the classes by p_hat in decreasing order. Then
    s(x,y) = (sum of p_hat(j|x) over classes j ranked above y) + U * p_hat(y|x), with U ~ Uniform(0,1).
  - [KNOWN] Same marginal guarantee as A1. The randomized version also has the 1/(n+1) upper bound.
  - [KNOWN] If p_hat is the true conditional probability, the oracle APS set has exact conditional coverage
    given X. This is why it is called "adaptive". Without randomization it over-covers.
- **RAPS.** Angelopoulos, Bates, Malik, Jordan. "Uncertainty sets for image classifiers using conformal
  prediction." ICLR 2021 (spotlight), arXiv 2009.14193. [SNIPPET; code README
  aangelopoulos/conformal_classification OPENED]
  - Score: s_RAPS(x,y) = s_APS(x,y) + lambda * max(0, rank(y) - k_reg).
  - Same marginal guarantee. Smaller, more stable sets when there are many classes; the paper reports mean
    size 2 against 19 for APS on ImageNet at alpha = 0.1.
  - [KNOWN] The paper also introduces the size-stratified coverage violation (SSCV): the worst coverage gap
    across bins of set size. Use it to check whether hard cells pay for easy ones.

### A3. Class-conditional (Mondrian) and group-conditional conformal
- Vovk. "Conditional validity of inductive conformal predictors." ACML 2012 (JMLR W&CP 25); journal version
  Machine Learning 92:349-376, 2013. DOI 10.1007/s10994-013-5355-6, arXiv 1209.2673. [SNIPPET]
  - Defines training-, object-, label- and example-conditional validity.
- Ding, Angelopoulos, Bates, Jordan, Tibshirani. "Class-conditional conformal prediction with many classes."
  NeurIPS 2023, arXiv 2306.09335. [SNIPPET] Clustered conformal: classes with similar score distributions
  share one quantile.
- Gibbs, Cherian, Candès. "Conformal prediction with conditional guarantees." JRSS-B 2025, pp. 1100-1126,
  DOI 10.1093/jrsssb/qkaf008, arXiv 2305.12616. [SNIPPET]
- Barber, Candès, Ramdas, Tibshirani. "The limits of distribution-free conditional predictive inference."
  Information and Inference 10:455-482, 2021. DOI 10.1093/imaiai/iaaa017. [SNIPPET]
- Bhattacharyya, Ding, Barber. "Conformal prediction with macro-coverage guarantees." arXiv 2606.28598,
  June 2026. [SNIPPET]
- Ding, Fermanian, Salmon. "Conformal prediction for long-tailed classification." ICLR 2026,
  arXiv 2507.06867. [SNIPPET]

Guarantees [KNOWN]:
- **Mondrian by class:** compute q_hat_y from the calibration cells with Y_i = y only, and put y in C(x) if
  s(x,y) <= q_hat_y. Then for every class y,
  P( Y in C(X) | Y = y ) >= 1 - alpha.
  - Each class needs n_y >= ceil(1/alpha) - 1 calibration cells.
  - Corollary used below: because the guarantee holds within every class, it also holds under any change in
    class proportions. Podkopaev & Ramdas (A6) make this point.
- **Mondrian by an observed group g(X):** the same construction gives P( Y in C | G = g ) >= 1 - alpha for
  every g, but only for groups that appear in the calibration data. A new donor or new lab is not such a
  group, so donor-Mondrian or lab-Mondrian cannot be applied to unseen donors or labs.
- **Gibbs-Cherian-Candès:** exact coverage over each group in a finite collection of groups given as
  covariates, and over a finite-dimensional class of covariate shifts.
- **Barber et al. 2021:** exact distribution-free coverage conditional on X = x is impossible with sets of
  finite size.
- **Ding et al. 2023:** cluster-conditional coverage, and approximate class-conditional coverage when the
  score distributions inside a cluster are similar.
- **Bhattacharyya et al. 2026:** finite-sample guarantee for macro-coverage, the unweighted mean of
  per-class coverages, through label-weighted conformal.

### A4. Covariate shift (weighted conformal)
Tibshirani, Barber, Candès, Ramdas. "Conformal prediction under covariate shift." NeurIPS 2019 (pages
2530-2540), arXiv 1904.06019. [SNIPPET]

Guarantee [KNOWN]:
- Assume test X ~ Q_X while calibration X ~ P_X, with P(Y|X) the same in both and w(x) = dQ_X/dP_X known.
- Give calibration point i the weight w(X_i) / (sum_j w(X_j) + w(x)). Give a point mass at +infinity the
  weight w(x) / (sum_j w(X_j) + w(x)). Set q_hat(x) to the (1-alpha) quantile of this weighted distribution.
- Then P( Y_{n+1} in C(X_{n+1}) ) >= 1 - alpha.
- With an estimated w, for example from a classifier that tells calibration cells from new-lab cells, the
  guarantee is only approximate.
- **Key assumption:** P(Y|X) is the same in both. It fails when labs annotate differently.

### A5. Beyond exchangeability (bounding the coverage gap)
Barber, Candès, Ramdas, Tibshirani. "Conformal prediction beyond exchangeability." Annals of Statistics
51(2):816-845, 2023. DOI 10.1214/23-AOS2276. [SNIPPET]

Guarantee [KNOWN, structure confirmed by snippet]:
- With fixed weights w_i in [0,1] and normalized w~_i = w_i / (1 + sum_j w_j),
  coverage >= 1 - alpha - sum_i w~_i * d_TV( Z, Z^i ).
- Here Z^i is the data with point i swapped with the test point, and d_TV is total variation distance.
- **Practical reading for M9:** if calibration scores come from distribution F and a new lab's test scores
  from G, the coverage loss is at most about d_TV(F, G).
  - This links H1 (random-cell calibration under-covers) to a measurable quantity: the shift between the
    calibration-score distribution and the held-out-lab score distribution.
  - Plot both empirical CDFs.

### A6. Label shift
Podkopaev & Ramdas. "Distribution-free uncertainty quantification for classification under label shift."
UAI 2021, PMLR 161:844-853, arXiv 2103.03323. [SNIPPET]

Guarantee [KNOWN, method confirmed by snippet]:
- Assume P(X|Y) is fixed while P(Y) changes from p to q.
- Weight calibration point i by w(Y_i) = q(Y_i) / p(Y_i) and use weighted conformal.
  - With known q, coverage >= 1 - alpha.
  - With q estimated from unlabeled target data (black-box shift estimation), the guarantee is asymptotic.
- Their alternative is label-conditional (Mondrian) conformal, which keeps 1 - alpha for any q (see A3).
- **Relevance:** cell-type proportions differ greatly between donors and labs, for example in OneK1K
  against COMBAT.

### A7. Hierarchical / grouped exchangeability (the core theory for this angle)
Model:
- Groups (donors or labs) k = 1..K are i.i.d. draws P_k from a population of distributions.
- Within group k, cells are exchangeable draws from P_k.
- Target: the coverage of a new cell from a new group.

Papers:
- Dunn, Wasserman, Ramdas. "Distribution-free prediction sets for two-layer hierarchical models."
  JASA 118(544):2491-2502, 2023. DOI 10.1080/01621459.2022.2060112, arXiv 1809.07441 (earlier title
  "Distribution-free prediction sets with random effects"). [SNIPPET]
- Lee, Barber, Willett. "Distribution-free inference with hierarchical data." arXiv 2306.06342
  (v1 June 2023, revised 2024). [SNIPPET]
- Duchi, Gupta, Jiang, Sur. "Predictive inference in multi-environment scenarios." arXiv 2403.16336
  (2024; NeurIPS 2024 workshop poster). [SNIPPET]
- Bhattacharyya & Barber. "Group-weighted conformal prediction." Electronic Journal of Statistics
  20:1171-1199, 2026, DOI 10.1214/26-EJS2506, arXiv 2401.17452. [SNIPPET]
- Liu, Levis, Normand, Han. "Multi-source conformal inference under distribution shift." ICML 2024,
  arXiv 2405.09331. [SNIPPET] Needs some labeled target data.

Guarantees:
- **Single subsampling** (Dunn et al.) [confirmed by snippet]: take one cell at random from each of the K
  calibration groups and run split conformal on those K scores. The K scores and a test cell from a new
  group are exchangeable, so P(cover) >= 1 - alpha.
  - This needs K >= ceil(1/alpha) - 1, i.e. 9 groups at alpha = 0.1.
- **Repeated subsampling** (Dunn et al.) [confirmed by snippet]: repeat single subsampling B times and
  aggregate.
  - Dunn et al. prove P(cover) >= 1 - 2 alpha, which tends to be about 1 - alpha in practice.
  - Lee et al. state that under hierarchical exchangeability the guarantee is 1 - alpha (snippet; not
    verified).
- **CDF pooling** (Dunn et al.): asymptotic coverage only.
- **Hierarchical conformal prediction (HCP)** (Lee, Barber, Willett) [existence confirmed; the weights are
  KNOWN, check against the PDF]:
  - Pool all calibration cells, but give group k total weight 1/(K+1), split equally across its n_k cells.
  - Put a mass of 1/(K+1) at +infinity for the test cell's own new group.
  - Use the (1-alpha) quantile of this weighted distribution. Coverage for a new cell in a new group is
    >= 1 - alpha.
  - **Feasibility:** the threshold is finite only if K/(K+1) >= 1 - alpha, i.e. K >= (1-alpha)/alpha.
    That is K >= 9 groups at alpha = 0.1 and K >= 4 at alpha = 0.2.
  - HCP2 targets repeated measurements and gives a stronger "second-moment" coverage that controls how
    much coverage varies from group to group.
- **Multi-environment split conformal** (Duchi et al.): assumes cells are exchangeable within each
  environment and environments are exchangeable. It gives coverage for a new environment and adds a step
  that shrinks sets using a little information from the test environment.
- **Jackknife+ / CV+** (Barber, Candès, Ramdas, Tibshirani, "Predictive inference with the jackknife+",
  Annals of Statistics 49(1):486-507, 2021, DOI 10.1214/20-AOS1965) [SNIPPET]: worst-case coverage
  1 - 2 alpha, and about 1 - alpha if the model is stable.
  - This is the closest theory to M9's "leave one training lab out, refit, pool the scores" design. That
    design is cross-conformal at the lab level. It would inherit at best a 1 - 2 alpha-type guarantee, and
    only if labs are exchangeable.

**What this means for M9 (important):**
- With 6 labs, a lab fold has 5 training labs. No method can give a finite-sample, distribution-free
  new-lab guarantee at alpha = 0.1 with K = 5 groups: 5/6 = 0.83 < 0.90.
  - At alpha = 0.2, HCP is feasible with K = 5.
- On PBMC (21 donors, val_fraction 0.2, so about 4 validation donors), donor calibration rests on about 4
  groups. That is also too few for a donor-level guarantee at alpha = 0.1. Pooling cells hides this but
  does not fix it.
- So lab-level calibration at alpha = 0.1 should be presented as an empirical correction, not a valid
  procedure. Report alpha = 0.2 as the setting where a group-level guarantee is possible.
- Donor-level calibration on the six-lab data (roughly 100 training donors) is where the hierarchical
  theory actually applies.

### A8. Outlier / novelty detection with conformal p-values
- Bates, Candès, Lei, Romano, Sesia. "Testing for outliers with conformal p-values." Annals of Statistics
  51(1):149-178, 2023. DOI 10.1214/22-AOS2244. [SNIPPET]
- Guan & Tibshirani. "Prediction and outlier detection in classification problems" (BCOPS). JRSS-B
  84(2):524-546, 2022. DOI 10.1111/rssb.12443. [SNIPPET]
- Hechtlinger, Póczos, Wasserman. "Cautious deep learning." arXiv 1805.09460, 2018. [SNIPPET]
  Uses p(x|y)-based conformal sets that return the empty set ("I don't know") for inputs unlike training.
- Angelopoulos, Bates, Fisch, Lei, Schuster. "Conformal risk control." ICLR 2024, arXiv 2208.02814.
  [SNIPPET] Controls any monotone loss; scConform uses it for ontology-consistent sets.

Guarantee [KNOWN]:
- Class-conditional conformal p-value: p_y(x) = (1 + #{ i in cal_y : s_i >= s(x,y) }) / (n_y + 1).
- For a test cell whose true class is y, P( p_y(X) <= t ) <= t. The marginal version with all cells is the
  same.
- Flag "novel" when max_y p_y(x) <= alpha. With Mondrian sets, this is exactly when C(x) is empty. The rate
  of known cells falsely flagged is then <= alpha.
- Bates et al. show the p-values for many test points are positively dependent, so Benjamini-Hochberg
  controls the FDR. They also give calibration-conditional p-values.
- Power (detecting real novel types) is not guaranteed. It depends entirely on the score.

---

## (b) Prior single-cell conformal and uncertainty work, and what our angle adds

### B1. Conformal cell-type annotation (direct prior work)

**1. Khatri & Bonn. "Uncertainty estimation for single-cell label transfer."** COPA 2022, PMLR 179
(pages 109-128 per PMLR). https://proceedings.mlr.press/v179/khatri22a.html [SNIPPET: abstract, plus slide
excerpts surfaced by search]
- Inductive conformal classifiers for label transfer. Pipeline: batch correction (Harmony, 50 PCs), then a
  classifier (SVM in the slides). Six human datasets (PBMC, pancreas).
- Finding: uncertainty is "drastically affected by the batch correction method".
- Stated future work: detecting unknown cell types, and "using conformal predictions to evaluate batch
  correction methods".
- Group-aware calibration: none seen in the snippets.
- Our multi-embedding comparison of set size at fixed coverage is the experiment they named as future work.

**2. López-De-Castro, García-Galindo, González-Gomariz, Armañanzas. "Conformal inference for reliable single
cell RNA-seq annotation."** Bioinformatics 41(10):btaf521, 2025. DOI 10.1093/bioinformatics/btaf521.
Code: https://github.com/digital-medicine-research-group-UNAV/conformalized_single_cell_annotator
[paper SNIPPET; code README OPENED]
- Two stages: an autoencoder-based out-of-distribution detector with a conformal hypothesis test (alpha_OOD
  can be estimated automatically), then a conformal classifier.
  - "Taxonomies": standard, classwise (Mondrian) and cluster.
  - Non-conformity functions from TorchCP, APS by default. Three nonconformity measures in the paper.
  - Underlying model: a torch MLP, CellTypist or scmap. Runs on any embedding; the example uses
    `X_pca_harmony`.
- Datasets: snippets disagree, one says 27 public datasets and another "10 batched experiments".
- Per the snippets, the authors state that batch effects and biological variation break exchangeability.
  Coverage was further from nominal for a tumour-patient query "where the biological batch clearly breaks
  the exchangeability assumption".
- From the README: calibration is a random split of the reference, and there is no donor- or lab-grouped
  calibration.
- **Overlap with our angle:** our parts (3) and (4) (Mondrian, novel types) and the qualitative half of (1).

**3. Corbetta, Geistlinger, Finos, Risso. "Conformal inference for cell type annotation with
graph-structured constraints."** arXiv 2410.23786 (Oct 2024); JRSS-C 2026, DOI 10.1093/jrsssc/qlag037.
R/Bioconductor package scConform (https://github.com/ccb-hms/scConform). [paper SNIPPET; package
DESCRIPTION and vignette OPENED]
- Split conformal with the LAC score (the vignette uses s = 1 - p_hat, a multinomial classifier and
  q_hat = ceil((1-alpha)(n+1))/n).
- Conformal risk control with the Cell Ontology: go up the ontology until the probability mass of the
  ancestor reaches lambda, then return its leaves.
- Data: MERFISH mouse ileum (15 types), a COVID-19 scRNA-seq dataset.
- The vignette says reference and query are "a subset of the same dataset", a toy split, and that "in real
  life examples, data integration is a necessary first step".
- Per the snippet, the paper says its "experiments do not focus on batch effects, and understanding how
  distributional shifts affect coverage remains an open question".
- This is the clearest published statement of the gap our part (1) targets.

**4. Engelmann, Hetzel, Palla, Sikkema, Luecken, Theis. "Uncertainty quantification for atlas-level cell type
transfer."** arXiv 2211.03793, 2022. [SNIPPET]
- Not conformal. Benchmarks four model classes on the Human Lung Cell Atlas.
- Finds current models lack calibration, robustness and actionable uncertainty scores.
- Models that quantify uncertainty detect unseen cell types better.

**5. Unreviewed GitHub projects, very close to the plan.** Cite as code only, never as evidence.
- **Vroon1703/cellconformal** (README OPENED). Not the owner's repository; the owner's remote is
  Varkot-dev.
  - LAC, randomized APS, Mondrian with an explicit +infinity when n_y is too small.
  - Leave-one-type-out novelty using 1 - max class-conditional LAC p-value. It argues APS is the wrong
    novelty statistic because the APS score of the top label is U * p_top.
  - Cross-dataset transfer: coverage 0.896 in domain, 0.694 after transfer, 0.902 after recalibrating on
    940 labeled target cells.
  - Data: simulated, pbmc68k_reduced and pbmc3k.
  - Novel NK cells undetected: AUROC 0.448, called CD8 T. No donor- or lab-grouped calibration.
- **aysegulsayineker/conformal-cell-annotation** (README OPENED).
  - LLM-based cluster annotation with conformal ontology back-off. PBMC3k and Segerstolpe pancreas.
  - Compares within-dataset, pooled and cross-dataset calibration. Pooled and cross-dataset calibration
    under-cover on pancreas (0.655 cross-dataset at a 0.80 target).
  - Notes that "pooling across datasets makes the threshold computable but does not establish
    exchangeability".

### B2. Non-conformal confidence and rejection in annotation tools (cite as heuristics without guarantees)

**CellTypist.** Domínguez Conde et al., Science 376:eabl5197, 2022. [README OPENED; paper SNIPPET]
- Logistic regression trained by SGD.
- `mode='prob match'` with `p_thres=0.5` assigns 0 ("Unassigned"), 1 or several labels per cell. It is a
  set-valued predictor with an arbitrary threshold and no coverage guarantee.
- `majority_voting=True` relabels cells by majority within Leiden over-clusters.
- Natural non-conformal baseline for our sets: a fixed probability threshold against a conformal threshold.

**scPred.** Alquicira-Hernández et al., Genome Biology 2019, DOI 10.1186/s13059-019-1862-5. [README OPENED;
paper SNIPPET]
- Classifies on a low-dimensional (PCA) embedding.
- Rejects ("unassigned") when the maximum class probability is below a threshold: 0.9 in the paper, 0.55
  as the current software default.

**scANVI.** Xu, Lopez, Mehlman, Regier, Jordan, Yosef, Mol. Syst. Biol. 17:e9620, 2021,
DOI 10.15252/msb.20209620. [SNIPPET] The posterior q(c | z) is returned by `predict(soft=True)`
(scvi-tools docs snippet). There is no coverage guarantee.

**scArches** (Lotfollahi et al., Nat. Biotechnol. 40:121-130, 2022, DOI 10.1038/s41587-021-01001-7)
**and the HLCA** (Sikkema et al., Nat. Med. 2023). [SNIPPET]
- Weighted kNN label transfer with an uncertainty score scaled to [0,1]. High-uncertainty cells can be
  called unknown.
- I could not confirm the HLCA's numeric cutoff, so do not quote one.

**popV.** Ergen et al., Nature Genetics 56(12), 2024, DOI 10.1038/s41588-024-01993-3. [SNIPPET] Consensus of
eight annotation methods; certainty is the level of agreement between them.

**Abdelaal et al.** Genome Biology 20:194, 2019. [SNIPPET]
- 22 classifiers on 27 datasets, including rejection options (SVM_rejection, scmap, scPred, Garnett) and
  dedicated rejection experiments.
- The general-purpose SVM was best overall.

**Novel or unseen types, other approaches** [SNIPPET]: scPOT (IJCAI 2023), scGAD (AAAI 2023), scDOT and
OVAAnno (Briefings in Bioinformatics 2024), MARS (cited, not opened), treeArches / scHPL (Michielsen et al.;
rejects on low posterior or large neighbour distance).

### B3. Linear probe or kNN: what embedding benchmarks actually use

| Benchmark | Classifier used | Source |
|---|---|---|
| scIB (Luecken et al., Nat. Methods 19:41-50, 2022) | No supervised classifier. Bio metrics are cell-type ASW, cLISI, ARI/NMI after clustering, isolated-label F1/ASW, HVG conservation, trajectory, cell cycle | SNIPPET (scib docs) |
| Kedzierska, Crawford, Amini, Lu (Genome Biology 2025, DOI 10.1186/s13059-025-03574-x) | No classifier for the main result. Zero-shot AvgBIO (ARI, NMI, ASW) and batch metrics against HVG, scVI and Harmony | SNIPPET |
| Boiarsky, Singh, Buendia, Getz, Sontag (bioRxiv 2023, "A deep dive into single-cell RNA sequencing foundation models") | L1-regularized logistic regression on genes against fine-tuned scBERT and scGPT; LR is competitive, including few-shot | SNIPPET |
| CELLxGENE Census model benchmarks (2024 article) | SVM, random forest and logistic regression on embeddings, 60/40 split, for bio labels and for batch labels | OPENED (`chanzuckerberg/cellxgene-census` docs) |
| cz-benchmarks (CZI) | Logistic regression, kNN and random forest, 5-fold CV; optional donor-level pooling; a cross-species task | OPENED (`cz-benchmarks` docs/assets.md) |
| Open Problems label projection | LR, MLP, kNN, XGBoost, Seurat, scANVI, scArches+scANVI (v1.0); later releases add scGPT, UCE, SCimilarity | SNIPPET |

Takeaways:
- Running both kNN and an LR probe (H6) matches current practice (cz-benchmarks runs both).
- scIB and Kedzierska do not use a supervised probe at all.
- None of these benchmarks puts conformal coverage or set size on embeddings, as far as the snippets show.
- The cz-benchmarks and Census splits are cell-level CV or random 60/40 splits, not donor- or lab-level. That
  is the leakage this project already documents.

### B4. What the angle adds and what it repeats

| Part of the plan | Status in the literature | Verdict |
|---|---|---|
| (1) Random-cell calibration fails on new donors or labs | Stated qualitatively: López-De-Castro 2025 (tumour query), cellconformal (simulated plus pbmc3k to pbmc68k), Corbetta (open question). No paper found that separates cell-, donor- and lab-level calibration on real multi-lab data with donor-level splits across many embeddings | **Partly new.** The new part is the controlled decomposition and its size, not the claim that exchangeability matters |
| (2) Group-aware (donor or lab) calibration restores coverage on held-out labs | Theory exists (A7: Dunn 2023, Lee et al., Duchi et al. 2024). No single-cell application found | **Most novel part.** It must be framed with the feasibility limit (K >= 9 groups at alpha = 0.1) |
| (3) Set size locates ambiguous types | Implicit in every conformal annotation paper. Ontology-aware sets (Corbetta) are the stronger published version | **Incremental.** Value comes from tying it to the repo's 18 shared confusions and testing whether set size predicts them out of sample |
| (4) Novel type held out | Done: López-De-Castro 2025 (dedicated OOD detector), cellconformal (leave-one-type-out, same failure for NK and CD8), Engelmann 2022, Hechtlinger 2018, Guan & Tibshirani 2022 | **Repeats prior work.** Keep as a secondary check and cite these works |
| Set size as an embedding metric (Census scVI, Geneformer, TranscriptFormer, VAE, PCA, Harmony) | Khatri & Bonn list it as future work, for batch correction | **New in this form** |

---

## (c) Recommended methods and scores to implement, and why

The core set is what M9 already declares. The additions are what reviewers will expect.

1. **Split conformal with three scores**, all on the same calibration and test cells:
   - LAC: smallest sets; empty sets are the novelty signal.
   - Randomized APS: adapts to ambiguity, with no empty sets in the non-randomized version.
   - RAPS with k_reg about 2 and lambda about 0.01-0.1, tuned on a separate split. Optional: with 12 to 19
     classes APS is rarely very large, but RAPS is cheap and is the default in the literature.
   - **For kNN (k = 15)** the probabilities are multiples of 1/15, so LAC scores have many ties. Break ties
     at random (the APS U term), or use distance-weighted kNN or larger k. Otherwise sets jump in steps and
     coverage overshoots.
2. **Mondrian per class**, with clustered conformal (Ding et al.) for classes with too few calibration
   cells. Two reasons:
   - Rare types such as Treg, plasmablasts and CD56-bright NK are exactly what marginal coverage hides.
   - Class-conditional sets keep their guarantee under label shift. Cell-type proportions differ across the
     six labs, so this is a principled partial fix for lab shift, assuming expression given cell type is the
     same across labs.
3. **Group-aware calibration**, implemented as distinct named methods so each matches a theorem:
   - (a) Donor-held-out pooled split conformal (current plan): the baseline.
   - (b) **HCP weighting** (Lee, Barber, Willett): each calibration donor or lab gets equal total weight, and
     a mass of 1/(K+1) goes at +infinity. Report when the threshold is infinite rather than replacing it.
   - (c) **Repeated subsampling** (Dunn et al.): a simple, valid comparison.
   - (d) **Leave-one-lab-out pooled scores** (current plan item 3): call it cross-conformal or CV+ at the
     lab level. Make clear it has no finite-sample guarantee at K = 5, alpha = 0.1.
   - (e) Optional, label-free adaptation to a new lab:
     - Weighted conformal under covariate shift (A4), with w estimated by a classifier on embeddings that
       tells training-lab cells from new-lab cells. Clip the weights and report the effective sample size
       (sum w)^2 / sum w^2.
     - Label-shift weighting (A6), with new-lab proportions estimated by black-box shift estimation from the
       classifier's confusion matrix.
   - (f) Optional: few-shot target recalibration with m = 25, 50, 100 or 200 labeled cells from the new lab.
     This shows the cost of the "honest fix" and connects to Duchi et al.'s resizing step.
4. **Novelty:**
   - Class-conditional conformal p-values on LAC scores and on a distance score: the mean distance to the
     k nearest training cells of class y, i.e. a kNN "conformity" score in embedding space.
   - Flag a cell when max_y p_y <= alpha. Report the false-flag rate on known cells (bounded by alpha) and
     AUROC on the held-out type.
   - Do not use APS scores for novelty (cellconformal's argument, which is correct).
5. **Metrics to report**, beyond those declared in M9:
   - Per-donor and per-lab coverage distributions (box plots), not only means. Under A7 the guarantee is
     averaged over groups, so single groups can fall below 1 - alpha.
   - SSCV, the size-stratified coverage violation.
   - Macro-coverage (mean of per-class coverage). Bhattacharyya, Ding & Barber 2026 give a method that
     targets it.
   - A **co-occurrence matrix** of labels inside sets of size 2 or more. It names the ambiguous pairs
     directly; for example, how often Treg and central memory CD4 T appear together.
   - The empirical CDFs of calibration scores and held-out-lab scores, with their KS or TV distance (A5),
     to explain any coverage gap.
   - The spread of coverage across repeated calibration draws, compared with the i.i.d. Beta spread (A1).
     The excess measures the donor effect.

---

## (d) Pitfalls reviewers would raise

**D1. Marginal, not conditional.** A 90 % guarantee is an average over cells and, in grouped designs, over
donors or labs. Under-coverage of one held-out lab is not by itself a failure of a valid method. Say which
event the probability is over every time.

**D2. Too few groups for the claim.**
- With 5 training labs, no distribution-free new-lab guarantee exists at alpha = 0.1 (needs K >= 9).
- PBMC has about 4 validation donors, which is also too few at the donor level.
- "Lab calibration restores coverage" can only be an empirical statement from 6 folds, with a pre-declared
  threshold. M9's H3 already compares worst-lab coverage; keep it descriptive.

**D3. Cells within a donor are not independent.**
- Pooling 2,000 cells from 4 donors gives an effective calibration size close to 4, not 2,000.
- Fold-to-fold variation will be much larger than the Beta formula predicts. Do not use the i.i.d. formula
  for error bars; bootstrap over donors.

**D4. Calibration and test must be disjoint from every fitted component.**
- If the embedding (PCA loadings, VAE, scVI) saw the calibration donors or lab, calibration scores are
  optimistic.
- Harmony is transductive and sees test donors.
- Census scVI was trained on Census cells that include these six datasets, and the Geneformer variant was
  fine-tuned with labels on CELLxGENE data. A "held-out lab" is therefore not new to those models.
- Report pretrained models separately and say this.

**D5. Concept shift from annotation.**
- Labs label at different granularity: COMBAT has "CD4-positive T cell" where Hao has naive, central memory
  and effector memory. Coverage is measured against each lab's own labels.
- No reweighting (A4 or A6) fixes a change in P(Y|X) caused by labelling conventions. Mapping to 12 lineages
  reduces this but does not remove it.
- Under-coverage in lab folds may come from labels, not biology. Report coverage per lab with that lab's
  label provenance.

**D6. Label shift and rare classes.**
- Marginal coverage can look fine while Treg or plasmablasts are covered 40 % of the time.
- Mondrian needs n_y >= 9 at alpha = 0.1 and n_y >= 19 at alpha = 0.05. When n_y is too small, report an
  infinite threshold or a cluster fallback, never a quietly finite threshold.

**D7. kNN probabilities are coarse.** Ties make non-randomized sets over-cover and make set size a step
function. Any comparison of kNN against LR on set size must use randomized tie-breaking, or both
comparisons will mislead.

**D8. Set size depends on the classifier.**
- Comparing embeddings by set size mixes embedding quality with classifier calibration and capacity: LR on
  2048-dimensional TranscriptFormer against LR on 10-dimensional VAE.
- Standardize, tune regularization on validation donors and report both probes.
- State that coverage holds by construction for every method on exchangeable data. Only efficiency (size)
  and robustness under shift differ.

**D9. Circularity in the ambiguity claim (H4).**
- Set size and the confusion matrix come from the same probabilities, so "large sets on confused types" is
  partly circular.
- Make it predictive: compute set size and co-occurrence on calibration or validation donors, then test on
  held-out donors or labs.
- Better still, use an orthogonal signal, for example FOXP3 or IL2RA detection rates, or protein data where
  a dataset has it. I did not check whether Census carries ADT for Hao.

**D10. Novel types: what conformal can and cannot say.**
- The true label of a removed type is not in the label space, so its "coverage" is 0 by definition.
- Only the false-flag rate on known cells is guaranteed (<= alpha). Detection power is not.
- Close relatives (NK against CD8 T, Treg against central memory CD4 T) are expected to be undetectable from
  softmax-type scores; cellconformal and the nature of the scores agree.
- Removing a type from classifier labels only, while the embedding still saw those cells unsupervised, is a
  softer test than removing it from the embedding too. Say which was done.

**D11. Multiple alphas, scores, calibrations, embeddings and folds.** This is a large grid. The M9
hypotheses are pre-declared, which is good. Keep any further findings labelled exploratory and keep the
5-of-6 / 4-of-5 rule style.

**D12. Randomized sets.** Randomized APS gives different sets on reruns. Fix and record the seed for U, and
report results averaged over several U draws.

**D13. Novelty of the method.** Reviewers will cite López-De-Castro et al. 2025, Corbetta et al.
2024/2026, Khatri & Bonn 2022 and Engelmann et al. 2022.
- The contribution should be framed as (i) a controlled measurement of how grouped structure breaks
  coverage, (ii) the first single-cell use of hierarchical conformal methods with their feasibility limits,
  and (iii) conformal efficiency as an embedding benchmark.
- It should not be framed as "conformal prediction for cell annotation".

**D14. Wording.** "Distribution-free" does not mean "assumption-free": exchangeability at the right level
(cells, donors or labs) is the assumption. Every sentence in the README that reports coverage should name
the level.

---

## Sources actually accessed

**OPENED** (read in full this session):
- https://github.com/digital-medicine-research-group-UNAV/conformalized_single_cell_annotator (README)
- https://github.com/ccb-hms/scConform (README, DESCRIPTION, vignettes/scConform.Rmd)
- https://github.com/Vroon1703/cellconformal (README)
- https://github.com/aysegulsayineker/conformal-cell-annotation (README)
- https://github.com/Teichlab/celltypist (README)
- https://github.com/powellgenomicslab/scPred (README)
- https://github.com/aangelopoulos/conformal_classification (README)
- https://github.com/msesia/arc (README)
- https://github.com/chanzuckerberg/cellxgene-census, docs/articles/2024/20240710_embedding_metrics_dec_2023_lts.md
- https://github.com/chanzuckerberg/cz-benchmarks, docs/source/assets.md

**SNIPPET only** (search abstracts and metadata): every paper cited above that is not listed under OPENED.
