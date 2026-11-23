# Journal

Daily log of what was built, what was found, and what was learned.

| Date | Focus | Key finding |
|---|---|---|
| [2026-11-17](2026-11-17/README.md) | M0 data pipeline, M1 baselines and evaluation harness | Harmony beats PCA on scIB in 5 of 5 donor splits (+0.077 aggregate) and loses fine-label kNN accuracy in 5 of 5 (-0.037) |
| [2026-11-17](2026-11-17/README.md) (later) | M2 VAE, M3 parity setup, M5 transformer, M6 pretrained embeddings, atlas | No method beat both simple baselines; the hand-written VAE matches 50-dim PCA with 10 dims; published foundation-model embeddings lose to PCA |
| [2026-11-18](2026-11-18/README.md) | M4 six labs, held-out lab and cell type, leaky-split and input-probe experiments | No method passed the rule on six labs; the VAE gained most from more data; the transformer ignores expression amounts |
| [2026-11-19](2026-11-19/README.md) | Conformal prediction, contrastive transformer (in progress) | Coverage holds for new donors (0.901) and erodes for a new lab (0.876); lab-level calibration restores it with sets twice as big |
| [2026-11-22](2026-11-22/README.md) | Contrastive transformer finished, model audit | The objective lifts the transformer by 0.13 on 5 of 5 splits, still 0.16 below PCA; the donor shortcut is cohort and chemistry, not HLA-DRB5 |
| [2026-11-23](2026-11-23/README.md) | Audit part 3 (sparse autoencoder), merge to main | Clean cell-type features, no batch feature; deleting batch-linked features does not mix donors |
