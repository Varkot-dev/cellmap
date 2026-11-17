# cellmap

![Donor-level split of the PBMC dataset](assets/m0_split_pbmc.png)

cellmap asks how much deep models add over simple baselines for representing single cells
from gene-expression data, using public data and splits that prevent leakage.
It is a work in progress: only the data pipeline (milestone M0) is finished, and no model has
been trained or compared yet.
The full plan and its status are in [docs/PLAN.md](docs/PLAN.md).

## Status

| Milestone | State |
|---|---|
| M0 data pipeline: load, quality control, donor-level split, normalize | built, tests pass |
| M1 to M7 (baselines, VAE, transformer, pretrained models, atlas) | not started or in progress, no results |

## M0 result: the dataset and its split

Data: healthy-donor blood cells (PBMCs) from Schulte-Schrepping et al., Cell 2020, downloaded
from CZ CELLxGENE Census release 2025-01-30 (dataset `5e717147-0f75-4de1-8bd2-6fda01b8d75f`).
Every number below was written by `scripts/m0_prepare_data.py` to
[`results/m0/pbmc/`](results/m0/pbmc/).

| | cells | donors | fine cell types |
|---|---|---|---|
| downloaded | 45,787 | 21 | 20 |
| after quality control | 44,756 | 21 | 19 |
| train | 33,720 | 13 | 19 |
| val | 5,676 | 4 | 19 |
| test | 5,360 | 4 | 19 |

Donors shared between any two splits: 0 (checked in code on every run, and by unit tests).
After filtering, 18,002 genes remain; the gene filter is fitted on training donors only.

Known limits of this split, stated up front:

- Both donors sequenced with the 10x 3' v3 chemistry landed in train, so this split does not
  test generalization to an unseen chemistry.
- One donor contributes 11,195 of the 33,720 training cells.
- Some cell types are very rare in the held-out donors (for example 3 platelets in test), so
  per-type scores for those types will be noisy.
- This is one split with one seed. Later milestones repeat over several donor splits.

## Reproduce

```bash
pip install -e ".[dev,data,figures]"
python scripts/m0_prepare_data.py --config configs/pbmc.yaml   # downloads about 450 MB once
python scripts/m0_figures.py --name pbmc
pytest -q
```

`configs/tiny.yaml` runs the same pipeline on 150 cells per donor in a few seconds once the
download is cached.
