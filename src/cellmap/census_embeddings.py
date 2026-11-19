"""Fetch precomputed model embeddings that CZ CELLxGENE publishes for Census cells.

For each Census release, CZI hosts embeddings of every cell from several pretrained models
(scVI, Geneformer, TranscriptFormer and others) as arrays indexed by the cell's `soma_joinid`.
Reading them avoids running the large models at all.

These embeddings are NOT held-out in the sense the rest of this project uses: the models were
trained on Census data, which includes the cells evaluated here. See PRETRAINED for what is
known about each one, and report that next to any score.
"""

from typing import Any

import numpy as np
import pandas as pd

from cellmap.census import tiledb_config

EMBEDDING_ROOT = "s3://cellxgene-contrib-public/contrib/cell-census/soma"

# Embeddings published for Census release 2025-01-30 (ids from the Census contributions
# manifest). "seen_cells" and "used_labels" come from each embedding's own published metadata.
PRETRAINED: dict[str, dict[str, Any]] = {
    "census_scvi": {
        "id": "CxG-czi-10",
        "label": "Census scVI",
        "n_features": 50,
        "seen_cells": "trained on all primary Census cells, so on every cell evaluated here",
        "used_labels": "no cell-type labels; batch covariates: assay, dataset, donor, suspension",
    },
    "geneformer_ft": {
        "id": "CxG-czi-11",
        "label": "Geneformer (fine-tuned classifier)",
        "n_features": 512,
        "seen_cells": "fine-tuned on CELLxGENE Census data",
        "used_labels": "YES: fine-tuned with multi-task supervision that includes cell type, "
        "so this is not a zero-shot embedding",
    },
    "tf_sapiens": {
        "id": "CxG-czi-12",
        "label": "TranscriptFormer TF-Sapiens",
        "n_features": 2048,
        "seen_cells": "pretrained on 57 million human cells; overlap with these cells not checked",
        "used_labels": "no cell-type labels (generative pretraining); inference used the assay",
    },
}


def census_joinids(census_cfg: dict[str, Any]) -> pd.DataFrame:
    """`soma_joinid`, donor and cell type for the config's cells, sorted by `soma_joinid`.

    `cellxgene_census.get_anndata` returns cells in the same order, so row i here is row i of
    the raw download. The donor and cell-type columns let the caller verify that.
    """
    import cellxgene_census
    import tiledbsoma

    context = tiledbsoma.SOMATileDBContext(tiledb_config=tiledb_config(census_cfg["region"]))
    obs_filter = f"dataset_id == '{census_cfg['dataset_id']}' and {census_cfg['obs_filter']}"
    with cellxgene_census.open_soma(uri=census_cfg["uri"], context=context) as census:
        obs = (
            census["census_data"]["homo_sapiens"]
            .obs.read(
                value_filter=obs_filter, column_names=["soma_joinid", "donor_id", "cell_type"]
            )
            .concat()
            .to_pandas()
        )
    return obs.sort_values("soma_joinid").reset_index(drop=True)


def fetch_embedding(
    census_cfg: dict[str, Any], embedding_id: str, joinids: np.ndarray, chunk: int = 5000
) -> np.ndarray:
    """Dense float32 embedding (len(joinids) x n_features), rows in the order of `joinids`.

    Raises if any requested cell has no embedding, so a silent gap cannot turn into a row of
    zeros that would then be scored as if it were real.
    """
    import tiledbsoma

    context = tiledbsoma.SOMATileDBContext(tiledb_config=tiledb_config(census_cfg["region"]))
    uri = f"{EMBEDDING_ROOT}/{census_cfg['version']}/{embedding_id}"
    joinids = np.asarray(joinids, dtype=np.int64)
    row_of = pd.Series(np.arange(len(joinids)), index=joinids)
    with tiledbsoma.open(uri, context=context) as array:
        out = np.zeros((len(joinids), int(array.shape[1])), dtype=np.float32)
        found = np.zeros(len(joinids), dtype=bool)
        for start in range(0, len(joinids), chunk):
            ids = joinids[start : start + chunk]
            table = array.read(coords=(ids,)).tables().concat()
            rows = row_of.loc[table["soma_dim_0"].to_numpy()].to_numpy()
            out[rows, table["soma_dim_1"].to_numpy()] = table["soma_data"].to_numpy()
            found[rows] = True
    if not found.all():
        raise ValueError(
            f"{(~found).sum()} of {len(joinids)} cells have no {embedding_id} embedding"
        )
    return out
