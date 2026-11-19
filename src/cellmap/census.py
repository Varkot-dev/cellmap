"""Download raw counts from the CZ CELLxGENE Census."""

import os
import re
from typing import Any
from urllib.parse import urlparse

import anndata as ad
import numpy as np
import pandas as pd

OBS_COLUMNS = [
    "dataset_id",
    "donor_id",
    "assay",
    "cell_type",
    "cell_type_ontology_term_id",
    "tissue",
    "disease",
    "sex",
    "development_stage",
]


def tiledb_config(region: str) -> dict[str, str]:
    """TileDB settings for anonymous reads of the public Census bucket.

    TileDB does not read the usual HTTPS_PROXY / SSL_CERT_FILE environment variables, so when
    they are set (for example on a machine behind a proxy) they are passed on explicitly.
    """
    cfg = {"vfs.s3.region": region, "vfs.s3.no_sign_request": "true"}
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        parsed = urlparse(proxy)
        cfg["vfs.s3.proxy_scheme"] = parsed.scheme or "http"
        cfg["vfs.s3.proxy_host"] = parsed.hostname or ""
        cfg["vfs.s3.proxy_port"] = str(parsed.port or 80)
    ca_file = os.environ.get("SSL_CERT_FILE")
    if ca_file:
        cfg["vfs.s3.ca_file"] = ca_file
    return cfg


def download(census_cfg: dict[str, Any]) -> ad.AnnData:
    """Fetch one dataset's cells as an AnnData with raw integer counts in X.

    `census_cfg` is the `census` section of a config file. Genes with no counts in the
    downloaded cells are removed, because the Census pads every dataset to the full gene list.
    """
    import cellxgene_census
    import tiledbsoma

    context = tiledbsoma.SOMATileDBContext(tiledb_config=tiledb_config(census_cfg["region"]))
    obs_filter = f"dataset_id == '{census_cfg['dataset_id']}' and {census_cfg['obs_filter']}"
    with cellxgene_census.open_soma(uri=census_cfg["uri"], context=context) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism=census_cfg["organism"],
            obs_value_filter=obs_filter,
            obs_column_names=OBS_COLUMNS,
        )

    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    adata.var_names = adata.var["feature_id"].astype(str).to_numpy()
    # Columns arrive with every category in the whole Census; keep only the ones present.
    for col in adata.obs.select_dtypes("category"):
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()

    detected = adata.X.getnnz(axis=0) > 0
    adata = adata[:, detected].copy()
    adata.uns["census_version"] = census_cfg["version"]
    return adata


def is_adult_stage(stage: str) -> bool:
    """True unless the development stage says the donor was a child (or gives an age under 18).

    "unknown" and broad adult stages pass. Used to keep age from becoming a second batch variable.
    """
    match = re.match(r"(\d+)-year-old", stage)
    if match:
        return int(match.group(1)) >= 18
    return not re.search(r"child|juvenile|newborn|infant|pediatric|postnatal|fetal|embryo", stage)


def sample_cells(
    obs: pd.DataFrame, datasets: list[dict[str, Any]], max_cells_per_donor: int,
    min_cells_per_donor: int, seed: int,
) -> pd.DataFrame:  # fmt: skip
    """Choose donors and cells: rows of `obs` to download, with a "dataset" name column.

    Per dataset: donors with at least `min_cells_per_donor` cells are eligible; if the dataset
    entry has `max_donors`, that many are drawn at random. Then each donor contributes at most
    `max_cells_per_donor` cells, drawn at random. Capping cells per donor stops the two very
    large datasets (and the few donors with 20,000 cells) from dominating training.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for entry in datasets:
        part = obs[obs["dataset_id"] == entry["dataset_id"]]
        sizes = part.groupby("donor_id", observed=True).size()
        donors = np.sort(sizes[sizes >= min_cells_per_donor].index.astype(str).to_numpy())
        if entry.get("max_donors") and len(donors) > entry["max_donors"]:
            donors = np.sort(rng.choice(donors, size=entry["max_donors"], replace=False))
        for donor in donors:
            rows = part.index[part["donor_id"].astype(str) == donor].to_numpy()
            take = min(len(rows), max_cells_per_donor)
            chosen.append(part.loc[np.sort(rng.choice(rows, size=take, replace=False))])
            chosen[-1] = chosen[-1].assign(dataset=entry["name"])
    return pd.concat(chosen)


def download_many(cfg: dict[str, Any]) -> ad.AnnData:
    """Download a sample of cells from several Census datasets into one AnnData of raw counts.

    Only genes measured in every chosen dataset are kept. The Census pads every dataset to the
    full gene list, so a gene one lab never measured would otherwise appear as zero in all of its
    cells, which looks like "not expressed" and would be learned as a difference between labs.
    """
    import cellxgene_census
    import tiledbsoma

    census_cfg = cfg["census"]
    context = tiledbsoma.SOMATileDBContext(tiledb_config=tiledb_config(census_cfg["region"]))
    ids = [d["dataset_id"] for d in cfg["datasets"]]
    id_list = ", ".join(f"'{i}'" for i in ids)
    obs_filter = f"dataset_id in [{id_list}] and {census_cfg['obs_filter']}"
    with cellxgene_census.open_soma(uri=census_cfg["uri"], context=context) as census:
        human = census["census_data"]["homo_sapiens"]
        obs = (
            human.obs.read(
                value_filter=obs_filter,
                column_names=["soma_joinid", "dataset_id", "donor_id", "development_stage"],
            )
            .concat()
            .to_pandas()
        )
        obs = obs[obs["development_stage"].astype(str).map(is_adult_stage)]
        chosen = sample_cells(
            obs, cfg["datasets"], cfg["max_cells_per_donor"], cfg["min_cells_per_donor"],
            cfg["seed"],
        )  # fmt: skip

        info = census["census_info"]["datasets"].read().concat().to_pandas()
        rows = info.set_index("dataset_id").loc[ids, "soma_joinid"].to_numpy()
        presence = cellxgene_census.get_presence_matrix(census, census_cfg["organism"])
        measured_everywhere = np.asarray(presence[rows].min(axis=0).todense()).ravel() > 0
        var_ids = np.flatnonzero(measured_everywhere)

        adata = cellxgene_census.get_anndata(
            census,
            organism=census_cfg["organism"],
            obs_coords=np.sort(chosen["soma_joinid"].to_numpy()),
            var_coords=var_ids,
            obs_column_names=["soma_joinid", *OBS_COLUMNS],
        )

    names = chosen.set_index("soma_joinid")["dataset"]
    adata.obs["dataset"] = names.loc[adata.obs["soma_joinid"].to_numpy()].to_numpy()
    adata.obs_names = [f"cell{j}" for j in adata.obs["soma_joinid"]]
    adata.var_names = adata.var["feature_id"].astype(str).to_numpy()
    for col in adata.obs.select_dtypes("category"):
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
    adata.uns["census_version"] = census_cfg["version"]
    adata.uns["genes_measured_in_all_datasets"] = int(measured_everywhere.sum())
    return adata
