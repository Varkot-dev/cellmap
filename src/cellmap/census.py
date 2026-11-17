"""Download raw counts from the CZ CELLxGENE Census."""

import os
from typing import Any
from urllib.parse import urlparse

import anndata as ad

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
