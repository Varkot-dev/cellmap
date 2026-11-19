import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def toy_adata() -> ad.AnnData:
    """60 cells from 10 donors, 8 genes (2 mitochondrial), Poisson counts."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 8
    x = sp.csr_matrix(rng.poisson(5.0, size=(n_cells, n_genes)).astype(np.float32))
    obs = pd.DataFrame(
        {
            "donor_id": [f"d{i % 10}" for i in range(n_cells)],
            "cell_type": ["T cell" if i % 3 else "B cell" for i in range(n_cells)],
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"feature_name": ["MT-CO1", "MT-ND1", "CD3E", "CD19", "LYZ", "NKG7", "MS4A1", "ACTB"]},
        index=[f"ENSG{i}" for i in range(n_genes)],
    )
    return ad.AnnData(X=x, obs=obs, var=var)
