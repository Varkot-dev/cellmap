import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellmap.qc import add_qc_metrics, cell_filter_report, genes_expressed_in, rare_cell_types


def hand_adata() -> ad.AnnData:
    # 3 cells x 4 genes; the first gene is mitochondrial.
    x = sp.csr_matrix(np.array([[10, 0, 30, 60], [0, 0, 5, 0], [1, 1, 1, 1]], dtype=np.float32))
    var = pd.DataFrame({"feature_name": ["MT-CO1", "CD3E", "LYZ", "ACTB"]}, index=list("abcd"))
    obs = pd.DataFrame({"cell_type": ["T", "T", "B"]}, index=["c0", "c1", "c2"])
    return ad.AnnData(X=x, obs=obs, var=var)


def test_qc_metrics_match_hand_computation():
    adata = hand_adata()
    add_qc_metrics(adata)
    assert adata.obs["total_counts"].tolist() == [100, 5, 4]
    assert adata.obs["n_genes"].tolist() == [3, 1, 4]
    np.testing.assert_allclose(adata.obs["pct_mt"], [10.0, 0.0, 25.0])


def test_cell_filter_counts_each_failure():
    adata = hand_adata()
    add_qc_metrics(adata)
    keep, report = cell_filter_report(adata.obs, min_counts=5, min_genes=2, max_pct_mt=20.0)
    # c0 passes; c1 fails min_genes; c2 fails min_counts and max_pct_mt.
    assert keep.tolist() == [True, False, False]
    assert report == {
        "cells_before": 3,
        "fail_min_counts": 1,
        "fail_min_genes": 1,
        "fail_max_pct_mt": 1,
        "cells_after": 1,
    }


def test_rare_cell_types():
    obs = pd.DataFrame({"cell_type": ["T"] * 5 + ["B"] * 2 + ["NK"]})
    assert rare_cell_types(obs, min_cells=3) == ["B", "NK"]
    assert rare_cell_types(obs, min_cells=1) == []


def test_gene_filter_only_looks_at_selected_cells():
    adata = hand_adata()
    train = np.array([True, True, False])
    # Gene "b" (CD3E) is detected only in cell c2, which is not a training cell.
    assert genes_expressed_in(adata, train, min_cells=1).tolist() == [True, False, True, True]
    assert genes_expressed_in(adata, train, min_cells=2).tolist() == [False, False, True, False]
