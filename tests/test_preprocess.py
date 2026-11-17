import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellmap.preprocess import normalize_log1p


def test_normalize_matches_hand_computation():
    # Cell 0 has 4 UMIs, cell 1 has 40 with the same proportions.
    adata = ad.AnnData(X=sp.csr_matrix(np.array([[1, 3, 0], [10, 30, 0]], dtype=np.float32)))
    normalize_log1p(adata, target_sum=100)
    expected = np.log1p(np.array([[25.0, 75.0, 0.0], [25.0, 75.0, 0.0]]))
    np.testing.assert_allclose(adata.X.toarray(), expected, rtol=1e-6)


def test_raw_counts_are_kept(toy_adata):
    before = toy_adata.X.toarray().copy()
    normalize_log1p(toy_adata)
    np.testing.assert_array_equal(toy_adata.layers["counts"].toarray(), before)
    assert not np.allclose(toy_adata.X.toarray(), before)


def test_each_cell_is_normalized_independently(toy_adata):
    # Normalizing a subset of cells gives the same values as normalizing all, then subsetting.
    subset = toy_adata[:10].copy()
    normalize_log1p(toy_adata)
    normalize_log1p(subset)
    np.testing.assert_allclose(subset.X.toarray(), toy_adata.X[:10].toarray(), rtol=1e-6)


def test_zero_count_cell_does_not_divide_by_zero():
    adata = ad.AnnData(X=sp.csr_matrix(np.array([[0, 0], [2, 2]], dtype=np.float32)))
    normalize_log1p(adata, target_sum=10)
    assert np.isfinite(adata.X.toarray()).all()
    assert adata.X.toarray()[0].tolist() == [0.0, 0.0]
