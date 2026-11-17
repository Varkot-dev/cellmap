import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellmap.baselines import harmony_correct, pca_embed, select_hvg
from cellmap.preprocess import normalize_log1p


def make_adata(seed: int = 0, n_cells: int = 400, n_genes: int = 120) -> ad.AnnData:
    """Log-normalized counts from 8 donors and 2 cell types; the first 20 genes mark the types.

    Donors d0..d4 are train, d5 and d6 val, d7 test, so cells of each split are interleaved.
    """
    rng = np.random.default_rng(seed)
    cell_type = rng.integers(0, 2, n_cells)
    donor = np.arange(n_cells) % 8
    rate = np.tile(rng.gamma(2.0, 1.0, n_genes), (n_cells, 1))
    rate[:, :10] *= np.where(cell_type == 0, 8.0, 0.2)[:, None]
    rate[:, 10:20] *= np.where(cell_type == 1, 8.0, 0.2)[:, None]
    obs = pd.DataFrame(
        {
            "donor_id": [f"d{d}" for d in donor],
            "cell_type": np.where(cell_type == 0, "T cell", "B cell"),
            "split": np.select([donor < 5, donor < 7], ["train", "val"], "test"),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])
    adata = ad.AnnData(X=sp.csr_matrix(rng.poisson(rate).astype(np.float32)), obs=obs, var=var)
    normalize_log1p(adata)
    return adata


def scramble_non_training_cells(adata: ad.AnnData, train: np.ndarray) -> ad.AnnData:
    """Copy of `adata` whose non-training rows are replaced by unrelated values."""
    x = adata.X.toarray()
    rng = np.random.default_rng(99)
    x[~train] = rng.gamma(1.0, 3.0, size=x[~train].shape) * rng.integers(0, 2, x[~train].shape)
    out = adata.copy()
    out.X = sp.csr_matrix(x.astype(np.float32))
    return out


@pytest.fixture
def adata() -> ad.AnnData:
    return make_adata()


@pytest.fixture
def train(adata) -> np.ndarray:
    return (adata.obs["split"] == "train").to_numpy()


@pytest.mark.parametrize("batch_key", ["donor_id", None])
def test_hvg_returns_requested_number_and_finds_marker_genes(adata, train, batch_key):
    mask = select_hvg(adata, train, n_top_genes=30, batch_key=batch_key)
    assert mask.dtype == bool and mask.shape == (adata.n_vars,)
    assert mask.sum() == 30
    # 20 of the 120 genes differ between the two cell types; chance would pick about 5 of them.
    assert mask[:20].sum() >= 15


@pytest.mark.parametrize("batch_key", ["donor_id", None])
def test_hvg_ignores_non_training_cells(adata, train, batch_key):
    scrambled = scramble_non_training_cells(adata, train)
    assert not np.allclose(scrambled.X.toarray(), adata.X.toarray())
    np.testing.assert_array_equal(
        select_hvg(adata, train, n_top_genes=30, batch_key=batch_key),
        select_hvg(scrambled, train, n_top_genes=30, batch_key=batch_key),
    )
    # The test has teeth: selecting on all cells of the scrambled data gives a different answer.
    everyone = np.ones(adata.n_obs, dtype=bool)
    assert not np.array_equal(
        select_hvg(adata, everyone, n_top_genes=30, batch_key=batch_key),
        select_hvg(scrambled, everyone, n_top_genes=30, batch_key=batch_key),
    )


def test_hvg_does_not_modify_input(adata, train):
    before = adata.X.toarray().copy()
    select_hvg(adata, train, n_top_genes=30)
    np.testing.assert_array_equal(adata.X.toarray(), before)
    assert list(adata.var.columns) == [] and "hvg" not in adata.uns


def test_masks_must_be_boolean(adata, train):
    with pytest.raises(ValueError, match="boolean mask"):
        select_hvg(adata, np.flatnonzero(train), n_top_genes=30)
    with pytest.raises(ValueError, match="boolean mask"):
        pca_embed(adata, np.ones(adata.n_vars, dtype=bool), np.flatnonzero(train), n_comps=5)


def test_pca_shape_dtype_and_centered_on_training_cells(adata, train):
    genes = select_hvg(adata, train, n_top_genes=40)
    emb = pca_embed(adata, genes, train, n_comps=10)
    assert emb.shape == (adata.n_obs, 10) and emb.dtype == np.float32
    np.testing.assert_allclose(emb[train].mean(axis=0), 0.0, atol=1e-4)
    # Components are uncorrelated and ordered by variance on the cells they were fitted on.
    cov = np.cov(emb[train], rowvar=False)
    variances = np.diag(cov)
    assert (np.diff(variances) <= 1e-4).all()
    np.testing.assert_allclose(cov - np.diag(variances), 0.0, atol=1e-3)
    # The first component is the cell-type axis.
    is_t = (adata.obs["cell_type"] == "T cell").to_numpy()
    assert abs(emb[is_t, 0].mean() - emb[~is_t, 0].mean()) > 5 * emb[is_t, 0].std()


def test_pca_fit_ignores_non_training_cells(adata, train):
    genes = np.zeros(adata.n_vars, dtype=bool)
    genes[:40] = True
    scrambled = scramble_non_training_cells(adata, train)
    emb = pca_embed(adata, genes, train, n_comps=10)
    emb_scrambled = pca_embed(scrambled, genes, train, n_comps=10)
    np.testing.assert_array_equal(emb[train], emb_scrambled[train])
    assert not np.allclose(emb[~train], emb_scrambled[~train])


def test_pca_embeds_a_cell_the_same_whether_or_not_others_are_present(adata, train):
    # Applying the frozen transform is row-wise: dropping test cells cannot move the others.
    genes = np.zeros(adata.n_vars, dtype=bool)
    genes[:40] = True
    keep = (adata.obs["split"] != "test").to_numpy()
    emb = pca_embed(adata, genes, train, n_comps=10)
    emb_subset = pca_embed(adata[keep].copy(), genes, train[keep], n_comps=10)
    np.testing.assert_allclose(emb[keep], emb_subset, atol=1e-5)


def test_pca_matches_direct_svd_up_to_sign():
    rng = np.random.default_rng(1)
    # Distinct column scales give well-separated singular values, so the axes are well defined.
    x = (rng.normal(size=(30, 6)) * np.array([9.0, 7.0, 5.0, 3.0, 2.0, 1.0])) @ rng.normal(
        size=(6, 6)
    )
    x = x + rng.normal(size=6) * 4.0
    adata = ad.AnnData(X=sp.csr_matrix(x.astype(np.float32)))
    train = np.arange(30) < 20
    genes = np.array([True, True, True, True, True, False])

    emb = pca_embed(adata, genes, train, n_comps=3, max_value=np.inf)

    sel = x[:, genes]
    z = (sel - sel[train].mean(axis=0)) / sel[train].std(axis=0, ddof=1)
    _, s, vt = np.linalg.svd(z[train], full_matrices=False)
    expected = z @ vt[:3].T
    signs = np.sign((emb * expected).sum(axis=0))
    np.testing.assert_allclose(emb, expected * signs, rtol=1e-3, atol=1e-3)
    # Variance along each axis on training cells is the squared singular value over n - 1.
    np.testing.assert_allclose(emb[train].var(axis=0, ddof=1), s[:3] ** 2 / 19, rtol=1e-3)


def test_pca_clips_extreme_values_and_handles_constant_genes():
    rng = np.random.default_rng(2)
    x = rng.poisson(3.0, size=(200, 5)).astype(np.float32)
    x[:, 3] = 0.0  # never expressed in anyone
    x[:, 4] = 0.0
    x[0, 4] = 1.0  # expressed in a single training cell: z-score about 14 before clipping
    adata = ad.AnnData(X=sp.csr_matrix(x))
    train = np.arange(200) < 150
    genes = np.ones(5, dtype=bool)

    clipped = pca_embed(adata, genes, train, n_comps=4, max_value=10.0)
    unclipped = pca_embed(adata, genes, train, n_comps=4, max_value=np.inf)
    assert np.isfinite(clipped).all() and np.isfinite(unclipped).all()
    assert np.abs(unclipped[0]).max() > 12
    assert np.abs(clipped[0]).max() < np.abs(unclipped[0]).max()


def test_pca_rejects_too_many_components(adata, train):
    genes = np.zeros(adata.n_vars, dtype=bool)
    genes[:8] = True
    with pytest.raises(ValueError, match="n_comps"):
        pca_embed(adata, genes, train, n_comps=9)


def make_batch_shifted_pcs(seed: int = 0, n_cells: int = 600):
    """Two cell types 10 apart on axis 0; batch "b" is shifted by 3 on axis 1."""
    rng = np.random.default_rng(seed)
    cell_type = rng.integers(0, 2, n_cells)
    batch = np.where(rng.integers(0, 2, n_cells) == 0, "a", "b")
    pcs = rng.normal(size=(n_cells, 8)).astype(np.float32)
    pcs[:, 0] += 10.0 * cell_type
    pcs[:, 1] += 3.0 * (batch == "b")
    return pcs, batch, cell_type


def batch_gap(x: np.ndarray, batch: np.ndarray, cells: np.ndarray) -> float:
    """Distance between the two batches' centroids among the selected cells."""
    return float(
        np.linalg.norm(x[cells & (batch == "a")].mean(axis=0) - x[cells & (batch == "b")].mean(0))
    )


def test_harmony_removes_batch_shift_and_keeps_cell_types_apart():
    pcs, batch, cell_type = make_batch_shifted_pcs()
    before = pcs.copy()
    corrected = harmony_correct(pcs, batch, seed=0)

    assert corrected.shape == pcs.shape and corrected.dtype == np.float32
    np.testing.assert_array_equal(pcs, before)  # input not modified
    for t in (0, 1):
        cells = cell_type == t
        assert batch_gap(pcs, batch, cells) > 2.5
        assert batch_gap(corrected, batch, cells) < 0.5 * batch_gap(pcs, batch, cells)
    type_gap = np.linalg.norm(
        corrected[cell_type == 0].mean(axis=0) - corrected[cell_type == 1].mean(axis=0)
    )
    assert type_gap > 8.0


def test_harmony_preserves_row_order():
    pcs, batch, _ = make_batch_shifted_pcs()
    corrected = harmony_correct(pcs, batch, seed=0)
    # Each cell stays closest to its own uncorrected position on the untouched axes (2..7 carry
    # neither cell type nor batch), which would fail if rows were returned in another order.
    np.testing.assert_allclose(corrected[:, 2:], pcs[:, 2:], atol=0.5)
    # And permuting the input permutes the output the same way (up to the algorithm's own noise).
    order = np.random.default_rng(5).permutation(len(pcs))
    permuted = harmony_correct(pcs[order], batch[order], seed=0)
    assert np.abs(permuted - corrected[order]).mean() < 0.2


def test_harmony_is_deterministic_for_a_fixed_seed():
    pcs, batch, _ = make_batch_shifted_pcs()
    np.testing.assert_array_equal(
        harmony_correct(pcs, batch, seed=3), harmony_correct(pcs, batch, seed=3)
    )


def test_harmony_rejects_bad_input():
    pcs, batch, _ = make_batch_shifted_pcs()
    with pytest.raises(ValueError, match="same cells"):
        harmony_correct(pcs, batch[:-1])
    with pytest.raises(ValueError, match="two batches"):
        harmony_correct(pcs, np.full(len(pcs), "a"))
    with pytest.raises(ValueError, match="square"):
        harmony_correct(pcs[:8], batch[:8])
