import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp
import torch

from cellmap.linear_ae import LinearAE, embed, fit_scaler, reconstruction_error, train_linear_ae

N_LATENT = 3
TRAIN_KWARGS = dict(n_latent=N_LATENT, n_epochs=150, batch_size=64, lr=3e-3)


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    """Run torch on one thread in this file, and put the setting back afterwards.

    These tensors are far too small to gain from threads, and when other jobs keep the cores
    busy, threads fighting over them make the training tests many times slower.
    """
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def make_low_rank(n_cells: int = 600, n_genes: int = 20, seed: int = 0) -> np.ndarray:
    """Three strong hidden factors of clearly different strength, plus a little noise."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_cells, N_LATENT)) * np.array([4.0, 2.5, 1.5])
    loadings = rng.normal(size=(N_LATENT, n_genes))
    x = factors @ loadings + 0.3 * rng.normal(size=(n_cells, n_genes)) + rng.normal(size=n_genes)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def x_train() -> np.ndarray:
    return make_low_rank()


@pytest.fixture(scope="module")
def trained(x_train):
    return train_linear_ae(x_train, seed=0, **TRAIN_KWARGS)


def scaled(model: LinearAE, x: np.ndarray) -> np.ndarray:
    return model.scale(torch.from_numpy(x)).numpy().astype(np.float64)


def pca(z: np.ndarray, d: int) -> tuple[np.ndarray, float]:
    """Top-d principal axes (genes x d) of centered `z` and PCA's reconstruction error.

    The error is the squared distance to the best d-dimensional subspace, summed over genes
    and averaged over cells: the sum of the discarded squared singular values over n_cells.
    """
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    return vt[:d].T, float((singular[d:] ** 2).sum() / len(z))


def test_loss_decreases_and_history_has_one_entry_per_epoch(trained):
    _, history = trained
    assert set(history) == {"train_mse", "seconds"}
    assert len(history["train_mse"]) == 150
    assert history["train_mse"][-1] < 0.1 * history["train_mse"][0]


def test_optimum_spans_the_pca_subspace(x_train, trained):
    model, _ = trained
    z = scaled(model, x_train)
    axes, pca_error = pca(z, N_LATENT)

    # Largest principal angle between the decoder's column space and the top-3 PCA subspace.
    decoder = model.decoder.weight.detach().numpy().astype(np.float64)  # genes x d
    angles = np.degrees(scipy.linalg.subspace_angles(decoder, axes))
    assert angles.max() < 2.0

    # Same subspace means (nearly) the same reconstruction error. PCA's is the best possible,
    # so the autoencoder can only be above it; the few percent left are the jitter of
    # mini-batch steps at a constant learning rate, tiny next to the starting error.
    error = reconstruction_error(model, x_train)
    assert pca_error <= error * (1 + 1e-6)
    assert error == pytest.approx(pca_error, rel=0.05)
    assert error - pca_error < 1e-3 * (z**2).sum(axis=1).mean()

    # The code is the PCA code up to an invertible linear map: a least-squares map from one to
    # the other leaves almost nothing unexplained.
    emb = embed(model, x_train).astype(np.float64)
    scores = z @ axes
    residual = scores - emb @ np.linalg.lstsq(emb, scores, rcond=None)[0]
    assert (residual**2).sum() / (scores**2).sum() < 1e-3


def test_embed_shape_dtype_and_sparse_input(x_train, trained):
    model, _ = trained
    emb = embed(model, x_train)
    assert emb.shape == (600, N_LATENT) and emb.dtype == np.float32
    np.testing.assert_allclose(embed(model, sp.csr_matrix(x_train), batch_size=7), emb, atol=1e-5)


def test_scaling_uses_training_cells_only(x_train, trained):
    model, _ = trained
    np.testing.assert_allclose(model.mean.numpy(), x_train.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(model.std.numpy(), x_train.std(axis=0, ddof=1), rtol=1e-5)
    np.testing.assert_allclose(scaled(model, x_train).mean(axis=0), 0.0, atol=1e-5)

    # Other cells are embedded with the stored training statistics: a cell's code is the same
    # whether it is embedded alone, with similar cells, or with wildly different ones.
    others = make_low_rank(n_cells=50, seed=7)
    wild = 100.0 * others + 50.0
    alone = embed(model, others[:1])
    np.testing.assert_allclose(embed(model, others)[:1], alone, atol=1e-5)
    np.testing.assert_allclose(embed(model, np.vstack([others[:1], wild]))[:1], alone, atol=1e-5)

    # Scoring validation cells during training does not change what is learned.
    with_val, history = train_linear_ae(x_train, seed=0, x_val=wild, **TRAIN_KWARGS)
    assert len(history["val_mse"]) == 150
    for name in ("mean", "std", "center"):
        np.testing.assert_array_equal(getattr(with_val, name).numpy(), getattr(model, name).numpy())
    np.testing.assert_array_equal(embed(with_val, x_train), embed(model, x_train))


def test_fit_scaler_clips_recenters_and_handles_constant_genes():
    # Gene 0: 99 zeros and one 100. mean 1, std 10, so the outlier's z-score is 9.9 -> not
    #         clipped at 10 but clipped at max_value=5, which shifts the mean of the z-scores
    #         from 0 to (99 * -0.1 + 5) / 100 = -0.049.
    # Gene 1: constant, so std is set to 1 and every scaled value is 0.
    x = np.zeros((100, 2), dtype=np.float32)
    x[0, 0] = 100.0
    x[:, 1] = 3.0
    mean, std, center = fit_scaler(x, max_value=5.0, chunk=30)
    np.testing.assert_allclose(mean, [1.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(std, [10.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(center, [-0.049, 0.0], atol=1e-6)
    for got, want in zip(
        fit_scaler(sp.csr_matrix(x), max_value=5.0), (mean, std, center), strict=True
    ):
        np.testing.assert_array_equal(got, want)

    model = LinearAE(n_genes=2, n_latent=1, max_value=5.0)
    for name, values in zip(("mean", "std", "center"), (mean, std, center), strict=True):
        getattr(model, name).copy_(torch.from_numpy(values))
    z = model.scale(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(z[0], [5.049, 0.0], atol=1e-5)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-6)


def test_same_seed_is_identical_and_other_seed_differs(x_train, trained):
    kwargs = {**TRAIN_KWARGS, "n_epochs": 3}
    first, _ = train_linear_ae(x_train, seed=0, **kwargs)
    again, _ = train_linear_ae(x_train, seed=0, **kwargs)
    other, _ = train_linear_ae(x_train, seed=1, **kwargs)
    np.testing.assert_array_equal(embed(first, x_train), embed(again, x_train))
    assert not np.allclose(embed(first, x_train), embed(other, x_train), atol=1e-3)


def test_rejects_impossible_sizes(x_train):
    with pytest.raises(ValueError, match="n_latent"):
        LinearAE(n_genes=5, n_latent=6)
    with pytest.raises(ValueError, match="different numbers of genes"):
        train_linear_ae(x_train, n_epochs=1, x_val=x_train[:, :5])
    with pytest.raises(ValueError, match="two training cells"):
        fit_scaler(x_train[:1])
