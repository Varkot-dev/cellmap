import math

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.stats
import torch

from cellmap.vae import (
    VAE,
    VAR_EPS,
    batch_codes,
    elbo_terms,
    embed,
    kl_normal,
    kl_weight_at,
    nb_log_likelihood,
    train_vae,
    vae_loss,
)

F64 = torch.float64


# ---------------------------------------------------------------- negative binomial likelihood


@pytest.mark.parametrize(
    ("x", "mu", "theta"),
    [(0, 0.5, 1.0), (3, 2.0, 0.7), (10, 4.0, 5.0), (0, 30.0, 0.1), (250, 180.0, 12.0)],
)
def test_nb_log_likelihood_matches_scipy(x, mu, theta):
    # scipy's nbinom(n, p) counts failures before the n-th success. It is the same distribution
    # with n = theta and p = theta / (theta + mu): mean n(1-p)/p = mu, variance mu + mu^2/theta.
    expected = scipy.stats.nbinom.logpmf(x, n=theta, p=theta / (theta + mu))
    got = nb_log_likelihood(*(torch.tensor(v, dtype=F64) for v in (x, mu, theta)))
    assert got.item() == pytest.approx(expected, abs=1e-6)


def test_nb_log_likelihood_float32_and_broadcasting():
    x = torch.tensor([[0.0, 1.0, 7.0], [2.0, 0.0, 40.0]])
    mu = torch.tensor([[0.3, 2.0, 5.0], [1.0, 0.1, 55.0]])
    theta = torch.tensor([0.5, 2.0, 8.0])  # one value per gene, broadcast over cells
    got = nb_log_likelihood(x, mu, theta)
    expected = scipy.stats.nbinom.logpmf(x.numpy(), n=theta.numpy(), p=(theta / (theta + mu)))
    assert got.shape == (2, 3) and got.dtype == torch.float32
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-4)


def test_nb_probabilities_sum_to_one():
    x = torch.arange(0, 400, dtype=F64)
    log_p = nb_log_likelihood(x, torch.tensor(2.0, dtype=F64), torch.tensor(1.5, dtype=F64))
    assert log_p.exp().sum().item() == pytest.approx(1.0, abs=1e-6)


def test_nb_mean_and_variance_are_mu_and_mu_plus_mu_squared_over_theta():
    mu, theta = 3.0, 2.0
    x = torch.arange(0, 2000, dtype=F64)
    p = nb_log_likelihood(x, torch.tensor(mu, dtype=F64), torch.tensor(theta, dtype=F64)).exp()
    mean = (p * x).sum().item()
    variance = (p * (x - mean) ** 2).sum().item()
    assert mean == pytest.approx(mu, abs=1e-6)
    assert variance == pytest.approx(mu + mu**2 / theta, abs=1e-5)  # 3 + 9/2 = 7.5


def test_nb_approaches_poisson_as_theta_grows():
    x = torch.tensor([0.0, 1.0, 4.0, 12.0], dtype=F64)
    mu = torch.tensor(3.5, dtype=F64)
    poisson = scipy.stats.poisson.logpmf(x.numpy(), 3.5)
    far = nb_log_likelihood(x, mu, torch.tensor(1.0, dtype=F64)).numpy()
    near = nb_log_likelihood(x, mu, torch.tensor(1e7, dtype=F64)).numpy()
    np.testing.assert_allclose(near, poisson, atol=1e-4)
    assert np.abs(far - poisson).max() > 0.1  # at theta = 1 it is clearly not a Poisson


def test_nb_is_finite_when_the_mean_is_zero():
    out = nb_log_likelihood(torch.tensor([0.0, 3.0]), torch.tensor(0.0), torch.tensor(1.0))
    assert torch.isfinite(out).all()
    assert out[0].item() == pytest.approx(0.0, abs=1e-6)  # a zero count is certain
    assert out[1].item() < -30  # a positive count is (numerically) impossible


# ------------------------------------------------------------------------------------ KL term


def test_kl_normal_matches_hand_computed_value():
    # mean = [1, 0], variance = [1, 4]. Per dimension 0.5 * (mean^2 + var - log(var) - 1):
    #   dim 0: 0.5 * (1 + 1 - 0        - 1) = 0.5
    #   dim 1: 0.5 * (0 + 4 - log(4)   - 1) = 0.5 * (3 - 1.3862944) = 0.8068528
    #   sum = 1.3068528
    kl = kl_normal(torch.tensor([[1.0, 0.0]]), torch.log(torch.tensor([[1.0, 4.0]])))
    assert kl.shape == (1,)
    assert kl.item() == pytest.approx(1.3068528, abs=1e-6)


def test_kl_normal_is_zero_at_the_prior():
    assert kl_normal(torch.zeros(3, 5), torch.zeros(3, 5)).abs().max().item() == 0.0


def test_kl_normal_is_never_negative_and_matches_torch():
    generator = torch.Generator().manual_seed(0)
    mean = torch.randn(200, 6, generator=generator, dtype=F64) * 3
    logvar = torch.randn(200, 6, generator=generator, dtype=F64) * 2
    kl = kl_normal(mean, logvar)
    assert (kl >= 0).all()
    q = torch.distributions.Normal(mean, torch.exp(0.5 * logvar))
    prior = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    reference = torch.distributions.kl_divergence(q, prior).sum(dim=-1)
    torch.testing.assert_close(kl, reference, atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------- the complete loss


def test_vae_loss_matches_hand_computed_example():
    # Two cells, two genes, two latent dimensions. theta = [1, 2] per gene.
    #
    # Reconstruction. With p = theta / (theta + mu), P(x) = C(x + theta - 1, x) p^theta (1-p)^x.
    #   cell A, gene 1: x=2, mu=1, theta=1 -> p=1/2, P = 1 * (1/2)   * (1/2)^2 = 1/8
    #   cell A, gene 2: x=0, mu=3, theta=2 -> p=2/5, P = 1 * (2/5)^2            = 4/25
    #     -log(1/8 * 4/25) = -log(0.02) = 3.9120230
    #   cell B, gene 1: x=1, mu=2, theta=1 -> p=1/3, P = 1 * (1/3)   * (2/3)   = 2/9
    #   cell B, gene 2: x=1, mu=1, theta=2 -> p=2/3, P = 2 * (2/3)^2 * (1/3)   = 8/27
    #     -log(2/9 * 8/27) = -log(16/243) = 2.7204727
    #   recon = (3.9120230 + 2.7204727) / 2 = 3.3162479
    #
    # KL.
    #   cell A: mean=[1, 0], var=[1, 4] -> 1.3068528 (worked out in the KL test above)
    #   cell B: mean=[0, 0], var=[1, 1] -> 0
    #   kl = (1.3068528 + 0) / 2 = 0.6534264
    #
    # Total with kl_weight = 0.5: 3.3162479 + 0.5 * 0.6534264 = 3.6429611
    x = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=F64)
    mu = torch.tensor([[1.0, 3.0], [2.0, 1.0]], dtype=F64)
    theta = torch.tensor([1.0, 2.0], dtype=F64)
    mean = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=F64)
    logvar = torch.log(torch.tensor([[1.0, 4.0], [1.0, 1.0]], dtype=F64))

    recon, kl, total = vae_loss(x, mu, theta, mean, logvar, kl_weight=0.5)
    assert recon.item() == pytest.approx(-(math.log(0.02) + math.log(16 / 243)) / 2, abs=1e-7)
    assert recon.item() == pytest.approx(3.3162479, abs=1e-5)
    assert kl.item() == pytest.approx(0.6534264, abs=1e-5)
    assert total.item() == pytest.approx(3.6429611, abs=1e-5)

    # kl_weight only changes the total, and kl_weight = 1 is minus the ELBO.
    recon1, kl1, total1 = vae_loss(x, mu, theta, mean, logvar)
    assert (recon1, kl1) == (recon, kl)
    assert total1.item() == pytest.approx(3.3162479 + 0.6534264, abs=1e-5)


def test_kl_weight_warm_up_is_a_linear_ramp_from_zero():
    assert [kl_weight_at(e, 2.0, 4) for e in range(6)] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.0]
    assert kl_weight_at(0, 2.0, 0) == 2.0  # no warm-up: full weight from the first epoch


# -------------------------------------------------------------------------- model structure


@pytest.fixture
def model() -> VAE:
    torch.manual_seed(0)
    return VAE(n_genes=12, n_batches=3, n_latent=4, n_hidden=16).eval()


@pytest.fixture
def counts() -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(0).poisson(3.0, (5, 12)).astype(np.float32))


def test_encoder_ignores_the_batch_label(model, counts):
    outputs = []
    for donor in (0, 2):
        torch.manual_seed(1)  # same noise for z, so any difference comes from the donor
        outputs.append(model(counts, torch.full((5,), donor)))
    torch.testing.assert_close(outputs[0].mean, outputs[1].mean, atol=0, rtol=0)
    torch.testing.assert_close(outputs[0].logvar, outputs[1].logvar, atol=0, rtol=0)
    assert not torch.allclose(outputs[0].mu, outputs[1].mu)  # the decoder does use the donor


def test_proportions_sum_to_one_and_nb_mean_sums_to_library_size(model, counts):
    batch = torch.tensor([0, 1, 2, 0, -1])
    proportions = model.decode(torch.randn(5, 4), batch)
    assert (proportions > 0).all()
    torch.testing.assert_close(proportions.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=0)

    out = model(counts, batch)
    torch.testing.assert_close(out.mu.sum(dim=-1), counts.sum(dim=-1), atol=0, rtol=1e-5)
    assert out.theta.shape == (12,) and (out.theta > 0).all()


def test_posterior_variance_has_a_floor(model, counts):
    with torch.no_grad():
        model.z_raw_var.bias.fill_(-100.0)  # push the encoder's variance output to zero
    _, logvar = model.encode(counts)
    torch.testing.assert_close(
        logvar, torch.full_like(logvar, math.log(VAR_EPS)), atol=1e-4, rtol=0
    )


def test_unseen_donor_is_decoded_as_the_average_training_donor(model):
    vectors = model.batch_vector(torch.tensor([1, -1]))
    torch.testing.assert_close(vectors, torch.tensor([[0.0, 1.0, 0.0], [1 / 3, 1 / 3, 1 / 3]]))
    for bad in (3, -2):
        with pytest.raises(ValueError, match="batch codes"):
            model.batch_vector(torch.tensor([bad]))


def test_batch_codes_maps_unknown_donors_to_minus_one():
    codes = batch_codes(np.array(["d2", "d0", "new", "d1"]), np.array(["d0", "d1", "d2"]))
    assert codes.tolist() == [2, 0, -1, 1]


# ---------------------------------------------------------------------------------- training


def make_counts(n_cells: int = 360, n_genes: int = 60, seed: int = 0):
    """NB counts from 3 planted cell types and 2 donors.

    Each type has its own block of 10 marker genes (8 times higher). Donor 1 has another block
    of 10 genes doubled, a batch effect unrelated to type. Library sizes vary threefold.
    """
    rng = np.random.default_rng(seed)
    cell_type = np.arange(n_cells) % 3
    donor = (np.arange(n_cells) // 3) % 2
    base = rng.gamma(2.0, 1.0, n_genes)
    rate = np.tile(base, (n_cells, 1))
    for t in range(3):
        rate[cell_type == t, 10 * t : 10 * (t + 1)] *= 8.0
    rate[donor == 1, 30:40] *= 2.0
    rate /= rate.sum(axis=1, keepdims=True)
    library = rng.uniform(300, 900, n_cells)
    mean = rate * library[:, None]
    counts = rng.negative_binomial(5.0, 5.0 / (5.0 + mean)).astype(np.float32)
    return counts, donor, cell_type


TRAIN_KWARGS = dict(n_latent=4, n_hidden=32, n_epochs=40, batch_size=64, kl_warmup_epochs=5)


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


@pytest.fixture(scope="module")
def data():
    return make_counts()


@pytest.fixture(scope="module")
def trained(data):
    counts, donor, _ = data
    return train_vae(counts, donor, seed=0, **TRAIN_KWARGS)


def nearest_centroid_accuracy(emb: np.ndarray, labels: np.ndarray) -> float:
    """Fit centroids on even rows, classify odd rows by the nearest centroid."""
    fit, score = np.arange(len(emb)) % 2 == 0, np.arange(len(emb)) % 2 == 1
    classes = np.unique(labels)
    centroids = np.stack([emb[fit & (labels == c)].mean(axis=0) for c in classes])
    distances = ((emb[score][:, None, :] - centroids[None]) ** 2).sum(axis=-1)
    return float((classes[distances.argmin(axis=1)] == labels[score]).mean())


def test_training_lowers_the_loss_and_separates_planted_types(data, trained):
    counts, _, cell_type = data
    model, history = trained
    assert set(history) == {"kl_weight", "train_recon", "train_kl", "train_total", "seconds"}
    assert all(len(values) == 40 for values in history.values())
    assert history["kl_weight"][:6] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert history["train_recon"][-1] < history["train_recon"][0] - 10
    assert history["train_total"][-1] < history["train_total"][5]  # same weight from epoch 5 on
    for e in (0, 3, 39):
        total = history["train_recon"][e] + history["kl_weight"][e] * history["train_kl"][e]
        assert history["train_total"][e] == pytest.approx(total, rel=1e-5)

    emb = embed(model, counts)
    assert emb.shape == (360, 4) and emb.dtype == np.float32
    assert not model.training
    assert nearest_centroid_accuracy(emb, cell_type) > 0.95


def test_embedding_does_not_depend_on_companion_cells_or_batch_size(data, trained):
    counts, _, _ = data
    model, _ = trained
    full = embed(model, counts)
    np.testing.assert_allclose(embed(model, counts[:7], batch_size=3), full[:7], atol=1e-5)
    np.testing.assert_array_equal(embed(model, sp.csr_matrix(counts)), full)


def test_very_large_kl_weight_collapses_the_posterior(data, trained):
    counts, donor, cell_type = data
    collapsed, history = train_vae(
        counts, donor, seed=0, kl_weight=500.0, **{**TRAIN_KWARGS, "kl_warmup_epochs": 0}
    )
    emb = embed(collapsed, counts)
    healthy = embed(trained[0], counts)
    # Every cell gets (almost) the same code: the prior mean. Nothing is left to tell types apart.
    assert emb.var(axis=0).max() < 0.01
    assert healthy.var(axis=0).min() > 0.2
    assert history["train_kl"][-1] < 0.05  # about 3 nats per cell at kl_weight = 1
    _, kl = elbo_terms(collapsed, counts, donor)
    assert kl < 0.05
    # And the reconstruction pays for it: the decoder can only use the donor and library size.
    assert history["train_recon"][-1] > trained[1]["train_recon"][-1] + 5


def test_zero_kl_weight_lets_the_kl_term_grow(data, trained):
    counts, donor, cell_type = data
    free, history = train_vae(counts, donor, seed=0, kl_weight=0.0, **TRAIN_KWARGS)
    assert history["train_total"] == history["train_recon"]  # the KL is measured but not charged
    _, kl_free = elbo_terms(free, counts, donor)
    _, kl_one = elbo_terms(trained[0], counts, donor)
    assert kl_free > 3 * kl_one
    assert nearest_centroid_accuracy(embed(free, counts), cell_type) > 0.95  # still an autoencoder


def test_same_seed_is_identical_and_other_seed_differs(data, trained):
    counts, donor, _ = data
    model, history = trained
    again, history_again = train_vae(counts, donor, seed=0, **TRAIN_KWARGS)
    other, _ = train_vae(counts, donor, seed=1, **TRAIN_KWARGS)
    np.testing.assert_array_equal(embed(again, counts), embed(model, counts))
    for key in ("train_recon", "train_kl", "train_total"):
        assert history_again[key] == history[key]
    assert not np.allclose(embed(other, counts), embed(model, counts), atol=1e-3)


def test_validation_is_recorded_and_does_not_change_training(data, trained):
    counts, donor, _ = data
    val_counts, _, _ = make_counts(n_cells=90, seed=5)
    val_batch = np.full(90, -1)  # donors never seen in training
    model, history = train_vae(
        sp.csr_matrix(counts),  # sparse training input, to check it matches the dense run
        donor,
        seed=0,
        counts_val=sp.csr_matrix(val_counts),
        batch_val=val_batch,
        **TRAIN_KWARGS,
    )
    assert len(history["val_recon"]) == len(history["val_kl"]) == 40
    assert history["val_recon"][-1] < history["val_recon"][0]
    assert history["val_total"][-1] == pytest.approx(
        history["val_recon"][-1] + history["val_kl"][-1]
    )
    assert history["train_recon"] == trained[1]["train_recon"]
    np.testing.assert_array_equal(embed(model, counts), embed(trained[0], counts))

    # elbo_terms is repeatable and leaves the model in the mode it found it in.
    assert elbo_terms(model, val_counts, val_batch) == elbo_terms(model, val_counts, val_batch)
    assert elbo_terms(model, val_counts, val_batch) == pytest.approx(
        (history["val_recon"][-1], history["val_kl"][-1])
    )
    assert not model.training


def test_train_vae_rejects_bad_batch_input(data):
    counts, donor, _ = data
    with pytest.raises(ValueError, match="do not match"):
        train_vae(counts, donor[:-1], n_epochs=1)
    with pytest.raises(ValueError, match="integer codes"):
        train_vae(counts, donor.astype(str), n_epochs=1)
    with pytest.raises(ValueError, match="known donor"):
        train_vae(counts, np.full(len(donor), -1), n_epochs=1)
    with pytest.raises(ValueError, match="without batch_val"):
        train_vae(counts, donor, n_epochs=1, counts_val=counts)
