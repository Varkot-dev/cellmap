import numpy as np
import pytest
import torch

from cellmap.sae import (
    SparseAutoencoder,
    activations,
    eta_squared,
    explained_variance,
    reconstruct,
    remove_features,
    train_sae,
)


def test_decoder_columns_have_unit_length():
    model = SparseAutoencoder(8, 32)
    assert torch.allclose(model.W_dec.norm(dim=0), torch.ones(32), atol=1e-6)


def test_worked_example_reconstruction_and_removal():
    model = SparseAutoencoder(2, 2)
    with torch.no_grad():
        model.W_dec.copy_(torch.tensor([[1.0, 0.6], [0.0, 0.8]]))
        model.b_dec.zero_()
        # encoder that returns f = (2, 0.5) for the input below
        model.enc.weight.copy_(torch.eye(2))
        model.enc.bias.copy_(torch.tensor([0.0, 0.0]))
    x = np.array([[2.0, 0.5]], dtype=np.float32)
    assert activations(model, x).tolist() == [[2.0, 0.5]]
    assert reconstruct(model, x) == pytest.approx(np.array([[2.3, 0.4]]))
    # removing feature 2 subtracts 0.5 * (0.6, 0.8) from x itself
    assert remove_features(model, x, [1]) == pytest.approx(np.array([[1.7, 0.1]]))
    assert remove_features(model, x, []) == pytest.approx(x)


def test_sae_recovers_a_sparse_planted_dictionary():
    rng = np.random.default_rng(0)
    atoms = rng.normal(size=(16, 8))
    atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    codes = rng.exponential(size=(3000, 16)) * (rng.random((3000, 16)) < 0.1)
    x = (codes @ atoms).astype(np.float32)
    model = train_sae(x, n_features=64, l1=0.03, n_epochs=60, batch_size=64, lr=3e-3, seed=0)
    assert explained_variance(x, reconstruct(model, x)) > 0.9
    active = (activations(model, x) > 0).sum(axis=1).mean()
    assert active < 32  # sparse: well under half of the 64 features per row


def test_explained_variance_and_eta_squared():
    x = np.array([[0.0], [2.0]])
    assert explained_variance(x, x) == 1.0
    assert explained_variance(x, np.array([[1.0], [1.0]])) == 0.0
    assert eta_squared(np.array([1.0, 1.0, 3.0, 3.0]), np.array([0, 0, 1, 1])) == 1.0
    assert eta_squared(np.array([1.0, 3.0, 1.0, 3.0]), np.array([0, 0, 1, 1])) == 0.0
    assert eta_squared(np.ones(4), np.array([0, 0, 1, 1])) == 0.0
