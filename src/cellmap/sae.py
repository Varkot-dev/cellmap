"""A sparse autoencoder (SAE) for reading what an embedding stores, written from scratch.

Idea. A 128-number embedding can hold more than 128 "things" if each cell only uses a few of
them. An SAE learns a dictionary of many directions (features) and writes each cell as a
non-negative, mostly-zero combination of them:

    f     = ReLU(W_enc (x - b_dec) + b_enc)      feature activations, most of them 0
    x_hat = W_dec f + b_dec                      the reconstruction

trained to minimise  mean ||x - x_hat||^2 + lambda * mean sum(f).  The squared error keeps the
reconstruction faithful; the L1 term (the sum of activations, which are non-negative) pushes
most activations to exactly zero, so each cell is explained by a handful of features. Each
column of W_dec is kept at length 1, otherwise the model could shrink f and grow W_dec to dodge
the penalty for free.

Worked example: with 2 features, W_dec columns d1 = (1, 0) and d2 = (0.6, 0.8), b_dec = 0, and
activations f = (2, 0.5), the reconstruction is 2 * (1, 0) + 0.5 * (0.6, 0.8) = (2.3, 0.4).
Removing feature 2 from a cell means subtracting 0.5 * (0.6, 0.8).

This follows the dictionary-learning setup of Anthropic's "Towards Monosemanticity" (Bricken et
al. 2023) in its simplest form: one hidden layer, ReLU, L1 penalty, unit-norm decoder columns.
"""

import numpy as np
import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, n_features: int) -> None:
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, n_features)
        self.W_dec = nn.Parameter(torch.empty(d_in, n_features))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            # start with the encoder as the decoder's transpose, a common stable choice
            self.normalize_decoder()
            self.enc.weight.copy_(self.W_dec.T)
            self.enc.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x - self.b_dec))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f


def train_sae(
    x: np.ndarray,
    n_features: int = 1024,
    l1: float = 0.01,
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> SparseAutoencoder:
    """Fit an SAE on the rows of `x` (already centred and scaled by the caller)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = torch.as_tensor(x, dtype=torch.float32)
    model = SparseAutoencoder(data.shape[1], n_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        order = torch.as_tensor(rng.permutation(len(data)))
        for start in range(0, len(data), batch_size):
            batch = data[order[start : start + batch_size]]
            x_hat, f = model(batch)
            loss = ((batch - x_hat) ** 2).sum(dim=1).mean() + l1 * f.sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_decoder()
    return model.eval()


@torch.no_grad()
def activations(model: SparseAutoencoder, x: np.ndarray) -> np.ndarray:
    """Feature activations, shape (cells, features)."""
    return model.encode(torch.as_tensor(x, dtype=torch.float32)).numpy()


@torch.no_grad()
def reconstruct(model: SparseAutoencoder, x: np.ndarray) -> np.ndarray:
    return model(torch.as_tensor(x, dtype=torch.float32))[0].numpy()


def explained_variance(x: np.ndarray, x_hat: np.ndarray) -> float:
    """1 - sum of squared errors / sum of squared deviations from the mean of x."""
    x, x_hat = np.asarray(x, dtype=np.float64), np.asarray(x_hat, dtype=np.float64)
    return float(1.0 - ((x - x_hat) ** 2).sum() / ((x - x.mean(axis=0)) ** 2).sum())


def remove_features(
    model: SparseAutoencoder, x: np.ndarray, features: list[int] | np.ndarray
) -> np.ndarray:
    """x minus the named features' contributions: x - sum_f a_f(x) * W_dec[:, f].

    The reconstruction error is kept, so every other part of x is left exactly as it was.
    """
    idx = np.asarray(features, dtype=int)
    if len(idx) == 0:
        return np.asarray(x, dtype=np.float32).copy()
    a = activations(model, x)[:, idx]
    d = model.W_dec.detach().numpy()[:, idx]
    return (np.asarray(x, dtype=np.float32) - a @ d.T).astype(np.float32)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance of `values` explained by group means (0 to 1; 0 if constant)."""
    values = np.asarray(values, dtype=np.float64)
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    between = sum(
        (groups == g).sum() * (values[groups == g].mean() - values.mean()) ** 2
        for g in np.unique(groups)
    )
    return float(between / total)
