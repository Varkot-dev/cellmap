"""A linear autoencoder: method 3 in the comparison, the stepping stone between PCA and the VAE.

The model is two matrices and nothing else. The encoder maps a cell's scaled expression x
(genes) to a code z = E x (d numbers); the decoder maps it back, x_hat = D z. Training
minimizes the mean squared error between x and x_hat over training cells.

The known result (Baldi and Hornik 1989; Bourlard and Kamp 1988): for centered data, every
global minimum of that loss has the decoder's columns spanning the same subspace as the top d
principal axes, and there are no other local minima, only saddle points. So a trained linear
autoencoder reconstructs exactly as well as PCA with d components. What it does not recover is
the principal axes themselves: any invertible d x d matrix A can be inserted (z -> A z,
D -> D A^-1) without changing the reconstruction, so the code is the PCA code up to an
invertible linear map. Its axes are not ordered by variance and need not be orthogonal.

Why it is in the comparison anyway: it is trained exactly like the VAE (same optimizer, same
mini-batches, same code size) but has no nonlinearity, no count likelihood, no prior on z and
no donor input. If it lands on PCA's numbers, the training machinery is sound, and any gap
between it and the VAE is due to what the VAE adds and not to how it was optimized.

Scaling follows `baselines.pca_embed`: z-score each gene with the mean and standard deviation
of the training cells, clip to [-10, 10], subtract the training mean of the clipped values.
The three per-gene vectors are stored inside the model, so `embed` applies training-cell
statistics to any cell and no statistic is ever computed from validation or test cells.
"""

import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


class LinearAE(nn.Module):
    """Linear encoder and decoder without bias terms, plus the frozen training-cell scaling.

    No bias: the scaled training data has mean zero in every gene, and for centered data the
    best reconstruction passes through the origin, so a bias would have nothing to learn.

    `mean`, `std` and `center` are buffers, not parameters: they are fitted once from training
    cells by `fit_scaler`, saved with the model, and never touched by the optimizer.
    """

    def __init__(self, n_genes: int, n_latent: int = 10, max_value: float = 10.0) -> None:
        super().__init__()
        if not 0 < n_latent <= n_genes:
            raise ValueError(f"n_latent={n_latent} must be in 1..{n_genes}")
        self.max_value = max_value
        self.encoder = nn.Linear(n_genes, n_latent, bias=False)
        self.decoder = nn.Linear(n_latent, n_genes, bias=False)
        self.register_buffer("mean", torch.zeros(n_genes))
        self.register_buffer("std", torch.ones(n_genes))
        self.register_buffer("center", torch.zeros(n_genes))

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """z-score with training statistics, clip, re-center. Same steps as `pca_embed`."""
        z = ((x - self.mean) / self.std).clamp(-self.max_value, self.max_value)
        return z - self.center

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled input and its reconstruction, from log-normalized expression `x`."""
        scaled = self.scale(x)
        return scaled, self.decoder(self.encoder(scaled))


def _dense_rows(x: np.ndarray | sp.spmatrix, rows: np.ndarray | slice) -> torch.Tensor:
    """The selected rows as a dense float32 tensor. Only these rows are ever densified."""
    block = x[rows]
    if sp.issparse(block):
        block = block.toarray()
    return torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))


def fit_scaler(
    x_train: np.ndarray | sp.spmatrix, max_value: float = 10.0, chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-gene (mean, std, center) of the training cells, as float32 arrays.

    `mean` and `std` (sample standard deviation, ddof=1) are of the log-normalized values;
    genes that are constant in training get std 1 so they stay at zero. `center` is the mean
    of the z-scores after clipping to [-max_value, max_value]: clipping cuts mostly the high
    side, which moves the mean slightly below zero, and subtracting `center` puts it back.

    Two passes over the rows in chunks, accumulating in float64, so a sparse matrix is never
    densified as a whole.
    """
    n_cells, n_genes = x_train.shape
    if n_cells < 2:
        raise ValueError("need at least two training cells to estimate a standard deviation")
    total, total_sq = np.zeros(n_genes), np.zeros(n_genes)
    for start in range(0, n_cells, chunk):
        block = _dense_rows(x_train, slice(start, start + chunk)).numpy().astype(np.float64)
        total += block.sum(axis=0)
        total_sq += (block**2).sum(axis=0)
    mean = total / n_cells
    var = np.maximum(total_sq - n_cells * mean**2, 0.0) / (n_cells - 1)
    std = np.sqrt(var)
    # A gene that is constant in training has variance 0 up to rounding of the two sums.
    std[var <= 1e-12 * np.maximum(mean**2, 1.0)] = 1.0

    clipped_total = np.zeros(n_genes)
    for start in range(0, n_cells, chunk):
        block = _dense_rows(x_train, slice(start, start + chunk)).numpy().astype(np.float64)
        clipped_total += np.clip((block - mean) / std, -max_value, max_value).sum(axis=0)
    center = clipped_total / n_cells
    return mean.astype(np.float32), std.astype(np.float32), center.astype(np.float32)


def train_linear_ae(
    x_train: np.ndarray | sp.spmatrix,
    n_latent: int = 10,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_value: float = 10.0,
    seed: int = 0,
    x_val: np.ndarray | sp.spmatrix | None = None,
) -> tuple[LinearAE, dict[str, list[float]]]:
    """Train a `LinearAE` on training cells and return it with its per-epoch history.

    `x_train` holds the log-normalized expression of the selected genes (cells x genes, dense
    or scipy sparse) for TRAINING cells only. The scaling statistics and the weights are both
    fitted from these rows and nothing else. `x_val`, if given, is only scored.

    Loss: squared error summed over genes and averaged over cells, on the scaled values. That
    is the quantity PCA minimizes, so `train_mse` can be compared directly with the variance
    PCA leaves unexplained at the same number of components.

    History: `train_mse` (average over the epoch's mini-batches while the weights were still
    moving), `val_mse` (after the epoch, if `x_val` was given) and `seconds` per epoch.

    Defaults: 10 code dimensions, to match the VAE's latent size so the two are compared at
    equal capacity (the PCA baseline uses 50, so pass `n_latent=50` to compare with that).
    Adam with lr 1e-3 and mini-batches of 128, as for the VAE, but with Adam's standard eps
    and no weight decay: weight decay would pull the optimum away from the PCA subspace, and
    the point of this model is to land on it.

    Determinism: `torch.manual_seed(seed)` fixes the initial weights and the shuffling. It
    resets PyTorch's global random stream.
    """
    n_cells, n_genes = x_train.shape
    if x_val is not None and x_val.shape[1] != n_genes:
        raise ValueError("validation and training data have different numbers of genes")

    torch.manual_seed(seed)
    model = LinearAE(n_genes, n_latent, max_value)
    for name, values in zip(["mean", "std", "center"], fit_scaler(x_train, max_value), strict=True):
        getattr(model, name).copy_(torch.from_numpy(values))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_mse": [], "seconds": []}
    if x_val is not None:
        history["val_mse"] = []

    for _ in range(n_epochs):
        started = time.perf_counter()
        loss_sum = 0.0
        for rows in torch.randperm(n_cells).split(batch_size):
            rows = np.sort(rows.numpy())
            scaled, reconstructed = model(_dense_rows(x_train, rows))
            loss = (scaled - reconstructed).pow(2).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(rows)
        history["train_mse"].append(loss_sum / n_cells)
        if x_val is not None:
            history["val_mse"].append(reconstruction_error(model, x_val))
        history["seconds"].append(time.perf_counter() - started)
    return model, history


def reconstruction_error(
    model: LinearAE, x: np.ndarray | sp.spmatrix, batch_size: int = 1024
) -> float:
    """Squared reconstruction error on the scaled values, summed over genes, mean over cells."""
    error_sum = 0.0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            scaled, reconstructed = model(_dense_rows(x, slice(start, start + batch_size)))
            error_sum += (scaled - reconstructed).pow(2).sum().item()
    return error_sum / x.shape[0]


def embed(model: LinearAE, x: np.ndarray | sp.spmatrix, batch_size: int = 1024) -> np.ndarray:
    """The code z = E * scale(x) for every cell (float32, cells x n_latent).

    `x` is log-normalized expression of the same genes the model was trained on, for any
    cells. Each cell is scaled with the stored training statistics and encoded on its own, so
    its embedding does not depend on which other cells are passed in.
    """
    codes = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            scaled = model.scale(_dense_rows(x, slice(start, start + batch_size)))
            codes.append(model.encoder(scaled).numpy())
    return np.concatenate(codes).astype(np.float32)
