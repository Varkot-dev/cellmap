"""A variational autoencoder for count data with a negative binomial likelihood, by hand.

The model in one paragraph. An encoder reads a cell's counts (as log(1 + count)) and outputs a
distribution over a short code z: a mean and a variance per latent dimension. A decoder reads a
sample of z together with the cell's donor and outputs, for every gene, the share of the cell's
RNA it expects that gene to have. Share times the cell's total count is the expected count, and
the observed count is scored under a negative binomial around it. Training maximizes

    ELBO = E_q[ log p(x | z, donor) ] - kl_weight * KL( q(z | x) || N(0, I) )

that is, it minimizes `recon + kl_weight * kl` as returned by `vae_loss`.

Why the pieces are the way they are:

- Negative binomial, not Gaussian: counts are non-negative integers whose variance grows faster
  than their mean. The NB has variance mu + mu^2 / theta, with one learned theta per gene.
- The donor goes to the decoder only. The decoder can then explain donor-specific shifts itself,
  so z is not pushed to carry them (that is the batch-correction mechanism). The encoder never
  sees the donor, so a cell from a donor that was not in training can still be embedded.
- The cell's total count ("library size") is observed, not learned. How many RNA molecules were
  captured is mostly technical, and handing it to the decoder keeps it out of z.

This follows scVI (Lopez et al. 2018). scvi-tools is deliberately not imported here: the point
is a version small enough to read line by line. Where the defaults match or differ from
scvi-tools 1.5.2 is noted next to each choice.
"""

import time
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

# Floor added to the encoder's variance, as in scvi-tools (`var_eps`). It bounds the log-variance
# below at log(1e-4) = -9.2, which keeps the KL term finite when kl_weight is 0 and nothing
# else stops the encoder from shrinking its variance toward zero.
VAR_EPS = 1e-4

# Batch code for "this cell's donor was not in training". See `VAE.batch_vector`.
UNSEEN_BATCH = -1


def nb_log_likelihood(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Elementwise log probability of counts `x` under a negative binomial.

    Parameterization: mean `mu` > 0 and inverse dispersion `theta` > 0, so that

        variance = mu + mu^2 / theta

    A large theta gives a Poisson (variance = mean); a small theta gives heavy overdispersion.
    The probability of a count x is

        P(x) = Gamma(x + theta) / (Gamma(theta) * x!)
               * (theta / (theta + mu))^theta * (mu / (theta + mu))^x

    and the log of that, which is what is returned, is

        lgamma(x + theta) - lgamma(theta) - lgamma(x + 1)
        + theta * (log(theta) - log(theta + mu))
        + x * (log(mu) - log(theta + mu))

    In scipy's terms this is nbinom(n=theta, p=theta / (theta + mu)).

    Stability: Gamma overflows quickly, so the log-gamma function is used directly, and `eps`
    is added inside each log so that mu = 0 gives a very negative number and not minus infinity
    times zero. The three inputs broadcast against each other (theta is usually one row of
    per-gene values). Same expression as scvi-tools' `log_nb_positive`.
    """
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )


def kl_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mean, diag(exp(logvar))) || N(0, I) ) for each cell (sum over the last axis).

    Closed form, with var = exp(logvar), summed over latent dimensions:

        KL = 0.5 * sum( mean^2 + var - log(var) - 1 )

    Each dimension contributes 0 exactly when mean = 0 and var = 1 and something positive
    otherwise (because v - log(v) - 1 >= 0 for every v > 0), so the result is never negative.
    Read it as the price a cell pays for an informative code: moving the mean away from 0 or
    shrinking the variance below 1 both cost.
    """
    return 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1)


def vae_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three loss numbers for a mini-batch: (recon, kl, total), each a mean over cells.

    recon = - sum over genes of log NB(x | mu, theta): how badly the counts are reconstructed.
    kl    = KL( q(z | x) || N(0, I) ): how far the cell's code distribution is from the prior.
    total = recon + kl_weight * kl, the number that is minimized. With kl_weight = 1 it is
            minus the ELBO.

    Both terms are summed within a cell (over genes, over latent dimensions) before averaging
    over cells. That keeps them on the scale of "nats per cell", which is the scale on which
    kl_weight = 1 is the true ELBO. Averaging over genes instead would silently change the
    meaning of kl_weight by a factor of the number of genes.
    """
    recon = -nb_log_likelihood(x, mu, theta).sum(dim=-1).mean()
    kl = kl_normal(mean, logvar).mean()
    return recon, kl, recon + kl_weight * kl


def kl_weight_at(epoch: int, kl_weight: float, kl_warmup_epochs: int) -> float:
    """KL weight used during `epoch` (0-based): a linear ramp from 0 up to `kl_weight`.

    Epoch 0 gets weight 0, epoch `kl_warmup_epochs` and every later one gets the full weight.
    With `kl_warmup_epochs` = 0 the full weight is used from the start.

    Why warm up: at the start the decoder does not yet use z, so the cheapest way to lower the
    loss is to switch latent dimensions off (set their posterior to the prior, KL = 0), and a
    dimension switched off early tends to stay off. A low weight at first lets the code become
    useful before it is charged for.
    """
    if kl_warmup_epochs <= 0 or epoch >= kl_warmup_epochs:
        return kl_weight
    return kl_weight * epoch / kl_warmup_epochs


class VAEOutput(NamedTuple):
    """What one forward pass produces; exactly the arguments `vae_loss` needs besides x."""

    mu: torch.Tensor  # NB mean per cell and gene: proportion * library size
    theta: torch.Tensor  # NB inverse dispersion per gene
    mean: torch.Tensor  # posterior mean of z per cell
    logvar: torch.Tensor  # posterior log-variance of z per cell


def _block(n_in: int, n_out: int, use_batch_norm: bool, dropout: float) -> nn.Sequential:
    """One hidden layer: linear, optional batch norm, ReLU, optional dropout."""
    layers: list[nn.Module] = [nn.Linear(n_in, n_out)]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(n_out))
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class VAE(nn.Module):
    """Encoder q(z | x) and decoder p(x | z, donor) with a negative binomial likelihood.

    Sizes (defaults): genes -> 128 -> latent 10 for the encoder, latent + donors -> 128 -> genes
    for the decoder. One hidden layer of 128 and 10 latent dimensions are scvi-tools' defaults;
    they are small enough to train on a laptop CPU and a second layer can be switched on with
    `n_layers` once there is evidence it helps.

    Choices, and how they relate to scvi-tools 1.5.2:

    - Encoder input is log(1 + count). Raw counts span four orders of magnitude and a few
      highly expressed genes would dominate the first layer. (Same: `log_variational=True`.)
    - The encoder gets no donor label. (Same: `encode_covariates=False`.)
    - The donor one-hot vector is appended to the input of every decoder layer, so with more
      than one layer the donor information cannot be washed out on the way.
      (Same: `deeply_inject_covariates=True`. With one layer it is plain concatenation with z.)
    - Proportions come from a softmax, so they are positive and sum to 1 over genes; the NB mean
      is proportion * observed library size. (Same: `use_observed_lib_size=True`.)
    - One inverse dispersion per gene, stored as its log so that it stays positive under any
      gradient step. (Same as `dispersion="gene"`. Different start: here log theta = 0, that
      is theta = 1 for every gene, which makes the start reproducible and neutral; scvi-tools
      draws log theta from a standard normal.)
    - Likelihood is NB. (Different: scvi-tools defaults to the zero-inflated NB; for parity it
      has to be run with `gene_likelihood="nb"`.)
    - Posterior variance is exp(raw output) + 1e-4. (Same.)
    - Batch norm in encoder and decoder, dropout 0.1 in the encoder only. Batch norm keeps the
      hidden activations on a steady scale while the weights move, which lets this plain
      network train at lr 1e-3 without tuning; dropout on the encoder is a mild regularizer
      against memorizing training cells. (Same switches as scvi-tools. Different numbers:
      PyTorch's default batch-norm momentum 0.1 and eps 1e-5 here, 0.01 and 1e-3 there. With
      momentum 0.01 the running averages used at embedding time need several hundred steps
      to forget their starting values, which silently hurts short runs on small data.)
    - Linear layers use PyTorch's default initialization. (Same.)

    Because of batch norm and dropout the model behaves differently in training and evaluation
    mode. `embed` and `elbo_terms` switch to evaluation mode themselves.
    """

    def __init__(
        self,
        n_genes: int,
        n_batches: int,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if min(n_genes, n_batches, n_latent, n_hidden, n_layers) < 1:
            raise ValueError("n_genes, n_batches, n_latent, n_hidden and n_layers must be >= 1")
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.n_latent = n_latent

        encoder_sizes = [n_genes] + [n_hidden] * n_layers
        self.encoder = nn.Sequential(
            *[
                _block(n_in, n_out, use_batch_norm, dropout)
                for n_in, n_out in zip(encoder_sizes[:-1], encoder_sizes[1:], strict=True)
            ]
        )
        self.z_mean = nn.Linear(n_hidden, n_latent)
        self.z_raw_var = nn.Linear(n_hidden, n_latent)

        # Every decoder layer sees its normal input plus the donor vector. No dropout here:
        # the reconstruction is the training signal and there is no reason to add noise to it.
        decoder_sizes = [n_latent] + [n_hidden] * n_layers
        self.decoder = nn.ModuleList(
            [
                _block(n_in + n_batches, n_out, use_batch_norm, dropout=0.0)
                for n_in, n_out in zip(decoder_sizes[:-1], decoder_sizes[1:], strict=True)
            ]
        )
        self.gene_logits = nn.Linear(n_hidden, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and log-variance of z from raw counts `x` (cells x genes)."""
        h = self.encoder(torch.log1p(x))
        var = torch.exp(self.z_raw_var(h)) + VAR_EPS
        return self.z_mean(h), torch.log(var)

    @staticmethod
    def sample_z(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """A sample from N(mean, exp(logvar)), written so that gradients can pass through it.

        The reparameterization trick: draw noise from N(0, I), which involves no parameters,
        and compute z = mean + std * noise. z is then an ordinary differentiable function of
        mean and logvar, so the reconstruction loss can tell the encoder how to move them.
        Sampling z "directly" would be a random step with no derivative.
        """
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def batch_vector(self, batch: torch.Tensor) -> torch.Tensor:
        """Decoder input for integer donor codes `batch`: one row per cell, n_batches columns.

        A training donor gets its one-hot row. A donor that was not in training (code -1) has
        no column of its own, so it gets the uniform row 1 / n_batches in every column. In the
        first decoder layer that selects the average of the training donors' learned offsets,
        which reads as "decode this cell as an average training donor". A row of zeros would
        instead be an input the decoder never met in training.

        This only matters when something is decoded (validation loss, reconstructions). The
        embedding does not depend on it, because the encoder never sees the donor.
        """
        batch = batch.long()
        if batch.min() < UNSEEN_BATCH or batch.max() >= self.n_batches:
            raise ValueError(f"batch codes must be in -1..{self.n_batches - 1}")
        unseen = batch == UNSEEN_BATCH
        one_hot = nn.functional.one_hot(batch.clamp(min=0), self.n_batches).float()
        one_hot[unseen] = 1.0 / self.n_batches
        return one_hot

    def decode(self, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Per-gene proportions (cells x genes, each row sums to 1) from z and donor codes."""
        donor = self.batch_vector(batch)
        h = z
        for layer in self.decoder:
            h = layer(torch.cat([h, donor], dim=-1))
        return torch.softmax(self.gene_logits(h), dim=-1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> VAEOutput:
        """Encode, sample z, decode. `x` holds raw counts, `batch` integer donor codes."""
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        library = x.sum(dim=-1, keepdim=True)
        mu = self.decode(z, batch) * library
        return VAEOutput(mu=mu, theta=torch.exp(self.log_theta), mean=mean, logvar=logvar)


def batch_codes(labels: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Integer donor codes: position of each label in `categories`, or -1 if it is not there.

    `categories` should be the donors of the training cells (for example `np.unique` of their
    ids), so that validation and test donors come out as -1, the "unseen donor" code.
    """
    lookup = {str(c): i for i, c in enumerate(np.asarray(categories))}
    return np.array([lookup.get(str(label), UNSEEN_BATCH) for label in np.asarray(labels)])


def _dense_rows(counts: np.ndarray | sp.spmatrix, rows: np.ndarray | slice) -> torch.Tensor:
    """The selected rows as a dense float32 tensor. Only these rows are ever densified."""
    block = counts[rows]
    if sp.issparse(block):
        block = block.toarray()
    return torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))


def _check_inputs(counts: np.ndarray | sp.spmatrix, batch: np.ndarray, name: str) -> np.ndarray:
    """Validate one (counts, batch) pair and return the batch codes as an int64 array."""
    batch = np.asarray(batch)
    if counts.ndim != 2 or batch.shape != (counts.shape[0],):
        raise ValueError(f"{name}: counts {counts.shape} and batch {batch.shape} do not match")
    if not np.issubdtype(batch.dtype, np.integer):
        raise ValueError(f"{name}: batch must hold integer codes (see batch_codes)")
    return batch.astype(np.int64)


def elbo_terms(
    model: VAE,
    counts: np.ndarray | sp.spmatrix,
    batch: np.ndarray,
    batch_size: int = 1024,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean reconstruction term and mean KL term per cell, in evaluation mode.

    Evaluation mode means dropout is off and batch norm uses its running averages, so every
    cell is scored on its own. One z is sampled per cell, from a private generator seeded with
    `seed`, so calling this neither depends on nor disturbs the random stream used in training.
    """
    batch = _check_inputs(counts, batch, "elbo_terms")
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    recon_sum, kl_sum = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, counts.shape[0], batch_size):
            rows = slice(start, start + batch_size)
            x = _dense_rows(counts, rows)
            mean, logvar = model.encode(x)
            noise = torch.randn(mean.shape, generator=generator)
            z = mean + torch.exp(0.5 * logvar) * noise
            mu = model.decode(z, torch.from_numpy(batch[rows])) * x.sum(dim=-1, keepdim=True)
            theta = torch.exp(model.log_theta)
            recon_sum += -nb_log_likelihood(x, mu, theta).sum().item()
            kl_sum += kl_normal(mean, logvar).sum().item()
    model.train(was_training)
    n_cells = counts.shape[0]
    return recon_sum / n_cells, kl_sum / n_cells


def train_vae(
    counts_train: np.ndarray | sp.spmatrix,
    batch_train: np.ndarray,
    n_batches: int | None = None,
    n_latent: int = 10,
    n_hidden: int = 128,
    n_layers: int = 1,
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    kl_weight: float = 1.0,
    kl_warmup_epochs: int = 10,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    adam_eps: float = 0.01,
    seed: int = 0,
    counts_val: np.ndarray | sp.spmatrix | None = None,
    batch_val: np.ndarray | None = None,
) -> tuple[VAE, dict[str, list[float]]]:
    """Train a `VAE` on training cells and return it (in evaluation mode) with its history.

    `counts_train` holds raw counts of the selected genes (cells x genes, dense or scipy
    sparse) for cells of TRAINING donors only; the caller does the selecting, this function
    never sees other cells except the optional validation set, which is only scored. Sparse
    input is densified one mini-batch at a time. `batch_train` holds integer donor codes
    0..n_batches-1 (see `batch_codes`).

    Validation cells (`counts_val`, `batch_val`) are scored after every epoch with
    `elbo_terms` and never used for a gradient step. Cells of donors that were not in training
    carry code -1 and are decoded as an average training donor (see `VAE.batch_vector`), so
    their reconstruction term is a little worse than it would be with a fitted donor offset.
    That gap is the honest cost of a new donor, not a bug.

    History: a dict of equal-length lists with one entry per epoch. `kl_weight` is the weight
    in force during the epoch (after warm-up). `train_recon`, `train_kl` and `train_total` are
    averages over the epoch's mini-batches as they were trained on: dropout on, weights still
    moving, total using the epoch's weight. `val_recon`, `val_kl` and `val_total` are measured
    after the epoch in evaluation mode, and `val_total` uses the final `kl_weight`, not the
    warm-up one, so it is comparable across epochs. Train and validation numbers are therefore
    not measured the same way; call `elbo_terms` on the training cells for a like-for-like
    comparison. `seconds` is the wall time of the epoch including validation.

    Optimizer: Adam with lr 1e-3, weight decay 1e-6 and eps 0.01, mini-batches of 128. These
    are scvi-tools' values. The large eps makes Adam behave more like plain momentum SGD for
    parameters whose gradients are tiny, which avoids huge early steps on the per-gene
    dispersions. What differs from scvi-tools on purpose:

    - Epochs: 100 by default here; scvi-tools uses min(400, round(400 * 20000 / n_cells)).
    - KL warm-up: linear over the first 10 epochs here. scvi-tools ramps over 400 epochs, so
      a default run of fewer than 400 epochs ends before its weight has reached 1.
    - Data: all training cells every epoch, reshuffled; scvi-tools holds out a random 10% of
      cells for validation unless told otherwise. Here validation is by whole donor.

    A final mini-batch with a single cell is skipped, because batch norm cannot compute a
    variance from one cell.

    Determinism: `torch.manual_seed(seed)` is called first, so weight initialization, shuffling,
    dropout and the sampling of z all follow from `seed`. Note that this resets PyTorch's global
    random stream. On CPU with a fixed number of threads two runs with the same seed give
    identical results; no claim is made about GPUs.
    """
    batch_train = _check_inputs(counts_train, batch_train, "train")
    if batch_train.min() < 0:
        raise ValueError("training cells must all have a known donor code (>= 0)")
    if n_batches is None:
        n_batches = int(batch_train.max()) + 1
    has_val = counts_val is not None
    if has_val:
        if batch_val is None:
            raise ValueError("counts_val was given without batch_val")
        batch_val = _check_inputs(counts_val, batch_val, "val")
        if counts_val.shape[1] != counts_train.shape[1]:
            raise ValueError("validation and training counts have different numbers of genes")

    torch.manual_seed(seed)
    model = VAE(
        n_genes=counts_train.shape[1],
        n_batches=n_batches,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=adam_eps)

    keys = ["kl_weight", "train_recon", "train_kl", "train_total", "seconds"]
    if has_val:
        keys += ["val_recon", "val_kl", "val_total"]
    history: dict[str, list[float]] = {key: [] for key in keys}
    n_cells = counts_train.shape[0]

    for epoch in range(n_epochs):
        started = time.perf_counter()
        weight = kl_weight_at(epoch, kl_weight, kl_warmup_epochs)
        model.train()
        sums = np.zeros(3)
        n_seen = 0
        for rows in torch.randperm(n_cells).split(batch_size):
            if len(rows) < 2:
                continue
            # Sorted row order within the mini-batch: same cells, faster sparse row lookup.
            rows = np.sort(rows.numpy())
            x = _dense_rows(counts_train, rows)
            out = model(x, torch.from_numpy(batch_train[rows]))
            recon, kl, total = vae_loss(x, out.mu, out.theta, out.mean, out.logvar, weight)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            sums += len(rows) * np.array([recon.item(), kl.item(), total.item()])
            n_seen += len(rows)

        history["kl_weight"].append(weight)
        for key, value in zip(["train_recon", "train_kl", "train_total"], sums, strict=True):
            history[key].append(float(value / n_seen))
        if has_val:
            val_recon, val_kl = elbo_terms(model, counts_val, batch_val, seed=seed)
            history["val_recon"].append(val_recon)
            history["val_kl"].append(val_kl)
            history["val_total"].append(val_recon + kl_weight * val_kl)
        history["seconds"].append(time.perf_counter() - started)

    model.eval()
    return model, history


def embed(model: VAE, counts: np.ndarray | sp.spmatrix, batch_size: int = 1024) -> np.ndarray:
    """Posterior mean of z for every cell (float32, cells x latent). Nothing is sampled.

    The mean is the single most representative point of q(z | x), and using it makes the
    embedding a deterministic function of the cell. No donor label is needed, so this works
    for cells of any donor, seen in training or not. Runs in evaluation mode (dropout off,
    batch norm using its running averages), so a cell's embedding does not depend on which
    other cells are embedded with it.
    """
    was_training = model.training
    model.eval()
    means = []
    with torch.no_grad():
        for start in range(0, counts.shape[0], batch_size):
            mean, _ = model.encode(_dense_rows(counts, slice(start, start + batch_size)))
            means.append(mean.numpy())
    model.train(was_training)
    return np.concatenate(means).astype(np.float32)
