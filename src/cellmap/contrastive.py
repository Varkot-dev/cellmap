"""Contrastive training for the cell transformer: learn a cell embedding directly.

Why this exists. The masked-gene transformer (`cellmap.transformer`) learned its own task, and
after the level encoding was fixed it learned it well (validation accuracy 0.78 against a 0.27
baseline), yet its cell embedding did not improve. Predicting a hidden gene's level can be
solved from a few co-expressed genes without organising whole cells by type. This module trains
the same architecture with an objective that acts on the whole-cell embedding instead.

The objective (SimCLR; Chen, Kornblith, Norouzi and Hinton, ICML 2020), for one mini-batch of
N cells:

1. Make two random "views" of every cell by corrupting its tokens (see `make_view`): drop a
   random share of its expressed genes and nudge a few levels by one bin. Both views are still
   the same cell, but they share fewer than all of their tokens.
2. Run both views through the transformer and take the [CLS] state of each: h_1 .. h_2N.
3. A small projection head maps each h to z, scaled to unit length.
4. For view i, its partner j(i) is the other view of the same cell; the other 2N - 2 views in
   the batch act as negatives. The loss for view i is

       l_i = - log( exp(z_i . z_j(i) / tau) / sum over k != i of exp(z_i . z_k / tau) )

   that is, a softmax over "which of the other 2N - 1 views is my partner?", with temperature
   tau. The loss is the mean of l_i over all 2N views (NT-Xent).

What the model learns: an embedding in which a cell is close to itself under corruption and far
from other cells. Corruption by gene dropout mimics what actually differs between two
measurements of the same cell type (which genes happen to be captured), so the embedding is
pushed to keep what is stable about a cell, its type, and to ignore capture noise. The [CLS]
state h, not the projection z, is used as the embedding, as in SimCLR: the projection head is
free to throw away information that the contrastive loss does not need.

Pitfall worth knowing: two different cells of the SAME type are also treated as negatives
("false negatives"). With 19 types and batches of 256 cells, most negatives are of a different
type, but some are not. Supervised contrastive learning would fix that, at the price of using
labels, which this project keeps out of representation learning.

Optionally the masked-gene loss of `cellmap.transformer` is added (`mgm_weight` > 0), so the
model keeps its gene-level task while the contrastive loss shapes the cell embedding.

Embeddings come from `cellmap.transformer.embed` with pooling "cls", unchanged.
"""

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from cellmap.transformer import (
    CellTokens,
    CellTransformer,
    TokenBatch,
    _adamw,
    _masked_step,
    bin_baselines,
    choose_mask,
    fit_value_edges,
    length_batches,
    lr_at,
    make_batch,
    tokenize,
)


def make_view(
    tokens: CellTokens,
    rows: np.ndarray,
    rng: np.random.Generator,
    max_len: int = 256,
    keep_range: tuple[float, float] = (0.5, 0.9),
    jitter: float = 0.2,
) -> TokenBatch:
    """One corrupted view of each cell in `rows`, padded like `make_batch`.

    Per cell, a keep rate is drawn uniformly from `keep_range` and every token is kept with
    that probability (at least one token is always kept). With probability `jitter` a kept
    token's level bin moves one step up or down, clipped to the valid bins. Cells longer than
    `max_len` after dropout are cut to a random `max_len` subset. A token's bin is the one
    computed from the whole cell, so dropping genes does not change the levels of the others.
    """
    rows = np.asarray(rows)
    kept_genes, kept_bins = [], []
    for row in rows:
        start, stop = tokens.indptr[row], tokens.indptr[row + 1]
        idx = np.arange(start, stop)
        if len(idx):
            rate = rng.uniform(*keep_range)
            keep = rng.random(len(idx)) < rate
            if not keep.any():
                keep[rng.integers(len(idx))] = True
            idx = idx[keep]
            if len(idx) > max_len:
                idx = np.sort(rng.choice(idx, size=max_len, replace=False))
        bins = tokens.bins[idx].astype(np.int64)
        if jitter > 0 and len(bins):
            move = rng.random(len(bins)) < jitter
            step = rng.choice([-1, 1], size=len(bins))
            bins = np.where(move, np.clip(bins + step, 0, tokens.n_bins - 1), bins)
        kept_genes.append(tokens.genes[idx].astype(np.int64))
        kept_bins.append(bins)
    width = max(max((len(g) for g in kept_genes), default=0), 1)
    genes = np.full((len(rows), width), tokens.n_genes, dtype=np.int64)
    bins = np.full((len(rows), width), tokens.n_bins + 1, dtype=np.int64)
    for i, (g, b) in enumerate(zip(kept_genes, kept_bins, strict=True)):
        genes[i, : len(g)] = g
        bins[i, : len(b)] = b
    return TokenBatch(
        genes=torch.from_numpy(genes),
        bins=torch.from_numpy(bins),
        pad=torch.from_numpy(genes == tokens.n_genes),
    )


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR loss for two views z1, z2 (N x d each, any scale; normalized here)."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    partner = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, partner)


class ProjectionHead(nn.Module):
    """h -> z: Linear, ReLU, Linear, as in SimCLR."""

    def __init__(self, d_model: int, d_out: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_out))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def contrastive_validation(
    model: CellTransformer,
    head: ProjectionHead,
    tokens: CellTokens,
    temperature: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Contrastive loss and partner-retrieval accuracy on held-out cells, in evaluation mode.

    Accuracy: the share of views whose most similar other view in the batch is their partner.
    The views come from a private generator seeded with `seed`, so epochs are comparable.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    head.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for start in range(0, tokens.n_cells - 1, batch_size):
            rows = np.arange(start, min(start + batch_size, tokens.n_cells))
            if len(rows) < 2:
                continue
            z = [
                head(
                    model(*(t.to(device) for t in make_view(tokens, rows, rng, model.max_len)))[
                        :, 0
                    ]
                )
                for _ in range(2)
            ]
            losses.append(nt_xent(z[0], z[1], temperature).item() * len(rows))
            zn = F.normalize(torch.cat(z), dim=1)
            sim = (zn @ zn.T).fill_diagonal_(-2.0)
            n = len(rows)
            partner = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(device)
            correct += (sim.argmax(1) == partner).sum().item()
            total += 2 * n
    model.train()
    head.train()
    return {"loss": float(np.sum(losses) / max(total / 2, 1)), "accuracy": correct / max(total, 1)}


def train_contrastive(
    counts_train: np.ndarray | sp.spmatrix,
    n_bins: int = 10,
    max_len: int = 256,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    temperature: float = 0.1,
    proj_dim: int = 64,
    keep_range: tuple[float, float] = (0.5, 0.9),
    jitter: float = 0.2,
    mgm_weight: float = 0.0,
    mask_fraction: float = 0.15,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    warmup: float = 0.05,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    counts_val: np.ndarray | sp.spmatrix | None = None,
    level_encoding: str = "rank",
) -> tuple[CellTransformer, dict[str, list[float]]]:
    """Train a `CellTransformer` with the SimCLR loss (plus `mgm_weight` x masked-gene loss).

    Same inputs and conventions as `cellmap.transformer.train_transformer`: counts of the
    selected genes for TRAINING donors only, validation cells only scored, the same optimizer,
    learning-rate schedule, gradient clipping and seeding. Returns the model in eval mode and a
    history with `train_contrastive_loss`, `val_contrastive_loss`, `val_partner_accuracy`
    (chance is 1 / (2 * batch - 1)), and, with `mgm_weight` > 0, `train_mgm_loss`.
    """
    if level_encoding not in ("rank", "value"):
        raise ValueError('level_encoding must be "rank" or "value"')
    device = torch.device(device)
    edges = fit_value_edges(counts_train, n_bins) if level_encoding == "value" else None
    train = tokenize(counts_train, n_bins, edges)
    val = tokenize(counts_val, n_bins, edges) if counts_val is not None else None
    baselines = bin_baselines(train)

    torch.manual_seed(seed)
    model = CellTransformer(
        n_genes=train.n_genes,
        n_bins=n_bins,
        max_len=max_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    ).to(device)
    model.value_edges = edges
    head = ProjectionHead(d_model, proj_dim).to(device)
    params = nn.ModuleList([model, head])
    optimizer = _adamw(params, lr, weight_decay)
    rng = np.random.default_rng(seed)
    mask_generator = torch.Generator().manual_seed(seed)

    lengths = np.minimum(np.diff(train.indptr), max_len)
    steps_per_epoch = math.ceil(train.n_cells / batch_size)
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = int(round(warmup * total_steps))
    history: dict[str, list[float]] = {}
    step = 0
    for _ in range(n_epochs):
        started = time.perf_counter()
        model.train()
        head.train()
        sums = {"contrastive": 0.0, "mgm": 0.0}
        n_batches = 0
        # Plain random batches: grouping by length would put similar cells together and make
        # the negatives too easy or too hard in a way that depends on depth.
        for rows in length_batches(lengths, batch_size, rng, bucket_batches=1):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, warmup_steps, lr)
            step += 1
            if len(rows) < 2:
                continue
            views = [make_view(train, rows, rng, max_len, keep_range, jitter) for _ in range(2)]
            z = [head(model(*(t.to(device) for t in v))[:, 0]) for v in views]
            loss = nt_xent(z[0], z[1], temperature)
            sums["contrastive"] += loss.item()
            if mgm_weight > 0:
                batch = make_batch(train, rows, max_len, rng)
                mask = choose_mask(batch.pad, mask_fraction, mask_generator)
                if mask.any():
                    token_loss, cls_loss, _ = _masked_step(model, batch, mask, baselines, device)
                    mgm = token_loss + cls_loss
                    loss = loss + mgm_weight * mgm
                    sums["mgm"] += mgm.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params.parameters(), grad_clip)
            optimizer.step()
            n_batches += 1
        row = {"train_contrastive_loss": sums["contrastive"] / max(n_batches, 1)}
        if mgm_weight > 0:
            row["train_mgm_loss"] = sums["mgm"] / max(n_batches, 1)
        if val is not None:
            scores = contrastive_validation(model, head, val, temperature, batch_size, seed, device)
            row["val_contrastive_loss"] = scores["loss"]
            row["val_partner_accuracy"] = scores["accuracy"]
        row["lr"] = optimizer.param_groups[0]["lr"]
        row["seconds"] = time.perf_counter() - started
        for key, value in row.items():
            history.setdefault(key, []).append(float(value))
    model.eval()
    return model, history
