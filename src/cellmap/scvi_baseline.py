"""The scvi-tools version of the count VAE, trained and embedded under this project's rules.

Why this exists. `cellmap.vae` is a negative binomial VAE written by hand. The way to show it is
correct is to train the reference implementation (scvi.model.SCVI) on the same cells, the same
genes and the same settings, score both with the same harness, and see whether they agree
within seed-to-seed noise. This module is the scvi-tools side of that comparison and nothing
else: no tuning, no extra features.

Three rules it enforces:

- Train-only fitting. The model is set up and trained on an AnnData that contains the training
  cells only. Held-out donors never enter training, not even as an internal validation set.
- Matched settings. Every setting that `cellmap.vae.train_vae` has is set explicitly here, also
  where the library default already matches, so that a changed default in a later scvi-tools
  release cannot silently change the comparison. See `scvi_settings`.
- Unseen donors are embedded without their donor label. See `_embed_all_cells`.

Checked against the installed scvi-tools 1.5.2 source, not against memory of older versions.
"""

import logging
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scvi

from cellmap.baselines import _as_cell_mask

# Name of the donor column inside the private AnnData copies handed to scvi-tools.
_BATCH = "batch"

# Loggers that print progress chatter at INFO level (device banner, tips, "setup transferred").
_NOISY_LOGGERS = ("scvi", "lightning", "lightning.pytorch", "lightning.fabric")

# Warnings that say nothing about correctness here. Matched by message, so that any other
# warning from the library still reaches the caller.
_HARMLESS_WARNINGS = (
    # One loader worker is intended: the data is in memory and extra processes would compete
    # for cores with other jobs.
    r".*does not have many workers.*",
    # Internal deprecation inside lightning's own use of a torch utility.
    r".*isinstance\(treespec, LeafSpec\).*",
)


def scvi_settings(
    n_latent: int = 10,
    n_hidden: int = 128,
    n_layers: int = 1,
    dropout: float = 0.1,
    max_epochs: int = 100,
    kl_warmup_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    adam_eps: float = 0.01,
    drop_last: bool = False,
) -> dict[str, dict]:
    """The exact keyword arguments given to scvi-tools, grouped by the call that takes them.

    `model` goes to `scvi.model.SCVI(...)`, `train` to `SCVI.train(...)`. Each line says what
    the hand-written model does, because that is what the value is matched to.

    Two things here are not scvi-tools defaults: the likelihood ("zinb" by default, the
    hand-written model has no zero inflation) and the KL warm-up length (400 epochs by default,
    10 here). `train_size=1.0` is also not the default (0.9): the library otherwise holds out a
    random 10% of the cells it is given, and the hand-written model trains on all of them.
    """
    model = {
        "n_latent": n_latent,
        "n_hidden": n_hidden,
        "n_layers": n_layers,
        "dropout_rate": dropout,  # encoder only, the scvi decoder has none; same by hand
        "gene_likelihood": "nb",  # plain negative binomial, no zero inflation
        "dispersion": "gene",  # one inverse dispersion per gene
        "use_observed_lib_size": True,  # library size is the cell's observed total count
        "latent_distribution": "normal",
        "log_variational": True,  # encoder input is log(1 + count)
        "encode_covariates": False,  # the encoder never sees the donor
        "deeply_inject_covariates": True,  # donor one-hot enters every decoder layer
        "batch_representation": "one-hot",
        "use_batch_norm": "both",  # encoder and decoder
        "use_layer_norm": "none",
    }
    plan = {
        "optimizer": "Adam",
        "lr": lr,
        "weight_decay": weight_decay,
        "eps": adam_eps,
        # Linear ramp over epochs: weight = epoch / n during epoch < n, then max_kl_weight.
        # Identical to cellmap.vae.kl_weight_at. None switches the ramp off (0 would too).
        "n_epochs_kl_warmup": kl_warmup_epochs if kl_warmup_epochs > 0 else None,
        "n_steps_kl_warmup": None,
        "max_kl_weight": 1.0,
        "min_kl_weight": 0.0,
        "reduce_lr_on_plateau": False,  # constant learning rate
    }
    train = {
        "max_epochs": max_epochs,
        "accelerator": "cpu",
        "devices": 1,
        "train_size": 1.0,  # every cell that is passed in is trained on
        "validation_size": None,  # with train_size 1.0 this leaves 0 validation cells
        "shuffle_set_split": True,  # irrelevant with train_size 1.0, set for the record
        "batch_size": batch_size,
        "early_stopping": False,  # always run max_epochs
        "datasplitter_kwargs": {"drop_last": drop_last},
        "plan_kwargs": plan,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "check_val_every_n_epoch": None,  # no validation loop at all
        # Logging only, no effect on training. The default of 10 skips the per-step KL weight
        # in epochs with fewer than 10 mini-batches, which leaves holes in the history.
        "log_every_n_steps": 1,
    }
    return {"model": model, "train": train}


def scvi_embed(
    adata: ad.AnnData,
    gene_mask: np.ndarray,
    train_mask: np.ndarray,
    batch_key: str = "donor_id",
    n_latent: int = 10,
    n_hidden: int = 128,
    n_layers: int = 1,
    max_epochs: int = 100,
    kl_warmup_epochs: int = 10,
    batch_size: int = 128,
    seed: int = 0,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    adam_eps: float = 0.01,
    counts_layer: str = "counts",
    return_model: bool = False,
) -> tuple[np.ndarray, dict]:
    """Train scvi.model.SCVI on the training cells and embed every cell.

    Returns `(embedding, info)`. `embedding` is the posterior mean of z for ALL cells of
    `adata`, float32, n_obs x n_latent, in `adata`'s row order. `info` records what was done:

    - settings: the exact arguments passed to scvi-tools (see `scvi_settings`), plus seed.
    - history: one entry per epoch of `train_recon`, `train_kl` (nats per cell, averaged over
      the epoch's mini-batches as trained on: dropout on, batch norm in batch mode, weights
      moving) and `kl_weight`. Same meaning as the history of `cellmap.vae.train_vae`.
    - train_seconds, seconds_per_epoch, scvi_tools_version, n_train_cells, n_genes,
      n_train_batches, train_batches (the donors the decoder knows, in code order).
    - unseen_batch_cells and unseen_batch_max_abs_diff: see below.
    - model (only with `return_model=True`): the trained scvi.model.SCVI, for diagnostics.

    What the model is trained on: raw counts from `adata.layers[counts_layer]`, columns in
    `gene_mask`, rows in `train_mask`, with `adata.obs[batch_key]` as the batch. A private
    AnnData is built from that slice, so `adata` itself is never written to (scvi-tools'
    `setup_anndata` adds columns and an `uns` entry to whatever object it is given).

    Cells of donors that were not in training. scvi-tools refuses a batch label it was not set
    up with. The encoder, however, does not take the batch at all (`encode_covariates=False`),
    so the posterior mean of z cannot depend on it. Such cells are therefore given the label of
    a training donor purely to get past the label check. As a guard against a future library
    change, the embedding of those cells is computed a second time with a different training
    donor's label and the largest absolute difference is stored in `unseen_batch_max_abs_diff`;
    anything other than exactly 0 raises.

    The final small mini-batch. `cellmap.vae.train_vae` trains on a short last mini-batch
    unless it holds a single cell, which batch norm cannot handle. scvi-tools would crash on a
    one-cell batch, so `drop_last` is switched on in exactly that case (number of training
    cells = 1 modulo batch_size) and off otherwise. Both then skip the same thing.

    Randomness. `scvi.settings.seed = seed` seeds Python, NumPy and PyTorch globally (that is
    how the library does it), so this call resets the caller's global random streams, as
    `train_vae` does for PyTorch. On CPU with a fixed thread count the result is reproducible.
    """
    train_mask = _as_cell_mask(train_mask, adata.n_obs)
    gene_mask = np.asarray(gene_mask)
    if gene_mask.dtype != bool or gene_mask.shape != (adata.n_vars,):
        raise ValueError(f"gene_mask must be a boolean mask of shape ({adata.n_vars},)")
    if counts_layer not in adata.layers:
        raise ValueError(f"adata.layers has no {counts_layer!r}; raw counts are required")

    counts = sp.csr_matrix(adata.layers[counts_layer][:, gene_mask], dtype=np.float32)
    donors = adata.obs[batch_key].astype(str).to_numpy()
    n_train = int(train_mask.sum())
    settings = scvi_settings(
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout=dropout,
        max_epochs=max_epochs,
        kl_warmup_epochs=kl_warmup_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        adam_eps=adam_eps,
        drop_last=n_train % batch_size == 1,
    )

    with _quiet():
        scvi.settings.seed = seed
        train_adata = _private_adata(counts[train_mask], donors[train_mask])
        started = time.perf_counter()
        model = _train(train_adata, settings)
        train_seconds = time.perf_counter() - started
        emb, unseen_diff = _embed_all_cells(model, counts, donors, train_mask)

    train_batches = [str(c) for c in _batch_categories(model)]
    info = {
        "settings": {**settings, "seed": seed, "batch_key": batch_key, "layer": counts_layer},
        "history": _history(model),
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / max(max_epochs, 1),
        "scvi_tools_version": version("scvi-tools"),
        "n_train_cells": n_train,
        "n_genes": int(gene_mask.sum()),
        "n_train_batches": len(train_batches),
        "train_batches": train_batches,
        "unseen_batch_cells": int((~np.isin(donors, train_batches)).sum()),
        "unseen_batch_max_abs_diff": unseen_diff,
    }
    if return_model:
        info["model"] = model
    return emb, info


def _private_adata(counts: sp.csr_matrix, batch: np.ndarray) -> ad.AnnData:
    """A fresh AnnData holding only what scvi-tools needs: counts in X and one batch column.

    `counts` is a new matrix made by the caller's slicing, so nothing in it is shared with the
    caller's AnnData and scvi-tools can write its bookkeeping into this object freely.
    """
    obs = pd.DataFrame({_BATCH: np.asarray(batch, dtype=str)})
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=counts, obs=obs)


def _train(train_adata: ad.AnnData, settings: dict[str, dict]) -> scvi.model.SCVI:
    """Set up, build and train the model on `train_adata` (all of its cells)."""
    scvi.model.SCVI.setup_anndata(train_adata, batch_key=_BATCH)
    model = scvi.model.SCVI(train_adata, **settings["model"])
    model.train(**settings["train"])
    if len(model.train_indices) != train_adata.n_obs or len(model.validation_indices) > 0:
        raise RuntimeError("scvi-tools held cells out of training despite train_size=1.0")
    return model


def _batch_categories(model: scvi.model.SCVI) -> np.ndarray:
    """The batch labels the model was set up with, in the order of their integer codes."""
    registry = model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.BATCH_KEY)
    return np.asarray(registry.categorical_mapping)


def _embed_all_cells(
    model: scvi.model.SCVI, counts: sp.csr_matrix, donors: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, float]:
    """Posterior mean of z for every row of `counts`, and the unseen-donor check value.

    Cells whose donor the model knows keep their label. The others get a known donor's label as
    a stand-in, because `get_latent_representation` validates labels before it runs the
    encoder, and the encoder then ignores them. The stand-in is the first known donor; the
    check repeats the unknown-donor cells with the last known donor and returns the largest
    absolute difference between the two embeddings (0.0 when there is nothing to compare).

    The module is in evaluation mode here (scvi-tools switches it after training), so dropout
    is off and batch norm uses its running averages: a cell's embedding does not depend on
    which other cells share its mini-batch.
    """
    known = _batch_categories(model)
    unseen = ~np.isin(donors, known)
    if (unseen & train_mask).any():
        raise RuntimeError("a training cell has a donor the model was not set up with")

    labels = donors.copy()
    labels[unseen] = known[0]
    emb = model.get_latent_representation(_private_adata(counts, labels), give_mean=True)
    emb = np.asarray(emb, dtype=np.float32)

    diff = 0.0
    if unseen.any() and len(known) > 1:
        other = _private_adata(counts[unseen], np.full(int(unseen.sum()), known[-1]))
        again = model.get_latent_representation(other, give_mean=True)
        diff = float(np.abs(np.asarray(again, dtype=np.float32) - emb[unseen]).max())
        if diff != 0.0:
            raise RuntimeError(
                f"embedding of unseen-donor cells changed by {diff:g} with the stand-in batch "
                "label; the encoder is expected to ignore the batch"
            )
    return emb, diff


def _history(model: scvi.model.SCVI) -> dict[str, list[float]]:
    """Per-epoch training curves from the library's logger, under this project's names.

    scvi-tools logs `reconstruction_loss_train` and `kl_local_train` as sums over the epoch's
    cells divided by the number of cells, measured on the mini-batches as they were trained
    on. `kl_weight` is logged per step and reduced to the epoch's last value by its logger;
    with an epoch-based warm-up it is constant within an epoch.
    """
    names = {
        "train_recon": "reconstruction_loss_train",
        "train_kl": "kl_local_train",
        "kl_weight": "kl_weight",
    }
    history = model.history
    return {
        ours: [float(v) for v in history[theirs].to_numpy().ravel()]
        for ours, theirs in names.items()
        if theirs in history
    }


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence progress chatter from scvi-tools and lightning; keep real warnings and errors.

    INFO-level log lines (device banner, tips, "AnnData setup transferred") are dropped by
    raising the level of the named loggers to WARNING for the duration and then restoring it.
    Warnings are filtered one known message at a time. The progress bar is switched off in the
    training arguments, not here.
    """
    loggers = [logging.getLogger(name) for name in _NOISY_LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            for message in _HARMLESS_WARNINGS:
                warnings.filterwarnings("ignore", message=message)
            yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)
