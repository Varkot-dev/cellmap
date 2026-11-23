import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from cellmap.scvi_baseline import _batch_categories, _private_adata, scvi_embed, scvi_settings

# Other jobs share the machine; the model is tiny, so one thread is also the fastest choice.
torch.set_num_threads(1)

N_TYPES, N_DONORS, CELLS_PER_GROUP, GENES_PER_TYPE = 3, 6, 20, 20
TRAIN_DONORS = ("d0", "d1", "d2", "d3")  # d4 and d5 are never trained on
FAST = {"max_epochs": 40, "kl_warmup_epochs": 10, "batch_size": 32, "n_hidden": 32, "n_latent": 5}


def make_adata(seed: int = 0) -> ad.AnnData:
    """360 cells: 3 planted cell types x 6 donors x 20 cells, 75 genes, Poisson counts.

    Each type has its own block of 20 marker genes expressed 6 times higher than elsewhere;
    15 genes are background. Every donor multiplies every gene by its own factor (a batch
    effect) and every cell has its own depth. X holds log-normalized values and
    layers["counts"] the raw counts, like the project's processed files. Rows are shuffled so
    that no test passes by relying on cells being grouped.
    """
    rng = np.random.default_rng(seed)
    n_genes = N_TYPES * GENES_PER_TYPE + 15
    cell_type = np.repeat(np.arange(N_TYPES), N_DONORS * CELLS_PER_GROUP)
    donor = np.tile(np.repeat(np.arange(N_DONORS), CELLS_PER_GROUP), N_TYPES)
    order = rng.permutation(len(cell_type))
    cell_type, donor = cell_type[order], donor[order]

    profile = np.ones((N_TYPES, n_genes))
    for t in range(N_TYPES):
        profile[t, t * GENES_PER_TYPE : (t + 1) * GENES_PER_TYPE] = 6.0
    donor_factor = np.exp(rng.normal(0.0, 0.3, size=(N_DONORS, n_genes)))
    depth = rng.uniform(0.5, 2.0, size=(len(cell_type), 1))
    counts = rng.poisson(profile[cell_type] * donor_factor[donor] * depth).astype(np.float32)

    obs = pd.DataFrame(
        {
            "donor_id": pd.Categorical([f"d{d}" for d in donor]),
            "cell_type": pd.Categorical([f"type{t}" for t in cell_type]),
        },
        index=[f"cell{i}" for i in range(len(cell_type))],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    total = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    adata = ad.AnnData(X=sp.csr_matrix(np.log1p(counts / total * 1e4)), obs=obs, var=var)
    adata.layers["counts"] = sp.csr_matrix(counts)
    return adata


def masks(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """(gene_mask, train_mask): the last 5 background genes are left out, 4 donors train."""
    gene_mask = np.ones(adata.n_vars, dtype=bool)
    gene_mask[-5:] = False
    return gene_mask, adata.obs["donor_id"].isin(TRAIN_DONORS).to_numpy()


def run(adata: ad.AnnData, **overrides) -> tuple[np.ndarray, dict]:
    gene_mask, train_mask = masks(adata)
    return scvi_embed(adata, gene_mask, train_mask, **{**FAST, **overrides})


def state(info: dict) -> dict[str, np.ndarray]:
    """Every learned weight and batch-norm running average of the trained model."""
    return {k: v.numpy().copy() for k, v in info["model"].module.state_dict().items()}


@pytest.fixture(scope="module")
def adata() -> ad.AnnData:
    return make_adata()


@pytest.fixture(scope="module")
def fitted(adata) -> tuple[np.ndarray, dict]:
    """One training run shared by the tests that only read its outputs."""
    return run(adata, return_model=True)


def test_shape_dtype_and_finite(adata, fitted):
    emb, info = fitted
    assert emb.shape == (adata.n_obs, FAST["n_latent"])
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()
    assert info["n_train_cells"] == 240 and info["n_genes"] == adata.n_vars - 5
    assert info["train_batches"] == list(TRAIN_DONORS)
    assert info["unseen_batch_cells"] == 120


def test_info_records_settings_and_history(fitted):
    _, info = fitted
    settings = info["settings"]
    assert settings["model"]["gene_likelihood"] == "nb"
    assert settings["model"]["encode_covariates"] is False
    assert settings["train"]["train_size"] == 1.0
    assert settings["train"]["early_stopping"] is False
    assert settings["train"]["plan_kwargs"]["n_epochs_kl_warmup"] == FAST["kl_warmup_epochs"]
    assert settings["seed"] == 0
    history = info["history"]
    assert {len(v) for v in history.values()} == {FAST["max_epochs"]}
    # Linear warm-up: 0, 0.1, ..., 0.9, then 1 for the rest. Same as cellmap.vae.kl_weight_at.
    expected = np.minimum(np.arange(FAST["max_epochs"]) / FAST["kl_warmup_epochs"], 1.0)
    np.testing.assert_allclose(history["kl_weight"], expected, atol=1e-6)
    assert history["train_recon"][-1] < history["train_recon"][0]
    assert info["train_seconds"] > 0 and info["scvi_tools_version"]


def test_model_was_trained_on_all_training_cells_and_nothing_else(fitted):
    model = fitted[1]["model"]
    assert model.adata.n_obs == 240
    assert sorted(model.train_indices) == list(range(240))
    assert len(model.validation_indices) == 0 and len(model.test_indices) == 0
    assert list(_batch_categories(model)) == list(TRAIN_DONORS)


def test_callers_adata_is_not_modified(adata):
    before = adata.copy()
    run(adata, max_epochs=2)
    assert list(adata.obs.columns) == list(before.obs.columns)
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    assert list(adata.uns.keys()) == list(before.uns.keys()) == []
    assert list(adata.obsm.keys()) == []
    assert list(adata.layers.keys()) == list(before.layers.keys())
    assert (adata.X != before.X).nnz == 0
    assert (adata.layers["counts"] != before.layers["counts"]).nnz == 0


def test_rows_follow_adata_order(adata, fitted):
    # Reversing the held-out rows leaves the training slice untouched, so the same model is
    # trained; the held-out embeddings must come back reversed along with their cells.
    emb, _ = fitted
    _, train_mask = masks(adata)
    order = np.arange(adata.n_obs)
    order[~train_mask] = order[~train_mask][::-1]
    emb_moved, _ = run(adata[order].copy())
    np.testing.assert_array_equal(emb_moved, emb[order])


def test_unseen_donor_embedding_ignores_the_batch_label(adata, fitted):
    emb, info = fitted
    gene_mask, train_mask = masks(adata)
    assert info["unseen_batch_max_abs_diff"] == 0.0
    counts = sp.csr_matrix(adata.layers["counts"][~train_mask][:, gene_mask], dtype=np.float32)
    for donor in TRAIN_DONORS:
        stand_in = _private_adata(counts, np.full(counts.shape[0], donor))
        again = info["model"].get_latent_representation(stand_in)
        np.testing.assert_array_equal(again, emb[~train_mask])


def test_unseen_donor_label_is_rejected_by_the_library_itself(adata, fitted):
    # The reason a stand-in label is needed at all. If scvi-tools ever starts accepting
    # unknown batches, this fails and the stand-in can be removed.
    gene_mask, train_mask = masks(adata)
    counts = sp.csr_matrix(adata.layers["counts"][~train_mask][:, gene_mask], dtype=np.float32)
    donors = adata.obs["donor_id"].astype(str).to_numpy()[~train_mask]
    with pytest.raises(ValueError, match="not found in source registry"):
        fitted[1]["model"].get_latent_representation(_private_adata(counts, donors))


def test_non_training_cells_do_not_influence_the_model(adata, fitted):
    emb, info = fitted
    _, train_mask = masks(adata)
    changed = adata.copy()
    rng = np.random.default_rng(1)
    noise = rng.poisson(3.0, size=(int((~train_mask).sum()), adata.n_vars)).astype(np.float32)
    counts = changed.layers["counts"].toarray()
    counts[~train_mask] = noise
    changed.layers["counts"] = sp.csr_matrix(counts)
    changed.obs["donor_id"] = changed.obs["donor_id"].astype(str)
    changed.obs.loc[~train_mask, "donor_id"] = "someone_else"

    emb_changed, info_changed = run(changed, return_model=True)
    np.testing.assert_array_equal(emb_changed[train_mask], emb[train_mask])
    assert not np.allclose(emb_changed[~train_mask], emb[~train_mask])
    before, after = state(info), state(info_changed)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)


def test_planted_cell_types_separate_on_held_out_donors(adata, fitted):
    emb, _ = fitted
    _, train_mask = masks(adata)
    labels = adata.obs["cell_type"].to_numpy()
    classes = np.unique(labels)
    centroids = np.stack([emb[train_mask & (labels == c)].mean(axis=0) for c in classes])
    held_out = emb[~train_mask]
    distances = np.linalg.norm(held_out[:, None, :] - centroids[None, :, :], axis=-1)
    predicted = classes[distances.argmin(axis=1)]
    assert (predicted == labels[~train_mask]).mean() > 0.95


def test_same_seed_reproduces_and_other_seed_differs(adata, fitted):
    emb, info = fitted
    emb_again, info_again = run(adata)
    np.testing.assert_array_equal(emb_again, emb)
    assert info_again["history"] == info["history"]
    emb_other, _ = run(adata, seed=1, max_epochs=5)
    assert not np.allclose(emb_other, emb)


def test_one_cell_last_batch_is_dropped_instead_of_crashing(adata):
    # 240 training cells in batches of 239 leave a single cell, which batch norm cannot take.
    emb, info = run(adata, batch_size=239, max_epochs=2)
    assert info["settings"]["train"]["datasplitter_kwargs"] == {"drop_last": True}
    assert np.isfinite(emb).all()
    assert scvi_settings()["train"]["datasplitter_kwargs"] == {"drop_last": False}


def test_bad_inputs_are_rejected(adata):
    gene_mask, train_mask = masks(adata)
    with pytest.raises(ValueError, match="boolean mask"):
        scvi_embed(adata, gene_mask, np.flatnonzero(train_mask))
    with pytest.raises(ValueError, match="gene_mask"):
        scvi_embed(adata, np.flatnonzero(gene_mask), train_mask)
    with pytest.raises(ValueError, match="raw counts"):
        scvi_embed(adata, gene_mask, train_mask, counts_layer="missing")


def test_scvi_weights_in_the_hand_written_vae_give_the_same_embedding(adata, fitted):
    """Weight transplant, the strongest of the three VAE parity checks (docs/NOTES.md).

    Same architecture and forward pass means every layer of the scvi-tools model has a
    counterpart in cellmap.vae.VAE; copying the trained weights across must reproduce the
    library's posterior means. First run as a scratch script on the tiny dataset; an
    independent review found nothing in the repo recorded it, so it is a test now.
    """
    from cellmap.vae import VAE, embed

    emb_lib, info = fitted
    lib = info["model"].module
    gene_mask, _ = masks(adata)
    hand = VAE(
        n_genes=int(gene_mask.sum()),
        n_batches=info["n_train_batches"],
        n_latent=FAST["n_latent"],
        n_hidden=FAST["n_hidden"],
    )

    def copy(dst: torch.nn.Module, src: torch.nn.Module) -> None:
        dst.load_state_dict(src.state_dict())

    enc, dec = lib.z_encoder, lib.decoder
    copy(hand.encoder[0][0], enc.encoder.fc_layers[0][0])
    copy(hand.encoder[0][1], enc.encoder.fc_layers[0][1])
    copy(hand.z_mean, enc.mean_encoder)
    copy(hand.z_raw_var, enc.var_encoder)
    copy(hand.decoder[0][0], dec.px_decoder.fc_layers[0][0])
    copy(hand.decoder[0][1], dec.px_decoder.fc_layers[0][1])
    copy(hand.gene_logits, dec.px_scale_decoder[0])
    hand.log_theta.data.copy_(lib.px_r.data)
    for bn in (hand.encoder[0][1], hand.decoder[0][1]):
        bn.eps = 1e-3  # scvi-tools' batch-norm eps; it matters in evaluation mode
    hand.eval()

    counts = adata.layers["counts"][:, gene_mask].astype(np.float32)
    emb_hand = embed(hand, counts)
    np.testing.assert_allclose(emb_hand, emb_lib, atol=1e-5 * np.abs(emb_lib).max())
