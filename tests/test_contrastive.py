import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.neighbors import KNeighborsClassifier

from cellmap.contrastive import (
    alignment_uniformity,
    debiased_nt_xent,
    group_batches,
    make_view,
    nt_xent,
    project,
    train_contrastive,
)
from cellmap.transformer import embed, tokenize

torch.set_num_threads(1)


def planted_counts(n_cells: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three cell types over 40 genes in 4 modules (same design as tests/test_transformer.py)."""
    rng = np.random.default_rng(seed)
    cell_type = rng.integers(0, 3, size=n_cells)
    module_rates = np.full((3, 4), 0.7)
    module_rates[np.arange(3), np.arange(3)] = 6.0
    module_rates[:, 3] = 2.5
    rates = np.repeat(module_rates[cell_type], 10, axis=1)
    depth = rng.uniform(0.5, 2.0, size=(n_cells, 1))
    return rng.poisson(rates * depth).astype(np.float32), cell_type


def test_views_are_subsets_of_the_cell_with_valid_levels():
    counts, _ = planted_counts(50, 0)
    tokens = tokenize(counts, n_bins=5)
    rng = np.random.default_rng(0)
    view = make_view(tokens, np.arange(50), rng, max_len=256, keep_range=(0.5, 0.9), jitter=0.2)
    for i in range(50):
        own = set(tokens.genes[tokens.indptr[i] : tokens.indptr[i + 1]])
        kept = view.genes[i][~view.pad[i]].numpy()
        assert len(kept) >= 1 and set(kept) <= own and len(kept) < len(own) + 1
        bins = view.bins[i][~view.pad[i]].numpy()
        assert bins.min() >= 0 and bins.max() <= 4
    # on average about 70 percent of tokens survive
    n_view = (~view.pad).sum().item()
    n_cell = tokens.indptr[50]
    assert 0.6 < n_view / n_cell < 0.8


def test_view_keeps_a_token_even_at_a_tiny_keep_rate():
    tokens = tokenize(np.array([[0, 3.0, 0, 1.0]]), n_bins=2)
    view = make_view(tokens, np.array([0]), np.random.default_rng(0), keep_range=(0.0, 0.0))
    assert (~view.pad).sum().item() == 1


def test_nt_xent_matches_a_hand_computed_value():
    # two cells, orthogonal; views identical to their cell
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = nt_xent(z, z.clone(), temperature=1.0)
    # for view 0: partner similarity 1, others: cell 1 twice at similarity 0
    expected = -math.log(math.e / (math.e + 1 + 1))
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_nt_xent_is_low_when_partners_match_and_high_when_shuffled():
    torch.manual_seed(0)
    z = torch.randn(64, 16)
    good = nt_xent(z, z + 0.01 * torch.randn_like(z))
    bad = nt_xent(z, z[torch.randperm(64)])
    assert good < bad


@pytest.fixture(scope="module")
def trained():
    counts_train, _ = planted_counts(480, seed=0)
    counts_val, _ = planted_counts(240, seed=1)
    model, history = train_contrastive(
        sp.csr_matrix(counts_train),
        counts_val=counts_val,
        n_bins=5,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_epochs=25,
        batch_size=64,
        lr=2e-3,
        seed=0,
    )
    return model, history


def test_history_and_learning(trained):
    model, history = trained
    assert not model.training
    assert {"train_contrastive_loss", "val_contrastive_loss", "val_partner_accuracy"} <= set(
        history
    )
    assert all(len(v) == 25 for v in history.values())
    assert history["val_contrastive_loss"][-1] < history["val_contrastive_loss"][0]
    # chance of picking the partner among 127 other views is under 1 percent
    assert history["val_partner_accuracy"][-1] > 0.05


def test_cls_embedding_separates_planted_types_on_new_cells(trained):
    model, _ = trained
    counts_a, type_a = planted_counts(300, seed=2)
    counts_b, type_b = planted_counts(300, seed=3)
    knn = KNeighborsClassifier(15).fit(embed(model, counts_a), type_a)
    assert knn.score(embed(model, counts_b), type_b) > 0.9


def test_combined_objective_runs_and_records_both_losses():
    counts, _ = planted_counts(128, seed=4)
    _, history = train_contrastive(
        counts, n_bins=5, d_model=16, n_layers=1, n_heads=2, n_epochs=2, batch_size=32,
        mgm_weight=1.0, seed=0,
    )  # fmt: skip
    assert "train_mgm_loss" in history and np.isfinite(history["train_mgm_loss"]).all()


def test_debiased_loss_equals_nt_xent_when_no_negative_shares_the_type():
    torch.manual_seed(1)
    z1, z2 = torch.randn(32, 8), torch.randn(32, 8)
    # tau+ = 0 means no correction; a high temperature keeps the floor inactive
    assert debiased_nt_xent(z1, z2, 1.0, 0.0).item() == pytest.approx(
        nt_xent(z1, z2, 1.0).item(), rel=1e-5
    )
    # a positive tau+ removes part of the negative mass, so the loss falls
    assert debiased_nt_xent(z1, z2, 1.0, 0.2) < nt_xent(z1, z2, 1.0)


def test_group_batches_keep_one_group_per_batch_and_use_every_cell_once():
    groups = np.repeat(["a", "b", "c"], [70, 25, 5])
    batches = group_batches(groups, 32, np.random.default_rng(0))
    assert all(len(set(groups[b])) == 1 for b in batches)
    assert sorted(np.concatenate(batches).tolist()) == list(range(100))


def test_alignment_is_zero_for_identical_views_and_uniformity_falls_with_spread():
    torch.manual_seed(0)
    z = torch.randn(64, 16)
    assert alignment_uniformity(z, z.clone())["alignment"] == pytest.approx(0.0, abs=1e-6)
    tight = torch.randn(64, 16) * 0.01 + 5.0  # all near one direction
    assert (
        alignment_uniformity(z, z)["uniformity"] < alignment_uniformity(tight, tight)["uniformity"]
    )


def test_ablation_options_run_and_return_the_head():
    counts, cell_type = planted_counts(128, seed=5)
    model, history, head = train_contrastive(
        counts, n_bins=5, d_model=16, n_layers=1, n_heads=2, n_epochs=2, batch_size=32,
        loss="debiased", batch_groups=cell_type, return_head=True, counts_val=counts, seed=0,
    )  # fmt: skip
    assert {"val_alignment", "val_uniformity"} <= set(history)
    assert project(model, head, counts).shape == (128, 64)
    with pytest.raises(ValueError):
        train_contrastive(counts, n_epochs=1, batch_groups=cell_type[:10])
