import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from cellmap.transformer import (
    CellTransformer,
    bin_baselines,
    bin_values,
    choose_mask,
    count_parameters,
    embed,
    length_batches,
    lr_at,
    make_batch,
    masked_scores,
    tokenize,
    tokenize_cell,
    train_transformer,
)

# Other jobs share this machine, and one thread also keeps results reproducible.
torch.set_num_threads(1)

# One hand-written cell over 8 genes (gene id = column number), and one cell with nothing in it.
#   gene id:   0  1  2  3  4  5  6  7
#   count:     0  3  1  0  1  7  2  1
# Expressed genes: 1, 2, 4, 5, 6, 7 (six of them). For each value, the number of expressed
# genes with a strictly lower value: count 1 -> 0, count 2 -> 3, count 3 -> 4, count 7 -> 5.
# With n_bins = 4, bin = floor(4 * lower / 6): count 1 -> 0, 2 -> floor(12/6) = 2,
# 3 -> floor(16/6) = 2, 7 -> floor(20/6) = 3.
# With n_bins = 10, bin = floor(10 * lower / 6): count 1 -> 0, 2 -> 5, 3 -> 6, 7 -> 8.
# Tokens are ordered from highest to lowest count, ties by gene id: genes 5, 1, 6, 2, 4, 7.
HAND_COUNTS = np.array([[0, 3, 1, 0, 1, 7, 2, 1], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
HAND_GENES = [5, 1, 6, 2, 4, 7]
HAND_BINS_4 = [3, 2, 2, 0, 0, 0]
HAND_BINS_10 = [8, 6, 5, 0, 0, 0]


def small_model(n_genes: int = 30, seed: int = 0, **kwargs) -> CellTransformer:
    torch.manual_seed(seed)
    defaults = dict(n_bins=5, max_len=16, d_model=16, n_layers=2, n_heads=2, dropout=0.1)
    return CellTransformer(n_genes, **{**defaults, **kwargs}).eval()


def random_counts(n_cells: int, n_genes: int, seed: int, rate: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(rate, size=(n_cells, n_genes)).astype(np.float32)


# ------------------------------------------------------------------------------ tokenization


def test_bin_values_docstring_example():
    # lower counts [0, 0, 2, 3, 4], n = 5, n_bins = 4: floor(4 * lower / 5) = [0, 0, 1, 2, 3]
    np.testing.assert_array_equal(bin_values(np.array([1, 1, 2, 5, 9]), 4), [0, 0, 1, 2, 3])


def test_bin_values_single_value_and_all_equal_are_bin_zero():
    np.testing.assert_array_equal(bin_values(np.array([4.0]), 10), [0])
    np.testing.assert_array_equal(bin_values(np.array([2.0, 2.0, 2.0]), 10), [0, 0, 0])


def test_bin_values_use_every_bin_when_values_are_distinct():
    bins = bin_values(np.arange(1, 101), 10)  # 100 distinct values: 10 per bin
    np.testing.assert_array_equal(np.bincount(bins), np.full(10, 10))


def test_tokenize_cell_hand_example():
    genes, bins = tokenize_cell(np.arange(8), HAND_COUNTS[0], n_bins=4)
    np.testing.assert_array_equal(genes, HAND_GENES)
    np.testing.assert_array_equal(bins, HAND_BINS_4)


@pytest.mark.parametrize("as_sparse", [False, True])
def test_tokenize_hand_matrix(as_sparse):
    counts = sp.csr_matrix(HAND_COUNTS) if as_sparse else HAND_COUNTS
    tokens = tokenize(counts, n_bins=10)
    assert (tokens.n_cells, tokens.n_genes, tokens.n_bins) == (2, 8, 10)
    np.testing.assert_array_equal(tokens.indptr, [0, 6, 6])  # the second cell has no token
    np.testing.assert_array_equal(tokens.genes, HAND_GENES)
    np.testing.assert_array_equal(tokens.bins, HAND_BINS_10)


def test_tokenize_ignores_stored_zeros_and_column_order_of_sparse_input():
    # The same cell stored with an explicit zero and with unsorted column indices.
    messy = sp.csr_matrix(
        (np.array([1.0, 0.0, 7.0, 3.0, 1.0, 2.0, 1.0]), np.array([7, 0, 5, 1, 2, 6, 4]), [0, 7]),
        shape=(1, 8),
    )
    tokens = tokenize(messy, n_bins=10)
    np.testing.assert_array_equal(tokens.genes, HAND_GENES)
    np.testing.assert_array_equal(tokens.bins, HAND_BINS_10)


def test_tokenize_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        tokenize(np.array([[1.0, -1.0]]))


def test_tokens_do_not_change_when_a_cell_is_scaled_or_log_normalized():
    counts = random_counts(20, 40, seed=1, rate=1.5)
    base = tokenize(counts)
    deeper = tokenize(counts * 7.0)  # the same cell sequenced "7 times deeper"
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    normalized = tokenize(np.log1p(counts / totals * 1e4))  # the project's normalization
    for other in (deeper, normalized):
        np.testing.assert_array_equal(other.indptr, base.indptr)
        np.testing.assert_array_equal(other.genes, base.genes)
        np.testing.assert_array_equal(other.bins, base.bins)


def test_make_batch_pads_to_the_longest_cell():
    tokens = tokenize(HAND_COUNTS, n_bins=10)
    batch = make_batch(tokens, np.array([1, 0]), max_len=256)
    pad_gene, pad_bin = 8, 11  # n_genes and n_bins + 1
    np.testing.assert_array_equal(batch.genes.numpy(), [[pad_gene] * 6, HAND_GENES])
    np.testing.assert_array_equal(batch.bins.numpy(), [[pad_bin] * 6, HAND_BINS_10])
    np.testing.assert_array_equal(batch.pad.numpy(), [[True] * 6, [False] * 6])


def test_make_batch_truncation_is_top_genes_without_rng_and_a_random_subset_with_rng():
    tokens = tokenize(HAND_COUNTS, n_bins=10)
    top = make_batch(tokens, np.array([0]), max_len=3)
    np.testing.assert_array_equal(top.genes.numpy(), [HAND_GENES[:3]])  # genes 5, 1, 6
    np.testing.assert_array_equal(top.bins.numpy(), [HAND_BINS_10[:3]])

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        batch = make_batch(tokens, np.array([0]), max_len=3, rng=rng)
        genes, bins = batch.genes.numpy()[0], batch.bins.numpy()[0]
        assert len(set(genes)) == 3 and set(genes) <= set(HAND_GENES)
        # every kept gene still carries the bin it has in the full cell
        assert all(HAND_BINS_10[HAND_GENES.index(g)] == b for g, b in zip(genes, bins, strict=True))
        seen.add(tuple(genes))
    assert len(seen) > 5  # different subsets on different draws


def test_length_batches_use_every_cell_once_and_group_similar_lengths():
    lengths = np.random.default_rng(0).integers(1, 200, size=103)
    batches = length_batches(lengths, batch_size=10, rng=np.random.default_rng(1))
    assert sorted(np.concatenate(batches).tolist()) == list(range(103))
    assert max(len(b) for b in batches) == 10
    spread = np.mean([np.ptp(lengths[b]) for b in batches])
    assert spread < 0.25 * np.ptp(lengths)  # random batches of 10 would span most of the range
    again = length_batches(lengths, batch_size=10, rng=np.random.default_rng(1))
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))


# ----------------------------------------------------------------------------------- masking


def test_choose_mask_counts_and_never_touches_padding():
    lengths = [1, 2, 3, 4, 10, 20, 40, 0]
    pad = torch.tensor([[j >= n for j in range(40)] for n in lengths])
    generator = torch.Generator().manual_seed(0)
    # round(0.15 * n), at least 1, at most n: 1, 1, 1, 1, 2, 3, 6, and 0 for the empty cell
    expected = [1, 1, 1, 1, 2, 3, 6, 0]
    for _ in range(20):
        mask = choose_mask(pad, 0.15, generator)
        assert mask.shape == pad.shape and mask.dtype == torch.bool
        assert not (mask & pad).any()
        assert mask.sum(dim=1).tolist() == expected


def test_choose_mask_is_random_and_follows_its_generator():
    pad = torch.zeros(4, 30, dtype=torch.bool)
    a = choose_mask(pad, 0.15, torch.Generator().manual_seed(0))
    b = choose_mask(pad, 0.15, torch.Generator().manual_seed(0))
    c = choose_mask(pad, 0.15, torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_model_hides_the_level_at_exactly_the_masked_positions():
    model = small_model()
    tokens = tokenize(random_counts(6, 30, seed=2), n_bins=5)
    batch = make_batch(tokens, np.arange(6), max_len=16)
    mask = choose_mask(batch.pad, 0.3, torch.Generator().manual_seed(0))

    seen = {}
    hooks = [
        model.value_emb.register_forward_hook(lambda _m, args, _o: seen.update(values=args[0])),
        model.gene_emb.register_forward_hook(lambda _m, args, _o: seen.update(genes=args[0])),
    ]
    hidden = model(batch.genes, batch.bins, batch.pad, mask)
    for hook in hooks:
        hook.remove()

    # What the level table was asked for: [MASK] (id n_bins) where masked, the true bin
    # elsewhere, the padding id at padding. The gene ids are passed through untouched.
    assert (seen["values"][mask] == model.mask_id).all()
    assert torch.equal(seen["values"][~mask], batch.bins[~mask])
    assert torch.equal(seen["genes"], batch.genes)
    # [CLS] is an extra position in front that no mask column refers to.
    assert hidden.shape == (6, 1 + batch.genes.shape[1], 16)

    # The true level of a masked token is invisible: changing it changes nothing ...
    other_bins = batch.bins.clone()
    other_bins[mask] = (other_bins[mask] + 1) % model.n_bins
    assert torch.equal(model(batch.genes, other_bins, batch.pad, mask), hidden)
    # ... while the level of a visible token does matter.
    visible = (~mask & ~batch.pad).nonzero()[0]
    other_bins = batch.bins.clone()
    other_bins[visible[0], visible[1]] = (other_bins[visible[0], visible[1]] + 1) % model.n_bins
    assert not torch.allclose(model(batch.genes, other_bins, batch.pad, mask), hidden, atol=1e-4)


# ------------------------------------------------------------------------------------- model


def test_default_parameter_count():
    # gene table (1000 + 1) * 128, level table (10 + 2) * 128, [CLS] 128
    # per block: qkv 128 * 384 + 384, out 128 * 128 + 128, MLP 128 * 512 + 512 + 512 * 128 + 128,
    #            two layer norms 2 * 256                                  -> 198,272
    # final norm 256, token head 128 * 10 + 10
    # [CLS] head with inner size 32: cell_to_bins 128 * 320 + 320, gene_query 128 * 32 + 32
    expected = 1001 * 128 + 12 * 128 + 128 + 4 * 198_272 + 256 + 1290 + 41_280 + 4128
    assert expected == 969_834
    assert count_parameters(CellTransformer(n_genes=1000)) == expected


def test_padding_does_not_change_a_cells_embedding():
    model = small_model()
    counts = random_counts(8, 30, seed=3, rate=0.4)
    counts[0] = 0
    counts[0, [2, 5, 11]] = [4, 1, 2]  # a short cell: 3 tokens
    counts[1] = 3  # a long cell: all 30 genes expressed, cut to max_len = 16
    for pooling in ("cls", "mean"):
        alone = embed(model, counts[:1], pooling=pooling)
        together = embed(model, counts, pooling=pooling)
        assert alone.shape == (1, 16) and together.shape == (8, 16)
        np.testing.assert_allclose(together[0], alone[0], atol=1e-5)
        # batch size and the order of the other cells do not matter either
        one_by_one = embed(model, counts, batch_size=1, pooling=pooling)
        np.testing.assert_allclose(together, one_by_one, atol=1e-5)
        np.testing.assert_allclose(
            embed(model, counts[::-1], pooling=pooling)[::-1], together, 0, 1e-5
        )


def test_embedding_ignores_the_order_of_the_tokens():
    model = small_model()
    tokens = tokenize(random_counts(5, 30, seed=4), n_bins=5)
    batch = make_batch(tokens, np.arange(5), max_len=16)
    shuffle = torch.randperm(batch.genes.shape[1], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        hidden = model(*batch)
        shuffled = model(batch.genes[:, shuffle], batch.bins[:, shuffle], batch.pad[:, shuffle])
    # [CLS] is unchanged, and each token's own state just moves with the token.
    assert torch.allclose(shuffled[:, 0], hidden[:, 0], atol=1e-5)
    assert torch.allclose(shuffled[:, 1:], hidden[:, 1:][:, shuffle], atol=1e-5)
    assert not torch.allclose(hidden[:, 1], hidden[:, 2], atol=1e-3)  # tokens do differ


def test_embedding_depends_on_the_cell_but_not_on_its_depth():
    model = small_model()
    counts = random_counts(4, 30, seed=5, rate=1.5)
    emb = embed(model, counts)
    np.testing.assert_array_equal(embed(model, counts * 3.0), emb)
    assert np.abs(emb[0] - emb[1]).max() > 1e-3


def test_embed_output_type_determinism_and_mode():
    model = small_model().train()
    counts = sp.csr_matrix(random_counts(7, 30, seed=6))
    first = embed(model, counts)
    assert model.training  # the mode the caller had is restored
    assert first.dtype == np.float32 and first.shape == (7, 16) and np.isfinite(first).all()
    np.testing.assert_array_equal(embed(model, counts), first)  # dropout is off inside embed
    np.testing.assert_array_equal(embed(model, counts.toarray()), first)
    assert not np.allclose(embed(model, counts, pooling="mean"), first, atol=1e-3)
    with pytest.raises(ValueError, match="pooling"):
        embed(model, counts, pooling="max")
    with pytest.raises(ValueError, match="genes"):
        embed(model, counts[:, :10])


def test_cell_without_any_expressed_gene_gets_a_finite_embedding():
    model = small_model()
    counts = np.zeros((2, 30), dtype=np.float32)
    counts[1, 3] = 2
    assert np.isfinite(embed(model, counts)).all()
    np.testing.assert_array_equal(embed(model, counts, pooling="mean")[0], np.zeros(16))


def test_untrained_loss_is_close_to_log_n_bins():
    # Small initial weights give nearly uniform predictions, whose cross-entropy is log(n_bins)
    # whatever the true bins are.
    for n_bins in (5, 10):
        torch.manual_seed(0)
        model = CellTransformer(n_genes=50, n_bins=n_bins, d_model=32, n_layers=2, n_heads=2)
        tokens = tokenize(random_counts(64, 50, seed=7, rate=2.0), n_bins)
        scores = masked_scores(model, tokens)
        assert scores["loss"] == pytest.approx(math.log(n_bins), abs=0.1)
        assert scores["cls_loss"] == pytest.approx(math.log(n_bins), abs=0.1)


# ---------------------------------------------------------------------------------- training


def test_lr_schedule_warms_up_linearly_then_follows_a_cosine():
    lr, total, warm = 1.0, 110, 10
    assert lr_at(0, total, warm, lr) == pytest.approx(0.1)
    assert lr_at(4, total, warm, lr) == pytest.approx(0.5)
    assert lr_at(9, total, warm, lr) == pytest.approx(1.0)
    assert lr_at(10, total, warm, lr) == pytest.approx(1.0)  # cosine starts at the peak
    assert lr_at(60, total, warm, lr) == pytest.approx(0.5)  # halfway down
    assert lr_at(110, total, warm, lr) == pytest.approx(0.0, abs=1e-12)
    rates = [lr_at(s, total, warm, lr) for s in range(total)]
    assert all(a > b for a, b in zip(rates[10:], rates[11:], strict=False))
    assert lr_at(0, 50, 0, 2.0) == pytest.approx(2.0)  # no warm-up


def test_bin_baselines_hand_example():
    # Two cells, 3 genes, n_bins = 2 (bin = floor(2 * lower / n)).
    # cell A counts [5, 1, 1]: lower counts 2, 0, 0 of n = 3 -> bins gene0: 1, gene1: 0, gene2: 0
    # cell B counts [3, 0, 9]: lower counts 0, -, 1 of n = 2 -> bins gene0: 0, gene2: 1
    # All tokens: bin 0 three times, bin 1 twice -> overall most common bin is 0.
    # gene0: one of each (tie -> lowest bin, 0); gene1: bin 0; gene2: one of each -> 0.
    overall, per_gene = bin_baselines(tokenize(np.array([[5.0, 1, 1], [3, 0, 9]]), n_bins=2))
    assert overall == 0
    np.testing.assert_array_equal(per_gene, [0, 0, 0])
    # A gene that is high in every cell gets its own guess; a gene never seen gets the overall.
    overall, per_gene = bin_baselines(tokenize(np.array([[9.0, 1, 0, 1], [8, 2, 0, 2]]), n_bins=2))
    assert overall == 0
    np.testing.assert_array_equal(per_gene, [1, 0, 0, 0])


def planted_counts(n_cells: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three cell types over 40 genes in 4 modules of 10 genes.

    Modules 0, 1, 2 are each high (rate 6) in one type and low (rate 0.7) in the other two;
    module 3 is medium (rate 2.5) everywhere. Every cell also gets a random depth factor
    between 0.5 and 2, which the per-cell binning should make irrelevant. The level of a gene
    is therefore predictable from the other genes of its cell, and not from the gene alone.
    """
    rng = np.random.default_rng(seed)
    cell_type = rng.integers(0, 3, size=n_cells)
    module_rates = np.full((3, 4), 0.7)
    module_rates[np.arange(3), np.arange(3)] = 6.0
    module_rates[:, 3] = 2.5
    rates = np.repeat(module_rates[cell_type], 10, axis=1)
    depth = rng.uniform(0.5, 2.0, size=(n_cells, 1))
    return rng.poisson(rates * depth).astype(np.float32), cell_type


TRAIN_KWARGS = dict(
    n_bins=5, d_model=32, n_layers=2, n_heads=2, n_epochs=40, batch_size=16, lr=2e-3, seed=0
)


@pytest.fixture(scope="module")
def planted():
    counts_train, type_train = planted_counts(480, seed=0)
    counts_val, type_val = planted_counts(240, seed=1)
    model, history = train_transformer(
        sp.csr_matrix(counts_train), counts_val=counts_val, device="cpu", **TRAIN_KWARGS
    )
    return model, history, (counts_train, type_train, counts_val, type_val)


def test_training_history_has_one_entry_per_epoch(planted):
    model, history, _ = planted
    assert not model.training
    names = ("loss", "accuracy", "cls_loss", "cls_accuracy", "majority", "gene_majority")
    expected = {f"{part}_{name}" for part in ("train", "val") for name in names}
    assert set(history) == expected | {"lr", "seconds_train", "seconds"}
    assert all(len(values) == 40 for values in history.values())
    assert all(np.isfinite(values).all() for values in history.values())
    assert all(
        s >= t > 0 for s, t in zip(history["seconds"], history["seconds_train"], strict=True)
    )
    assert history["lr"][-1] < 0.01 * 2e-3  # the cosine has come down


def test_training_beats_both_model_free_baselines_on_held_out_cells(planted):
    _, history, _ = planted
    assert history["val_loss"][-1] < history["val_loss"][0] < math.log(5)
    assert history["train_loss"][-1] < history["train_loss"][0]
    # The trivial guess (most common bin) and the per-gene guess, on the same masked tokens.
    assert history["val_accuracy"][-1] > history["val_majority"][-1] + 0.15
    assert history["val_accuracy"][-1] > history["val_gene_majority"][-1] + 0.10
    # The [CLS] head, which sees the cell only through one vector, must have learned too.
    assert history["val_cls_accuracy"][-1] > history["val_gene_majority"][-1] + 0.07


def test_cell_types_are_separable_in_the_embedding(planted):
    # Only the default [CLS] embedding is held to this. In development runs on this data the
    # mean of the token states separated the types clearly worse (57% to 71% nearest-centroid
    # accuracy in the four runs tried), presumably because no loss reads the final states of
    # unmasked tokens.
    model, _, (counts_train, type_train, counts_val, type_val) = planted
    emb_train, emb_val = embed(model, counts_train), embed(model, counts_val)
    # Nearest centroid: centroids from training cells, each held-out cell goes to the closest.
    centroids = np.stack([emb_train[type_train == t].mean(axis=0) for t in range(3)])
    distances = np.linalg.norm(emb_val[:, None, :] - centroids[None, :, :], axis=2)
    assert (distances.argmin(axis=1) == type_val).mean() > 0.9  # chance is 1/3


def test_same_seed_reproduces_and_another_seed_differs():
    counts, _ = planted_counts(96, seed=2)
    kwargs = {**TRAIN_KWARGS, "n_epochs": 2}
    timing = ("seconds", "seconds_train")

    def run(seed: int, device):
        model, history = train_transformer(
            counts, counts_val=counts[:32], device=device, **{**kwargs, "seed": seed}
        )
        return embed(model, counts, device=device), {
            key: values for key, values in history.items() if key not in timing
        }

    emb_a, history_a = run(0, "cpu")
    emb_b, history_b = run(0, torch.device("cpu"))  # both spellings of the device are accepted
    emb_c, history_c = run(1, "cpu")
    np.testing.assert_array_equal(emb_a, emb_b)
    assert history_a == history_b
    assert not np.allclose(emb_a, emb_c, atol=1e-4)
    assert history_a["train_loss"] != history_c["train_loss"]


def test_training_never_updates_on_validation_cells():
    counts, _ = planted_counts(64, seed=3)
    other, _ = planted_counts(32, seed=4)
    kwargs = {**TRAIN_KWARGS, "n_epochs": 2}
    with_val, _ = train_transformer(counts, counts_val=other, **kwargs)
    without_val, _ = train_transformer(counts, **kwargs)
    for a, b in zip(with_val.parameters(), without_val.parameters(), strict=True):
        assert torch.equal(a, b)


def test_long_cells_and_bad_arguments():
    counts, _ = planted_counts(48, seed=5)  # about 30 expressed genes per cell
    model, history = train_transformer(counts, **{**TRAIN_KWARGS, "n_epochs": 1, "max_len": 8})
    assert model.max_len == 8 and np.isfinite(history["train_loss"]).all()
    assert np.isfinite(embed(model, counts)).all()
    with pytest.raises(ValueError, match="mask_fraction"):
        train_transformer(counts, mask_fraction=0.0)
    with pytest.raises(ValueError, match="numbers of genes"):
        train_transformer(counts, counts_val=counts[:, :5], n_epochs=1)


def test_value_encoding_separates_depths_that_rank_encoding_ties():
    from cellmap.transformer import fit_value_edges, tokenize

    # Two cells with the same gene pattern; cell 1 is sequenced 10 times deeper.
    counts = np.array([[1, 1, 2, 0], [10, 10, 20, 0]], dtype=np.float32)
    rank = tokenize(counts, n_bins=4)
    assert np.array_equal(rank.bins[:3], rank.bins[3:])  # rank bins ignore depth
    edges = fit_value_edges(counts, n_bins=4)
    value = tokenize(counts, n_bins=4, edges=edges)
    # Normalized by depth the two cells are identical too, so the bins must match...
    assert np.array_equal(value.bins[:3], value.bins[3:])
    # ...but a lone count of 1 in a shallow cell and in a deep cell now differ.
    shallow_deep = np.array([[1, 1, 1, 1], [1, 30, 30, 30]], dtype=np.float32)
    tok = tokenize(shallow_deep, n_bins=4, edges=fit_value_edges(shallow_deep, 4))
    gene0_shallow = tok.bins[: tok.indptr[1]][tok.genes[: tok.indptr[1]] == 0][0]
    gene0_deep = tok.bins[tok.indptr[1] :][tok.genes[tok.indptr[1] :] == 0][0]
    assert gene0_shallow > gene0_deep


def test_value_edges_are_quantiles_of_training_values():
    from cellmap.transformer import fit_value_edges

    counts = np.array([[1, 2, 3, 4]], dtype=np.float32)
    edges = fit_value_edges(counts, n_bins=2)
    levels = np.log1p(1e4 * np.array([1, 2, 3, 4]) / 10)
    assert np.isclose(edges[0], np.median(levels))


def test_value_encoding_trains_and_embeds():
    from cellmap.transformer import embed, train_transformer

    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(60, 30)).astype(np.float32)
    model, history = train_transformer(
        counts, n_epochs=1, batch_size=16, d_model=16, n_layers=1, n_heads=2,
        level_encoding="value",
    )  # fmt: skip
    assert model.value_edges is not None and len(model.value_edges) == 9
    emb = embed(model, counts)
    assert emb.shape == (60, 16) and np.isfinite(emb).all()
