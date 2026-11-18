"""A small transformer for single cells, trained with masked gene modeling, by hand.

The model in one paragraph. A cell is turned into a short, unordered set of tokens, one per
expressed gene. Each token says "this gene, at about this level": a learned vector for the gene
plus a learned vector for the level. A transformer encoder lets every token look at every other
token of the same cell. During training the level of some tokens is hidden and the model has
to predict it from the rest of the cell. To do that it must learn which genes go up and down
together, which is what distinguishes cell types. The state of an extra [CLS] token is then
used as the embedding of the cell.

How a cell becomes a sequence (the choices made here, and what else could have been done)

1. Which genes are tokens: only the EXPRESSED ones (count > 0) among the selected highly
   variable genes. With 1,000 selected genes a blood cell in this project expresses about 55
   of them, so sequences are short. Attention costs grow with the square of the sequence
   length, so a cell with 55 tokens is roughly 300 times cheaper than the same cell with all
   1,000 genes as tokens. The price: the model is never told "gene X is off". Absence is only
   implicit (the gene is not in the set), and the model cannot be asked to predict that a
   gene is silent. The alternative, one token per gene including zeros, makes zeros explicit
   but spends most of the compute on them, and many zeros in this kind of data are genes that
   were present but not captured.

2. Too many expressed genes: at most `max_len` tokens are kept. In training a different random
   subset is drawn every time the cell is used, which also acts as data augmentation. For
   embedding the choice must be deterministic, so the highest-expressed genes are kept (ties
   broken by lower gene index). Alternative: always keep the top genes, which is deterministic
   everywhere but never shows the model a cell's weakly expressed genes when it is long.

3. Expression level: a BIN computed inside the cell. A token's bin is

       bin = floor( n_bins * (number of the cell's expressed genes with a strictly lower value)
                    / (number of the cell's expressed genes) )

   so bin 0 holds the cell's lowest values and bin n_bins - 1 its highest. The bin depends
   only on the ORDER of the values within the cell. That has three consequences. It is robust
   to sequencing depth: a cell sequenced twice as deeply has, to a first approximation, all
   counts doubled, and its tokens do not change at all. It makes normalization irrelevant:
   dividing by the cell's total and taking log(1 + x) is an increasing function, so raw counts
   and log-normalized values give exactly the same tokens (which is why this module takes the
   same raw count matrix as `cellmap.vae`). And tied values always share a bin, so the training
   target is never an arbitrary tie-break. The price: absolute level is thrown away, and so is
   the cell's total count. Because the counts are small integers, most expressed genes in a
   shallow cell have count 1 and all of those land in bin 0, so the bins are far from equally
   filled (about 60% of tokens are in bin 0 on this project's blood data). That is why the
   training history reports the accuracy of always guessing the most common bin.

   The bins are computed from ALL of the cell's expressed selected genes before any random
   subset is taken, so a gene's bin in a given cell is always the same.

   Alternatives. (a) A continuous value passed through a small network (scFoundation style)
   keeps full resolution but ties the model to one normalization and is sensitive to depth.
   (b) Rank ordering (Geneformer style): no value at all, the cell is the list of its genes
   sorted by expression, and the position in the list carries the level. That also is robust
   to depth, but it needs positional embeddings and long sequences, and ties between small
   integer counts have to be broken somehow.

4. No positional embedding. A sentence has an order; the genes of a cell do not. Any order we
   could impose (chromosome position, alphabet, gene index) would be a made-up signal that the
   model could latch on to. Without positional embeddings the encoder treats the tokens as a
   set: shuffling them changes nothing in the output, which is the correct symmetry and is
   tested. Geneformer does use positions, but there the position IS the expression rank.

5. [CLS]: one extra learned vector is put in front of every cell. It carries no gene and no
   value; its job is to collect a summary of the cell through attention. Padding tokens fill
   shorter cells up to the longest cell in the mini-batch and are masked out of attention, so
   a cell's output does not depend on which cells it is batched with.

Training objective: masked gene modeling. In each cell 15% of the tokens (at least one) have
their level replaced by a learned [MASK] level; the gene identity stays visible. The model
predicts the hidden bin and is scored with cross-entropy. See `CellTransformer` and
`masked_scores` for why the identity is kept, and for the alternative objectives.

A second objective makes [CLS] a bottleneck: the level of every gene token of the cell must
be recoverable from the [CLS] state alone. Without it nothing would train the final [CLS]
state at all. See `CellTransformer.cls_logits`.

What this model does NOT do. The donor label is never an input, in training or in embedding,
and nothing in the objective asks for donors to be mixed. There is no batch-correction
mechanism at all. The only protection against technical differences is the per-cell binning,
which removes sequencing depth and nothing else. This is a real limitation and has to be
stated next to this model's numbers when it is compared with Harmony (which corrects for the
donor explicitly) and with the VAE (which hands the donor to its decoder).

Background, from memory and NOT verified against the papers or code for this project (neither
package is imported here): scGPT (Cui et al. 2024) uses gene tokens plus per-cell binned values
(51 bins), no positional embedding, a cell token, and predicts masked values with a regression
loss plus extra objectives that train the cell embedding. Geneformer (Theodoris et al. 2023)
uses rank ordering with each gene's count first divided by that gene's typical level across its
corpus, sequences of up to 2,048 genes, positional embeddings, masks gene identities, and takes
the mean of token states as the cell embedding. Check both before quoting any of this.
"""

import math
import time
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------- tokenization


class CellTokens(NamedTuple):
    """Every cell's tokens, stored back to back (the same layout as a CSR sparse matrix).

    Cell i owns positions indptr[i] to indptr[i + 1] of `genes` and `bins`. Within a cell the
    tokens are sorted from highest to lowest expression (ties: lower gene index first), so the
    first `max_len` tokens of a cell are its deterministic "top genes" selection.
    """

    indptr: np.ndarray  # int64, length n_cells + 1
    genes: np.ndarray  # int32 gene ids in 0..n_genes-1 (column numbers of the count matrix)
    bins: np.ndarray  # int16 expression bins in 0..n_bins-1
    n_genes: int
    n_bins: int

    @property
    def n_cells(self) -> int:
        return len(self.indptr) - 1


class TokenBatch(NamedTuple):
    """A padded mini-batch: three (cells x longest cell) tensors. [CLS] is not in here; the
    model adds it itself, so nothing that operates on a batch can mask or pad it by mistake."""

    genes: torch.Tensor  # int64 gene ids; the padding id is n_genes
    bins: torch.Tensor  # int64 bins; the padding id is n_bins + 1 (n_bins is [MASK])
    pad: torch.Tensor  # bool, True where the position is padding


def bin_values(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bins in 0..n_bins-1 for one cell's nonzero values, by the cell's own quantiles.

        bin = floor( n_bins * (how many of the cell's values are strictly lower) / n_values )

    Only the order of the values matters, so multiplying the cell by a constant or applying
    log(1 + x) changes nothing. Equal values get equal bins. The lowest value is always in
    bin 0. Integer arithmetic is used so there is no rounding to argue about.

    Worked example with n_bins = 4 and values [1, 1, 2, 5, 9]: the numbers of strictly lower
    values are [0, 0, 2, 3, 4], so the bins are floor(4 * [0, 0, 2, 3, 4] / 5) = [0, 0, 1, 2, 3].
    """
    values = np.asarray(values)
    n_lower = np.searchsorted(np.sort(values), values, side="left")
    return (n_bins * n_lower) // max(len(values), 1)


def normalized_levels(values: np.ndarray, total: float) -> np.ndarray:
    """log(1 + 10,000 * value / total): a cell's counts on a depth-independent scale."""
    return np.log1p(1e4 * np.asarray(values, dtype=np.float64) / max(float(total), 1.0))


def fit_value_edges(counts: np.ndarray | sp.spmatrix, n_bins: int) -> np.ndarray:
    """Cut points (n_bins - 1 of them) for the "value" level encoding, fitted on these cells.

    Every nonzero count is put on the depth-normalized scale of `normalized_levels` (each cell
    divided by its own total over the selected genes), and the cut points are the quantiles of
    all those values, so each bin holds about the same number of training tokens. Pass the
    TRAINING cells only: the cut points are a statistic of the data.
    """
    csr = sp.csr_matrix(counts)
    totals = np.asarray(csr.sum(axis=1)).ravel()
    per_entry_total = np.repeat(totals, np.diff(csr.indptr))
    nonzero = csr.data > 0
    levels = np.log1p(1e4 * csr.data[nonzero] / np.maximum(per_entry_total[nonzero], 1.0))
    if len(levels) == 0:
        raise ValueError("no nonzero counts to fit cut points on")
    return np.quantile(levels, np.arange(1, n_bins) / n_bins)


def tokenize_cell(
    gene_ids: np.ndarray,
    values: np.ndarray,
    n_bins: int,
    edges: np.ndarray | None = None,
    total: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(gene ids, bins) of one cell's expressed genes, highest expression first.

    Entries with value 0 are dropped: a gene that was not seen is not a token. Ties in
    expression are ordered by gene id so that the result is fully deterministic.

    Two level encodings:

    - `edges` is None ("rank", the default): bins from the cell's own ranking (`bin_values`).
      Independent of sequencing depth, but in sparse data most nonzero counts are 1 and tie,
      so most tokens land in bin 0 and the levels carry little information.
    - `edges` given ("value"): each value is put on the depth-normalized scale of
      `normalized_levels` (using the cell's `total`) and binned with cut points shared by all
      cells (`fit_value_edges`). A count of 1 in a shallow cell and in a deep cell now get
      different bins, because they mean different things.
    """
    gene_ids, values = np.asarray(gene_ids), np.asarray(values)
    expressed = values > 0
    gene_ids, values = gene_ids[expressed], values[expressed]
    order = np.lexsort((gene_ids, -values))  # last key is the primary one
    if edges is None:
        bins = bin_values(values, n_bins)
    else:
        if total is None:
            raise ValueError("the value encoding needs the cell's total count")
        bins = np.searchsorted(edges, normalized_levels(values, total), side="right")
    return gene_ids[order], np.asarray(bins)[order]


def tokenize(
    counts: np.ndarray | sp.spmatrix, n_bins: int = 10, edges: np.ndarray | None = None
) -> CellTokens:
    """Tokens of every cell of a (cells x selected genes) matrix of non-negative values.

    The matrix is never densified: it is converted to CSR (a no-op if it already is) and read
    one row at a time, and the result needs about 6 bytes per nonzero entry. One pass in plain
    Python, done once before training (tens of thousands of cells per second).
    """
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-dimensional, got shape {counts.shape}")
    if not 2 <= n_bins <= 1000:
        raise ValueError("n_bins must be in 2..1000")
    csr = sp.csr_matrix(counts)
    if csr.nnz and csr.data.min() < 0:
        raise ValueError("counts must be non-negative")

    if edges is not None and len(edges) != n_bins - 1:
        raise ValueError(f"expected {n_bins - 1} cut points, got {len(edges)}")
    totals = np.asarray(csr.sum(axis=1)).ravel()
    genes, bins, lengths = [], [], [0]
    for row in range(csr.shape[0]):
        span = slice(csr.indptr[row], csr.indptr[row + 1])
        cell_genes, cell_bins = tokenize_cell(
            csr.indices[span], csr.data[span], n_bins, edges, totals[row]
        )
        genes.append(cell_genes)
        bins.append(cell_bins)
        lengths.append(len(cell_genes))
    return CellTokens(
        indptr=np.cumsum(lengths, dtype=np.int64),
        genes=np.concatenate(genes).astype(np.int32) if genes else np.zeros(0, np.int32),
        bins=np.concatenate(bins).astype(np.int16) if bins else np.zeros(0, np.int16),
        n_genes=csr.shape[1],
        n_bins=n_bins,
    )


def make_batch(
    tokens: CellTokens,
    rows: np.ndarray,
    max_len: int = 256,
    rng: np.random.Generator | None = None,
) -> TokenBatch:
    """Padded token tensors for the cells `rows`, as wide as the longest of them (<= max_len).

    A cell with more than `max_len` tokens is cut down. With `rng` (training) a random subset
    without replacement is drawn, different each time. Without `rng` (embedding, validation)
    the first `max_len` tokens are taken, which are the highest-expressed ones.

    Padding to the longest cell in the mini-batch rather than to `max_len` matters for speed:
    most cells are much shorter than `max_len`.
    """
    rows = np.asarray(rows)
    starts, stops = tokens.indptr[rows], tokens.indptr[rows + 1]
    width = max(int(np.minimum(stops - starts, max_len).max(initial=0)), 1)
    genes = np.full((len(rows), width), tokens.n_genes, dtype=np.int64)
    bins = np.full((len(rows), width), tokens.n_bins + 1, dtype=np.int64)
    for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        keep = np.arange(start, stop)
        if len(keep) > max_len:
            if rng is None:
                keep = keep[:max_len]
            else:
                keep = np.sort(rng.choice(keep, size=max_len, replace=False))
        genes[i, : len(keep)] = tokens.genes[keep]
        bins[i, : len(keep)] = tokens.bins[keep]
    return TokenBatch(
        genes=torch.from_numpy(genes),
        bins=torch.from_numpy(bins),
        pad=torch.from_numpy(genes == tokens.n_genes),
    )


def length_batches(
    lengths: np.ndarray, batch_size: int, rng: np.random.Generator, bucket_batches: int = 50
) -> list[np.ndarray]:
    """One epoch of training mini-batches (arrays of cell numbers) of similar-length cells.

    A mini-batch is padded to its longest cell and attention costs grow with the square of
    that width, so one long cell makes a whole batch of short cells expensive. Remedy: shuffle
    all cells, cut the shuffled list into buckets of `bucket_batches` mini-batches, sort the
    cells of each bucket by length, cut each bucket into mini-batches, and finally shuffle the
    order of the mini-batches. Every cell is still used exactly once per epoch and the
    grouping changes every epoch.

    The price: the cells of a mini-batch are no longer a fully random sample. Cells with
    similar numbers of expressed genes tend to be of similar type and depth, so the gradient
    of one step is a little less representative of the whole data set. When the data set is
    smaller than one bucket (`bucket_batches` * `batch_size` cells), the whole epoch is sorted
    by length. Set `bucket_batches` to 1 for plain random mini-batches.
    """
    order = rng.permutation(len(lengths))
    bucket = batch_size * bucket_batches
    batches: list[np.ndarray] = []
    for start in range(0, len(order), bucket):
        cells = order[start : start + bucket]
        cells = cells[np.argsort(lengths[cells], kind="stable")]
        batches += [cells[i : i + batch_size] for i in range(0, len(cells), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def choose_mask(
    pad: torch.Tensor, mask_fraction: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Bool tensor shaped like `pad`: True at the tokens whose expression level will be hidden.

    Per cell, round(mask_fraction * number of real tokens) positions are chosen uniformly at
    random among the real tokens, and at least one (a cell with nothing masked would give no
    training signal). Padding is never chosen. [CLS] cannot be chosen because it is not part
    of the batch tensors at all. A cell without any token gets no mask.

    How: give every position a random score, push padding to the end, mask the lowest scores.
    """
    n_tokens = (~pad).sum(dim=1)
    n_mask = torch.floor(mask_fraction * n_tokens + 0.5).long().clamp(min=1)
    n_mask = torch.minimum(n_mask, n_tokens)
    scores = torch.rand(pad.shape, generator=generator)
    scores[pad] = 2.0  # larger than any random score, so padding sorts last
    rank = scores.argsort(dim=1).argsort(dim=1)
    return rank < n_mask[:, None]


# ------------------------------------------------------------------------------------ model


class SelfAttention(nn.Module):
    """Multi-head self-attention: every token gathers information from all tokens of its cell.

    For one head, each token i makes a query q_i and every token j a key k_j and a value v_j
    (three linear maps of the token states). Token i's output is a weighted average of the
    values, with weights

        w_ij = softmax over j of ( q_i . k_j / sqrt(head_dim) )

    so a token attends most to the tokens whose key matches its query. Dividing by
    sqrt(head_dim) keeps the dot products at a scale where the softmax is not saturated.
    `n_heads` heads do this in parallel on separate slices of the vector, so that different
    heads can look for different things, and a final linear map mixes the heads back together.

    `keep` is False at padding tokens: their weight is forced to zero for every query, so
    padding can never influence a real token.

    No dropout is applied to the attention weights themselves (many implementations do). It
    needs one random number per pair of tokens, which on a CPU made a training step several
    times slower in this project's measurements; dropout on the output of each branch (see
    `Block`) is kept.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """`x`: (cells, tokens, d_model). `keep`: bool (cells, 1, 1, tokens), True = real."""
        n_cells, n_tokens, d_model = x.shape
        # (cells, tokens, 3 * d) -> three tensors of shape (cells, heads, tokens, head_dim)
        q, k, v = (
            self.qkv(x)
            .view(n_cells, n_tokens, 3, self.n_heads, d_model // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # Computes softmax(q k^T / sqrt(head_dim)) v with masked keys given zero weight.
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return self.out(mixed.transpose(1, 2).reshape(n_cells, n_tokens, d_model))


class Block(nn.Module):
    """One transformer layer: attention, then a small MLP, each added to its own input.

        x = x + attention(norm(x))      tokens exchange information
        x = x + mlp(norm(x))            each token processes what it gathered, on its own

    The additions (residual connections) mean each layer only has to learn a correction, and
    they give gradients a direct path to the early layers. "Pre-norm" means the layer norm sits
    inside the branch, before attention and before the MLP, so that the residual path itself is
    never rescaled. That trains more stably than the original post-norm layout and is the
    reason a modest learning-rate warm-up is enough. The MLP widens to `mlp_ratio` * d_model
    and back, with GELU in between.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm_attn(x), keep))
        return x + self.drop(self.mlp(self.norm_mlp(x)))


class CellTransformer(nn.Module):
    """Token embeddings, a stack of `Block`s, and two small prediction heads.

    Input of a token = gene embedding + expression-level embedding. Adding (rather than
    concatenating) is the usual choice: both live in the same d_model-dimensional space and the
    first layer can pull them apart again. The level table has n_bins + 2 rows: one per bin,
    one for [MASK] ("level hidden, please predict") and one for padding (fixed at zero).

    Masked gene modeling, and why the gene identity stays visible. For a masked token the
    model sees which gene it is and must say how highly that gene is expressed in this cell.
    The question "how much CD3E does this cell have, given everything else it expresses?" is
    exactly a question about cell identity. If the identity were hidden as well, the token
    would carry no information and, with no positions, all masked tokens of a cell would be
    indistinguishable. Unlike BERT, the masked input is always [MASK] (no "10% random, 10%
    unchanged" variants), to keep the objective easy to state.

    Alternative objectives, not implemented:

    - Predict a continuous value with mean squared error. Keeps the full resolution of the
      data and treats "off by one bin" as better than "off by five", which cross-entropy over
      bins does not. But squared error assumes symmetric noise around the prediction, a poor
      fit for small counts, and the model outputs a single number rather than a distribution.
    - Geneformer-style rank ordering: the sequence is the genes sorted by expression, and the
      gene IDENTITY at a masked position is predicted (a choice among all genes in the
      vocabulary). No binning is needed and the target is rich, but the output layer is as
      large as the vocabulary, positional embeddings are required, and the model is asked
      "which gene has this rank", which is harder to relate to expression level.

    Size with the defaults (d_model 128, 4 layers, 4 heads, MLP ratio 4) and 1,000 genes:
    about 0.97 million parameters, of which the gene table is 128 * 1,001. `count_parameters`
    gives the exact number. This is deliberately small: the training sets here have tens of
    thousands of cells, not tens of millions.

    Initialization: all weight matrices and embeddings from N(0, 0.02^2), biases zero, as in
    BERT. One useful consequence: an untrained model outputs almost uniform probabilities, so
    its loss is close to log(n_bins), which is tested.
    """

    def __init__(
        self,
        n_genes: int,
        n_bins: int = 10,
        max_len: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(n_genes, max_len, d_model, n_layers, n_heads, mlp_ratio) < 1 or n_bins < 2:
            raise ValueError("sizes must be >= 1 (n_bins >= 2)")
        self.n_genes, self.n_bins, self.max_len, self.d_model = n_genes, n_bins, max_len, d_model
        self.mask_id = n_bins  # row of `value_emb` used for a hidden level
        pad_gene, pad_value = n_genes, n_bins + 1

        self.gene_emb = nn.Embedding(n_genes + 1, d_model, padding_idx=pad_gene)
        self.value_emb = nn.Embedding(n_bins + 2, d_model, padding_idx=pad_value)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, n_bins)
        # The [CLS] head, see `cls_logits`. Its inner size is kept small on purpose.
        self.cls_rank = max(d_model // 4, 8)
        self.cell_to_bins = nn.Linear(d_model, n_bins * self.cls_rank)
        self.gene_query = nn.Linear(d_model, self.cls_rank)
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def forward(
        self,
        genes: torch.Tensor,
        bins: torch.Tensor,
        pad: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Final token states, shape (cells, 1 + tokens, d_model). Position 0 is [CLS].

        `genes`, `bins`, `pad` are the fields of a `TokenBatch`. Where `mask` is True the
        level is replaced by [MASK] before it is embedded; the gene id is left alone.
        """
        values = bins if mask is None else bins.masked_fill(mask, self.mask_id)
        tokens = self.gene_emb(genes) + self.value_emb(values)
        cls = self.cls_token.expand(len(tokens), -1, -1)
        x = self.drop(torch.cat([cls, tokens], dim=1))
        # [CLS] is always a real token, so every row has at least one position to attend to.
        real = torch.cat([torch.ones_like(pad[:, :1]), ~pad], dim=1)
        keep = real[:, None, None, :]
        for block in self.blocks:
            x = block(x, keep)
        return self.norm(x)

    def token_logits(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Bin scores (n_masked x n_bins) for the masked tokens, from their own final states.

        This is the masked gene modeling head: one linear layer on the token's state.
        """
        return self.token_head(hidden[:, 1:][mask])

    def cls_logits(
        self, hidden: torch.Tensor, genes: torch.Tensor, where: torch.Tensor
    ) -> torch.Tensor:
        """Bin scores for the tokens selected by `where`, predicted from the [CLS] state only.

        Why this second head exists. The masked-token loss reads the final states of the
        masked tokens. Nothing reads the final state of [CLS], so with that loss alone the
        last layer's [CLS] output would receive no gradient at all: it would be an untrained
        by-product, and using it as the cell embedding would be unjustified. (In this project's
        synthetic test data the masked tokens solve their task by attending to each other
        directly, and [CLS] ends up carrying almost nothing.) BERT solves this with a
        sentence-level task; scGPT (background, not verified) with objectives that predict
        expression from the cell embedding.

        The version here turns [CLS] into a bottleneck: from the [CLS] state alone, the level
        of EVERY gene token of the cell has to be recovered, the visible ones (compression,
        as in an autoencoder) and the masked ones (prediction). One vector that can answer
        "how high is gene g in this cell?" for any g is a summary of the cell, which is what
        an embedding should be. Asking only about the masked tokens was tried as well; on the
        synthetic test data it trained [CLS] more slowly and separated the cell types worse.

        How the question is asked. The [CLS] state is mapped to one short vector per bin (think
        "what the genes in bin b of this cell look like"), the gene's embedding is mapped to a
        short query vector, and the score of bin b is their dot product:

            score[b] = (A_b cls) . (Q gene_embedding)

        This product form lets the answer depend on the cell and the gene jointly. A plain MLP
        on the sum cls + gene_embedding was tried first and learned far more slowly on the
        synthetic test data.

        Alternative: drop this head (`cls_weight` = 0) and use the mean of the token states as
        the embedding (`embed(..., pooling="mean")`, what Geneformer does). Supported too.
        """
        per_bin = self.cell_to_bins(hidden[:, 0]).view(-1, self.n_bins, self.cls_rank)
        query = self.gene_query(self.gene_emb(genes[where]))
        cell_of_token = where.nonzero()[:, 0]
        return torch.einsum("nbr,nr->nb", per_bin[cell_of_token], query)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------- training


def lr_at(step: int, total_steps: int, warmup_steps: int, lr: float) -> float:
    """Learning rate for optimizer step `step` (0-based): linear warm-up, then cosine decay.

    Steps 0..warmup_steps-1 climb linearly, reaching `lr` at the last warm-up step. After that
    the rate follows half a cosine wave from `lr` down toward 0 at `total_steps`.

    Why warm up: Adam's per-parameter step sizes come from running averages of the gradients
    that are unreliable for the first steps, and attention with random weights produces large,
    erratic gradients. Small first steps avoid wrecking the initialization. Why decay: large
    steps find a good region quickly, small steps at the end settle into it.
    """
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def bin_baselines(tokens: CellTokens) -> tuple[int, np.ndarray]:
    """Two guesses that use no model: (most common bin overall, most common bin per gene).

    Both are estimated from the tokens given, which should be the training cells. The first
    is the trivial "always say bin 0" guess. The second knows which gene is asked about but
    nothing about the cell; a model that only matches it has learned each gene's typical level
    and has not used the context. A gene with no token at all falls back to the overall bin.
    """
    table = np.zeros((tokens.n_genes, tokens.n_bins), dtype=np.int64)
    np.add.at(table, (tokens.genes, tokens.bins), 1)
    overall = int(table.sum(axis=0).argmax())
    per_gene = table.argmax(axis=1)
    per_gene[table.sum(axis=1) == 0] = overall
    return overall, per_gene


class _Tally:
    """Running sums over masked tokens; `means` turns them into the numbers for the history."""

    names = ("loss", "accuracy", "cls_loss", "cls_accuracy", "majority", "gene_majority")

    def __init__(self) -> None:
        self.sums = np.zeros(len(self.names))
        self.n_masked = 0

    def add(self, values: list[float], n_masked: int) -> None:
        self.sums += np.array(values)
        self.n_masked += n_masked

    def means(self) -> dict[str, float]:
        return dict(zip(self.names, self.sums / max(self.n_masked, 1), strict=True))


def _masked_step(
    model: CellTransformer,
    batch: TokenBatch,
    mask: torch.Tensor,
    baselines: tuple[int, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Forward pass with `mask` applied: (token loss, [CLS] loss, sums for a `_Tally`).

    token loss: cross-entropy of the masked-token head, averaged over the masked tokens.
    [CLS] loss: cross-entropy of the [CLS] head, averaged over ALL real tokens of the batch,
    masked or not (see `CellTransformer.cls_logits`).

    The sums are all totals over the MASKED tokens only, so that the two heads and the
    model-free guesses are compared on the same held-back values: loss and number correct for
    the token head, the same for the [CLS] head, then the number of times each model-free
    guess would have been right.
    """
    genes, bins, pad, mask = (t.to(device) for t in (*batch, mask))
    hidden = model(genes, bins, pad, mask)
    target = bins[mask]
    token_logits = model.token_logits(hidden, mask)
    token_loss = F.cross_entropy(token_logits, target)

    real = ~pad
    cls_logits = model.cls_logits(hidden, genes, real)
    cls_loss = F.cross_entropy(cls_logits, bins[real])
    cls_logits_masked = cls_logits[mask[real]]  # rows of the masked tokens, in the same order

    overall, per_gene = baselines
    target_cpu = batch.bins[mask.cpu()].numpy()
    gene_guess = per_gene[batch.genes[mask.cpu()].numpy()]
    sums = [
        token_loss.item() * len(target),
        (token_logits.argmax(dim=1) == target).sum().item(),
        F.cross_entropy(cls_logits_masked, target, reduction="sum").item(),
        (cls_logits_masked.argmax(dim=1) == target).sum().item(),
        float((target_cpu == overall).sum()),
        float((target_cpu == gene_guess).sum()),
    ]
    return token_loss, cls_loss, sums


def masked_scores(
    model: CellTransformer,
    tokens: CellTokens,
    baselines: tuple[int, np.ndarray] | None = None,
    mask_fraction: float = 0.15,
    batch_size: int = 256,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Masked-token loss and accuracy on `tokens` in evaluation mode, with model-free baselines.

    Keys: `loss` (cross-entropy in nats; log(n_bins) is what uniform guessing gets) and
    `accuracy` for the masked-token head, `cls_loss` and `cls_accuracy` for the [CLS] head,
    `majority` (accuracy of always predicting the most common bin) and `gene_majority`
    (accuracy of predicting each gene's most common bin), both on exactly the same masked
    tokens. `baselines` comes from `bin_baselines` on the TRAINING cells; if omitted it is
    estimated from `tokens` themselves, which flatters the baselines slightly.

    The masked positions come from a private generator seeded with `seed`, so the same
    positions are masked on every call and the numbers are comparable across epochs. Long
    cells are cut to their top `max_len` genes, deterministically. Dropout is off.
    """
    device = torch.device(device)
    if baselines is None:
        baselines = bin_baselines(tokens)
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    tally = _Tally()
    by_length = np.argsort(np.diff(tokens.indptr), kind="stable")  # short cells first: less padding
    with torch.no_grad():
        for start in range(0, tokens.n_cells, batch_size):
            rows = by_length[start : start + batch_size]
            batch = make_batch(tokens, rows, model.max_len)
            mask = choose_mask(batch.pad, mask_fraction, generator)
            if mask.any():
                tally.add(_masked_step(model, batch, mask, baselines, device)[2], int(mask.sum()))
    model.train(was_training)
    return tally.means()


def _adamw(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with weight decay on the weight matrices of linear layers only.

    Weight decay shrinks a parameter toward zero a little at every step, which discourages the
    network from relying on a few very large weights. It is not applied to biases, layer-norm
    scales, embeddings or the [CLS] vector: pulling those toward zero has no regularizing
    meaning (a rarely seen gene's embedding would just fade away). "AdamW" is Adam with the
    decay applied directly to the weights instead of being mixed into the gradient.
    """
    decay = [p for name, p in model.named_parameters() if p.ndim == 2 and "emb" not in name]
    decay_ids = {id(p) for p in decay}
    no_decay = [p for p in model.parameters() if id(p) not in decay_ids]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def train_transformer(
    counts_train: np.ndarray | sp.spmatrix,
    n_bins: int = 10,
    max_len: int = 256,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    mask_fraction: float = 0.15,
    cls_weight: float = 1.0,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    warmup: float = 0.05,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    counts_val: np.ndarray | sp.spmatrix | None = None,
    level_encoding: str = "rank",
) -> tuple[CellTransformer, dict[str, list[float]]]:
    """Train a `CellTransformer` with masked gene modeling; return it (eval mode) and its history.

    `counts_train` holds the counts of the selected genes (cells x genes, dense or scipy
    sparse) for cells of TRAINING donors only, exactly as for `cellmap.vae.train_vae`. No
    donor label is taken, because the model has no use for one (see the module docstring).
    Validation cells are only scored, never trained on. Sparse input stays sparse: cells are
    tokenized once and mini-batches are padded to the longest cell in the batch.

    Loss minimized: token_loss + cls_weight * cls_loss. The first is the cross-entropy of the
    masked tokens' bins predicted from their own states; the second is the cross-entropy of
    every token's bin predicted from the [CLS] state alone (see `CellTransformer.cls_logits`).
    With `cls_weight` = 0 it is plain masked gene modeling, and then the embedding should be
    taken with `pooling="mean"`.

    Optimization: AdamW; the learning rate warms up linearly over the first `warmup` fraction
    of all steps and then decays along a cosine (`lr_at`); the gradient norm is clipped to
    `grad_clip`, so that one unusual mini-batch cannot throw the weights far away. A new random
    mask (and, for long cells, a new random subset of genes) is drawn every time a cell is used.
    Mini-batches group cells of similar length to save padding (see `length_batches`).

    History: a dict of equal-length lists, one entry per epoch.

    - `train_loss`, `train_accuracy`: masked-token cross-entropy (nats) and accuracy, averaged
      over the epoch's masked tokens as they were trained on (dropout on, weights moving).
    - `train_cls_loss`, `train_cls_accuracy`: the [CLS] head on the same masked tokens.
    - `train_majority`: accuracy of always predicting the most common training bin, on the same
      masked tokens. `train_gene_majority`: accuracy of predicting each gene's most common
      training bin. The model has learned something only if it beats the first, and it uses
      the rest of the cell only if it beats the second.
    - `val_*`: the same six numbers on validation cells, measured after the epoch by
      `masked_scores` (dropout off, the same masked positions every epoch).
    - `lr`: the learning rate at the epoch's last step.
    - `seconds_train`: wall time of the epoch's training steps. `seconds`: including validation.

    Determinism: `torch.manual_seed(seed)` is called first (this resets PyTorch's global random
    stream) and covers initialization and dropout; shuffling, gene subsets and masks come from
    private generators seeded with `seed`. On CPU with a fixed number of threads two runs with
    the same seed give identical results. That has not been checked on a GPU, where some
    operations are not bit-reproducible.
    """
    if not 0.0 < mask_fraction < 1.0 or not 0.0 <= warmup < 1.0:
        raise ValueError("mask_fraction must be in (0, 1) and warmup in [0, 1)")
    if level_encoding not in ("rank", "value"):
        raise ValueError('level_encoding must be "rank" or "value"')
    device = torch.device(device)
    edges = fit_value_edges(counts_train, n_bins) if level_encoding == "value" else None
    train = tokenize(counts_train, n_bins, edges)
    val = None
    if counts_val is not None:
        if counts_val.shape[1] != counts_train.shape[1]:
            raise ValueError("validation and training counts have different numbers of genes")
        val = tokenize(counts_val, n_bins, edges)
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
    # Plain attribute, not a buffer, so models saved before this option existed still load.
    # `embed` reads it to tokenize new cells with the training cut points.
    model.value_edges = edges
    optimizer = _adamw(model, lr, weight_decay)
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
        tally = _Tally()
        for rows in length_batches(lengths, batch_size, rng):
            batch = make_batch(train, rows, max_len, rng)
            mask = choose_mask(batch.pad, mask_fraction, mask_generator)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, warmup_steps, lr)
            step += 1
            if not mask.any():  # only possible if no cell in the batch expresses any gene
                continue
            token_loss, cls_loss, sums = _masked_step(model, batch, mask, baselines, device)
            optimizer.zero_grad()
            (token_loss + cls_weight * cls_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tally.add(sums, int(mask.sum()))
        seconds_train = time.perf_counter() - started

        row = {f"train_{name}": value for name, value in tally.means().items()}
        if val is not None:
            scores = masked_scores(model, val, baselines, mask_fraction, seed=seed, device=device)
            row.update({f"val_{name}": value for name, value in scores.items()})
        row["lr"] = optimizer.param_groups[0]["lr"]
        row["seconds_train"] = seconds_train
        row["seconds"] = time.perf_counter() - started
        for key, value in row.items():
            history.setdefault(key, []).append(float(value))

    model.eval()
    return model, history


def embed(
    model: CellTransformer,
    counts: np.ndarray | sp.spmatrix,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    pooling: str = "cls",
) -> np.ndarray:
    """One float32 vector of length d_model per cell. Nothing is masked, nothing is random.

    `counts` has the same gene columns, in the same order, as the training matrix. No donor
    label is needed or used. The model runs in evaluation mode (dropout off), and a cell with
    more than `max_len` expressed genes is represented by its `max_len` highest-expressed ones,
    so the embedding is a deterministic function of the cell. Padding is masked in attention,
    so a cell's embedding does not depend on the other cells in its mini-batch (up to
    floating-point rounding of about 1e-6).

    `pooling`:

    - "cls" (default): the final state of the [CLS] token. It is the default because it is
      the only vector that a loss trains to summarize the cell: the level of every gene of the
      cell has to be recoverable from it alone (see `CellTransformer.cls_logits`). Only
      meaningful if the model was trained with `cls_weight` > 0.
    - "mean": the average of the final states of the cell's gene tokens ([CLS] and padding
      excluded). Needs no extra objective, but no loss reads the final state of an unmasked
      token, and at embedding time nothing is masked, so this averages vectors that were
      never trained to mean anything in particular. A cell with no expressed gene has no
      tokens and gets a vector of zeros.

    Evidence so far, all from development runs and not a result: on synthetic data with three
    planted cell types "cls" separated the types almost perfectly and "mean" did not; on the
    3,000-cell development subset the two were about equal. Compare them on validation donors
    before reporting either.
    """
    if pooling not in ("cls", "mean"):
        raise ValueError('pooling must be "cls" or "mean"')
    if counts.shape[1] != model.n_genes:
        raise ValueError(f"counts has {counts.shape[1]} genes, the model expects {model.n_genes}")
    device = torch.device(device)
    tokens = tokenize(counts, model.n_bins, getattr(model, "value_edges", None))
    was_training = model.training
    model.eval().to(device)
    # Cells are processed from shortest to longest so that little padding is needed, and put
    # back in their original order at the end.
    by_length = np.argsort(np.diff(tokens.indptr), kind="stable")
    out = []
    with torch.no_grad():
        for start in range(0, tokens.n_cells, batch_size):
            rows = by_length[start : start + batch_size]
            genes, bins, pad = (t.to(device) for t in make_batch(tokens, rows, model.max_len))
            hidden = model(genes, bins, pad)
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                real = (~pad).unsqueeze(-1).float()
                pooled = (hidden[:, 1:] * real).sum(dim=1) / real.sum(dim=1).clamp(min=1.0)
            out.append(pooled.cpu().numpy())
    model.train(was_training)
    result = np.empty((tokens.n_cells, model.d_model), dtype=np.float32)
    if out:
        result[by_length] = np.concatenate(out)
    return result
