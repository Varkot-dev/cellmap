"""Data for the interactive pages: 3D coordinates per method, labels, genes and results.

Both pages (the atlas and the explainer) are single self-contained HTML files with this data
inlined, so they work from GitHub Pages or straight from disk with no server.
"""

import base64
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import umap

from cellmap.provenance import git_commit

# Order here is the order of the method switch in the page.
METHODS = {
    "pca": "HVG + PCA",
    "pca_harmony": "PCA + Harmony",
    "linear_ae": "Linear autoencoder",
    "vae": "VAE (hand-written)",
    "scvi": "scvi-tools",
    "transformer": "Transformer",
    "census_scvi": "Census scVI",
    "geneformer_ft": "Geneformer (fine-tuned)",
    "tf_sapiens": "TranscriptFormer",
}
MARKER_GENES = [
    "CD3E", "CD4", "CD8A", "CCR7", "IL7R", "FOXP3", "TRDC", "KLRB1", "NKG7", "GNLY",
    "MS4A1", "CD79A", "MZB1", "CD14", "LYZ", "FCGR3A", "FCER1A", "LILRA4", "PPBP", "XIST",
]  # fmt: skip


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def umap_3d(emb: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    coords = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(
        emb
    )
    np.save(cache, coords)
    return coords


def quantize_coords(coords: np.ndarray) -> np.ndarray:
    """Center, scale to the unit cube by the 99.9th percentile radius, store as int16."""
    centered = coords - np.median(coords, axis=0)
    scale = np.percentile(np.abs(centered), 99.9)
    return np.clip(centered / scale, -1, 1).__mul__(32767).astype("<i2")


def categorical(values) -> tuple[np.ndarray, list[str]]:
    """Codes ordered by frequency, most common first, so legends read top-down by size."""
    values = np.asarray(values).astype(str)
    names, counts = np.unique(values, return_counts=True)
    order = [str(n) for n in names[np.argsort(-counts, kind="stable")]]
    lookup = {n: i for i, n in enumerate(order)}
    return np.array([lookup[v] for v in values], dtype=np.uint8), order


def build_payload(root: Path, name: str) -> dict:
    """Everything the pages draw, for the methods with a saved split-seed-0 embedding.

    3D UMAP coordinates are cached under data/embeddings/<name>/ because they take minutes.
    """
    emb_dir = root / "data" / "embeddings" / name
    adata = ad.read_h5ad(root / "data" / "processed" / f"{name}.h5ad")
    obs = adata.obs
    split = np.load(emb_dir / "split_seed0.npy", allow_pickle=True)

    methods = []
    for key, label in METHODS.items():
        path = emb_dir / f"{key}.npy"
        if not path.exists():
            continue
        coords = umap_3d(np.load(path), emb_dir / f"umap3d_{key}.npy")
        methods.append({"key": key, "label": label, "xyz": b64(quantize_coords(coords))})

    chemistry = np.where(obs["assay"].astype(str).str.contains("v3"), "10x 3' v3", "10x 3' v2")
    fields = []
    for key, label, values in (
        ("coarse", "Cell type (10 lineages)", obs["cell_type_coarse"]),
        ("fine", "Cell type (19 fine types)", obs["cell_type"]),
        ("chemistry", "Sequencing chemistry", chemistry),
        ("split", "Train / val / test (split seed 0)", split),
        ("donor", "Donor", obs["donor_id"]),
    ):
        codes, names = categorical(values)
        fields.append({"key": key, "label": label, "names": names, "codes": b64(codes)})

    symbols = adata.var["feature_name"].astype(str).to_numpy()
    genes = []
    for gene in MARKER_GENES:
        idx = np.flatnonzero(symbols == gene)
        if len(idx) == 0:
            continue
        x = np.asarray(adata.X[:, idx[0]].todense()).ravel()
        top = float(np.percentile(x[x > 0], 99)) if (x > 0).any() else 1.0
        genes.append(
            {
                "name": gene,
                "max": round(top, 3),
                "pct": round(100 * float((x > 0).mean()), 1),
                "values": b64(np.clip(x / top, 0, 1).__mul__(255).astype(np.uint8)),
            }
        )

    return {
        "n": int(adata.n_obs),
        "dataset": "Healthy-donor PBMCs, Schulte-Schrepping et al. 2020, CELLxGENE Census",
        "donors": int(obs["donor_id"].nunique()),
        "commit": git_commit(root)[:7],
        "methods": methods,
        "fields": fields,
        "genes": genes,
    }


RESULT_FILES = ("m1", "m2", "m3", "m5", "m6")
RESULT_COLUMNS = (
    "knn_accuracy_fine", "knn_macro_f1_fine", "scib_bio_mean", "scib_batch_mean", "scib_total",
)  # fmt: skip


def build_results(root: Path, name: str) -> list[dict]:
    """Per-method means and per-split values, read from the saved results files only."""
    frames = [
        pd.read_csv(root / "results" / m / name / "metrics_by_split.csv")
        for m in RESULT_FILES
        if (root / "results" / m / name / "metrics_by_split.csv").exists()
    ]
    data = pd.concat(frames)
    out = []
    for method, group in data.groupby("method"):
        group = group.sort_values("split_seed")
        out.append(
            {
                "key": method,
                "label": METHODS.get(method, RESULT_LABELS.get(method, method)),
                "n_splits": int(len(group)),
                **{c: round(float(group[c].mean()), 3) for c in RESULT_COLUMNS},
                **{f"{c}_splits": [round(float(v), 3) for v in group[c]] for c in RESULT_COLUMNS},
            }
        )
    return out


RESULT_LABELS = {"pca_10": "HVG + PCA, 10 dims"}


def write_page(template: Path, data: dict, outputs: list[Path], marker: str) -> int:
    """Inline `data` into `template` at `marker`; write a full document to the first output and
    the bare fragment (for hosts that add their own document wrapper) to the others."""
    payload = json.dumps(data, separators=(",", ":"))
    page = template.read_text().replace(marker, payload)
    wrapped = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
    )
    outputs[0].write_text(wrapped)
    for extra in outputs[1:]:
        extra.write_text(page)
    return len(wrapped)
