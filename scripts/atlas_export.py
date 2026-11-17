"""Build the interactive 3D atlas as one self-contained HTML file.

Usage:
    python scripts/atlas_export.py --name pbmc

For every method with a saved split-seed-0 embedding under data/embeddings/<name>/, computes a
3D UMAP (cached under data/), then writes site/index.html: site/template.html with all data
inlined, so the page works from GitHub Pages or straight from disk with no server.

What is inlined, per cell: 3D coordinates per method (16-bit), cell type (fine and coarse),
donor, chemistry, split, and the expression of a small panel of marker genes (8-bit).
"""

import argparse
import base64
import json
from pathlib import Path

import anndata as ad
import numpy as np
import umap

from cellmap.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]

# Order here is the order of the method switch in the page.
METHODS = {
    "pca": "HVG + PCA",
    "pca_harmony": "PCA + Harmony",
    "linear_ae": "Linear autoencoder",
    "vae": "VAE (hand-written)",
    "scvi": "scvi-tools",
    "transformer": "Transformer",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pbmc")
    args = parser.parse_args()

    emb_dir = ROOT / "data" / "embeddings" / args.name
    adata = ad.read_h5ad(ROOT / "data" / "processed" / f"{args.name}.h5ad")
    obs = adata.obs
    split = np.load(emb_dir / "split_seed0.npy", allow_pickle=True)

    methods = []
    for key, label in METHODS.items():
        path = emb_dir / f"{key}.npy"
        if not path.exists():
            continue
        coords = umap_3d(np.load(path), emb_dir / f"umap3d_{key}.npy")
        methods.append({"key": key, "label": label, "xyz": b64(quantize_coords(coords))})
        print(f"method {key}: ok")

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
            print(f"gene {gene}: not in data, skipped")
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

    data = {
        "n": int(adata.n_obs),
        "dataset": "Healthy-donor PBMCs, Schulte-Schrepping et al. 2020, CELLxGENE Census",
        "donors": int(obs["donor_id"].nunique()),
        "commit": git_commit(ROOT)[:7],
        "methods": methods,
        "fields": fields,
        "genes": genes,
    }
    payload = json.dumps(data, separators=(",", ":"))
    template = (ROOT / "site" / "template.html").read_text()
    page = template.replace("/*__ATLAS_DATA__*/null", payload)
    wrapped = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
    )
    (ROOT / "site" / "index.html").write_text(wrapped)
    # The same page without the document wrapper, for hosts that add their own.
    (emb_dir / "atlas_fragment.html").write_text(page)
    print(f"wrote site/index.html ({len(wrapped) / 1e6:.1f} MB), {len(methods)} methods, "
          f"{len(genes)} genes, {data['n']:,} cells")  # fmt: skip


if __name__ == "__main__":
    main()
