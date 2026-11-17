"""Coarse cell-type labels.

The fine labels are the Cell Ontology names stored in CELLxGENE. Results are reported at both
levels: fine labels test whether close subtypes are separated, coarse labels test lineages.
"""

import pandas as pd

COARSE_PBMC = {
    "classical monocyte": "monocyte",
    "non-classical monocyte": "monocyte",
    "central memory CD4-positive, alpha-beta T cell": "CD4 T cell",
    "naive thymus-derived CD4-positive, alpha-beta T cell": "CD4 T cell",
    "regulatory T cell": "CD4 T cell",
    "effector memory CD8-positive, alpha-beta T cell": "CD8 T cell",
    "naive thymus-derived CD8-positive, alpha-beta T cell": "CD8 T cell",
    "mucosal invariant T cell": "other T cell",
    "gamma-delta T cell": "other T cell",
    "double negative thymocyte": "other T cell",
    "natural killer cell": "NK cell",
    "CD16-negative, CD56-bright natural killer cell, human": "NK cell",
    "naive B cell": "B cell",
    "memory B cell": "B cell",
    "transitional stage B cell": "B cell",
    "plasmablast": "plasmablast",
    "conventional dendritic cell": "dendritic cell",
    "plasmacytoid dendritic cell, human": "dendritic cell",
    "platelet": "platelet",
    # A generic label from the original study: these cells were not resolved further.
    "lymphocyte": "unresolved lymphocyte",
}


def to_coarse(fine: pd.Series, mapping: dict[str, str] = COARSE_PBMC) -> pd.Series:
    """Map fine cell-type names to coarse ones; raise if any fine name has no mapping."""
    fine = fine.astype(str)
    unmapped = sorted(set(fine) - set(mapping))
    if unmapped:
        raise KeyError(f"cell types without a coarse label: {unmapped}")
    return fine.map(mapping).astype("category")
