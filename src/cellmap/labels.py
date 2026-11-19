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


# Six labs: datasets label cells at different levels of detail (one lab writes "naive B cell",
# another only "B cell"), so fine labels cannot be compared across datasets. Every Cell Ontology
# name used by the six labs is mapped to one of 12 lineages that all labs resolve. Names that
# cannot be placed in one lineage ("T cell" could be CD4 or CD8) map to None and those cells are
# dropped before any model sees them.
LINEAGE = {
    # CD4 T cells
    "naive thymus-derived CD4-positive, alpha-beta T cell": "CD4 T cell",
    "central memory CD4-positive, alpha-beta T cell": "CD4 T cell",
    "effector memory CD4-positive, alpha-beta T cell": "CD4 T cell",
    "CD4-positive, alpha-beta T cell": "CD4 T cell",
    "CD4-positive, alpha-beta cytotoxic T cell": "CD4 T cell",
    "regulatory T cell": "CD4 T cell",
    "naive regulatory T cell": "CD4 T cell",
    "memory regulatory T cell": "CD4 T cell",
    "T-helper 1 cell": "CD4 T cell",
    "T-helper 2 cell": "CD4 T cell",
    "T-helper 17 cell": "CD4 T cell",
    "T-helper 22 cell": "CD4 T cell",
    "T follicular helper cell": "CD4 T cell",
    # CD8 T cells
    "naive thymus-derived CD8-positive, alpha-beta T cell": "CD8 T cell",
    "central memory CD8-positive, alpha-beta T cell": "CD8 T cell",
    "effector memory CD8-positive, alpha-beta T cell": "CD8 T cell",
    "effector CD8-positive, alpha-beta T cell": "CD8 T cell",
    "CD8-positive, alpha-beta T cell": "CD8 T cell",
    "CD8-positive, alpha-beta cytotoxic T cell": "CD8 T cell",
    "CD8-positive, alpha-beta memory T cell": "CD8 T cell",
    # Other T cells (unconventional or neither CD4 nor CD8)
    "gamma-delta T cell": "other T cell",
    "mucosal invariant T cell": "other T cell",
    "mature NK T cell": "other T cell",
    "double negative thymocyte": "other T cell",
    "double-positive, alpha-beta thymocyte": "other T cell",
    "double negative T regulatory cell": "other T cell",
    # NK cells and other innate lymphoid cells
    "natural killer cell": "NK cell",
    "CD16-positive, CD56-dim natural killer cell, human": "NK cell",
    "CD16-negative, CD56-bright natural killer cell, human": "NK cell",
    "innate lymphoid cell": "innate lymphoid cell",
    "ILC1, human": "innate lymphoid cell",
    "group 2 innate lymphoid cell, human": "innate lymphoid cell",
    # B cells
    "B cell": "B cell",
    "naive B cell": "B cell",
    "memory B cell": "B cell",
    "transitional stage B cell": "B cell",
    "mature B cell": "B cell",
    "immature B cell": "B cell",
    "class switched memory B cell": "B cell",
    "unswitched memory B cell": "B cell",
    # Antibody-secreting cells
    "plasmablast": "plasma cell",
    "plasma cell": "plasma cell",
    "IgA plasma cell": "plasma cell",
    "IgG plasma cell": "plasma cell",
    "IgM plasma cell": "plasma cell",
    # Monocytes (one dataset does not separate classical from non-classical)
    "monocyte": "monocyte",
    "CD14-positive monocyte": "monocyte",
    "classical monocyte": "monocyte",
    "CD14-low, CD16-positive monocyte": "monocyte",
    "non-classical monocyte": "monocyte",
    # Dendritic cells
    "dendritic cell": "dendritic cell",
    "dendritic cell, human": "dendritic cell",
    "conventional dendritic cell": "dendritic cell",
    "myeloid dendritic cell": "dendritic cell",
    "myeloid dendritic cell, human": "dendritic cell",
    "CD1c-positive myeloid dendritic cell": "dendritic cell",
    "CD141-positive myeloid dendritic cell": "dendritic cell",
    "plasmacytoid dendritic cell": "dendritic cell",
    "plasmacytoid dendritic cell, human": "dendritic cell",
    # Platelets, red cells, progenitors
    "platelet": "platelet",
    "erythrocyte": "erythrocyte",
    "enucleated reticulocyte": "erythrocyte",
    "hematopoietic stem cell": "progenitor",
    "hematopoietic precursor cell": "progenitor",
    "CD34-positive, CD38-negative hematopoietic stem cell": "progenitor",
    "erythroid progenitor cell, mammalian": "progenitor",
    "megakaryocyte-erythroid progenitor cell": "progenitor",
    "myeloid lineage restricted progenitor cell": "progenitor",
    # Too vague to place in one lineage, or too rare to evaluate: dropped
    "T cell": None,
    "lymphocyte": None,
    "blood cell": None,
    "peripheral blood mononuclear cell": None,
    "unknown": None,
    "mast cell": None,
}


def to_lineage(cell_type: pd.Series) -> pd.Series:
    """Map Cell Ontology names to the 12 six-lab lineages; unplaceable names become NaN.

    Raises on a name that is not in the table at all, so a new dataset cannot slip in a label
    nobody has looked at.
    """
    names = cell_type.astype(str)
    unknown = sorted(set(names) - set(LINEAGE))
    if unknown:
        raise KeyError(f"cell types not in the lineage table: {unknown}")
    return names.map(LINEAGE)
