import pandas as pd
import pytest

from cellmap.labels import to_coarse


def test_to_coarse_maps_known_types():
    fine = pd.Series(["naive B cell", "memory B cell", "classical monocyte"])
    assert to_coarse(fine).tolist() == ["B cell", "B cell", "monocyte"]


def test_to_coarse_raises_on_unknown_type():
    with pytest.raises(KeyError, match="made-up cell"):
        to_coarse(pd.Series(["naive B cell", "made-up cell"]))


def test_to_lineage_maps_and_drops():
    from cellmap.labels import to_lineage

    out = to_lineage(pd.Series(["naive B cell", "T cell", "classical monocyte"]))
    assert out[0] == "B cell"
    assert pd.isna(out[1])  # "T cell" could be CD4 or CD8, so it is dropped
    assert out[2] == "monocyte"


def test_to_lineage_raises_on_unlisted_name():
    from cellmap.labels import to_lineage

    with pytest.raises(KeyError, match="new cell"):
        to_lineage(pd.Series(["naive B cell", "new cell"]))


def test_lineage_table_has_twelve_lineages():
    from cellmap.labels import LINEAGE

    assert len({v for v in LINEAGE.values() if v is not None}) == 12
