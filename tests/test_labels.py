import pandas as pd
import pytest

from cellmap.labels import to_coarse


def test_to_coarse_maps_known_types():
    fine = pd.Series(["naive B cell", "memory B cell", "classical monocyte"])
    assert to_coarse(fine).tolist() == ["B cell", "B cell", "monocyte"]


def test_to_coarse_raises_on_unknown_type():
    with pytest.raises(KeyError, match="made-up cell"):
        to_coarse(pd.Series(["naive B cell", "made-up cell"]))
