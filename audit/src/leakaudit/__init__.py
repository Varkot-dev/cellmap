"""leakaudit: leakage and shortcut checks for machine learning on biological data.

Three checks, each a small function:

1. Group-aware splitting with guarantees: `group_split`, `group_kfold`,
   `assert_no_group_overlap`, `split_report`.
2. Leakage inflation: `measure_inflation` (random row split versus group split) and
   `measure_preprocessing_leak` (preprocessing fitted on all rows versus training rows).
3. Shortcut probes: `shortcut_probe` (damage one named input block at a time).
"""

from leakaudit.inflation import (
    InflationResult,
    measure_inflation,
    measure_preprocessing_leak,
    random_split,
)
from leakaudit.probes import ProbeResult, shortcut_probe
from leakaudit.splits import (
    GroupLeakError,
    SplitReport,
    assert_no_group_overlap,
    group_kfold,
    group_split,
    split_report,
)

__version__ = "0.1.0"

__all__ = [
    "GroupLeakError",
    "InflationResult",
    "ProbeResult",
    "SplitReport",
    "assert_no_group_overlap",
    "group_kfold",
    "group_split",
    "measure_inflation",
    "measure_preprocessing_leak",
    "random_split",
    "shortcut_probe",
    "split_report",
]
