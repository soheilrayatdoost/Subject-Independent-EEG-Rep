"""Paired comparison of two methods across test subjects (paper Sec. 6: t-test, p < .05).

The unit of analysis is the test subject: each method contributes one score per subject
(:func:`sieeg.evaluation.metrics.per_subject_metrics`), and the scores are paired by subject.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Keys of :func:`paired_comparison` results, in order.
COMPARISON_FIELDS = ("n", "mean_a", "mean_b", "mean_diff", "t", "p_ttest", "p_wilcoxon")

#: Paired score differences are rounded to this many decimals of a percentage point (1e-9) before testing.
_DIFF_DECIMALS = 9

#: Starts of scipy's warnings about its normal approximation, a caveat :func:`paired_comparison`
#: documents: fewer than 10 subjects (SciPy < 1.15) and zero differences (SciPy < 1.13).
_APPROXIMATION_WARNINGS = (
    "Sample size too small for normal approximation",
    "Exact p-value calculation does not work if there are zeros",
)


def paired_comparison(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    """Paired t-test and Wilcoxon signed-rank test of ``a`` vs ``b``.

    ``a`` and ``b`` are per-subject scores in percent, indexed by subject; only the subjects
    present in both are compared. ``t > 0`` and ``mean_diff = mean(a - b) > 0`` mean ``a``
    scores higher. Both tests use the differences ``a - b`` rounded to 1e-9 percentage points:
    float noise (around 1e-14 on scores up to 100) would otherwise break ties between equal
    differences and turn zero differences into ranked ones, which changes Wilcoxon p-values.
    Differences that round to the same value count as equal. The t-test is
    ``scipy.stats.ttest_1samp`` of the differences, i.e. ``ttest_rel(a, b)``.
    Where a test is undefined its values are NaN: ``t`` and ``p_ttest`` when every subject has
    the same difference (no spread, so scipy would report ``t = inf``, ``p = 0``), and
    ``p_wilcoxon`` when every difference is zero. The Wilcoxon test drops zero differences and
    keeps scipy's default method, which depends on the SciPy version: exact for up to 50
    subjects without zero differences (SciPy >= 1.15: nor tied ones); with them, SciPy >= 1.15
    runs an exact permutation test for up to 13 subjects. Otherwise it is the normal
    approximation (rough for fewer than 10 subjects; ``n`` gives the count).

    Raises:
        ValueError: if a subject appears twice in ``a`` or ``b``, or fewer than 2 subjects are shared.
    """
    for name, scores in (("a", a), ("b", b)):
        if not scores.index.is_unique:
            duplicated = scores.index[scores.index.duplicated()].unique().tolist()
            raise ValueError(f"Scores {name} list subjects more than once: {duplicated}")
    common = a.index.intersection(b.index)
    if len(common) < 2:
        raise ValueError(f"A paired test needs at least 2 common subjects, got {len(common)}")
    x = a.loc[common].to_numpy(dtype=float)
    y = b.loc[common].to_numpy(dtype=float)
    diff = np.round(x - y, _DIFF_DECIMALS)  # without float noise: equal differences tie, zero ones are 0
    if np.ptp(diff) == 0:  # every subject has the same difference: no spread
        t = p_ttest = math.nan
    else:
        t, p_ttest = stats.ttest_1samp(diff, 0.0)
    p_wilcoxon = math.nan
    if np.any(diff != 0):
        with warnings.catch_warnings():  # the docstring states the caveat; n is reported
            for message in _APPROXIMATION_WARNINGS:
                warnings.filterwarnings("ignore", message, UserWarning)
            p_wilcoxon = float(stats.wilcoxon(diff).pvalue)
    return {
        "n": len(common),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "t": float(t),
        "p_ttest": float(p_ttest),
        "p_wilcoxon": p_wilcoxon,
    }
