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


def paired_comparison(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    """Paired t-test (``scipy.stats.ttest_rel``) and Wilcoxon signed-rank test of ``a`` vs ``b``.

    ``a`` and ``b`` are per-subject scores indexed by subject; only the subjects present in
    both are compared. ``t > 0`` and ``mean_diff = mean(a - b) > 0`` mean ``a`` scores higher.
    Where a test is undefined its values are NaN: ``t`` and ``p_ttest`` when every subject has
    the same difference (no spread, so scipy would report ``t = inf``, ``p = 0``), and
    ``p_wilcoxon`` when every difference is zero. The Wilcoxon test keeps scipy's defaults:
    exact for up to 50 subjects without a zero difference, otherwise the normal approximation
    with the zero differences dropped (rough for fewer than 10 subjects; ``n`` gives the count).

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
    diff = x - y
    if np.ptp(diff) <= 1e-9 * max(1.0, float(np.abs(diff).max())):  # equal up to float rounding
        t = p_ttest = math.nan
    else:
        t, p_ttest = stats.ttest_rel(x, y)
    p_wilcoxon = math.nan
    if np.any(diff != 0):
        with warnings.catch_warnings():  # the docstring states the caveat; n is reported
            warnings.filterwarnings("ignore", "Sample size too small for normal approximation", UserWarning)
            p_wilcoxon = float(stats.wilcoxon(x, y).pvalue)
    return {
        "n": len(common),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "t": float(t),
        "p_ttest": float(p_ttest),
        "p_wilcoxon": p_wilcoxon,
    }
