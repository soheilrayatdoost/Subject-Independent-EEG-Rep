"""Paired comparison across test subjects (paper Sec. 6; docs/DESIGN.md 8.2)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sieeg.evaluation.stats import COMPARISON_FIELDS, paired_comparison


def _wilcoxon_p(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """``scipy.stats.wilcoxon(x, y).pvalue``, whether or not this SciPy version warns about it."""
    with warnings.catch_warnings():  # SciPy < 1.15 warns on its normal approximation, later versions do not
        warnings.simplefilter("ignore")
        return float(stats.wilcoxon(x, y).pvalue)


def test_paired_comparison_aligns_on_common_subjects():
    a = pd.Series({"S1": 70.0, "S2": 65.0, "S3": 80.0, "S4": 60.0, "S9": 10.0})
    b = pd.Series({"S4": 58.0, "S3": 75.0, "S2": 66.0, "S1": 62.0, "S7": 99.0})  # other order, other extras
    result = paired_comparison(a, b)

    x = np.array([70.0, 65.0, 80.0, 60.0])  # S1..S4
    y = np.array([62.0, 66.0, 75.0, 58.0])
    assert tuple(result) == COMPARISON_FIELDS
    assert result["n"] == 4
    assert result["mean_a"] == pytest.approx(x.mean())
    assert result["mean_b"] == pytest.approx(y.mean())
    assert result["mean_diff"] == pytest.approx((x - y).mean())
    expected = stats.ttest_rel(x, y)
    assert result["t"] == pytest.approx(expected.statistic)
    assert result["p_ttest"] == pytest.approx(expected.pvalue)
    assert result["p_wilcoxon"] == pytest.approx(stats.wilcoxon(x, y).pvalue)


def test_paired_comparison_sign_follows_a_minus_b():
    a = pd.Series([71.0, 66.0, 74.0, 69.5, 70.0], index=list("ABCDE"))
    b = a - np.array([2.0, 1.0, 3.5, 0.5, 2.5])
    better = paired_comparison(a, b)
    worse = paired_comparison(b, a)
    assert better["t"] > 0 and better["mean_diff"] == pytest.approx(1.9)
    assert worse["t"] == pytest.approx(-better["t"]) and worse["mean_diff"] == pytest.approx(-1.9)
    assert worse["p_ttest"] == pytest.approx(better["p_ttest"])


def test_paired_comparison_needs_two_common_subjects():
    a = pd.Series({"S1": 1.0, "S2": 2.0})
    with pytest.raises(ValueError, match="at least 2 common subjects, got 1"):
        paired_comparison(a, pd.Series({"S2": 3.0, "S3": 4.0}))
    with pytest.raises(ValueError, match="got 0"):
        paired_comparison(a, pd.Series({"S3": 4.0}))


def test_paired_comparison_identical_scores_give_nan_wilcoxon():
    a = pd.Series({"S1": 70.0, "S2": 65.0, "S3": 80.0})
    result = paired_comparison(a, a.copy())
    assert result["n"] == 3 and result["mean_diff"] == 0.0
    assert math.isnan(result["p_wilcoxon"])
    assert math.isnan(result["p_ttest"])  # 0 / 0: scipy's ttest_rel is undefined too


def test_paired_comparison_constant_differences_give_nan_t_test():
    a = pd.Series({"S1": 70.0, "S2": 80.0, "S3": 30.000000000000004})  # 100 * (0.1 + 0.2) in float64
    b = pd.Series({"S1": 60.0, "S2": 70.0, "S3": 20.0})  # every subject 10 points lower: no spread
    result = paired_comparison(a, b)
    assert result["mean_diff"] == pytest.approx(10.0)
    assert math.isnan(result["t"]) and math.isnan(result["p_ttest"])  # scipy: t = inf, p = 0
    assert result["p_wilcoxon"] == pytest.approx(_wilcoxon_p(np.full(3, 10.0)))


def test_paired_comparison_keeps_scipy_p_values_without_its_small_sample_warning():
    a = pd.Series({"S1": 69.0, "S2": 64.0})
    b = pd.Series({"S1": 69.0, "S2": 45.0})  # a zero difference: SciPy < 1.15 warns on its normal approximation
    expected = _wilcoxon_p(a.to_numpy(), b.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = paired_comparison(a, b)
    assert result["p_wilcoxon"] == pytest.approx(expected)
    assert result["p_ttest"] == pytest.approx(stats.ttest_rel(a.to_numpy(), b.to_numpy()).pvalue)


def test_paired_comparison_ties_differences_that_float_noise_tells_apart():
    # CR in percent as classification_rate computes it: 8 subjects 15/15 vs 14/15 correct trials,
    # 2 subjects 9/15 vs 10/15. Every difference is 20/3 points in size, but not in float64.
    cr = {k: 100 * (k / 15) for k in (9, 10, 14, 15)}
    assert abs(cr[15] - cr[14]) != abs(cr[9] - cr[10])
    subjects = [f"S{i}" for i in range(1, 11)]
    a = pd.Series([cr[15]] * 8 + [cr[9]] * 2, index=subjects)
    b = pd.Series([cr[14]] * 8 + [cr[10]] * 2, index=subjects)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = paired_comparison(a, b)

    exact = np.r_[np.full(8, 20 / 3), np.full(2, -20 / 3)]  # all ten tied in size
    assert result["p_wilcoxon"] == pytest.approx(_wilcoxon_p(exact))
    assert result["p_wilcoxon"] > 0.05  # ranks that float noise breaks give p < 0.01
    expected = stats.ttest_1samp(exact, 0.0)
    assert result["t"] == pytest.approx(expected.statistic)
    assert result["p_ttest"] == pytest.approx(expected.pvalue)
    assert result["mean_diff"] == pytest.approx(4.0)


def test_paired_comparison_drops_differences_that_are_zero_up_to_float_noise():
    cr = {k: 100 * (k / 15) for k in (0, 1, 5, 6)}  # CR in percent as classification_rate computes it
    # S1 scores 20% with both methods, as the mean of 2 seeds: 0 and 6 of 15 trials vs 1 and 5 of 15
    a = pd.Series({"S1": (cr[0] + cr[6]) / 2, "S2": 70.0, "S3": 80.0, "S4": 65.0, "S5": 90.0})
    b = pd.Series({"S1": (cr[1] + cr[5]) / 2, "S2": 60.0, "S3": 75.0, "S4": 66.0, "S5": 82.0})
    assert a["S1"] != b["S1"]  # 20.0 vs 19.999999999999996
    with warnings.catch_warnings():  # the zero makes SciPy < 1.15 warn on its normal approximation
        warnings.simplefilter("error")
        result = paired_comparison(a, b)
    assert result["p_wilcoxon"] == pytest.approx(_wilcoxon_p(np.array([0.0, 10.0, 5.0, -1.0, 8.0])))


def test_paired_comparison_rejects_duplicate_subjects():
    a = pd.Series([1.0, 2.0, 3.0], index=["S1", "S1", "S2"])
    b = pd.Series({"S1": 1.5, "S2": 2.5})
    with pytest.raises(ValueError, match=r"Scores a .*\['S1'\]"):
        paired_comparison(a, b)
    with pytest.raises(ValueError, match="Scores b"):
        paired_comparison(b, a)
