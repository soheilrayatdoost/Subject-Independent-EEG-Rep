"""CR, weighted F1 and their per-subject summaries (paper Sec. 5.3, Table 1; docs/DESIGN.md 8.1, D17)."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from sieeg.evaluation.metrics import classification_rate, per_subject_metrics, summarize_predictions, weighted_f1


def _pred(rows: list[tuple]) -> pd.DataFrame:
    """predictions.csv-like frame from (subject, y_true, y_pred) tuples."""
    return pd.DataFrame(rows, columns=["subject", "y_true", "y_pred"])


#: S10: 2/2 correct, S2: 3/4 correct, S1: 0/1 correct (7 trials, 5 correct).
PRED = _pred(
    [
        ("S10", 1, 1), ("S10", 0, 0),
        ("S2", 1, 0), ("S2", 1, 1), ("S2", 0, 0), ("S2", 0, 0),
        ("S1", 0, 1),
    ]
)


def test_classification_rate_is_accuracy_in_percent():
    assert classification_rate([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(75.0)
    assert classification_rate(np.array([1, 0]), np.array([1, 0])) == 100.0
    assert isinstance(classification_rate([1], [1]), float)


def test_weighted_f1_hand_computed():
    # class 0: P = R = 2/3 -> F1 = 2/3 (support 3); class 1: P = R = 1/2 -> F1 = 1/2 (support 2)
    expected = 100 * (3 * (2 / 3) + 2 * 0.5) / 5
    assert weighted_f1([0, 0, 0, 1, 1], [0, 0, 1, 1, 0]) == pytest.approx(expected)  # 60 %


def test_weighted_f1_matches_sklearn(rng):
    y_true, y_pred = rng.integers(0, 2, 50), rng.integers(0, 2, 50)
    assert weighted_f1(y_true, y_pred) == pytest.approx(100 * f1_score(y_true, y_pred, average="weighted"))


def test_weighted_f1_scores_a_never_predicted_class_as_zero_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = weighted_f1([0, 0, 1, 1], [0, 0, 0, 0])
    # class 0: P = 1/2, R = 1 -> F1 = 2/3; class 1 is never predicted -> F1 = 0; equal supports
    assert score == pytest.approx(100 / 3)


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="empty"):
        classification_rate([], [])
    with pytest.raises(ValueError, match="empty"):
        weighted_f1(np.array([]), np.array([]))


def test_per_subject_metrics_in_natural_subject_order():
    table = per_subject_metrics(PRED)
    assert list(table.columns) == ["subject", "n", "cr", "f1"]
    assert table["subject"].tolist() == ["S1", "S2", "S10"]
    assert table["n"].tolist() == [1, 4, 2]
    assert table["cr"].tolist() == pytest.approx([0.0, 75.0, 100.0])
    # S2: y_true [1, 1, 0, 0], y_pred [0, 1, 0, 0] -> class 0 F1 = 0.8, class 1 F1 = 2/3, equal supports
    assert table["f1"].tolist() == pytest.approx([0.0, 100 * (0.8 + 2 / 3) / 2, 100.0])


def test_per_subject_metrics_on_concatenated_fold_frames():
    folds = [PRED[PRED.subject == s] for s in ("S10", "S2", "S1")]
    stacked = pd.concat([f.reset_index(drop=True) for f in folds])  # index labels repeat across folds
    pd.testing.assert_frame_equal(per_subject_metrics(stacked), per_subject_metrics(PRED))


def test_per_subject_metrics_orders_numeric_ids_naturally_as_strings():
    pred = _pred([(10, 1, 1), (9, 0, 0), (1, 1, 0), (10, 0, 0)])
    table = per_subject_metrics(pred)
    assert table["subject"].tolist() == ["1", "9", "10"]
    assert table["n"].tolist() == [1, 1, 2]


def test_summarize_predictions_keys_and_values():
    summary = summarize_predictions(PRED)
    assert list(summary) == ["n_trials", "n_subjects", "pooled", "subject_mean"]
    assert list(summary["pooled"]) == ["cr", "f1"]
    assert list(summary["subject_mean"]) == ["cr", "f1", "cr_std", "f1_std"]
    assert (summary["n_trials"], summary["n_subjects"]) == (7, 3)

    assert summary["pooled"]["cr"] == pytest.approx(100 * 5 / 7)
    assert summary["pooled"]["f1"] == pytest.approx(100 * f1_score(PRED.y_true, PRED.y_pred, average="weighted"))
    per_subject = per_subject_metrics(PRED)
    for metric in ("cr", "f1"):
        assert summary["subject_mean"][metric] == pytest.approx(np.mean(per_subject[metric]))
        assert summary["subject_mean"][f"{metric}_std"] == pytest.approx(np.std(per_subject[metric], ddof=1))
    assert summary["subject_mean"]["cr"] == pytest.approx(175 / 3)  # mean of 0, 75 and 100 %
    json.dumps(summary, allow_nan=False)  # plain Python numbers only


def test_summarize_predictions_single_subject_has_nan_sd():
    summary = summarize_predictions(_pred([("S1", 1, 1), ("S1", 0, 1)]))
    assert summary["n_subjects"] == 1
    assert summary["subject_mean"]["cr"] == summary["pooled"]["cr"] == 50.0
    assert math.isnan(summary["subject_mean"]["cr_std"]) and math.isnan(summary["subject_mean"]["f1_std"])
