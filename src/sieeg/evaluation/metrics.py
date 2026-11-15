"""Classification rate (CR) and weighted-average F1, the metrics of the paper (Sec. 5.3, Table 1).

Both are reported in percent, pooled over all test trials and as mean +- sd over test
subjects (docs/DESIGN.md D17). The per-subject scores feed the paired tests of
:mod:`sieeg.evaluation.stats`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score

from ..utils import natural_key

PER_SUBJECT_COLUMNS = ("subject", "n", "cr", "f1")


def _non_empty(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        raise ValueError("Cannot score an empty set of predictions")
    return y_true, y_pred


def classification_rate(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Classification rate (accuracy) in percent."""
    return 100.0 * float(accuracy_score(*_non_empty(y_true, y_pred)))


def weighted_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Per-class F1 averaged with weights = true class support, in percent.

    A class that is never predicted has F1 0 (``zero_division=0``), without a warning.
    """
    y_true, y_pred = _non_empty(y_true, y_pred)
    return 100.0 * float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def per_subject_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """CR and weighted F1 (percent) of every test subject, in natural subject order.

    ``pred`` needs the ``subject``, ``y_true`` and ``y_pred`` columns of ``predictions.csv``.
    Returns the columns ``subject`` (str), ``n`` (number of trials), ``cr`` and ``f1``.
    """
    subjects = pred["subject"].astype(str).to_numpy()  # plain values: no index alignment
    rows = []
    for subject, part in pred.groupby(subjects, sort=False):
        y_true, y_pred = part["y_true"], part["y_pred"]
        rows.append((subject, len(part), classification_rate(y_true, y_pred), weighted_f1(y_true, y_pred)))
    rows.sort(key=lambda row: natural_key(row[0]))
    return pd.DataFrame(rows, columns=list(PER_SUBJECT_COLUMNS))


def summarize_predictions(pred: pd.DataFrame) -> dict[str, Any]:
    """Scores of one run (DESIGN D17).

    ``pooled`` scores all test trials together; ``subject_mean`` averages the per-subject
    scores and gives their sample standard deviation (ddof=1, so NaN for a single subject).

    Returns ``{"n_trials", "n_subjects", "pooled": {"cr", "f1"},
    "subject_mean": {"cr", "f1", "cr_std", "f1_std"}}`` with plain Python numbers.
    """
    per_subject = per_subject_metrics(pred)
    return {
        "n_trials": len(pred),
        "n_subjects": len(per_subject),
        "pooled": {
            "cr": classification_rate(pred["y_true"], pred["y_pred"]),
            "f1": weighted_f1(pred["y_true"], pred["y_pred"]),
        },
        "subject_mean": {
            "cr": float(per_subject["cr"].mean()),
            "f1": float(per_subject["f1"].mean()),
            "cr_std": float(per_subject["cr"].std(ddof=1)),
            "f1_std": float(per_subject["f1"].std(ddof=1)),
        },
    }
