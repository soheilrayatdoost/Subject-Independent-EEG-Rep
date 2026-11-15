"""Linear probes for subject information left in a learned representation (diagnostic, not in the paper).

The paper's claim is that adversarial training removes subject information from the
128-d representation z. Experiment 1 measures subject information in the *input*
features with random forests; these probes measure it in *z*, after training:

* :func:`subject_probe` - can a linear classifier tell subjects apart from z? Applied to
  the validation subjects, which the encoder never trained on, it asks whether the
  representation of new people still encodes who they are. Lower is more invariant.
* :func:`group_probe` - can a linear classifier separate in-group (training) trials from
  out-group (test) trials? This is exactly what SI-EEG's discriminator is trained on
  (Eq. 1), measured by an independent classifier. 0.5 is perfectly invariant.

Both use cross-validated, standardised logistic regression, so they score held-out
trials only. They never use emotion labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _probe_model(seed: int) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))


def _cv_predict(z: np.ndarray, labels: np.ndarray, n_splits: int, seed: int) -> np.ndarray | None:
    """Out-of-fold predictions, or None if some class has fewer than 2 samples."""
    _, counts = np.unique(labels, return_counts=True)
    folds = int(min(n_splits, counts.min()))
    if len(counts) < 2 or folds < 2:
        return None
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(_probe_model(seed), z, labels, cv=cv)


def subject_probe(z: np.ndarray, subjects: np.ndarray, *, n_splits: int = 5, seed: int = 0) -> dict[str, float]:
    """Cross-validated accuracy of decoding the subject from ``z``.

    Args:
        z: representations, shape (n, d).
        subjects: subject id of every row (any hashable values), shape (n,).

    Returns:
        ``{"subject_acc", "subject_chance", "n_subjects"}``. Accuracy and chance (the
        majority-subject rate) are fractions; both are NaN with fewer than 2 subjects or
        fewer than 2 trials in some subject.
    """
    z = np.asarray(z, dtype=np.float64)
    subjects = np.asarray(subjects)
    _, counts = np.unique(subjects, return_counts=True)
    pred = _cv_predict(z, subjects, n_splits, seed)
    return {
        "subject_acc": float(np.mean(pred == subjects)) if pred is not None else float("nan"),
        "subject_chance": float(counts.max() / len(subjects)) if pred is not None else float("nan"),
        "n_subjects": int(len(counts)),
    }


def group_probe(
    z_in: np.ndarray, z_out: np.ndarray, *, n_splits: int = 5, seed: int = 0
) -> dict[str, float]:
    """Cross-validated balanced accuracy of separating in-group from out-group representations.

    The larger group is undersampled (seeded) to the size of the smaller one so that every
    fold is balanced; the score is balanced accuracy (chance 0.5).

    Returns:
        ``{"group_bacc", "n_per_group"}``; ``group_bacc`` is NaN if a group has fewer than 2 rows.
    """
    z_in = np.asarray(z_in, dtype=np.float64)
    z_out = np.asarray(z_out, dtype=np.float64)
    n = min(len(z_in), len(z_out))
    if n < 2:
        return {"group_bacc": float("nan"), "n_per_group": int(n)}
    rng = np.random.default_rng(seed)
    keep_in = np.sort(rng.choice(len(z_in), size=n, replace=False))
    keep_out = np.sort(rng.choice(len(z_out), size=n, replace=False))
    z = np.concatenate([z_in[keep_in], z_out[keep_out]])
    labels = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    pred = _cv_predict(z, labels, n_splits, seed)
    bacc = balanced_accuracy_score(labels, pred) if pred is not None else float("nan")
    return {"group_bacc": float(bacc), "n_per_group": int(n)}
