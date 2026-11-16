"""Experiment 1: subject and database information in the EEG features (paper Sec. 5.3 and 6).

A cross-validated random forest on the flattened log band power identifies (a) the
participant among all subjects of one database (paper: 52 % for the 50 DAI-EF subjects,
chance 2 %) and (b) the database, DAI-EF vs MAHNOB (paper: 75 %, chance 50 %). For (b) the
larger database is undersampled so that both classes are equally frequent (docs/DESIGN.md
D18).

Cross-validation (``cv``):

* ``"trial"``: stratified k-fold over trials, the paper's "10-fold cross-validation". It is the
  protocol of subject identification and the paper-comparable one for database
  identification. For the database it also measures subject identity: every subject belongs
  to one database and has trials in both the training and the test folds, so a forest that
  recognises the person names the database even when the databases do not differ at all.
* ``"subject"`` (database identification only): stratified group k-fold with the subjects as
  groups, so no subject has trials in both the training and the test folds. This row
  measures the database bias itself, and only here is 50 % the accuracy without any
  database effect. Whole subjects cannot balance every fold exactly, so a fold's training
  majority is often its test minority; ``balanced_accuracy`` is the robust figure here.

``normalization="subject"`` applies the paper's per-participant z-score (Sec. 5.3) inside the
cross-validation: in every fold, each subject's mean and sd are fitted on that subject's
trials outside the test fold (on all of its trials if it has none there, i.e. a held-out
subject) and applied to all of its trials. Z-scoring every subject over all of its trials
before splitting would make the held-out trials of a subject anti-correlated with its
training trials and push the accuracy below chance. Per-subject z-scoring removes any offset
shared by a database's subjects (a mean shift, a units mismatch), so these rows show what is
left of the bias after the paper's normalisation, not the whole database difference.

Accuracies and chance levels are fractions in [0, 1]; ``*_std`` is the sample standard
deviation across folds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ..features.featureset import FeatureSet, feature_config_differences

logger = logging.getLogger(__name__)

NORMALIZATIONS = ("none", "subject")
CV_MODES = ("trial", "subject")
#: Gap in median log10 band power (decades) above which two databases look like different units.
UNITS_GAP_DECADES = 2.0

_SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
}


def _check_options(normalization: str, cv: str = "trial") -> None:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization {normalization!r}; use one of {NORMALIZATIONS}")
    if cv not in CV_MODES:
        raise ValueError(f"Unknown cv {cv!r}; use one of {CV_MODES}")


def _zscore_outside_fold(x: np.ndarray, subjects: np.ndarray, test: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-subject z-score fitted on each subject's rows outside ``test`` (all its rows if it has none there).

    Same rule as :func:`~sieeg.features.featureset.zscore_by_group`: features with (near) zero
    variance are only centred.
    """
    x = np.asarray(x, dtype=np.float64)
    fit_rows = np.ones(len(x), dtype=bool)
    fit_rows[test] = False
    out = np.empty_like(x)
    for subject in np.unique(subjects):
        rows = subjects == subject
        fit = rows & fit_rows
        reference = x[fit] if fit.any() else x[rows]
        sd = reference.std(axis=0)
        out[rows] = (x[rows] - reference.mean(axis=0)) / np.where(sd < eps, 1.0, sd)
    return out.astype(np.float32)


def _cv_scores(
    x: np.ndarray,
    y: np.ndarray,
    scoring: tuple[str, ...],
    *,
    n_splits: int,
    n_estimators: int,
    seed: int,
    n_jobs: int,
    groups: np.ndarray | None = None,
    normalize_by: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-fold test scores of a random forest under seeded stratified k-fold CV.

    ``groups`` (one id per row) makes the folds group-disjoint (stratified group k-fold).
    ``normalize_by`` (one subject id per row) z-scores every subject inside each fold with
    :func:`_zscore_outside_fold`.
    """
    if groups is None:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y)
    else:
        # StratifiedGroupKFold(shuffle=True) of scikit-learn 1.5 shuffles the per-group class counts
        # but not the group ids, so its folds are not stratified (single-database test folds are
        # common). shuffle=False on seeded random group codes gives a seeded, stratified split.
        _, inverse = np.unique(groups, return_inverse=True)
        codes = np.random.default_rng(seed).permutation(int(inverse.max()) + 1)[inverse]
        splits = StratifiedGroupKFold(n_splits=n_splits).split(x, y, codes)
    scores: dict[str, list[float]] = {name: [] for name in scoring}
    for train, test in splits:
        features = x if normalize_by is None else _zscore_outside_fold(x, normalize_by, test)
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
        predicted = forest.fit(features[train], y[train]).predict(features[test])
        for name in scoring:
            scores[name].append(float(_SCORERS[name](y[test], predicted)))
    return {name: np.array(values) for name, values in scores.items()}


def _majority_share(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def subject_identification(
    fs: FeatureSet,
    *,
    normalization: str = "none",
    n_splits: int = 10,
    n_estimators: int = 500,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Recognise the participant of every trial among all subjects of ``fs`` (Exp. 1).

    Trial-level stratified k-fold CV (the class is the subject, so folds cannot be grouped by
    subject); ``"subject"`` normalisation is fitted inside each fold (module docstring).

    Returns ``{"task": "subject", "normalization", "cv": "trial", "accuracy", "accuracy_std",
    "chance_uniform" (1/K), "chance_majority", "n_classes" (K), "n_samples", "n_splits"}``.
    """
    _check_options(normalization)
    x = fs.flat_features("none")
    y = fs.subjects
    scores = _cv_scores(
        x, y, ("accuracy",), n_splits=n_splits, n_estimators=n_estimators, seed=seed, n_jobs=n_jobs,
        normalize_by=y if normalization == "subject" else None,
    )
    n_classes = len(np.unique(y))
    result = {
        "task": "subject",
        "normalization": normalization,
        "cv": "trial",
        "accuracy": float(scores["accuracy"].mean()),
        "accuracy_std": float(scores["accuracy"].std(ddof=1)),
        "chance_uniform": 1.0 / n_classes,
        "chance_majority": _majority_share(y),
        "n_classes": n_classes,
        "n_samples": len(y),
        "n_splits": n_splits,
    }
    logger.info(
        "Subject identification on %s (%s): accuracy %.3f +- %.3f, chance %.3f (%d subjects)",
        fs.dataset, normalization, result["accuracy"], result["accuracy_std"], result["chance_uniform"], n_classes,
    )
    return result


def _common_channels(fs_a: FeatureSet, fs_b: FeatureSet) -> tuple[FeatureSet, FeatureSet]:
    """Both sets restricted to their shared channels (case-insensitive, in ``fs_a`` order)."""
    names_b = {name.lower() for name in fs_b.ch_names}
    common = [name for name in fs_a.ch_names if name.lower() in names_b]
    if not common:
        raise ValueError(f"{fs_a.dataset} and {fs_b.dataset} have no channel in common")
    if len(common) != fs_a.n_channels or len(common) != fs_b.n_channels:
        logger.warning(
            "Domain identification uses the %d channels common to %s (%d) and %s (%d)",
            len(common), fs_a.dataset, fs_a.n_channels, fs_b.dataset, fs_b.n_channels,
        )
    return fs_a.pick_channels(common), fs_b.pick_channels(common)


def _warn_if_not_comparable(fs_a: FeatureSet, fs_b: FeatureSet) -> None:
    """Warn about differences that unnormalised database identification picks up besides the EEG itself."""
    median_a, median_b = float(np.median(fs_a.band_power)), float(np.median(fs_b.band_power))
    if abs(median_a - median_b) > UNITS_GAP_DECADES:
        logger.warning(
            "The median log10 band power of %s (%.2f) and %s (%.2f) differs by %.1f decades, which looks like "
            "different units (microvolts read as volts add 12): unnormalised database identification then "
            "measures the units. Check `sieeg info` and the manifest's scale column",
            fs_a.dataset, median_a, fs_b.dataset, median_b, abs(median_a - median_b),
        )
    differences = feature_config_differences(fs_a, fs_b)
    if differences:
        logger.warning(
            "%s and %s were extracted with different settings (%s): unnormalised database identification "
            "also measures them",
            fs_a.dataset, fs_b.dataset, "; ".join(differences),
        )


def _undersample(n_rows: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``n`` of ``n_rows`` rows drawn without replacement (sorted); all rows if ``n_rows <= n``."""
    if n_rows <= n:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=n, replace=False))


def domain_identification(
    fs_a: FeatureSet,
    fs_b: FeatureSet,
    *,
    normalization: str = "none",
    cv: str = "trial",
    balance: bool = True,
    n_splits: int = 10,
    n_estimators: int = 500,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Recognise the database of every trial, ``fs_a`` vs ``fs_b`` (Exp. 1: DAI-EF vs MAHNOB).

    Both sets are restricted to their common channels and must have identical bands. If
    ``balance``, the larger set is undersampled at random (seeded) to the size of the smaller
    (DESIGN D18). The class label is the dataset name; the subjects are qualified by it
    (:func:`~sieeg.utils.qualify`), so equal ids in both sets stay different people.
    ``cv="trial"`` is the paper's stratified k-fold over trials, ``cv="subject"`` keeps every
    subject in one fold, with at most as many folds as the smaller database has subjects
    (module docstring). With ``normalization="none"``, a gap of more than
    :data:`UNITS_GAP_DECADES` between the sets' median log10 band power and differing
    extraction settings (``feature_config_differences``) are logged as warnings.

    Returns ``{"task": "domain", "normalization", "cv", "accuracy", "accuracy_std",
    "balanced_accuracy", "chance_uniform" (0.5), "chance_majority", "n_classes" (2),
    "n_samples", "n_splits" (folds used), "n_per_domain": {dataset: n_trials}}``.

    Raises:
        ValueError: if both sets have the same dataset name, different bands or no common
            channel, or (``cv="subject"``) a set has fewer than two subjects.
    """
    _check_options(normalization, cv)
    if fs_a.dataset == fs_b.dataset:
        raise ValueError(f"Both feature sets are named {fs_a.dataset!r}; the dataset name is the class label")
    if fs_a.bands != fs_b.bands:
        raise ValueError(f"Bands differ: {fs_a.dataset} has {fs_a.bands}, {fs_b.dataset} has {fs_b.bands}")
    fs_a, fs_b = _common_channels(fs_a, fs_b)
    if normalization == "none":
        _warn_if_not_comparable(fs_a, fs_b)
    keep_a, keep_b = np.arange(len(fs_a)), np.arange(len(fs_b))
    if balance:
        rng = np.random.default_rng(seed)
        n = min(len(fs_a), len(fs_b))
        keep_a, keep_b = _undersample(len(fs_a), n, rng), _undersample(len(fs_b), n, rng)
    x = np.concatenate([fs_a.flat_features("none")[keep_a], fs_b.flat_features("none")[keep_b]])
    y = np.array([fs_a.dataset] * len(keep_a) + [fs_b.dataset] * len(keep_b))
    subjects = np.concatenate([fs_a.qualified_subjects[keep_a], fs_b.qualified_subjects[keep_b]])

    n_folds = n_splits
    if cv == "subject":
        n_subjects = {name: len(np.unique(subjects[y == name])) for name in (fs_a.dataset, fs_b.dataset)}
        n_folds = min(n_splits, *n_subjects.values())
        if n_folds < 2:
            raise ValueError(
                f"Subject-grouped cross-validation needs at least 2 subjects per database, got {n_subjects}"
            )
        if n_folds < n_splits:
            logger.warning(
                "Subject-grouped database identification uses %d folds instead of %d (subjects: %s)",
                n_folds, n_splits, ", ".join(f"{name} {k}" for name, k in n_subjects.items()),
            )
    scores = _cv_scores(
        x, y, ("accuracy", "balanced_accuracy"), n_splits=n_folds, n_estimators=n_estimators, seed=seed,
        n_jobs=n_jobs, groups=subjects if cv == "subject" else None,
        normalize_by=subjects if normalization == "subject" else None,
    )
    result = {
        "task": "domain",
        "normalization": normalization,
        "cv": cv,
        "accuracy": float(scores["accuracy"].mean()),
        "accuracy_std": float(scores["accuracy"].std(ddof=1)),
        "balanced_accuracy": float(scores["balanced_accuracy"].mean()),
        "chance_uniform": 0.5,
        "chance_majority": _majority_share(y),
        "n_classes": 2,
        "n_samples": len(y),
        "n_splits": n_folds,
        "n_per_domain": {fs_a.dataset: len(keep_a), fs_b.dataset: len(keep_b)},
    }
    logger.info(
        "Domain identification %s vs %s (%s, cv %s): accuracy %.3f +- %.3f, balanced %.3f, majority %.3f",
        fs_a.dataset, fs_b.dataset, normalization, cv, result["accuracy"], result["accuracy_std"],
        result["balanced_accuracy"], result["chance_majority"],
    )
    return result


def run_bias_analysis(
    fs: FeatureSet,
    other: FeatureSet | None = None,
    *,
    normalizations: Sequence[str] = NORMALIZATIONS,
    domain_cv: Sequence[str] = CV_MODES,
    balance: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Experiment 1 table: subject identification on ``fs`` (one row per normalisation) and,
    given ``other``, domain identification ``fs`` vs ``other`` (one row per ``domain_cv``
    mode and normalisation, the paper-comparable ``"trial"`` rows first).

    ``kwargs`` (``n_splits``, ``n_estimators``, ``seed``, ``n_jobs``) go to both analyses,
    ``balance`` only to the domain one.
    """
    rows = [subject_identification(fs, normalization=norm, **kwargs) for norm in normalizations]
    if other is not None:
        rows += [
            domain_identification(fs, other, normalization=norm, cv=cv, balance=balance, **kwargs)
            for cv in domain_cv
            for norm in normalizations
        ]
    return pd.DataFrame(rows)
