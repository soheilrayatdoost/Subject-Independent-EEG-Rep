"""Experiment 1: random-forest subject and domain identification (paper Sec. 5.3 / 6; docs/DESIGN.md 9, D18)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from sieeg.analysis import bias
from sieeg.analysis.bias import domain_identification, run_bias_analysis, subject_identification
from sieeg.config import BIOSEMI32_CHANNELS, PAPER_BANDS, FeatureConfig, config_to_dict
from sieeg.data.synthetic import make_synthetic_featureset
from sieeg.features.featureset import FeatureSet

#: Small forests and 5 folds keep every test well under a second per analysis.
FAST = {"n_splits": 5, "n_estimators": 50, "seed": 0, "n_jobs": 1}
#: The paper's 10 folds, where z-scoring before the split pushes null data far below chance.
TEN_FOLDS = {**FAST, "n_splits": 10}


def _copy(fs: FeatureSet, *, dataset: str, order=None, ch_names: list[str] | None = None,
          rename: Callable[[str], str] = str, bands=None) -> FeatureSet:
    """``fs``'s trials as another dataset, channels reordered by ``order`` and renamed by ``rename``."""
    order = np.arange(fs.n_channels) if order is None else np.asarray(order)
    return FeatureSet(
        dataset=dataset, band_power=fs.band_power[:, order, :], subjects=fs.subjects, trials=fs.trials,
        valence=fs.valence, arousal=fs.arousal, ch_names=ch_names or [rename(fs.ch_names[i]) for i in order],
        bands=bands or fs.bands,
    )


def _null_featureset(dataset: str, n_subjects: int, n_trials: int, *, seed: int, prefix: str = "S") -> FeatureSet:
    """i.i.d. N(0, 1) log band power: no subject and no database information at all."""
    n = n_subjects * n_trials
    return FeatureSet(
        dataset=dataset,
        band_power=np.random.default_rng(seed).standard_normal((n, len(BIOSEMI32_CHANNELS), len(PAPER_BANDS))),
        subjects=np.repeat([f"{prefix}{s + 1:02d}" for s in range(n_subjects)], n_trials),
        trials=np.tile([f"T{t + 1:02d}" for t in range(n_trials)], n_subjects),
        valence=np.full(n, 5.0), arousal=np.full(n, 5.0), ch_names=list(BIOSEMI32_CHANNELS), bands=list(PAPER_BANDS),
    )


# ------------------------------------------------------------------ subject identification
def test_subject_identification_far_above_chance_until_subject_normalisation():
    fs = make_synthetic_featureset(n_subjects=8, n_trials=15, subject_offset=3.0, subject_style=0.0, seed=0)
    raw = subject_identification(fs, normalization="none", **FAST)
    assert list(raw) == ["task", "normalization", "cv", "accuracy", "accuracy_std", "chance_uniform",
                         "chance_majority", "n_classes", "n_samples", "n_splits"]
    assert (raw["task"], raw["normalization"], raw["cv"], raw["n_classes"], raw["n_samples"], raw["n_splits"]) == (
        "subject", "none", "trial", 8, 120, 5)
    assert raw["chance_uniform"] == pytest.approx(1 / 8) and raw["chance_majority"] == pytest.approx(1 / 8)
    assert raw["accuracy"] > 0.9  # chance is 0.125: the subject offsets make trials identifiable
    assert raw["accuracy_std"] >= 0.0

    z = subject_identification(fs, normalization="subject", **FAST)
    assert z["normalization"] == "subject"
    assert 0.05 < z["accuracy"] < 0.3  # the offsets are all there is: back to chance (not below it)


def test_subject_normalisation_keeps_subject_patterns():
    """Per-subject z-scoring removes offsets, not a subject-specific pattern (the synthetic "style")."""
    fs = make_synthetic_featureset(n_subjects=8, n_trials=15, subject_offset=3.0, subject_style=2.0, seed=0)
    assert subject_identification(fs, normalization="subject", **FAST)["accuracy"] > 0.45  # chance 0.125


def test_subject_normalisation_is_fitted_outside_the_test_fold():
    """Null data: z-scoring each subject over all its trials before the split gave about 2 % (chance 10 %)."""
    fs = _null_featureset("null", n_subjects=10, n_trials=40, seed=0)
    assert 0.06 < subject_identification(fs, normalization="subject", **TEN_FOLDS)["accuracy"] < 0.2


def test_subject_identification_majority_chance_and_determinism():
    fs = make_synthetic_featureset(n_subjects=3, n_trials=15, seed=1)
    fs = fs.subset(np.r_[0:15, 15:20, 30:45])  # S01: 15 trials, S02: 5, S03: 15
    result = subject_identification(fs, **FAST)
    assert result["n_samples"] == 35
    assert result["chance_uniform"] == pytest.approx(1 / 3)
    assert result["chance_majority"] == pytest.approx(15 / 35)
    assert subject_identification(fs, **FAST) == result
    with pytest.raises(ValueError, match="normalization"):
        subject_identification(fs, normalization="global", **FAST)


# ------------------------------------------------------------------ domain identification
def test_domain_identification_balances_databases_and_detects_a_shift():
    no_subject_effect = {"subject_offset": 0.0, "subject_style": 0.0}  # only a database shift can carry the label
    a = make_synthetic_featureset(n_subjects=4, n_trials=10, dataset="a", seed=1, subject_prefix="A",
                                  **no_subject_effect)
    b = make_synthetic_featureset(n_subjects=6, n_trials=10, dataset="b", seed=2, domain_shift=1.0, subject_prefix="B",
                                  **no_subject_effect)
    result = domain_identification(a, b, **FAST)
    assert list(result) == ["task", "normalization", "cv", "accuracy", "accuracy_std", "balanced_accuracy",
                            "chance_uniform", "chance_majority", "n_classes", "n_samples", "n_splits", "n_per_domain"]
    assert (result["task"], result["normalization"], result["cv"]) == ("domain", "none", "trial")
    assert result["n_classes"] == 2
    assert result["n_per_domain"] == {"a": 40, "b": 40} and result["n_samples"] == 80
    assert result["chance_majority"] == 0.5 and result["chance_uniform"] == 0.5
    assert result["accuracy"] > 0.75 and result["balanced_accuracy"] > 0.75
    assert domain_identification(a, b, **FAST) == result  # seeded undersampling and forest

    same = make_synthetic_featureset(n_subjects=6, n_trials=10, dataset="b", seed=2, subject_prefix="B",
                                     **no_subject_effect)
    assert domain_identification(a, same, **FAST)["accuracy"] < 0.65  # no shift, nothing to detect

    unbalanced = domain_identification(a, b, balance=False, **FAST)
    assert unbalanced["n_per_domain"] == {"a": 40, "b": 60} and unbalanced["n_samples"] == 100
    assert unbalanced["chance_majority"] == pytest.approx(0.6)


def test_subject_grouped_cv_measures_the_database_not_the_subjects():
    """Trial-level folds name the database by recognising the subjects; subject-grouped folds do not.

    Trials of one subject are correlated, so the subjects are the effective sample: with 30
    per database the null accuracy of subject-grouped folds has an SD of about 0.04 across
    seeds, and the +-0.15 band below is about 3.5 SDs wide (12 subjects gave an SD of 0.09).
    """
    a = make_synthetic_featureset(n_subjects=30, n_trials=4, dataset="a", seed=3, subject_prefix="A")
    same = make_synthetic_featureset(n_subjects=30, n_trials=4, dataset="b", seed=4, subject_prefix="B")
    shifted = make_synthetic_featureset(n_subjects=30, n_trials=4, dataset="b", seed=4, domain_shift=1.0,
                                        subject_prefix="B")
    assert domain_identification(a, same, cv="trial", **FAST)["accuracy"] > 0.8  # no database effect at all
    null = domain_identification(a, same, cv="subject", **FAST)
    assert null["cv"] == "subject" and null["n_per_domain"] == {"a": 120, "b": 120}
    assert abs(null["accuracy"] - 0.5) < 0.15 and abs(null["balanced_accuracy"] - 0.5) < 0.15
    shift = domain_identification(a, shifted, cv="subject", **FAST)
    assert shift["accuracy"] > 0.75 and shift["balanced_accuracy"] > 0.75


def test_subject_grouped_folds_are_subject_disjoint_and_stratified(monkeypatch):
    folds = []

    class SpyFolds(StratifiedGroupKFold):
        def split(self, x, y=None, groups=None):
            for train, test in super().split(x, y, groups):
                folds.append((np.asarray(y), np.asarray(groups), train, test))
                yield train, test

    monkeypatch.setattr(bias, "StratifiedGroupKFold", SpyFolds)
    a = make_synthetic_featureset(n_subjects=10, n_trials=6, dataset="a", seed=3)  # S01-S10 ...
    b = make_synthetic_featureset(n_subjects=12, n_trials=6, dataset="b", seed=4)  # ... other people, same ids
    domain_identification(a, b, cv="subject", **FAST)
    assert len(folds) == 5
    assert len(np.unique(folds[0][1])) == 22  # subjects are qualified by the dataset
    for y, groups, train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert 0.3 < np.mean(y[test] == "a") < 0.7  # every test fold holds both databases


def test_subject_grouped_cv_fold_count(caplog):
    a = make_synthetic_featureset(n_subjects=3, n_trials=10, dataset="a", seed=1)
    b = make_synthetic_featureset(n_subjects=4, n_trials=10, dataset="b", seed=2)
    with caplog.at_level(logging.WARNING, logger="sieeg.analysis.bias"):
        result = domain_identification(a, b, cv="subject", **FAST)
    assert (result["cv"], result["n_splits"]) == ("subject", 3)
    assert "uses 3 folds instead of 5 (subjects: a 3, b 4)" in caplog.text
    with pytest.raises(ValueError, match="at least 2 subjects per database"):
        domain_identification(a.select_subjects(["S01"]), b, cv="subject", **FAST)
    with pytest.raises(ValueError, match="cv"):
        domain_identification(a, b, cv="group", **FAST)


def test_domain_subject_normalisation_is_fitted_outside_the_test_fold():
    """Two databases from the same distribution; z-scoring before the split gave about 35 % (chance 50 %)."""
    a = _null_featureset("a", n_subjects=10, n_trials=20, seed=10, prefix="A")
    b = _null_featureset("b", n_subjects=10, n_trials=20, seed=20, prefix="B")
    for cv in ("trial", "subject"):
        assert 0.42 < domain_identification(a, b, normalization="subject", cv=cv, **TEN_FOLDS)["accuracy"] < 0.6


def test_domain_subject_normalisation_groups_are_qualified_subjects():
    """``S01`` of each database is a different person: a database offset must not survive z-scoring."""
    a = make_synthetic_featureset(n_subjects=10, n_trials=6, dataset="a", seed=3)
    b = make_synthetic_featureset(n_subjects=12, n_trials=6, dataset="b", seed=4)
    b.band_power += 1.5
    assert domain_identification(a, b, **FAST)["accuracy"] > 0.9
    for cv in ("trial", "subject"):
        assert domain_identification(a, b, normalization="subject", cv=cv, **FAST)["accuracy"] < 0.7


def test_domain_identification_warns_about_units_and_extraction_settings(caplog):
    a = make_synthetic_featureset(n_subjects=3, n_trials=10, dataset="a", seed=1)
    b = make_synthetic_featureset(n_subjects=3, n_trials=10, dataset="b", seed=2)
    with caplog.at_level(logging.WARNING, logger="sieeg.analysis.bias"):
        domain_identification(a, b, **FAST)
    assert caplog.text == ""

    microvolts = _copy(b, dataset="b")
    microvolts.band_power += 12.0  # microvolts read as volts
    with caplog.at_level(logging.WARNING, logger="sieeg.analysis.bias"):
        domain_identification(a, microvolts, **FAST)
    gap = abs(float(np.median(microvolts.band_power)) - float(np.median(a.band_power)))  # 12 + the datasets' own gap
    assert 11.0 < gap < 13.0
    assert f"differs by {gap:.1f} decades" in caplog.text and "scale column" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sieeg.analysis.bias"):
        domain_identification(a, microvolts, normalization="subject", **FAST)  # z-scoring removes the offset
    assert caplog.text == ""

    a.meta["feature_config"] = config_to_dict(FeatureConfig())
    b.meta["feature_config"] = config_to_dict(FeatureConfig(resample_sfreq=128.0, reference=None))
    with caplog.at_level(logging.WARNING, logger="sieeg.analysis.bias"):
        domain_identification(a, b, **FAST)
    assert "a and b were extracted with different settings" in caplog.text
    assert "resample_sfreq: a=256.0 vs b=128.0" in caplog.text and "reference: a='average' vs b=None" in caplog.text


def test_domain_identification_matches_channels_by_name_case_insensitively():
    a = make_synthetic_featureset(n_subjects=4, n_trials=10, dataset="a", seed=3)
    order = np.random.default_rng(0).permutation(a.n_channels)
    same = _copy(a, dataset="b", order=order, rename=str.upper)  # a's trials, channels shuffled and upper-cased
    assert domain_identification(a, same, **FAST)["accuracy"] < 0.6  # aligned by name: nothing to tell apart

    misaligned = _copy(a, dataset="b", order=order, ch_names=a.ch_names)  # names no longer follow the data
    assert domain_identification(a, misaligned, **FAST)["accuracy"] > 0.9  # what a positional match would give


def test_domain_identification_restricts_to_common_channels(caplog):
    a = make_synthetic_featureset(n_subjects=3, n_trials=10, dataset="a", seed=4)
    b = make_synthetic_featureset(n_subjects=3, n_trials=10, dataset="b", seed=5, domain_shift=1.0)
    b = b.pick_channels(a.ch_names[:20][::-1])
    with caplog.at_level(logging.WARNING, logger="sieeg.analysis.bias"):
        result = domain_identification(a, b, **FAST)
    assert "20 channels common to a (32) and b (20)" in caplog.text
    assert result["n_per_domain"] == {"a": 30, "b": 30}


def test_domain_identification_rejects_incompatible_sets():
    a = make_synthetic_featureset(n_subjects=2, n_trials=10, dataset="a", seed=6)
    with pytest.raises(ValueError, match="named 'a'"):
        domain_identification(a, a, **FAST)
    wider = _copy(a, dataset="b", bands=[(name, lo, hi + 1.0) for name, lo, hi in a.bands])
    with pytest.raises(ValueError, match="Bands differ"):
        domain_identification(a, wider, **FAST)
    with pytest.raises(ValueError, match="no channel in common"):
        domain_identification(a, _copy(a, dataset="b", rename=lambda name: f"X{name}"), **FAST)
    with pytest.raises(ValueError, match="normalization"):
        domain_identification(a, _copy(a, dataset="b"), normalization="global", **FAST)


# ------------------------------------------------------------------ the Experiment 1 table
def test_run_bias_analysis_rows_per_task_cv_and_normalisation(small_featureset):
    other = make_synthetic_featureset(
        n_subjects=3, n_trials=12, dataset="other", seed=5, domain_shift=1.0, subject_prefix="O"
    )
    table = run_bias_analysis(small_featureset, other, balance=False, **FAST)
    assert table[["task", "cv", "normalization"]].values.tolist() == [
        ["subject", "trial", "none"], ["subject", "trial", "subject"],
        ["domain", "trial", "none"], ["domain", "trial", "subject"],
        ["domain", "subject", "none"], ["domain", "subject", "subject"],
    ]
    assert table["chance_uniform"].tolist() == pytest.approx([1 / 6, 1 / 6, 0.5, 0.5, 0.5, 0.5])
    assert table["n_per_domain"].iloc[2:].tolist() == [{"synthetic": 72, "other": 36}] * 4  # balance=False reached it
    assert table["n_per_domain"].iloc[:2].isna().all()
    assert table["n_splits"].tolist() == [5, 5, 5, 5, 3, 3]  # "other" has 3 subjects
    assert table.loc[0, "accuracy"] == subject_identification(small_featureset, **FAST)["accuracy"]

    subject_only = run_bias_analysis(small_featureset, normalizations=("subject",), **FAST)
    assert subject_only[["task", "normalization"]].values.tolist() == [["subject", "subject"]]
    trial_only = run_bias_analysis(small_featureset, other, normalizations=("none",), domain_cv=("trial",), **FAST)
    assert trial_only[["task", "cv"]].values.tolist() == [["subject", "trial"], ["domain", "trial"]]
