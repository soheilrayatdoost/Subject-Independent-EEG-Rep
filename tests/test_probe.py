"""Linear subject probes on representations (sieeg.analysis.probe)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sieeg.analysis.probe import group_probe, subject_probe


def _clustered(n_subjects: int, n_per: int, spread: float, rng: np.random.Generator, d: int = 16):
    centres = rng.normal(0.0, spread, size=(n_subjects, d))
    z = np.concatenate([c + rng.normal(size=(n_per, d)) for c in centres])
    subjects = np.repeat([f"S{i}" for i in range(n_subjects)], n_per)
    return z, subjects


def test_subject_probe_detects_subject_clusters(rng):
    z, subjects = _clustered(5, 30, spread=4.0, rng=rng)
    out = subject_probe(z, subjects, seed=0)
    assert out["n_subjects"] == 5
    assert out["subject_chance"] == pytest.approx(0.2)
    assert out["subject_acc"] > 0.9


def test_subject_probe_is_near_chance_without_subject_information(rng):
    z = rng.normal(size=(150, 16))
    subjects = np.repeat(["a", "b", "c", "d", "e"], 30)
    out = subject_probe(z, subjects, seed=0)
    assert out["subject_acc"] < 0.4  # chance 0.2; held-out predictions only


def test_subject_probe_needs_two_subjects_with_two_trials(rng):
    out = subject_probe(rng.normal(size=(10, 4)), np.array(["a"] * 10))
    assert math.isnan(out["subject_acc"]) and math.isnan(out["subject_chance"]) and out["n_subjects"] == 1
    out = subject_probe(rng.normal(size=(3, 4)), np.array(["a", "a", "b"]))
    assert math.isnan(out["subject_acc"])


def test_subject_probe_uses_fewer_folds_for_small_subjects(rng):
    z, subjects = _clustered(3, 3, spread=5.0, rng=rng)
    out = subject_probe(z, subjects, n_splits=10, seed=0)  # only 3 trials per subject -> 3 folds
    assert out["subject_acc"] > 0.6


def test_group_probe_separates_shifted_groups_and_balances(rng):
    z_in = rng.normal(size=(200, 8))
    z_out = rng.normal(size=(20, 8)) + 3.0
    out = group_probe(z_in, z_out, seed=0)
    assert out["n_per_group"] == 20
    assert out["group_bacc"] > 0.9


def test_group_probe_is_near_chance_for_identical_distributions(rng):
    out = group_probe(rng.normal(size=(200, 8)), rng.normal(size=(40, 8)), seed=0)
    assert 0.3 < out["group_bacc"] < 0.7


def test_group_probe_is_deterministic_and_handles_tiny_groups(rng):
    z_in, z_out = rng.normal(size=(50, 4)), rng.normal(size=(10, 4)) + 1.0
    assert group_probe(z_in, z_out, seed=3) == group_probe(z_in, z_out, seed=3)
    assert math.isnan(group_probe(z_in, z_out[:1])["group_bacc"])


def test_empty_selection_gives_nan_like_a_single_subject():
    out = subject_probe(np.empty((0, 4)), np.array([], dtype=str))
    assert math.isnan(out["subject_acc"]) and math.isnan(out["subject_chance"]) and out["n_subjects"] == 0
    assert math.isnan(group_probe(np.empty((0, 4)), np.ones((5, 4)))["group_bacc"])
