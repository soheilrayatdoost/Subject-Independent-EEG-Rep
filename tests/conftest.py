"""Shared fixtures. Everything here is synthetic: no real EEG data is needed to run the tests."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# GPU numerics of the suite, set before TensorFlow starts and inherited by the workers of jobs > 1
# (no effect on a CPU, except that random TF ops need a seed):
# - memory grows on demand: this process trains too and would otherwise keep nearly all GPU memory;
# - deterministic kernels, for the tests that compare two trainings bit for bit. Process-wide, because
#   TensorFlow keeps the cuDNN algorithm it first chose for a convolution shape: enabling op
#   determinism inside one test does not cover the shapes that an earlier test has trained. The
#   variable is deprecated (enable_op_determinism replaces it) but is the only setting workers inherit;
# - plain float32 instead of TensorFloat-32 (default on Ampere and later GPUs, about 1e-3 relative
#   error), for the tests that check gradients and outputs to 1e-4 .. 1e-6.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np
import pytest

from sieeg.config import ExperimentConfig, apply_overrides
from sieeg.data.synthetic import make_synthetic_featureset, make_synthetic_trials

#: Small model / short training so tests that train stay fast on a laptop CPU.
TINY_OVERRIDES = [
    "model.conv_blocks=[[4], [8]]",
    "model.dense_units=[32, 16]",
    "model.classifier_hidden=8",
    "model.discriminator_hidden=8",
    "train.epochs=2",
    "train.batch_size=16",
    "protocol.n_val_subjects=1",
]


@pytest.fixture
def tiny_config() -> ExperimentConfig:
    return apply_overrides(ExperimentConfig(), TINY_OVERRIDES)


@pytest.fixture
def small_featureset():
    """6 subjects x 12 trials, 32 channels x 6 bands."""
    return make_synthetic_featureset(n_subjects=6, n_trials=12, seed=0)


@pytest.fixture
def source_target_featuresets():
    """A source / target pair with a database shift, for cross-database tests."""
    source = make_synthetic_featureset(n_subjects=5, n_trials=12, dataset="src", seed=1, subject_prefix="A")
    target = make_synthetic_featureset(n_subjects=3, n_trials=12, dataset="tgt", seed=2, domain_shift=0.5, subject_prefix="B")
    return source, target


@pytest.fixture
def raw_trials():
    """Raw synthetic trials (2 subjects x 3 trials, 30 s at 128 Hz)."""
    return list(make_synthetic_trials(n_subjects=2, n_trials=3, duration=30.0, sfreq=128.0, seed=0))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
