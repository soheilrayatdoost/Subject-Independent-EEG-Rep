"""Synthetic EEG for tests and smoke runs (no real data needed).

Both generators share one latent model of per-trial log10 band power::

    x[t] = base + offset[s] + a[t] * style[s] + gain[s] * k * (v[t] * P_val + r[t] * P_aro) + domain + noise

* ``offset[s]``  subject-specific mean. Per-subject z-scoring removes it, but it makes the
  raw features highly subject-identifiable (cf. the paper's Experiment 1).
* ``style[s]``   subject-specific pattern with a random per-trial amplitude ``a[t]``. It
  survives z-scoring, so a learned representation can still encode subject identity.
* ``P_val, P_aro`` emotion patterns shared by every subject *and every dataset* (fixed
  seed), so cross-dataset transfer is possible; ``domain_shift`` perturbs them per dataset.
* ``v[t], r[t]`` centred valence / arousal ratings ((rating - 5) / 2).

:func:`make_synthetic_featureset` returns features directly (fast; used by model and
protocol tests). :func:`make_synthetic_trials` renders the same latent model into raw
multichannel signals so the full extraction pipeline can be exercised.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from ..config import BIOSEMI32_CHANNELS, PAPER_BANDS
from ..features.featureset import FeatureSet
from .trial import Trial

_PATTERN_SEED = 20210606  # emotion patterns are shared across synthetic datasets


def _unit(p: np.ndarray) -> np.ndarray:
    return p / (np.sqrt(np.mean(p**2)) + 1e-12)


def _latent_rows(
    n_subjects: int,
    n_trials: int,
    n_channels: int,
    n_bands: int,
    *,
    emotion_effect: float,
    subject_offset: float,
    subject_style: float,
    domain_shift: float,
    noise: float,
    seed: int,
    subject_prefix: str,
) -> list[tuple[str, str, float, float, np.ndarray]]:
    pattern_rng = np.random.default_rng(_PATTERN_SEED)
    p_val = _unit(pattern_rng.standard_normal((n_channels, n_bands)))
    p_aro = _unit(pattern_rng.standard_normal((n_channels, n_bands)))

    rng = np.random.default_rng(seed)
    if domain_shift > 0:
        p_val = _unit(p_val + domain_shift * rng.standard_normal(p_val.shape))
        p_aro = _unit(p_aro + domain_shift * rng.standard_normal(p_aro.shape))
        domain = domain_shift * rng.standard_normal((n_channels, n_bands))
    else:
        domain = np.zeros((n_channels, n_bands))
    base = np.tile(np.linspace(1.0, -1.0, n_bands), (n_channels, 1))

    rows = []
    for s in range(n_subjects):
        offset = subject_offset * rng.standard_normal((n_channels, n_bands))
        style = _unit(rng.standard_normal((n_channels, n_bands)))
        gain = rng.uniform(0.6, 1.4)
        for t in range(n_trials):
            val = int(rng.integers(1, 10))
            aro = int(rng.integers(1, 10))
            v, r = (val - 5) / 2.0, (aro - 5) / 2.0
            a = subject_style * rng.standard_normal()
            x = (
                base
                + offset
                + a * style
                + gain * emotion_effect * (v * p_val + r * p_aro)
                + domain
                + noise * rng.standard_normal((n_channels, n_bands))
            )
            rows.append((f"{subject_prefix}{s + 1:02d}", f"T{t + 1:02d}", float(val), float(aro), x))
    return rows


def make_synthetic_featureset(
    n_subjects: int = 10,
    n_trials: int = 20,
    *,
    dataset: str = "synthetic",
    channels: Sequence[str] = BIOSEMI32_CHANNELS,
    bands: Sequence[tuple[str, float, float]] = PAPER_BANDS,
    emotion_effect: float = 0.25,
    subject_offset: float = 2.0,
    subject_style: float = 1.0,
    domain_shift: float = 0.0,
    noise: float = 1.0,
    seed: int = 0,
    subject_prefix: str = "S",
) -> FeatureSet:
    """Feature-level synthetic dataset (log10 band power), see module docstring."""
    rows = _latent_rows(
        n_subjects,
        n_trials,
        len(channels),
        len(bands),
        emotion_effect=emotion_effect,
        subject_offset=subject_offset,
        subject_style=subject_style,
        domain_shift=domain_shift,
        noise=noise,
        seed=seed,
        subject_prefix=subject_prefix,
    )
    return FeatureSet(
        dataset=dataset,
        band_power=np.stack([r[4] for r in rows]).astype(np.float32),
        subjects=np.array([r[0] for r in rows]),
        trials=np.array([r[1] for r in rows]),
        valence=np.array([r[2] for r in rows], dtype=np.float32),
        arousal=np.array([r[3] for r in rows], dtype=np.float32),
        ch_names=list(channels),
        bands=list(bands),
        meta={
            "generator": "make_synthetic_featureset",
            "n_subjects": n_subjects,
            "n_trials": n_trials,
            "emotion_effect": emotion_effect,
            "subject_offset": subject_offset,
            "subject_style": subject_style,
            "domain_shift": domain_shift,
            "noise": noise,
            "seed": seed,
        },
    )


def make_synthetic_trials(
    n_subjects: int = 3,
    n_trials: int = 4,
    *,
    dataset: str = "synthetic",
    channels: Sequence[str] = BIOSEMI32_CHANNELS,
    bands: Sequence[tuple[str, float, float]] = PAPER_BANDS,
    sfreq: float = 128.0,
    duration: float = 40.0,
    emotion_effect: float = 0.25,
    subject_offset: float = 0.5,
    subject_style: float = 0.5,
    domain_shift: float = 0.0,
    noise: float = 0.2,
    amplitude_volts: float = 20e-6,
    seed: int = 0,
    subject_prefix: str = "S",
) -> Iterator[Trial]:
    """Raw multichannel trials whose band power follows the latent model.

    Each channel is pink-ish Gaussian noise whose spectrum is scaled inside every
    band by ``10 ** (x / 2)`` (amplitude), so log10 band power shifts by ``x``
    relative to the 1/f background. Overlapping bands (slow alpha within alpha)
    multiply their gains.
    """
    rows = _latent_rows(
        n_subjects,
        n_trials,
        len(channels),
        len(bands),
        emotion_effect=emotion_effect,
        subject_offset=subject_offset,
        subject_style=subject_style,
        domain_shift=domain_shift,
        noise=noise,
        seed=seed,
        subject_prefix=subject_prefix,
    )
    n_times = int(round(duration * sfreq))
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    pink = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    signal_rng = np.random.default_rng(seed + 1)
    for subject, trial, val, aro, x in rows:
        spec = np.fft.rfft(signal_rng.standard_normal((len(channels), n_times)), axis=1) * pink
        for b, (_, lo, hi) in enumerate(bands):
            in_band = (freqs >= lo) & (freqs < hi)
            spec[:, in_band] *= (10.0 ** (x[:, b] / 2.0))[:, None]
        data = np.fft.irfft(spec, n=n_times, axis=1)
        data *= amplitude_volts / (data.std() + 1e-12)
        yield Trial(
            dataset=dataset,
            subject=subject,
            trial=trial,
            data=data,
            sfreq=sfreq,
            ch_names=list(channels),
            valence=val,
            arousal=aro,
            meta={"generator": "make_synthetic_trials", "latent_log_power": x.tolist()},
        )
