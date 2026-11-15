"""Raw trials -> per-trial log band power (paper Sec. 4; docs/DESIGN.md D1-D5).

Every trial gives one sample: the paper uses "single-trial classification ... to avoid
mixing samples from training and testing sets" (Sec. 5.3), so the Welch PSD is
estimated over the whole stimulus rather than over sliding windows.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from fractions import Fraction

import numpy as np
from scipy import signal

from .. import __version__
from ..config import FeatureConfig, config_to_dict
from ..data.trial import Trial
from .featureset import FeatureSet
from .psd import band_log_power, check_band_edges, welch_psd

logger = logging.getLogger(__name__)


def preprocess(trial: Trial, cfg: FeatureConfig) -> Trial:
    """Pick channels, re-reference, optionally band-pass and resample one trial (DESIGN D4, D5).

    1. Pick ``cfg.channels`` (case-insensitive, in config order); the result carries the
       config's channel names, so every dataset ends up with identical names.
    2. ``cfg.reference == "average"``: subtract the mean over the picked channels.
    3. If ``cfg.l_freq`` or ``cfg.h_freq`` is set: :func:`mne.filter.filter_data`.
    4. Resample to ``cfg.resample_sfreq`` with :func:`scipy.signal.resample_poly`
       (rational ``up / down``) if the rate differs.

    Raises:
        KeyError: listing the configured channels that the trial lacks.
        ValueError: for NaN / inf samples in the picked channels (they would poison the
            per-subject normalisation) or a reference other than ``"average"`` or ``None``.
    """
    lookup = {name.lower(): i for i, name in enumerate(trial.ch_names)}
    missing = [c for c in cfg.channels if c.lower() not in lookup]
    if missing:
        raise KeyError(f"{trial.dataset} subject {trial.subject} trial {trial.trial} lacks channels {missing}")
    data = trial.data[[lookup[c.lower()] for c in cfg.channels]]
    if not np.isfinite(data).all():
        raise ValueError(f"{trial.dataset} subject {trial.subject} trial {trial.trial} has non-finite samples")

    if cfg.reference == "average":
        data = data - data.mean(axis=0, keepdims=True)
    elif cfg.reference is not None:
        raise ValueError(f"Unknown reference {cfg.reference!r}; use 'average' or None")

    sfreq = trial.sfreq
    if cfg.l_freq is not None or cfg.h_freq is not None:
        import mne

        data = mne.filter.filter_data(data, sfreq, cfg.l_freq, cfg.h_freq, verbose="warning")

    if cfg.resample_sfreq is not None and cfg.resample_sfreq != sfreq:
        ratio = Fraction(cfg.resample_sfreq / sfreq).limit_denominator(1000)
        data = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
        sfreq = sfreq * ratio.numerator / ratio.denominator

    return dataclasses.replace(trial, data=data, sfreq=sfreq, ch_names=list(cfg.channels), meta=dict(trial.meta))


def trial_band_power(trial: Trial, cfg: FeatureConfig) -> np.ndarray:
    """``(n_channels, n_bands)`` log10 band power of one trial: preprocess -> Welch -> bands (Sec. 4).

    Raises:
        ValueError: if a band reaches above the Nyquist frequency of the preprocessed
            (resampled) trial (:func:`~sieeg.features.psd.check_band_edges`).
    """
    clean = preprocess(trial, cfg)
    check_band_edges(cfg.bands, clean.sfreq)
    freqs, psd = welch_psd(clean.data, clean.sfreq, segment_sec=cfg.welch_segment_sec, overlap=cfg.welch_overlap)
    return band_log_power(freqs, psd, cfg.bands)


def _skip(trial: Trial, reason: str) -> dict[str, str]:
    """Log a skipped trial and return its ``meta["skipped"]`` record."""
    logger.warning("Skipping %s subject %s trial %s: %s", trial.dataset, trial.subject, trial.trial, reason)
    return {"subject": trial.subject, "trial": trial.trial, "reason": reason}


def extract_features(
    trials: Iterable[Trial],
    cfg: FeatureConfig,
    *,
    dataset: str | None = None,
    on_error: str = "warn",
    progress: bool = True,
    loader_skipped: list[dict[str, str]] | None = None,
) -> FeatureSet:
    """Log band power of every trial as a :class:`FeatureSet` (one sample per trial).

    Trials shorter than ``cfg.min_duration_sec`` are skipped with a warning. A trial whose
    extraction raises is skipped with a warning (``on_error="warn"``) or re-raised
    (``"raise"``). Skipped trials are listed in ``meta["skipped"]`` with the reason.
    ``loader_skipped`` is the ``skipped`` list given to the loader that produces ``trials``
    (``iter_manifest_trials`` / ``iter_mahnob_trials``): its records, read once ``trials``
    is exhausted, come first in ``meta["skipped"]`` and count in ``meta["n_input_trials"]``.
    The dataset name is taken from the first trial unless ``dataset`` is given.
    ``progress`` logs every extracted trial at INFO level.

    Raises:
        ValueError: if ``on_error`` is unknown, a band reaches above the Nyquist frequency of
            ``cfg.resample_sfreq``, or no trial yields features.
    """
    if on_error not in ("warn", "raise"):
        raise ValueError(f"on_error must be 'warn' or 'raise', got {on_error!r}")
    if cfg.resample_sfreq is not None:
        check_band_edges(cfg.bands, cfg.resample_sfreq)  # a config error: fail before reading any trial
    rows: list[np.ndarray] = []
    subjects: list[str] = []
    trial_ids: list[str] = []
    valence: list[float] = []
    arousal: list[float] = []
    skipped: list[dict[str, str]] = []
    n_input = 0
    for trial in trials:
        n_input += 1
        if dataset is None:
            dataset = trial.dataset
        if trial.duration < cfg.min_duration_sec:
            reason = f"duration {trial.duration:.2f} s < min_duration_sec {cfg.min_duration_sec:g} s"
            skipped.append(_skip(trial, reason))
            continue
        try:
            power = trial_band_power(trial, cfg)
        except Exception as exc:
            if on_error == "raise":
                raise
            skipped.append(_skip(trial, f"{type(exc).__name__}: {exc}"))
            continue
        rows.append(power)
        subjects.append(trial.subject)
        trial_ids.append(trial.trial)
        valence.append(trial.valence)
        arousal.append(trial.arousal)
        if progress:
            logger.info("[%d] %s subject %s trial %s", len(rows), trial.dataset, trial.subject, trial.trial)

    # Loaders fill their list lazily, so it is complete only now that ``trials`` is exhausted.
    dropped_by_loader = [dict(record) for record in loader_skipped or ()]
    skipped = dropped_by_loader + skipped
    n_input += len(dropped_by_loader)
    if not rows:
        raise ValueError(f"No features extracted from {n_input} input trials ({len(skipped)} skipped)")
    if progress:
        logger.info("Extracted %d of %d trials (%d skipped)", len(rows), n_input, len(skipped))
    return FeatureSet(
        dataset=dataset,
        band_power=np.stack(rows),
        subjects=np.array(subjects),
        trials=np.array(trial_ids),
        valence=np.array(valence),
        arousal=np.array(arousal),
        ch_names=list(cfg.channels),
        bands=list(cfg.bands),
        meta={
            "feature_config": config_to_dict(cfg),
            "skipped": skipped,
            "n_input_trials": n_input,
            "sieeg_version": __version__,
        },
    )
