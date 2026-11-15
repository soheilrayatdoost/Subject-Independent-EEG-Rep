"""The unit every dataset loader yields: one continuous EEG trial with its ratings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_rating(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


@dataclass
class Trial:
    """One stimulus-locked EEG recording.

    Attributes:
        dataset: short dataset name, e.g. ``"mahnob"`` or ``"daief"``.
        subject: subject id within the dataset (string).
        trial: trial / session id within the subject (string).
        data: EEG in volts, shape ``(n_channels, n_times)``, already cropped to the stimulus.
        sfreq: sampling rate in Hz.
        ch_names: channel names matching the rows of ``data``.
        valence, arousal: self-reported ratings (1-9 SAM scale); NaN when missing.
        meta: free-form provenance (file path, session id, stimulus, ...).
    """

    dataset: str
    subject: str
    trial: str
    data: np.ndarray
    sfreq: float
    ch_names: list[str]
    valence: float = math.nan
    arousal: float = math.nan
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset = str(self.dataset)
        self.subject = str(self.subject)
        self.trial = str(self.trial)
        self.data = np.asarray(self.data, dtype=np.float64)
        self.ch_names = [str(c) for c in self.ch_names]
        self.sfreq = float(self.sfreq)
        self.valence = _as_rating(self.valence)
        self.arousal = _as_rating(self.arousal)
        if self.data.ndim != 2:
            raise ValueError(f"Trial data must be 2-D (n_channels, n_times), got shape {self.data.shape}")
        if self.data.shape[0] != len(self.ch_names):
            raise ValueError(f"{self.data.shape[0]} data rows but {len(self.ch_names)} channel names")
        if not self.sfreq > 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")

    @property
    def n_times(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.n_times / self.sfreq
