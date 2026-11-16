"""Per-trial log band-power features for one dataset, with metadata and npz I/O.

A :class:`FeatureSet` is the hand-off point between feature extraction (slow,
done once per dataset) and the experiments (fast, repeated). It stores the
*un-normalised* log10 band power so every normalisation choice can be made
later without re-reading raw EEG.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import check_band_names, check_channel_names, check_dataset_name, natural_sorted, qualify

FORMAT_VERSION = 1
LABELS = ("valence", "arousal")

#: Extraction settings that must agree before two databases are compared or combined.
PIPELINE_KEYS = ("resample_sfreq", "reference", "l_freq", "h_freq", "welch_segment_sec", "welch_overlap")


def feature_config_differences(a: FeatureSet, b: FeatureSet) -> list[str]:
    """Extraction settings that differ between two feature files, as readable strings.

    Compares ``meta["feature_config"]`` (written by ``extract_features``) on :data:`PIPELINE_KEYS`,
    and the channel set when both used an average reference (which depends on it). Returns an
    empty list when either file has no recorded extraction config (e.g. synthetic features).
    """
    ca, cb = a.meta.get("feature_config"), b.meta.get("feature_config")
    if not isinstance(ca, dict) or not isinstance(cb, dict):
        return []
    diffs = [
        f"{key}: {a.dataset}={ca.get(key)!r} vs {b.dataset}={cb.get(key)!r}"
        for key in PIPELINE_KEYS
        if ca.get(key) != cb.get(key)
    ]
    if ca.get("reference") == cb.get("reference") == "average":
        set_a = {str(c).lower() for c in ca.get("channels", ())}
        set_b = {str(c).lower() for c in cb.get("channels", ())}
        if set_a != set_b:
            diffs.append(f"average-reference channel sets differ ({len(set_a)} vs {len(set_b)} channels)")
    return diffs


def zscore_by_group(x: np.ndarray, groups: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score ``x`` along axis 0 separately within each group (e.g. each subject).

    Features with (near) zero variance inside a group are only mean-centred.
    """
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray(groups)
    out = np.empty_like(x)
    for g in np.unique(groups):
        m = groups == g
        mu = x[m].mean(axis=0)
        sd = x[m].std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        out[m] = (x[m] - mu) / sd
    return out.astype(np.float32)


@dataclass
class FeatureSet:
    """Log band power of every trial in one dataset.

    Attributes:
        dataset: dataset name (``"mahnob"``, ``"daief"``, ...); no ``":"``, so that
            :attr:`qualified_subjects` are unique (:func:`~sieeg.utils.check_dataset_name`).
        band_power: float32 array ``(n_trials, n_channels, n_bands)`` of log10 band power.
        subjects, trials: string arrays ``(n_trials,)``.
        valence, arousal: float32 ratings ``(n_trials,)``, NaN when missing.
        ch_names: channel names (axis 1); unique ignoring case.
        bands: ``(name, low_hz, high_hz)`` per band (axis 2); the names are unique.
        meta: JSON-serialisable provenance (extraction config, versions, ...).
    """

    dataset: str
    band_power: np.ndarray
    subjects: np.ndarray
    trials: np.ndarray
    valence: np.ndarray
    arousal: np.ndarray
    ch_names: list[str]
    bands: list[tuple[str, float, float]]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset = check_dataset_name(self.dataset)
        self.band_power = np.asarray(self.band_power, dtype=np.float32)
        self.subjects = np.asarray(self.subjects).astype(str)
        self.trials = np.asarray(self.trials).astype(str)
        self.valence = np.asarray(self.valence, dtype=np.float32)
        self.arousal = np.asarray(self.arousal, dtype=np.float32)
        self.ch_names = check_channel_names(self.ch_names)
        self.bands = [(str(b[0]), float(b[1]), float(b[2])) for b in self.bands]
        check_band_names(self.bands)
        n = self.band_power.shape[0]
        if self.band_power.ndim != 3:
            raise ValueError(f"band_power must be (n_trials, n_channels, n_bands), got {self.band_power.shape}")
        if self.band_power.shape[1:] != (len(self.ch_names), len(self.bands)):
            raise ValueError(
                f"band_power shape {self.band_power.shape} does not match "
                f"{len(self.ch_names)} channels x {len(self.bands)} bands"
            )
        for name in ("subjects", "trials", "valence", "arousal"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {getattr(self, name).shape}")

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self.band_power.shape[0])

    @property
    def n_channels(self) -> int:
        return len(self.ch_names)

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def band_names(self) -> list[str]:
        return [b[0] for b in self.bands]

    @property
    def subject_ids(self) -> list[str]:
        """Unique subject ids, naturally sorted."""
        return natural_sorted(self.subjects)

    @property
    def qualified_subjects(self) -> np.ndarray:
        """``"<dataset>:<subject>"`` per trial, unique across datasets (:func:`~sieeg.utils.qualify`)."""
        return np.array([qualify(self.dataset, s) for s in self.subjects])

    def __repr__(self) -> str:
        return (
            f"FeatureSet(dataset={self.dataset!r}, trials={len(self)}, subjects={len(self.subject_ids)}, "
            f"channels={self.n_channels}, bands={self.band_names})"
        )

    # ---------------------------------------------------------------- selection
    def subset(self, index: np.ndarray | Sequence[int] | slice) -> FeatureSet:
        """Rows selected by a boolean mask, integer indices or slice."""
        return FeatureSet(
            dataset=self.dataset,
            band_power=self.band_power[index],
            subjects=self.subjects[index],
            trials=self.trials[index],
            valence=self.valence[index],
            arousal=self.arousal[index],
            ch_names=list(self.ch_names),
            bands=list(self.bands),
            meta=dict(self.meta),
        )

    def select_subjects(self, subjects: Iterable[str]) -> FeatureSet:
        wanted = {str(s) for s in subjects}
        missing = wanted - set(self.subjects.tolist())
        if missing:
            raise KeyError(f"Subjects not in {self.dataset}: {natural_sorted(missing)}")
        return self.subset(np.isin(self.subjects, list(wanted)))

    def drop_subjects(self, subjects: Iterable[str]) -> FeatureSet:
        unwanted = [str(s) for s in subjects]
        return self.subset(~np.isin(self.subjects, unwanted))

    def pick_channels(self, ch_names: Sequence[str]) -> FeatureSet:
        """Keep (and reorder to) ``ch_names``; matching is case-insensitive.

        Raises:
            KeyError: listing the requested channels the set lacks.
            ValueError: if a channel is requested twice (ignoring case).
        """
        ch_names = check_channel_names(ch_names, "Requested channels")
        lookup = {c.lower(): i for i, c in enumerate(self.ch_names)}
        missing = [c for c in ch_names if c.lower() not in lookup]
        if missing:
            raise KeyError(f"Channels not in {self.dataset}: {missing}")
        idx = [lookup[c.lower()] for c in ch_names]
        picked = [self.ch_names[i] for i in idx]
        return replace(self, band_power=self.band_power[:, idx, :], ch_names=picked, meta=dict(self.meta))

    # ------------------------------------------------------------------ labels
    def ratings(self, label: str) -> np.ndarray:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {label!r}")
        return getattr(self, label)

    def has_rating(self, label: str) -> np.ndarray:
        return np.isfinite(self.ratings(label))

    def labels(self, label: str, threshold: float = 5.0) -> np.ndarray:
        """Binary labels: 1 if rating > threshold (high arousal / positive valence), else 0.

        Raises if any rating is missing; filter with :meth:`has_rating` first.
        """
        r = self.ratings(label)
        if not np.all(np.isfinite(r)):
            raise ValueError(f"{int((~np.isfinite(r)).sum())} trials have no {label} rating; filter with has_rating()")
        return (r > threshold).astype(np.int64)

    # ------------------------------------------------------------ normalisation
    def normalized(self, method: str = "subject") -> np.ndarray:
        """Band power as float32 ``(n_trials, n_channels, n_bands)``.

        ``method="subject"`` z-scores every (channel, band) feature within each
        participant (paper, Sec. 5.3); ``"none"`` returns the raw log power.
        """
        if method == "subject":
            flat = self.band_power.reshape(len(self), self.n_channels * self.n_bands)  # -1 fails for 0 trials
            return zscore_by_group(flat, self.subjects).reshape(self.band_power.shape)
        if method == "none":
            return self.band_power.astype(np.float32, copy=True)
        raise ValueError(f"Unknown normalization {method!r}; use 'subject' or 'none'")

    def flat_features(self, method: str = "subject") -> np.ndarray:
        """``(n_trials, n_channels * n_bands)`` matrix for classic classifiers."""
        return self.normalized(method).reshape(len(self), self.n_channels * self.n_bands)

    # ---------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        """Write a compressed ``.npz`` and return the path actually written (numpy appends ``.npz``)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array(FORMAT_VERSION),
            dataset=np.array(self.dataset),
            band_power=self.band_power,
            subjects=self.subjects.astype(str),
            trials=self.trials.astype(str),
            valence=self.valence,
            arousal=self.arousal,
            ch_names=np.array(self.ch_names, dtype=str),
            band_names=np.array(self.band_names, dtype=str),
            band_edges=np.array([[b[1], b[2]] for b in self.bands], dtype=np.float64),
            meta=np.array(json.dumps(self.meta, default=str)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureSet:
        with np.load(Path(path), allow_pickle=False) as z:
            version = int(z["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"{path}: feature format v{version}, expected v{FORMAT_VERSION}")
            edges = z["band_edges"]
            try:
                return cls(
                    dataset=str(z["dataset"]),
                    band_power=z["band_power"],
                    subjects=z["subjects"],
                    trials=z["trials"],
                    valence=z["valence"],
                    arousal=z["arousal"],
                    ch_names=[str(c) for c in z["ch_names"]],
                    bands=[(str(n), float(lo), float(hi)) for n, (lo, hi) in zip(z["band_names"], edges)],
                    meta=json.loads(str(z["meta"])),
                )
            except ValueError as err:  # name the file, as the format-version error does
                raise ValueError(f"{path}: {err}") from err
