"""Welch power spectral density and log band power (paper Sec. 4).

The paper computes the PSD "from 15 seconds time window with a 50% overlap through
the Welch method" and uses "the logarithms of PSD from six frequency bands". One PSD
is estimated per trial over the whole stimulus (docs/DESIGN.md, D1), and band power is
log10 of the mean PSD over ``[lo, hi)`` Hz (D3). Every band must lie below the Nyquist
frequency of the (resampled) signal (:func:`check_band_edges`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import signal


def welch_psd(
    data: np.ndarray,
    sfreq: float,
    *,
    segment_sec: float = 15.0,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of every channel (paper Sec. 4: 15 s windows with 50 % overlap).

    Args:
        data: signals ``(n_channels, n_times)``.
        sfreq: sampling rate in Hz.
        segment_sec: segment length in seconds. A signal shorter than one segment is
            analysed as a single segment of ``n_times`` samples.
        overlap: fraction of a segment shared by consecutive segments.
        window: window name for :func:`scipy.signal.welch`.

    Returns:
        ``(freqs, psd)``: ``(n_freqs,)`` frequencies in Hz and the one-sided density
        ``(n_channels, n_freqs)`` in (data unit)^2 / Hz.
    """
    data = np.asarray(data, dtype=np.float64)
    nperseg = min(int(round(segment_sec * sfreq)), data.shape[-1])
    # Rounding can reach nperseg for an overlap close to 1 on a short trial; scipy requires fewer.
    noverlap = min(int(round(nperseg * overlap)), max(nperseg - 1, 0))
    return signal.welch(
        data,
        fs=sfreq,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        average="mean",
        axis=-1,
    )


def check_band_edges(bands: Sequence[tuple[str, float, float]], sfreq: float) -> None:
    """Raise if a band reaches above the Nyquist frequency ``sfreq / 2`` (e.g. gamma 30-45 Hz at 64 Hz).

    :func:`band_log_power` would silently average only the bins below Nyquist, which also lie
    in the resampler's anti-aliasing roll-off, while the feature file still names the full band.

    Raises:
        ValueError: naming every such band.
    """
    nyquist = sfreq / 2.0
    above = [f"{name!r} [{lo:g}, {hi:g}) Hz" for name, lo, hi in bands if hi > nyquist * (1.0 + 1e-9)]
    if above:
        raise ValueError(
            f"Band(s) {', '.join(above)} reach above the Nyquist frequency ({nyquist:g} Hz) of a {sfreq:g} Hz "
            "signal; use a higher features.resample_sfreq or lower band edges"
        )


def band_log_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Sequence[tuple[str, float, float]],
    *,
    floor: float = 1e-30,
) -> np.ndarray:
    """log10 of the mean PSD inside each band ``[lo, hi)`` (paper Sec. 4; DESIGN D3).

    Args:
        freqs: ``(n_freqs,)`` frequencies of the PSD bins in Hz.
        psd: ``(n_channels, n_freqs)`` power spectral density.
        bands: ``(name, lo_hz, hi_hz)`` per band.
        floor: lower bound on the mean power, so a flat channel gives a finite value.

    Returns:
        ``(n_channels, n_bands)`` log10 band power.

    Raises:
        ValueError: if a band contains no frequency bin.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    power = []
    for name, lo, hi in bands:
        in_band = (freqs >= lo) & (freqs < hi)
        if not in_band.any():
            raise ValueError(
                f"Band {name!r} [{lo:g}, {hi:g}) Hz contains no frequency bin "
                f"({freqs.size} bins up to {freqs.max(initial=0.0):g} Hz)"
            )
        power.append(psd[..., in_band].mean(axis=-1))
    return np.log10(np.maximum(np.stack(power, axis=-1), floor))
