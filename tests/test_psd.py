"""Welch PSD and log band power (paper Sec. 4; docs/DESIGN.md 6.1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from sieeg.config import PAPER_BANDS
from sieeg.features.psd import band_log_power, check_band_edges, welch_psd

SFREQ = 256.0
BAND_NAMES = [b[0] for b in PAPER_BANDS]


def test_long_signal_uses_15_s_segments_with_half_overlap(rng):
    data = rng.standard_normal((2, int(60 * SFREQ)))
    freqs, psd = welch_psd(data, SFREQ)
    ref_freqs, ref_psd = signal.welch(
        data, fs=SFREQ, window="hann", nperseg=3840, noverlap=1920,
        detrend="constant", scaling="density", average="mean",
    )
    assert psd.shape == (2, 3840 // 2 + 1)
    assert np.isclose(freqs[1] - freqs[0], 1 / 15)
    np.testing.assert_array_equal(freqs, ref_freqs)
    np.testing.assert_allclose(psd, ref_psd)


def test_signal_shorter_than_15_s_is_one_segment(rng):
    n_times = int(10 * SFREQ)
    freqs, psd = welch_psd(rng.standard_normal((3, n_times)), SFREQ)
    assert psd.shape == (3, n_times // 2 + 1)
    assert np.isclose(freqs[1], SFREQ / n_times)


@pytest.mark.parametrize(
    ("freq", "band"),
    [(2.5, "delta"), (6.0, "theta"), (9.0, "slow_alpha"), (10.0, "alpha"), (11.0, "alpha"),
     (20.0, "beta"), (38.0, "gamma")],
)
def test_sinusoid_has_highest_log_power_in_its_band(freq, band, rng):
    t = np.arange(int(40 * SFREQ)) / SFREQ
    data = 10e-6 * np.sin(2 * np.pi * freq * t) + 0.1e-6 * rng.standard_normal((2, t.size))
    power = band_log_power(*welch_psd(data, SFREQ), PAPER_BANDS)
    assert power.shape == (2, len(PAPER_BANDS))
    assert [BAND_NAMES[i] for i in power.argmax(axis=1)] == [band, band]


def test_white_noise_matches_analytic_density(rng):
    sigma = 3e-6  # volts; the one-sided density of white noise is 2 sigma^2 / sfreq
    data = sigma * rng.standard_normal((8, int(120 * SFREQ)))
    power = band_log_power(*welch_psd(data, SFREQ), PAPER_BANDS)
    expected = np.log10(2 * sigma**2 / SFREQ)
    # Estimation sd is <= 0.03 per value and <= 0.01 for the mean over 8 channels.
    np.testing.assert_allclose(power, expected, atol=0.12)
    np.testing.assert_allclose(power.mean(axis=0), expected, atol=0.04)


def test_band_power_is_log10_mean_over_half_open_band():
    freqs = np.arange(0.0, 50.0, 0.5)
    psd = np.vstack([freqs + 1.0, 2 * (freqs + 1.0)])
    power = band_log_power(freqs, psd, [("delta", 1.0, 4.0), ("theta", 4.0, 8.0)])
    # delta averages 1.0 ... 3.5 Hz (mean 2.25), theta 4.0 ... 7.5 Hz (mean 5.75): lo included, hi excluded.
    expected = np.log10([[3.25, 6.75], [6.5, 13.5]])
    np.testing.assert_allclose(power, expected)


def test_flat_channel_is_floored():
    freqs = np.arange(0.0, 50.0, 0.5)
    power = band_log_power(freqs, np.zeros((1, freqs.size)), PAPER_BANDS)
    np.testing.assert_array_equal(power, -30.0)


@pytest.mark.parametrize("band", [("ultra", 200.0, 300.0), ("narrow", 10.01, 10.05)])
def test_band_without_frequency_bins_raises(band, rng):
    freqs, psd = welch_psd(rng.standard_normal((1, int(20 * SFREQ))), SFREQ)  # bins every 1/15 Hz up to 128 Hz
    with pytest.raises(ValueError, match=band[0]):
        band_log_power(freqs, psd, [*PAPER_BANDS, band])


def test_check_band_edges_against_nyquist(rng):
    check_band_edges(PAPER_BANDS, 90.0)  # gamma ends exactly at Nyquist: every bin is there
    with pytest.raises(ValueError, match=r"'gamma' \[30, 45\) Hz reach above the Nyquist frequency \(32 Hz\) of a 64"):
        check_band_edges(PAPER_BANDS, 64.0)
    with pytest.raises(ValueError, match=r"Band\(s\) 'beta' \[12, 30\) Hz, 'gamma' \[30, 45\) Hz reach"):
        check_band_edges(PAPER_BANDS, 50.0)
    # What the check prevents: band_log_power silently averages the bins below Nyquist only.
    freqs, psd = welch_psd(rng.standard_normal((1, int(20 * 64))), 64.0)
    assert np.isfinite(band_log_power(freqs, psd, PAPER_BANDS)).all()
