"""Synthetic EEG generators (sieeg.data.synthetic): shapes, determinism and the latent model they promise."""

from __future__ import annotations

import numpy as np
import pytest

from sieeg.config import BIOSEMI32_CHANNELS, PAPER_BANDS
from sieeg.data.synthetic import make_synthetic_featureset, make_synthetic_trials
from sieeg.features.featureset import FeatureSet

# A band centred on 10 Hz has the base -log10(10 / 10) = 0, so without effects its latent log power is exactly 0.
ALPHA_ONLY = {"n_subjects": 1, "n_trials": 2, "bands": [("alpha", 8.0, 12.0)], "duration": 10.0, "seed": 4}
NO_EFFECTS = {"emotion_effect": 0.0, "subject_offset": 0.0, "subject_style": 0.0, "noise": 0.0}


def _valence_pattern(fs: FeatureSet) -> np.ndarray:
    """Correlation of every per-subject z-scored feature with valence."""
    z = fs.flat_features("subject").astype(np.float64)
    z -= z.mean(axis=0)
    v = fs.valence - fs.valence.mean()
    return (z * v[:, None]).sum(axis=0) / (np.linalg.norm(z, axis=0) * np.linalg.norm(v))


def _held_out_alpha_valence_r(fs: FeatureSet) -> float:
    """Valence correlation of the z-scored alpha power on half the subjects, weighted by the other half."""
    alpha = fs.band_names.index("alpha")
    half = len(fs.subject_ids) // 2
    fit, test = fs.select_subjects(fs.subject_ids[:half]), fs.select_subjects(fs.subject_ids[half:])
    z_fit, z_test = (part.normalized("subject")[:, :, alpha].astype(np.float64) for part in (fit, test))
    weights = np.array([np.corrcoef(z_fit[:, c], fit.valence)[0, 1] for c in range(fs.n_channels)])
    return float(np.corrcoef(z_test @ weights, test.valence)[0, 1])


def test_featureset_shapes_ids_and_ratings():
    fs = make_synthetic_featureset(n_subjects=3, n_trials=4, dataset="syn", seed=0, subject_prefix="P")
    assert fs.band_power.shape == (12, len(BIOSEMI32_CHANNELS), len(PAPER_BANDS))
    assert (fs.dataset, fs.ch_names, fs.bands) == ("syn", list(BIOSEMI32_CHANNELS), list(PAPER_BANDS))
    assert fs.subject_ids == ["P01", "P02", "P03"]
    assert fs.trials[:4].tolist() == ["T01", "T02", "T03", "T04"]
    for ratings in (fs.valence, fs.arousal):
        assert set(np.unique(ratings)) <= set(range(1, 10))  # integer SAM ratings 1-9
    assert fs.meta["generator"] == "make_synthetic_featureset" and fs.meta["seed"] == 0
    custom = make_synthetic_featureset(2, 3, channels=["Cz", "Pz"], bands=[("alpha", 8.0, 12.0)])
    assert custom.band_power.shape == (6, 2, 1)


def test_featureset_is_deterministic_per_seed():
    a, b = make_synthetic_featureset(4, 5, seed=3), make_synthetic_featureset(4, 5, seed=3)
    np.testing.assert_array_equal(a.band_power, b.band_power)
    np.testing.assert_array_equal(a.valence, b.valence)
    assert not np.array_equal(a.band_power, make_synthetic_featureset(4, 5, seed=4).band_power)


def test_subject_offsets_make_raw_features_subject_specific():
    fs = make_synthetic_featureset(6, 20, subject_style=0.0, emotion_effect=0.0, seed=1)
    flat = fs.flat_features("none")
    means = np.array([flat[fs.subjects == s].mean(axis=0) for s in fs.subject_ids])
    within = np.mean([flat[fs.subjects == s].std(axis=0).mean() for s in fs.subject_ids])
    assert means.std(axis=0).mean() > 1.3 * within  # subject_offset 2.0 vs noise 1.0 (about 1.75x)
    z = fs.normalized("subject").reshape(len(fs), -1)
    np.testing.assert_allclose([z[fs.subjects == s].mean() for s in fs.subject_ids], 0.0, atol=1e-6)


def test_emotion_signal_survives_per_subject_z_scoring():
    with_effect = make_synthetic_featureset(10, 20, seed=0)
    assert _held_out_alpha_valence_r(with_effect) > 0.4  # alpha of new subjects predicts their valence
    without = make_synthetic_featureset(10, 20, seed=0, emotion_effect=0.0)
    assert abs(_held_out_alpha_valence_r(without)) < 0.25


def test_emotion_patterns_are_shared_across_datasets_until_a_domain_shift():
    a = make_synthetic_featureset(10, 20, dataset="a", seed=10)
    same = make_synthetic_featureset(10, 20, dataset="b", seed=20)
    shifted = make_synthetic_featureset(10, 20, dataset="b", seed=20, domain_shift=2.0)
    assert np.corrcoef(_valence_pattern(a), _valence_pattern(same))[0, 1] > 0.75
    assert np.corrcoef(_valence_pattern(a), _valence_pattern(shifted))[0, 1] < 0.6
    offset = np.sqrt(np.mean((a.band_power.mean(axis=0) - shifted.band_power.mean(axis=0)) ** 2))
    no_offset = np.sqrt(np.mean((a.band_power.mean(axis=0) - same.band_power.mean(axis=0)) ** 2))
    assert offset > no_offset + 0.5  # the shift also moves the database mean


def test_trials_render_the_same_latent_model():
    kwargs = {"emotion_effect": 0.3, "subject_offset": 0.5, "subject_style": 0.5, "noise": 0.2, "seed": 2}
    trials = list(make_synthetic_trials(2, 3, sfreq=128.0, duration=20.0, **kwargs))
    fs = make_synthetic_featureset(2, 3, **kwargs)
    assert [(t.subject, t.trial) for t in trials] == list(zip(fs.subjects, fs.trials))
    np.testing.assert_allclose(np.array([t.meta["latent_log_power"] for t in trials]), fs.band_power, rtol=1e-6)
    np.testing.assert_array_equal([t.valence for t in trials], fs.valence)
    for trial in trials:
        assert trial.data.shape == (32, 2560) and trial.sfreq == 128.0 and trial.ch_names == list(BIOSEMI32_CHANNELS)
    again = list(make_synthetic_trials(2, 3, sfreq=128.0, duration=20.0, **kwargs))
    np.testing.assert_array_equal(trials[-1].data, again[-1].data)


def test_a_reordered_or_reduced_montage_keeps_each_named_channel_and_band():
    """Values follow channel / band names, not array positions, so montages can be aligned by name."""
    full = make_synthetic_featureset(n_subjects=4, n_trials=5, seed=3)
    channels, bands = list(BIOSEMI32_CHANNELS[::-3]), list(PAPER_BANDS[::-1])
    part = make_synthetic_featureset(n_subjects=4, n_trials=5, seed=3, channels=channels, bands=bands)
    np.testing.assert_array_equal(part.valence, full.valence)  # the ratings do not depend on the montage
    np.testing.assert_array_equal(part.arousal, full.arousal)
    order = [full.band_names.index(name) for name in part.band_names]
    np.testing.assert_array_equal(part.band_power, full.pick_channels(channels).band_power[:, :, order])
    # Channel names are matched case-insensitively, as everywhere else in sieeg.
    upper = make_synthetic_featureset(n_subjects=4, n_trials=5, seed=3, channels=[c.upper() for c in channels])
    np.testing.assert_array_equal(upper.band_power, full.pick_channels(channels).band_power)


def test_raw_trials_of_a_reordered_or_reduced_montage_render_the_same_signals():
    kwargs = {"n_subjects": 1, "n_trials": 2, "duration": 10.0, "seed": 5}
    full = list(make_synthetic_trials(**kwargs))
    channels = list(BIOSEMI32_CHANNELS[::-3])  # 11 of the 32 channels, in reverse order
    part = list(make_synthetic_trials(channels=channels, **kwargs))
    for a, b in zip(full, part):
        assert b.ch_names == channels
        # The signals as rendered, not re-normalised; numpy's FFT may round rows in SIMD groups differently.
        np.testing.assert_allclose(b.data, a.data[::-3], rtol=0, atol=1e-12 * 20e-6)


def test_amplitude_volts_is_the_rms_of_the_background():
    """Without effects, a band centred on 10 Hz leaves the trials as the bare 1/f background."""
    background = list(make_synthetic_trials(**ALPHA_ONLY, **NO_EFFECTS))
    assert not np.any([t.meta["latent_log_power"] for t in background])
    # 64 channel-trials of 10 s: the pooled RMS has a standard deviation of about 0.5 %.
    np.testing.assert_allclose(np.sqrt(np.mean([t.data**2 for t in background])), 20e-6, rtol=0.03)
    doubled = list(make_synthetic_trials(amplitude_volts=40e-6, **ALPHA_ONLY, **NO_EFFECTS))
    np.testing.assert_array_equal(doubled[-1].data, 2.0 * background[-1].data)


def test_latent_values_shift_the_power_inside_their_band_and_nothing_else():
    """The gain is one constant, so a trial's latent values do not rescale the rest of its spectrum."""
    freqs = np.fft.rfftfreq(1280, d=1.0 / 128.0)
    in_band = (freqs >= 8.0) & (freqs < 12.0)
    background = make_synthetic_trials(**ALPHA_ONLY, **NO_EFFECTS)
    for reference, trial in zip(background, make_synthetic_trials(**ALPHA_ONLY)):  # the same white noise
        ref_power, power = (np.abs(np.fft.rfft(t.data, axis=1)) ** 2 for t in (reference, trial))
        np.testing.assert_allclose(power[:, ~in_band].sum(axis=1), ref_power[:, ~in_band].sum(axis=1), rtol=1e-9)
        shift = np.log10(power[:, in_band].sum(axis=1) / ref_power[:, in_band].sum(axis=1))
        np.testing.assert_allclose(shift, np.array(trial.meta["latent_log_power"])[:, 0], atol=1e-9)


def test_a_numpy_integer_seed_gives_the_same_data_as_a_python_int(tmp_path):
    """The streams are keyed by repr(); on numpy 2 repr(np.int64(4)) is 'np.int64(4)', not '4'."""
    a, b = make_synthetic_featureset(2, 3, seed=4), make_synthetic_featureset(2, 3, seed=np.int64(4))
    np.testing.assert_array_equal(a.band_power, b.band_power)
    assert FeatureSet.load(b.save(tmp_path / "f.npz")).meta["seed"] == 4  # not the string "4"
    kwargs = {"n_subjects": 1, "n_trials": 1, "duration": 2.0}
    x, y = next(make_synthetic_trials(seed=4, **kwargs)), next(make_synthetic_trials(seed=np.int64(4), **kwargs))
    np.testing.assert_array_equal(x.data, y.data)


@pytest.mark.parametrize("seed", [None, [1, 2]])
def test_any_seed_numpy_accepts_is_accepted(seed):
    assert len(make_synthetic_featureset(2, 3, seed=seed)) == 6
