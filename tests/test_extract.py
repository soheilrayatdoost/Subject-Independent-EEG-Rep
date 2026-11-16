"""Preprocessing and FeatureSet extraction (paper Sec. 4; docs/DESIGN.md 6.2)."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from sieeg import __version__
from sieeg.config import BIOSEMI32_CHANNELS, PAPER_BANDS, FeatureConfig, config_to_dict
from sieeg.data.trial import Trial
from sieeg.features.extract import extract_features, preprocess, trial_band_power
from sieeg.features.featureset import FeatureSet
from sieeg.features.psd import band_log_power, welch_psd

CHANNELS = list(BIOSEMI32_CHANNELS)


def _trial(ch_names, rng, *, sfreq=256.0, seconds=8.0, sine_hz=(), dataset="toy") -> Trial:
    """10 uV white noise per channel plus 20 uV sines shared by all channels.

    ``dataset="synthetic"`` lets it join the ``raw_trials`` fixture in one extraction (one dataset per extraction).
    """
    t = np.arange(int(seconds * sfreq)) / sfreq
    data = 10e-6 * rng.standard_normal((len(ch_names), t.size))
    for f in sine_hz:
        data += 20e-6 * np.sin(2 * np.pi * f * t)
    return Trial(dataset, "1", "1", data, sfreq, list(ch_names), valence=7, arousal=3, meta={"source": "unit"})


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# ------------------------------------------------------------------ preprocess
def test_preprocess_picks_channels_case_insensitively_in_config_order(rng):
    names = [c.upper() for c in reversed(CHANNELS)] + ["EXG1"]
    trial = _trial(names, rng)
    original = trial.data.copy()
    out = preprocess(trial, FeatureConfig(reference=None, resample_sfreq=None))
    assert out.ch_names == CHANNELS
    for i, name in enumerate(CHANNELS):
        np.testing.assert_array_equal(out.data[i], original[names.index(name.upper())])
    assert (out.sfreq, out.valence, out.arousal, out.meta) == (256.0, 7.0, 3.0, {"source": "unit"})
    np.testing.assert_array_equal(trial.data, original)  # the input trial is left untouched


def test_preprocess_missing_channels_raise_keyerror_listing_them(rng):
    trial = _trial([c for c in CHANNELS if c not in ("Cz", "O2")], rng)
    with pytest.raises(KeyError, match=r"\['Cz', 'O2'\]"):
        preprocess(trial, FeatureConfig())


def test_average_reference_gives_zero_channel_mean(rng):
    trial = _trial(CHANNELS, rng)
    out = preprocess(trial, FeatureConfig(resample_sfreq=None))
    np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-20)
    np.testing.assert_allclose(out.data, trial.data - trial.data.mean(axis=0))


def test_unknown_reference_raises(rng):
    with pytest.raises(ValueError, match="reference"):
        preprocess(_trial(CHANNELS, rng), FeatureConfig(reference="Cz"))


def test_non_finite_samples_raise_only_in_picked_channels(rng):
    trial = _trial([*CHANNELS, "EXG1"], rng)
    trial.data[-1, 10] = np.nan  # unused channel: fine
    assert preprocess(trial, FeatureConfig()).data.shape[0] == 32
    trial.data[CHANNELS.index("Pz"), 10] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        preprocess(trial, FeatureConfig())


def test_resample_512_to_256_halves_n_times(rng):
    trial = _trial(CHANNELS, rng, sfreq=512.0, seconds=10.0, sine_hz=(10.0,))
    out = preprocess(trial, FeatureConfig(reference=None, resample_sfreq=256.0))
    assert out.sfreq == 256.0
    assert out.n_times == trial.n_times // 2
    freqs, psd = welch_psd(out.data, out.sfreq)
    np.testing.assert_allclose(freqs[psd.argmax(axis=1)], 10.0)


def test_no_resampling_when_rates_match(rng):
    trial = _trial(CHANNELS, rng)
    out = preprocess(trial, FeatureConfig(reference=None, resample_sfreq=256.0))
    np.testing.assert_array_equal(out.data, trial.data)


def test_optional_lowpass_attenuates_out_of_band_power(rng):
    trial = _trial(CHANNELS[:2], rng, seconds=20.0, sine_hz=(6.0, 40.0))
    cfg = FeatureConfig(channels=tuple(CHANNELS[:2]), reference=None, resample_sfreq=None, h_freq=20.0)
    bands = [("low", 5.5, 6.5), ("high", 39.5, 40.5)]
    before = band_log_power(*welch_psd(trial.data, trial.sfreq), bands)
    after = band_log_power(*welch_psd(preprocess(trial, cfg).data, trial.sfreq), bands)
    np.testing.assert_allclose(after[:, 0], before[:, 0], atol=0.01)  # 6 Hz passes
    assert np.all(after[:, 1] < before[:, 1] - 3.0)  # 40 Hz down by more than 30 dB


# ------------------------------------------------------------------ extraction
def test_trial_band_power_shape(raw_trials):
    power = trial_band_power(raw_trials[0], FeatureConfig())
    assert power.shape == (32, 6)
    assert np.all(np.isfinite(power))


def test_extract_features_builds_featureset(raw_trials, tmp_path):
    cfg = FeatureConfig()
    fs = extract_features(raw_trials, cfg, progress=False)
    assert isinstance(fs, FeatureSet)
    assert fs.band_power.shape == (6, 32, 6)
    assert fs.dataset == "synthetic"
    assert fs.ch_names == CHANNELS
    assert fs.bands == list(PAPER_BANDS)
    assert fs.subjects.tolist() == [t.subject for t in raw_trials]
    assert fs.trials.tolist() == [t.trial for t in raw_trials]
    np.testing.assert_array_equal(fs.valence, [t.valence for t in raw_trials])
    np.testing.assert_array_equal(fs.arousal, [t.arousal for t in raw_trials])
    np.testing.assert_allclose(fs.band_power[2], trial_band_power(raw_trials[2], cfg), rtol=1e-6)
    assert fs.meta == {
        "feature_config": config_to_dict(cfg),
        "skipped": [],
        "n_input_trials": 6,
        "sieeg_version": __version__,
    }
    assert FeatureSet.load(fs.save(tmp_path / "features.npz")).meta == fs.meta  # meta is JSON-safe


def test_band_power_tracks_latent_log_power(raw_trials):
    fs = extract_features(raw_trials, FeatureConfig(), progress=False)
    latent = np.array([t.meta["latent_log_power"] for t in raw_trials])
    # Slow alpha (8-10 Hz) lies inside alpha (8-12 Hz), and the generator multiplies the
    # gains of overlapping bands, so slow-alpha bins carry both shifts.
    latent[:, :, fs.band_names.index("slow_alpha")] += latent[:, :, fs.band_names.index("alpha")]
    measured = fs.band_power - fs.band_power.mean(axis=0)
    latent -= latent.mean(axis=0)
    assert _corr(measured, latent) > 0.8
    for b in range(fs.n_bands):
        assert _corr(measured[:, :, b], latent[:, :, b]) > 0.7, fs.band_names[b]


def test_short_trials_are_skipped_and_recorded(raw_trials, caplog):
    short = dataclasses.replace(raw_trials[0], trial="short", data=raw_trials[0].data[:, : 3 * 128])
    with caplog.at_level(logging.WARNING, logger="sieeg.features.extract"):
        fs = extract_features([*raw_trials, short], FeatureConfig(), progress=False)
    assert len(fs) == 6
    assert "short" not in fs.trials
    assert fs.meta["n_input_trials"] == 7
    [record] = fs.meta["skipped"]
    assert (record["subject"], record["trial"]) == (short.subject, "short")
    assert "min_duration_sec" in record["reason"]
    assert "short" in caplog.text


def test_failing_trials_are_skipped_or_raised(raw_trials):
    broken = dataclasses.replace(raw_trials[1], trial="broken", data=raw_trials[1].data[1:], ch_names=CHANNELS[1:])
    fs = extract_features([broken, *raw_trials], FeatureConfig(), progress=False)
    assert len(fs) == 6
    [record] = fs.meta["skipped"]
    assert record["trial"] == "broken"
    assert record["reason"].startswith("KeyError") and "Fp1" in record["reason"]
    with pytest.raises(KeyError, match="Fp1"):
        extract_features([broken, *raw_trials], FeatureConfig(), on_error="raise", progress=False)


def test_dataset_name_can_be_overridden(raw_trials):
    fs = extract_features(raw_trials[:2], FeatureConfig(), dataset="custom", progress=False)
    assert fs.dataset == "custom"
    assert len(fs) == 2


def test_trials_of_two_datasets_are_rejected(raw_trials):
    """Subject ids repeat across databases (S01 in both), so one feature file would merge two people."""
    other = [dataclasses.replace(t, dataset="other") for t in raw_trials[:2]]
    too_short = dataclasses.replace(other[0], trial="short", data=other[0].data[:, :128])  # rejected, not skipped
    for mixed, trial in (([*raw_trials, *other], "T01"), ([*raw_trials, too_short], "short")):
        message = f"^other subject S01 trial {trial} follows trials of dataset synthetic: subject ids are unique"
        for kwargs in ({}, {"dataset": "renamed"}, {"on_error": "raise"}):
            with pytest.raises(ValueError, match=message):
                extract_features(mixed, FeatureConfig(), progress=False, **kwargs)


def test_dataset_names_cannot_contain_a_colon(raw_trials):
    """':' separates dataset and subject in qualified subject ids (sieeg.utils.qualify); subject ids may hold it."""
    with pytest.raises(ValueError, match=r"Dataset name 'a:b' contains ':'"):
        dataclasses.replace(raw_trials[0], dataset="a:b")
    assert dataclasses.replace(raw_trials[0], subject="b:c").subject == "b:c"
    read = []

    def trials():
        read.append(True)
        yield from raw_trials

    with pytest.raises(ValueError, match=r"Dataset name 'a:b' contains ':'"):
        extract_features(trials(), FeatureConfig(), dataset="a:b", progress=False)
    assert read == []  # a configuration error: no trial was read


def test_nothing_extracted_raises(raw_trials):
    short = dataclasses.replace(raw_trials[0], data=raw_trials[0].data[:, :128])
    with pytest.raises(ValueError, match="No features"):
        extract_features([short], FeatureConfig(), progress=False)
    with pytest.raises(ValueError, match="on_error"):
        extract_features(raw_trials, FeatureConfig(), on_error="ignore", progress=False)


def test_loader_skips_are_merged_after_the_trials_are_read(raw_trials, tmp_path):
    skipped: list[dict] = []

    def loader():  # like iter_manifest_trials / iter_mahnob_trials: records a drop only while iterating
        yield from raw_trials[:3]
        skipped.append({"subject": "S02", "trial": "T09", "reason": "FileNotFoundError: no such file"})
        yield from raw_trials[3:]

    short = dataclasses.replace(raw_trials[0], trial="short", data=raw_trials[0].data[:, : 3 * 128])
    fs = extract_features([*loader(), short], FeatureConfig(), progress=False, loader_skipped=skipped)
    assert len(fs) == 6
    assert fs.meta["n_input_trials"] == 8  # 6 extracted + 1 too short + 1 dropped by the loader
    assert [(s["trial"], s["reason"].split(":")[0]) for s in fs.meta["skipped"]] == [
        ("T09", "FileNotFoundError"), ("short", "duration 3.00 s < min_duration_sec 5 s")]
    assert FeatureSet.load(fs.save(tmp_path / "f.npz")).meta["skipped"] == fs.meta["skipped"]


def test_loader_skips_from_a_manifest(raw_trials, tmp_path, caplog):
    from sieeg.data.manifest import iter_manifest_trials

    rows = []
    for trial in raw_trials[:2]:
        name = f"{trial.subject}_{trial.trial}.npz"
        np.savez(tmp_path / name, data=trial.data, sfreq=trial.sfreq, ch_names=np.array(trial.ch_names))
        rows.append(f"{trial.subject},{trial.trial},{name},{trial.valence},{trial.arousal},,")
    rows += ["S03,T01,missing.npz,5,5,,", f"S03,T02,{raw_trials[0].subject}_{raw_trials[0].trial}.npz,5,5,10,99"]
    (tmp_path / "m.csv").write_text("subject,trial,path,valence,arousal,tmin,tmax\n" + "\n".join(rows) + "\n")
    skipped: list[dict] = []
    trials = iter_manifest_trials(tmp_path / "m.csv", skipped=skipped)
    with caplog.at_level(logging.INFO, logger="sieeg.features.extract"):
        fs = extract_features(trials, FeatureConfig(), loader_skipped=skipped)
    assert len(fs) == 2 and fs.meta["n_input_trials"] == 4
    assert [(s["subject"], s["trial"]) for s in fs.meta["skipped"]] == [("S03", "T01"), ("S03", "T02")]
    assert "Extracted 2 of 4 trials (2 skipped)" in caplog.text


def test_bands_above_nyquist_are_rejected(raw_trials, rng):
    read = []

    def trials():
        read.append(True)
        yield from raw_trials

    with pytest.raises(ValueError, match=r"'gamma' \[30, 45\) Hz reach above the Nyquist frequency \(32 Hz\)"):
        extract_features(trials(), FeatureConfig(resample_sfreq=64.0), progress=False)
    assert read == []  # a configuration error: no trial was read

    slow = _trial(CHANNELS, rng, sfreq=64.0, seconds=20.0, dataset="synthetic")  # no resampling: 64 Hz is too low
    with pytest.raises(ValueError, match="'gamma'.*Nyquist"):
        trial_band_power(slow, FeatureConfig(resample_sfreq=None))
    fs = extract_features([*raw_trials, slow], FeatureConfig(resample_sfreq=None), progress=False)
    assert len(fs) == 6
    [record] = fs.meta["skipped"]
    assert record["reason"].startswith("ValueError: Band(s) 'gamma'")
    with pytest.raises(ValueError, match=r"Nyquist frequency \(32 Hz\) of a 64 Hz signal"):
        trial_band_power(raw_trials[0], FeatureConfig(resample_sfreq=64.0))  # 128 Hz recording resampled to 64 Hz
    assert trial_band_power(raw_trials[0], FeatureConfig(resample_sfreq=90.0)).shape == (32, 6)  # 45 Hz = Nyquist


def test_bands_above_the_recorded_nyquist_frequency_are_rejected_after_upsampling(raw_trials, rng):
    """Upsampling a 64 Hz recording to 256 Hz cannot restore 32-45 Hz: gamma would measure the resampler."""
    cfg = FeatureConfig()  # resample_sfreq 256 Hz, gamma 30-45 Hz
    slow = _trial(CHANNELS, rng, sfreq=64.0, seconds=20.0, dataset="synthetic")
    message = (r"'gamma' \[30, 45\) Hz reach above the Nyquist frequency \(32 Hz\) of synthetic subject 1 trial 1, "
               "recorded at 64 Hz; resampling cannot add")
    with pytest.raises(ValueError, match=message):
        trial_band_power(slow, cfg)
    fs = extract_features([*raw_trials, slow], cfg, progress=False)  # raw_trials: 128 Hz, upsampled too
    assert len(fs) == 6
    [record] = fs.meta["skipped"]
    assert record["reason"].startswith("ValueError: Band(s) 'gamma'") and "recorded at 64 Hz" in record["reason"]
    with pytest.raises(ValueError, match=message):
        extract_features([*raw_trials, slow], cfg, on_error="raise", progress=False)
    at_limit = _trial(CHANNELS, rng, sfreq=90.0, seconds=20.0)  # 45 Hz = the recording's Nyquist frequency
    assert trial_band_power(at_limit, cfg).shape == (32, 6)
