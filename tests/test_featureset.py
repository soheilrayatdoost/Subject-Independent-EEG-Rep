"""FeatureSet: validation, selection, labels, per-subject normalisation and npz I/O (docs/DESIGN.md 4, D7)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sieeg.config import BIOSEMI32_CHANNELS, FeatureConfig, config_to_dict
from sieeg.features.featureset import FORMAT_VERSION, FeatureSet, feature_config_differences, zscore_by_group

CHANNELS = ["Fp1", "Cz", "O2"]
BANDS = [("theta", 4, 8), ("alpha", 8.0, 12.0)]


def _featureset(n_per_subject=(2, 3), *, dataset="toy", valence=None, meta=None) -> FeatureSet:
    """Subjects S1, S2, ... with ``n_per_subject`` trials; band power counts up from 0."""
    n = sum(n_per_subject)
    subjects = np.repeat([f"S{i + 1}" for i in range(len(n_per_subject))], n_per_subject)
    return FeatureSet(
        dataset=dataset,
        band_power=np.arange(n * 6, dtype=np.float64).reshape(n, 3, 2),
        subjects=subjects,
        trials=[f"T{i}" for i in range(n)],
        valence=np.linspace(1, 9, n) if valence is None else valence,
        arousal=np.full(n, 5),
        ch_names=CHANNELS,
        bands=BANDS,
        meta=meta or {},
    )


# ------------------------------------------------------------------ construction
def test_construction_normalises_types():
    fs = _featureset()
    assert fs.band_power.dtype == np.float32 and fs.valence.dtype == np.float32 and fs.arousal.dtype == np.float32
    assert fs.subjects.dtype.kind == "U" and fs.trials.tolist() == ["T0", "T1", "T2", "T3", "T4"]
    assert fs.bands == [("theta", 4.0, 8.0), ("alpha", 8.0, 12.0)]
    assert (len(fs), fs.n_channels, fs.n_bands, fs.band_names) == (5, 3, 2, ["theta", "alpha"])
    assert repr(fs) == "FeatureSet(dataset='toy', trials=5, subjects=2, channels=3, bands=['theta', 'alpha'])"
    numeric_ids = FeatureSet("x", np.zeros((2, 3, 2)), [1, 2], [1, 1], [5, 5], [5, 5], CHANNELS, BANDS)
    assert numeric_ids.subjects.tolist() == ["1", "2"] and numeric_ids.trials.tolist() == ["1", "1"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"band_power": np.zeros((5, 6))}, "must be \\(n_trials, n_channels, n_bands\\)"),
        ({"band_power": np.zeros((5, 3, 3))}, "does not match 3 channels x 2 bands"),
        ({"ch_names": ["Fp1", "Cz"]}, "does not match 2 channels"),
        ({"subjects": ["S1"] * 4}, "subjects must have shape \\(5,\\)"),
        ({"trials": ["T"] * 6}, "trials must have shape \\(5,\\)"),
        ({"valence": np.ones(4)}, "valence must have shape \\(5,\\)"),
        ({"arousal": np.ones((5, 1))}, "arousal must have shape \\(5,\\)"),
    ],
)
def test_construction_rejects_inconsistent_shapes(change, message):
    fs = _featureset()
    fields = {"dataset": "toy", "band_power": fs.band_power, "subjects": fs.subjects, "trials": fs.trials,
              "valence": fs.valence, "arousal": fs.arousal, "ch_names": CHANNELS, "bands": BANDS} | change
    with pytest.raises(ValueError, match=message):
        FeatureSet(**fields)


def test_subject_ids_and_qualified_subjects():
    fs = FeatureSet("mahnob", np.zeros((4, 3, 2)), ["10", "2", "10", "1"], list("abcd"), [5] * 4, [5] * 4,
                    CHANNELS, BANDS)
    assert fs.subject_ids == ["1", "2", "10"]  # natural order
    assert fs.qualified_subjects.tolist() == ["mahnob:10", "mahnob:2", "mahnob:10", "mahnob:1"]


# ------------------------------------------------------------------ selection
def test_subset_by_mask_indices_and_slice_copies_meta():
    fs = _featureset(meta={"source": "unit"})
    by_mask = fs.subset(fs.subjects == "S2")
    by_index = fs.subset([4, 2])
    by_slice = fs.subset(slice(1, 3))
    assert by_mask.trials.tolist() == ["T2", "T3", "T4"]
    assert by_index.trials.tolist() == ["T4", "T2"]
    np.testing.assert_array_equal(by_index.band_power, fs.band_power[[4, 2]])
    np.testing.assert_array_equal(by_index.valence, fs.valence[[4, 2]])
    assert by_slice.trials.tolist() == ["T1", "T2"]
    by_mask.meta["source"] = "changed"
    assert fs.meta == {"source": "unit"}  # the subset owns its meta


def test_select_and_drop_subjects():
    fs = _featureset((2, 3, 1))
    assert fs.select_subjects(["S3", "S1"]).trials.tolist() == ["T0", "T1", "T5"]  # file order kept
    with pytest.raises(KeyError, match=r"\['S4', 'S9'\]"):
        fs.select_subjects(["S1", "S9", "S4"])
    assert fs.drop_subjects(["S2", "unknown"]).subject_ids == ["S1", "S3"]
    with pytest.raises(KeyError, match=r"\['s1'\]"):
        fs.select_subjects(["s1"])  # subject ids match exactly (channel names do not)


def test_pick_channels_is_case_insensitive_and_reorders():
    fs = _featureset()
    picked = fs.pick_channels(["o2", "FP1"])
    assert picked.ch_names == ["O2", "Fp1"]  # the file's spelling, in the requested order
    np.testing.assert_array_equal(picked.band_power, fs.band_power[:, [2, 0], :])
    assert fs.ch_names == CHANNELS and fs.band_power.shape == (5, 3, 2)  # the original is unchanged
    with pytest.raises(KeyError, match=r"\['Pz'\]"):
        fs.pick_channels(["Cz", "Pz"])


# ------------------------------------------------------------------ labels
def test_labels_threshold_rating_five_is_low():
    fs = _featureset((2, 3), valence=np.array([1.0, 5.0, 5.5, 9.0, 4.99]))
    np.testing.assert_array_equal(fs.labels("valence"), [0, 0, 1, 1, 0])  # paper: 5 is low / negative
    np.testing.assert_array_equal(fs.labels("valence", threshold=4.0), [0, 1, 1, 1, 1])
    assert fs.labels("arousal").tolist() == [0] * 5 and fs.labels("valence").dtype == np.int64
    with pytest.raises(ValueError, match="label must be one of"):
        fs.labels("dominance")


def test_labels_refuse_missing_ratings():
    fs = _featureset((2, 3), valence=np.array([1.0, np.nan, 7.0, np.nan, 3.0]))
    np.testing.assert_array_equal(fs.has_rating("valence"), [True, False, True, False, True])
    with pytest.raises(ValueError, match="2 trials have no valence rating"):
        fs.labels("valence")
    rated = fs.subset(fs.has_rating("valence"))
    np.testing.assert_array_equal(rated.labels("valence"), [0, 1, 0])


# ------------------------------------------------------------------ normalisation
def test_zscore_by_group_standardises_each_group_and_centres_constant_features():
    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 6 + ["b"] * 4)
    x = rng.standard_normal((10, 3)) * [1.0, 5.0, 1.0] + [0.0, 100.0, 0.0]
    x[groups == "b", 2] = 7.0  # constant inside group b
    z = zscore_by_group(x, groups)
    assert z.dtype == np.float32
    for g in ("a", "b"):
        np.testing.assert_allclose(z[groups == g].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(z[groups == "a"].std(axis=0), 1.0, rtol=1e-5)
    np.testing.assert_allclose(z[groups == "b", :2].std(axis=0), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(z[groups == "b", 2], 0.0)  # centred only, not NaN
    np.testing.assert_allclose(z[groups == "a"], zscore_by_group(x[groups == "a"], groups[groups == "a"]))


def test_normalized_none_and_subject():
    fs = _featureset()
    raw = fs.normalized("none")
    np.testing.assert_array_equal(raw, fs.band_power)
    raw[0, 0, 0] = -1.0
    assert fs.band_power[0, 0, 0] == 0.0  # a copy
    z = fs.normalized("subject")
    assert z.shape == fs.band_power.shape and z.dtype == np.float32
    flat = fs.band_power.reshape(len(fs), -1)
    np.testing.assert_allclose(z.reshape(len(fs), -1), zscore_by_group(flat, fs.subjects))
    np.testing.assert_allclose(z[fs.subjects == "S1"].mean(axis=0), 0.0, atol=1e-6)
    with pytest.raises(ValueError, match="Unknown normalization 'global'"):
        fs.normalized("global")


def test_flat_features_are_channel_major():
    fs = _featureset()
    flat = fs.flat_features("none")
    assert flat.shape == (5, 6)
    for c in range(3):
        for b in range(2):
            np.testing.assert_array_equal(flat[:, c * 2 + b], fs.band_power[:, c, b])
    np.testing.assert_array_equal(fs.flat_features(), fs.normalized("subject").reshape(5, 6))  # default: subject


# ------------------------------------------------------------------ io
def test_save_load_round_trip_with_unicode_ids_and_meta(tmp_path):
    meta = {"feature_config": {"bands": [["alpha", 8.0, 12.0]], "reference": None}, "note": "Zoë – 日本",
            "skipped": [{"subject": "Zoë", "trial": "2", "reason": "ValueError: x"}], "n_input_trials": 6}
    fs = FeatureSet(
        dataset="dai-ef", band_power=np.random.default_rng(1).standard_normal((5, 3, 2)),
        subjects=["Zoë", "日本", "S 01", "01", "Zoë"], trials=["1", "01", "a b", "Ü", "2"],
        valence=[1.0, np.nan, 5.0, 9.0, 3.5], arousal=[np.nan, 2.0, 5.0, 6.0, 7.0], ch_names=CHANNELS, bands=BANDS,
        meta=meta,
    )
    back = FeatureSet.load(fs.save(tmp_path / "f.npz"))
    assert back.dataset == "dai-ef"
    assert back.subjects.tolist() == ["Zoë", "日本", "S 01", "01", "Zoë"]
    assert back.trials.tolist() == ["1", "01", "a b", "Ü", "2"]
    np.testing.assert_array_equal(back.band_power, fs.band_power)
    assert back.band_power.dtype == np.float32
    np.testing.assert_array_equal(back.valence, fs.valence)  # NaN where unrated
    np.testing.assert_array_equal(back.arousal, fs.arousal)
    assert (back.ch_names, back.bands, back.meta) == (CHANNELS, fs.bands, meta)


@pytest.mark.parametrize(("name", "written"), [("features", "features.npz"), ("feat.v2", "feat.v2.npz"),
                                               ("f.npz", "f.npz"), ("f.NPZ", "f.NPZ.npz")])
def test_save_returns_the_path_written(tmp_path, name, written):
    fs = _featureset()
    path = fs.save(tmp_path / "sub" / name)  # missing folders are created
    assert path == tmp_path / "sub" / written and path.exists()
    assert sorted(p.name for p in path.parent.iterdir()) == [written]
    np.testing.assert_array_equal(FeatureSet.load(path).band_power, fs.band_power)


def test_load_rejects_another_format_version(tmp_path):
    path = _featureset().save(tmp_path / "f.npz")
    with np.load(path) as z:
        fields = {key: z[key] for key in z.files}
    np.savez(path, **(fields | {"format_version": np.array(FORMAT_VERSION + 1)}))
    with pytest.raises(ValueError, match=f"feature format v{FORMAT_VERSION + 1}, expected v{FORMAT_VERSION}"):
        FeatureSet.load(path)
    assert json.loads(str(fields["meta"])) == {}


# ------------------------------------------------------------------ provenance
def test_feature_config_differences():
    a, b = _featureset(dataset="a"), _featureset(dataset="b")
    assert feature_config_differences(a, b) == []  # no recorded extraction config (e.g. synthetic)
    a.meta["feature_config"] = config_to_dict(FeatureConfig())
    assert feature_config_differences(a, b) == []
    b.meta["feature_config"] = config_to_dict(FeatureConfig(channels=tuple(c.upper() for c in BIOSEMI32_CHANNELS)))
    assert feature_config_differences(a, b) == []  # channel names compare case-insensitively
    b.meta["feature_config"] = config_to_dict(FeatureConfig(resample_sfreq=128.0, welch_segment_sec=10.0))
    assert feature_config_differences(a, b) == [
        "resample_sfreq: a=256.0 vs b=128.0", "welch_segment_sec: a=15.0 vs b=10.0"]
    b.meta["feature_config"] = config_to_dict(FeatureConfig(channels=BIOSEMI32_CHANNELS[:30]))
    assert feature_config_differences(a, b) == ["average-reference channel sets differ (32 vs 30 channels)"]
    b.meta["feature_config"] = config_to_dict(FeatureConfig(channels=BIOSEMI32_CHANNELS[:30], reference=None))
    assert feature_config_differences(a, b) == ["reference: a='average' vs b=None"]  # channels: 'average' only
