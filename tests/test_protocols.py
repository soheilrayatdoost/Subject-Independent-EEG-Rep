"""Within- and cross-database protocols (docs/DESIGN.md 8.4, D13-D16).

Tests marked "spy" replace the trainer by a recorder, so they check exactly what reaches
training (and nothing trains). The others train tiny models for a couple of epochs.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sieeg
from sieeg.config import BIOSEMI32_CHANNELS, PAPER_BANDS, apply_overrides
from sieeg.evaluation import protocols
from sieeg.evaluation.metrics import classification_rate, summarize_predictions, weighted_f1
from sieeg.evaluation.protocols import Split, loso_splits, run_cross_database, run_within_database
from sieeg.evaluation.report import SUMMARY_KEYS, RunResult, load_run, save_run
from sieeg.features.featureset import FeatureSet
from sieeg.features.topomap import images_from_config
from sieeg.training.methods import AdversarySpec, MethodSpec, available_methods, get_method
from sieeg.training.trainer import FitResult, LabeledData, UnlabeledData
from sieeg.utils import natural_key, natural_sorted

PREDICTION_COLUMNS = ["fold", "dataset", "subject", "trial", "rating", "y_true", "y_pred", "p1"]
PROBE_COLUMNS = ["probe_subject_acc", "probe_subject_chance", "probe_group_bacc"]
FOLD_COLUMNS = [
    "fold", "test_subjects", "val_subjects", "n_train", "n_val", "n_test", "n_outgroup", "n_target",
    "best_epoch", "best_val", "cr", "f1", "seconds", *PROBE_COLUMNS,
]
#: summary keys of a protocol run, in order: report.SUMMARY_KEYS with "probe" after "subject_mean".
RUN_SUMMARY_KEYS = [
    "protocol", "method", "tag", "label", "train_dataset", "test_dataset", "lambdas", "seed",
    "n_folds", "n_subjects", "n_trials", "pooled", "subject_mean", "probe", "created", "sieeg_version",
]
WITHIN_METHODS = ["cnn", "si_eeg", "sinn"]
DOMAIN_METHODS = ["dann", "sidann", "sida_eeg"]


@pytest.fixture
def spy(monkeypatch):
    """Replace AdversarialTrainer by a recorder, and log clear_session.

    The recorder predicts p1 from the image mean (or with ``log.p1(x)`` if set) and embeds an
    image as its maximum. Returns a namespace with ``events`` (call order) and ``trainers`` (one
    record per trainer: its constructor arguments, the data given to ``fit`` and the images
    given to ``predict_proba``).
    """
    import keras

    from sieeg.training import trainer as trainer_module

    log = SimpleNamespace(events=[], trainers=[], p1=None)

    class SpyTrainer:
        def __init__(self, method, input_shape, *, n_classes=2, n_subjects=0, model_cfg=None, train_cfg=None, seed=0):
            log.events.append("init")
            self.record = SimpleNamespace(
                method=method, input_shape=tuple(input_shape), n_classes=n_classes, n_subjects=n_subjects,
                model_cfg=model_cfg, train_cfg=train_cfg, seed=seed,
            )
            log.trainers.append(self.record)

        def fit(self, train, val=None, *, outgroup=None, target=None):
            log.events.append("fit")
            vars(self.record).update(train=train, val=val, outgroup=outgroup, target=target)
            return FitResult(history=pd.DataFrame({"epoch": [1]}), best_epoch=1, best_score=0.5, seconds=0.0)

        def predict_proba(self, x, batch_size=512):
            log.events.append("predict")
            self.record.test_x = x
            p1 = log.p1(x) if log.p1 else 1.0 / (1.0 + np.exp(-x.astype(np.float64).mean(axis=(1, 2, 3))))
            return np.column_stack([1.0 - p1, p1])

        def embed(self, x, batch_size=512):
            log.events.append("embed")
            return x.reshape(len(x), -1).max(axis=1, keepdims=True)

    monkeypatch.setattr(trainer_module, "AdversarialTrainer", SpyTrainer)
    monkeypatch.setattr(keras.backend, "clear_session", lambda *args, **kwargs: log.events.append("clear_session"))
    return log


@pytest.fixture
def op_determinism():
    """TensorFlow op determinism for one test; the previous setting is restored afterwards."""
    import tensorflow as tf
    from tensorflow.python.framework import config as tf_config

    enabled = tf_config.is_op_determinism_enabled()
    tf.config.experimental.enable_op_determinism()
    yield
    if not enabled:
        tf_config.disable_op_determinism()


def assert_valid_run(run: RunResult, fs, label: str, fold_names: list[str], test_subjects: list[str], cfg) -> None:
    """Schema of docs/DESIGN.md 8.3; predictions and fold scores consistent with the test rows of ``fs``."""
    assert isinstance(run, RunResult)
    summary, pred, folds = run.summary, run.predictions, run.folds
    assert list(summary) == RUN_SUMMARY_KEYS and set(SUMMARY_KEYS) <= set(summary)
    assert summary["label"] == label and summary["seed"] == cfg.seed and summary["n_folds"] == len(fold_names)
    assert summary["lambdas"] == {a.kind: a.lam for a in get_method(summary["method"]).adversaries}
    assert summary["sieeg_version"] == sieeg.__version__
    datetime.fromisoformat(summary["created"])
    scores = summarize_predictions(pred)
    assert {key: summary[key] for key in scores} == scores
    assert summary["n_subjects"] == len(test_subjects)

    assert list(pred.columns) == PREDICTION_COLUMNS
    rows = np.isin(fs.subjects, test_subjects)
    assert (pred["dataset"] == fs.dataset).all()
    assert pred["subject"].tolist() == fs.subjects[rows].tolist()
    assert pred["trial"].tolist() == fs.trials[rows].tolist()
    np.testing.assert_array_equal(pred["rating"], fs.ratings(label)[rows])
    np.testing.assert_array_equal(pred["y_true"], fs.labels(label)[rows])
    assert pred["p1"].between(0.0, 1.0).all()
    np.testing.assert_array_equal(pred["y_pred"], (pred["p1"] > 0.5).astype(int))
    assert list(dict.fromkeys(pred["fold"])) == fold_names

    assert list(folds.columns) == FOLD_COLUMNS
    assert folds["fold"].tolist() == fold_names
    assert list(run.histories) == fold_names
    if cfg.protocol.subject_probe:
        assert list(summary["probe"]) == ["subject_acc", "subject_chance", "group_bacc"]
    else:  # no probe: NaN columns, no summary
        assert summary["probe"] is None and folds[PROBE_COLUMNS].isna().all().all()
    for row in folds.itertuples(index=False):
        part = pred[pred["fold"] == row.fold]
        assert row.n_test == len(part)
        assert row.cr == classification_rate(part["y_true"], part["y_pred"])
        assert row.f1 == weighted_f1(part["y_true"], part["y_pred"])
        history = run.histories[row.fold]
        assert history["epoch"].tolist() == list(range(1, cfg.train.epochs + 1))
        assert 1 <= row.best_epoch <= cfg.train.epochs and row.seconds > 0
        assert row.best_val == history["val_f1"].iloc[row.best_epoch - 1]  # selected on validation F1


# ------------------------------------------------------------------ splits
def test_loso_splits_partition_the_subjects_in_natural_order():
    ordered = ["S1", "S2", "S3", "S4", "S10", "S11"]
    splits = loso_splits(["S10", "S2", "S1", "S3", "S2", "S11", "S4"], 2, seed=0)
    assert all(isinstance(split, Split) for split in splits)
    assert [split.test for split in splits] == [[s] for s in ordered]
    for split in splits:
        assert len(split.val) == 2 and len(split.train) == 3
        assert set(split.val).isdisjoint(split.train) and split.test[0] not in split.val + split.train
        assert sorted(split.test + split.val + split.train, key=natural_key) == ordered
        assert split.val == natural_sorted(split.val) and split.train == natural_sorted(split.train)


def test_loso_splits_draw_validation_per_test_subject():
    subjects = [f"P{i}" for i in range(1, 9)]
    splits = loso_splits(subjects, 3, seed=4)
    assert loso_splits(subjects[::-1], 3, seed=4) == splits  # deterministic, whatever the input order
    for i, split in enumerate(splits):
        others = [s for s in subjects if s != split.test[0]]
        assert split.val == natural_sorted(np.random.default_rng([4, i]).choice(others, 3, replace=False))
    assert [split.val for split in loso_splits(subjects, 3, seed=5)] != [split.val for split in splits]
    assert all(split.val == [] and len(split.train) == 7 for split in loso_splits(subjects, 0, seed=4))
    for n_val in (-1, 7):  # 7 validation subjects would leave nobody to train on
        with pytest.raises(ValueError, match="n_val_subjects"):
            loso_splits(subjects, n_val, seed=4)


# ------------------------------------------------------- within (spy)
@pytest.mark.parametrize("name", WITHIN_METHODS)
def test_within_trains_on_training_and_validation_labels_only(spy, tiny_config, small_featureset, name):
    fs = small_featureset
    cfg = dataclasses.replace(apply_overrides(tiny_config, ["protocol.n_val_subjects=2"]), seed=3)
    run_within_database(fs, name, "valence", cfg, folds=["S05", "S02"], progress=False)

    assert spy.events == ["clear_session", "init", "fit", "predict"] * 2  # a fresh Keras session per fold
    ids = fs.subject_ids
    splits = loso_splits(ids, 2, cfg.seed)
    images, labels = images_from_config(fs, cfg), fs.labels("valence")
    for record, i in zip(spy.trainers, [1, 4], strict=True):  # S02 and S05, in natural order
        split = splits[i]
        assert record.method == get_method(name) and record.seed == 3 * 1000 + i
        assert (record.n_classes, record.n_subjects, record.input_shape) == (2, 6, (32, 32, 6))
        assert record.model_cfg == cfg.model and record.train_cfg == cfg.train
        for data, subjects in ((record.train, split.train), (record.val, split.val)):
            rows = np.isin(fs.subjects, subjects)
            assert isinstance(data, LabeledData)
            np.testing.assert_array_equal(data.x, images[rows])
            np.testing.assert_array_equal(data.y, labels[rows])
            assert [ids[k] for k in data.subjects] == fs.subjects[rows].tolist()  # global index = natural order
        test = fs.subjects == ids[i]
        np.testing.assert_array_equal(record.test_x, images[test])
        if name == "cnn":
            assert record.outgroup is None
        else:  # the test subject is the out-group (Eq. 1) or the identity pool (D15), without labels
            assert type(record.outgroup) is UnlabeledData and not hasattr(record.outgroup, "y")
            np.testing.assert_array_equal(record.outgroup.x, images[test])
            np.testing.assert_array_equal(record.outgroup.subjects, i)
        assert record.target is None


def test_running_some_folds_changes_no_split_seed_or_prediction(spy, tiny_config, small_featureset):
    cfg = apply_overrides(tiny_config, ["protocol.n_val_subjects=2"])
    full = run_within_database(small_featureset, "si_eeg", "valence", cfg, progress=False)
    splits = loso_splits(small_featureset.subject_ids, 2, cfg.seed)
    assert full.folds["val_subjects"].tolist() == [split.val for split in splits]
    spy.trainers.clear()
    part = run_within_database(small_featureset, "si_eeg", "valence", cfg, folds=["S06", "S03"], progress=False)

    assert part.folds["fold"].tolist() == ["S03", "S06"]  # natural order, whatever the order asked for
    assert [record.seed for record in spy.trainers] == [2, 5]
    columns = ["fold", "test_subjects", "val_subjects", "n_train", "n_val", "n_test", "n_outgroup", "n_target"]
    expected = full.folds[full.folds["fold"].isin(["S03", "S06"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(part.folds[columns], expected[columns])
    expected = full.predictions[full.predictions["fold"].isin(["S03", "S06"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(part.predictions, expected)


def test_every_method_gets_the_same_validation_subjects(spy, tiny_config, small_featureset, source_target_featuresets):
    cfg = apply_overrides(tiny_config, ["protocol.n_val_subjects=2"])
    columns = ["fold", "test_subjects", "val_subjects", "n_train", "n_val", "n_test"]
    runs, val_rows = [], []
    for name in WITHIN_METHODS:
        spy.trainers.clear()
        runs.append(run_within_database(small_featureset, name, "valence", cfg, progress=False).folds[columns])
        val_rows.append([record.val.x for record in spy.trainers])
    for folds, rows in zip(runs[1:], val_rows[1:]):
        pd.testing.assert_frame_equal(folds, runs[0])
        for x, x0 in zip(rows, val_rows[0], strict=True):
            np.testing.assert_array_equal(x, x0)

    source, target = source_target_featuresets
    source_ids = source.subject_ids
    val = natural_sorted(np.random.default_rng(cfg.seed).choice(source_ids, 2, replace=False))
    train = [s for s in source_ids if s not in val]
    for name in available_methods():
        spy.trainers.clear()
        run = run_cross_database(source, target, name, "valence", cfg, progress=False)
        assert all(subjects == val for subjects in run.folds["val_subjects"])
        for record in spy.trainers:
            assert natural_sorted(source_ids[k] for k in record.val.subjects) == val
            assert natural_sorted(source_ids[k] for k in record.train.subjects) == train


@pytest.mark.parametrize("name", DOMAIN_METHODS)
def test_within_rejects_methods_with_a_domain_adversary(tiny_config, small_featureset, name):
    with pytest.raises(ValueError, match="domain"):
        run_within_database(small_featureset, name, "valence", tiny_config)


@pytest.mark.parametrize("protocol", ["within", "cross"])
def test_outgroup_can_include_the_validation_subjects(spy, tiny_config, small_featureset, source_target_featuresets,
                                                      protocol):
    cfg = apply_overrides(tiny_config, ["protocol.n_val_subjects=2", "protocol.outgroup_includes_val=true"])
    if protocol == "within":
        run = run_within_database(small_featureset, "si_eeg", "valence", cfg, folds=["S04"], progress=False)
        index = {s: i for i, s in enumerate(small_featureset.subject_ids)}
        test = [index["S04"]]
    else:
        source, target = source_target_featuresets
        run = run_cross_database(source, target, "si_eeg", "valence", cfg, folds=["B02"], progress=False)
        index = {s: i for i, s in enumerate(source.subject_ids)}
        test = [len(index) + 1]  # B02 follows the 5 source subjects in the global index
    (record,) = spy.trainers
    expected = test + [index[s] for s in run.folds.loc[0, "val_subjects"]]
    assert sorted(set(record.outgroup.subjects)) == sorted(expected)
    assert run.folds.loc[0, "n_outgroup"] == 12 * 3


def test_lambdas_come_from_the_config_or_the_method_spec(spy, tiny_config, small_featureset):
    cfg = apply_overrides(tiny_config, ["protocol.lambda_subject=0.35"])
    run = run_within_database(small_featureset, "si_eeg", "valence", cfg, folds=["S01"], progress=False)
    assert spy.trainers[0].method == get_method("si_eeg", lambda_subject=0.35)
    assert run.summary["lambdas"] == {"group": 0.35}
    custom = MethodSpec("si_eeg", "custom", (AdversarySpec("group", 0.5),))
    run = run_within_database(small_featureset, custom, "valence", cfg, folds=["S01"], progress=False)
    assert spy.trainers[1].method is custom and run.summary["lambdas"] == {"group": 0.5}


def test_trials_without_a_rating_are_left_out(spy, tiny_config, small_featureset):
    fs = dataclasses.replace(small_featureset, valence=small_featureset.valence.copy())
    fs.valence[:3] = np.nan  # three trials of S01
    fs.valence[fs.subjects == "S06"] = np.nan  # S06 has no valence rating at all
    run = run_within_database(fs, "si_eeg", "valence", tiny_config, progress=False)
    assert run.folds["fold"].tolist() == ["S01", "S02", "S03", "S04", "S05"]
    assert run.summary["n_trials"] == 5 * 12 - 3 and run.predictions["rating"].notna().all()
    assert run.folds.loc[0, "n_test"] == 9 and spy.trainers[0].n_subjects == 5
    with pytest.raises(ValueError, match="S06"):
        run_within_database(fs, "si_eeg", "valence", tiny_config, folds=["S06"])


def test_invalid_arguments(tiny_config, small_featureset, source_target_featuresets):
    fs = small_featureset
    source, target = source_target_featuresets
    with pytest.raises(ValueError, match="Unknown test subjects"):
        run_within_database(fs, "cnn", "valence", tiny_config, folds=["S01", "S99"])
    with pytest.raises(ValueError, match="no test subject"):
        run_within_database(fs, "cnn", "valence", tiny_config, folds=[])
    with pytest.raises(ValueError, match="label"):
        run_within_database(fs, "cnn", "dominance", tiny_config)
    with pytest.raises(ValueError, match="n_val_subjects"):
        run_within_database(fs, "cnn", "valence", apply_overrides(tiny_config, ["protocol.n_val_subjects=5"]))
    # Built without apply_overrides, whose validation would reject the mode before the protocol sees it.
    bad_mode = dataclasses.replace(
        tiny_config, protocol=dataclasses.replace(tiny_config.protocol, cross_per_subject="sometimes")
    )
    with pytest.raises(ValueError, match=r"cross_per_subject must be one of \("):
        run_cross_database(source, target, "si_eeg", "valence", bad_mode)
    with pytest.raises(ValueError, match="Unknown test subjects"):  # source subjects are never test subjects
        run_cross_database(source, target, "dann", "valence", tiny_config, folds=["A01"])
    with pytest.raises(ValueError, match="n_val_subjects"):  # 5 source subjects: nobody left to train on
        no_training = apply_overrides(tiny_config, ["protocol.n_val_subjects=5"])
        run_cross_database(source, target, "dann", "valence", no_training)
    other_bands = dataclasses.replace(target, bands=[(name, lo, hi + 1.0) for name, lo, hi in target.bands])
    with pytest.raises(ValueError, match="Bands differ"):
        run_cross_database(source, other_bands, "dann", "valence", tiny_config)


# -------------------------------------------------------- cross (spy)
@pytest.mark.parametrize(
    ("mode", "name", "per_subject"),
    [
        ("auto", "si_eeg", True),
        ("auto", "sida_eeg", True),
        ("auto", "cnn", False),
        ("auto", "sinn", False),
        ("auto", "dann", False),
        ("auto", "sidann", False),
        ("always", "dann", True),
        ("never", "si_eeg", False),
        ("never", "sida_eeg", False),
    ],
)
def test_cross_database_modes(spy, tiny_config, source_target_featuresets, mode, name, per_subject, caplog):
    source, target = source_target_featuresets
    cfg = dataclasses.replace(apply_overrides(tiny_config, [f"protocol.cross_per_subject={mode}"]), seed=2)
    with caplog.at_level(logging.WARNING, logger="sieeg.evaluation.protocols"):
        run = run_cross_database(source, target, name, "valence", cfg, progress=False)
    spec, target_ids = get_method(name), target.subject_ids
    # One out-group for the whole target: the group labels would equal the domain labels (D14).
    assert ("duplicates a domain adversary" in caplog.text) == (spec.has("group") and not per_subject)
    n_source = len(source.subject_ids)
    target_index = list(range(n_source, n_source + len(target_ids)))  # global index: source subjects first
    if per_subject:  # one model per target subject; its trials are the out-group (D14)
        assert run.folds["fold"].tolist() == target_ids
        assert run.folds["test_subjects"].tolist() == [[t] for t in target_ids]
        assert [record.seed for record in spy.trainers] == [2000, 2001, 2002]
        outgroups, n_test = [[k] for k in target_index], 12
    else:  # one model; the out-group is the whole target database
        assert run.folds["fold"].tolist() == ["all"]
        assert run.folds["test_subjects"].tolist() == [target_ids]
        assert [record.seed for record in spy.trainers] == [2000]
        outgroups, n_test = [target_index], 36
    assert run.predictions["subject"].tolist() == target.subjects.tolist()  # every target trial, tested once
    for record, outgroup in zip(spy.trainers, outgroups, strict=True):
        assert record.n_subjects == n_source + len(target_ids)  # identity classes: source and target subjects
        assert set(record.train.subjects) | set(record.val.subjects) <= set(range(n_source))  # labels: source only
        assert len(record.test_x) == n_test
        if spec.has("group"):
            assert type(record.outgroup) is UnlabeledData
            assert sorted(set(record.outgroup.subjects)) == outgroup
        else:
            assert record.outgroup is None
        if spec.adversaries:  # the target pool: every target trial (group adversary: Eq. 3, D14)
            assert type(record.target) is UnlabeledData and not hasattr(record.target, "y")
            np.testing.assert_array_equal(record.target.subjects, np.repeat(target_index, 12))
        else:
            assert record.target is None


def test_cross_database_uses_the_common_channels(spy, tiny_config, source_target_featuresets, caplog):
    source, target = source_target_featuresets
    upper = dataclasses.replace(target, ch_names=[c.upper() for c in target.ch_names])
    with caplog.at_level(logging.WARNING, logger="sieeg.evaluation.protocols"):
        run_cross_database(source, upper, "dann", "valence", tiny_config, progress=False)
    assert not [r for r in caplog.records if r.name == "sieeg.evaluation.protocols"]  # case-insensitive match

    common = source.ch_names[2:]
    with caplog.at_level(logging.WARNING, logger="sieeg.evaluation.protocols"):
        run = run_cross_database(source, upper.pick_channels(common), "dann", "valence", tiny_config, progress=False)
    assert "30 common channels" in caplog.text
    train = ~np.isin(source.subjects, run.folds.loc[0, "val_subjects"])
    expected = images_from_config(source.pick_channels(common), tiny_config)[train]
    np.testing.assert_array_equal(spy.trainers[-1].train.x, expected)


def test_cross_database_rejects_the_same_database_and_shared_trials(spy, tiny_config, source_target_featuresets):
    """A target trial that is also a source trial would be tested after training on its own label."""
    source, _ = source_target_featuresets  # A01-A05, 12 trials each
    with pytest.raises(ValueError, match="both named 'src'"):
        run_cross_database(source, source, "cnn", "valence", tiny_config)
    copy = dataclasses.replace(source, dataset="src_copy")  # the same recordings under another name
    with pytest.raises(ValueError, match="60 of 60 target trials are identical to source trials"):
        run_cross_database(source, copy, "cnn", "valence", tiny_config)
    reordered = dataclasses.replace(copy, band_power=copy.band_power[:, ::-1].copy(), ch_names=copy.ch_names[::-1])
    with pytest.raises(ValueError, match="60 of 60"):  # compared after aligning the channels to the source order
        run_cross_database(source, reordered, "cnn", "valence", tiny_config)
    overlapping = dataclasses.replace(source.drop_subjects(["A01", "A02"]), dataset="src_b")  # A03-A05
    with pytest.raises(ValueError, match="24 of 36"):  # A03 and A04 are in both
        run_cross_database(source.drop_subjects(["A05"]), overlapping, "sida_eeg", "valence", tiny_config)
    assert spy.trainers == []  # nothing trained

    disjoint = dataclasses.replace(source.select_subjects(["A04", "A05"]), dataset="src_b")
    run = run_cross_database(source.select_subjects(["A01", "A02", "A03"]), disjoint, "cnn", "valence", tiny_config,
                             progress=False)
    assert run.summary["n_trials"] == 24


def test_cross_database_warns_about_different_extraction_settings(spy, tiny_config, source_target_featuresets,
                                                                   caplog):
    source, target = source_target_featuresets
    config = {"channels": list(source.ch_names), "resample_sfreq": 256.0, "reference": "average", "l_freq": None,
              "h_freq": None, "welch_segment_sec": 15.0, "welch_overlap": 0.5}
    source = dataclasses.replace(source, meta={"feature_config": config})
    same = dataclasses.replace(target, meta={"feature_config": dict(config)})
    with caplog.at_level(logging.WARNING, logger="sieeg.evaluation.protocols"):
        run_cross_database(source, same, "dann", "valence", tiny_config, progress=False)
        run_cross_database(source, target, "dann", "valence", tiny_config, progress=False)  # no config recorded
    assert not caplog.records

    other = dataclasses.replace(target, meta={"feature_config": dict(config, reference=None, resample_sfreq=128.0)})
    with caplog.at_level(logging.WARNING, logger="sieeg.evaluation.protocols"):
        run_cross_database(source, other, "dann", "valence", tiny_config, progress=False)
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "resample_sfreq: src=256.0 vs tgt=128.0" in record.getMessage() and "reference" in record.getMessage()


def coded_featureset(dataset: str, subjects: list[str], n_trials: int, *, seed: int, offset: float) -> FeatureSet:
    """Rows interleaved across subjects (as in a manifest), ids as given, three valence ratings missing.

    Row k has the constant band power ``offset + (k + 1) / 1000``: with ``normalization=none`` its
    image is that code inside the electrodes' hull and 0 outside, so :func:`code_index` names the row.
    """
    rng = np.random.default_rng(seed)
    keys = [(s, f"T{t}") for s in subjects for t in range(1, n_trials + 1)]
    keys = [keys[i] for i in rng.permutation(len(keys))]
    n = len(keys)
    codes = offset + (np.arange(n) + 1) / 1000.0
    valence = rng.integers(1, 10, n).astype(np.float32)
    valence[rng.choice(n, 3, replace=False)] = np.nan
    return FeatureSet(
        dataset, np.broadcast_to(codes[:, None, None], (n, len(BIOSEMI32_CHANNELS), len(PAPER_BANDS))),
        [s for s, _ in keys], [t for _, t in keys], valence, rng.integers(1, 10, n).astype(np.float32),
        list(BIOSEMI32_CHANNELS), list(PAPER_BANDS),
    )


def code_index(values: np.ndarray) -> np.ndarray:
    """``round(1000 * code) - 1`` of images (their maximum) or codes: row k of a set with offset o gives 1000 o + k."""
    values = np.asarray(values, dtype=np.float64)
    codes = values.reshape(len(values), -1).max(axis=1) if values.ndim > 1 else values
    return np.rint(codes * 1000).astype(int) - 1


@pytest.mark.parametrize(
    ("protocol", "mode", "name"),
    [
        ("within", None, "cnn"),
        ("within", None, "si_eeg"),
        ("within", None, "sinn"),
        ("cross", "always", "si_eeg"),
        ("cross", "never", "sida_eeg"),
        ("cross", "auto", "sidann"),
    ],
)
def test_predictions_follow_their_rows_on_interleaved_data(spy, tiny_config, protocol, mode, name):
    """Rows interleaved across subjects with non-natural ids. The spy returns the code of each predicted
    image as p1: every prediction's subject, trial, rating and y_true must be those of that trial, and
    training must only see source rows with their own labels."""
    spy.p1 = lambda x: x.reshape(len(x), -1).max(axis=1).astype(np.float64)
    cfg = apply_overrides(tiny_config, ["protocol.normalization=none", "protocol.n_val_subjects=2"])
    source = coded_featureset("db_a", ["10", "2", "1", "S3", "7", "11"], 6, seed=0, offset=0.0)
    if protocol == "within":
        target, offset = source, 0
        run = run_within_database(source, name, "valence", cfg, progress=False)
    else:  # colliding subject ids: the target's "1" and "2" are other people than the source's
        target, offset = coded_featureset("db_b", ["2", "1", "3"], 5, seed=1, offset=0.5), 500
        cfg = apply_overrides(cfg, [f"protocol.cross_per_subject={mode}"])
        run = run_cross_database(source, target, name, "valence", cfg, progress=False)

    pred, rated = run.predictions, np.flatnonzero(target.has_rating("valence"))
    rows = code_index(pred["p1"].to_numpy()) - offset
    np.testing.assert_array_equal(np.sort(rows), rated)  # every rated target trial, once
    assert (pred["dataset"] == target.dataset).all()
    np.testing.assert_array_equal(pred["subject"], target.subjects[rows])
    np.testing.assert_array_equal(pred["trial"], target.trials[rows])
    np.testing.assert_array_equal(pred["rating"], target.valence[rows])
    np.testing.assert_array_equal(pred["y_true"], (target.valence[rows] > 5).astype(int))
    per_subject = protocol == "within" or mode == "always"
    np.testing.assert_array_equal(pred["fold"], pred["subject"] if per_subject else "all")
    for record, fold in zip(spy.trainers, run.folds.itertuples(), strict=True):
        for data in (record.train, record.val):
            source_rows = code_index(data.x)
            assert source_rows.max() < len(source)  # labeled data: source rows only (target rows give >= 500)
            np.testing.assert_array_equal(data.y, (source.valence[source_rows] > 5).astype(int))
            if protocol == "within":
                assert not np.isin(source.subjects[source_rows], fold.test_subjects).any()


@pytest.mark.parametrize("protocol", ["within", "cross"])
def test_subject_probe_sees_validation_in_group_and_test_rows(spy, monkeypatch, tiny_config, protocol):
    """The subject probe gets z of the validation rows with their subject indices; the group probe
    the in-group rows (training rows and the target-pool rows that are not tested) and the test rows.
    The spy embeds an image as its code, so the rows can be read back from z. No emotion label."""
    from sieeg.analysis import probe as probe_module

    calls = []

    def subject_probe(z, subjects, *, seed):
        calls.append(("subject", code_index(z[:, 0]), np.asarray(subjects), seed))
        return {"subject_acc": 0.3, "subject_chance": 0.25, "n_subjects": len(set(subjects))}

    def group_probe(z_in, z_out, *, seed):
        calls.append(("group", code_index(z_in[:, 0]), code_index(z_out[:, 0]), seed))
        return {"group_bacc": 0.6, "n_per_group": min(len(z_in), len(z_out))}

    monkeypatch.setattr(probe_module, "subject_probe", subject_probe)
    monkeypatch.setattr(probe_module, "group_probe", group_probe)
    cfg = apply_overrides(tiny_config, ["protocol.normalization=none", "protocol.n_val_subjects=2",
                                        "protocol.subject_probe=true"])
    source = coded_featureset("db_a", ["10", "2", "1", "S3", "7", "11"], 6, seed=0, offset=0.0)
    rated = source.has_rating("valence")
    ids = natural_sorted(source.subjects[rated])
    if protocol == "within":
        run = run_within_database(source, "si_eeg", "valence", cfg, folds=["7"], progress=False)
        test_rows, other_target_rows = np.flatnonzero(rated & (source.subjects == "7")), []
    else:
        target = coded_featureset("db_b", ["2", "1", "3"], 5, seed=1, offset=0.5)
        run = run_cross_database(source, target, "sida_eeg", "valence", cfg, folds=["1"], progress=False)
        tested = target.has_rating("valence") & (target.subjects == "1")
        test_rows = 500 + np.flatnonzero(tested)
        other_target_rows = 500 + np.flatnonzero(target.has_rating("valence") & ~tested)
    fold = run.folds.iloc[0]
    val_rows = np.flatnonzero(rated & np.isin(source.subjects, fold.val_subjects))
    not_training = [*fold.val_subjects, *(fold.test_subjects if protocol == "within" else [])]  # cross: ids collide
    train_rows = np.flatnonzero(rated & ~np.isin(source.subjects, not_training))
    seed = spy.trainers[0].seed

    (subject_call,) = [call for call in calls if call[0] == "subject"]
    np.testing.assert_array_equal(np.sort(subject_call[1]), val_rows)
    np.testing.assert_array_equal(subject_call[2], [ids.index(s) for s in source.subjects[subject_call[1]]])
    (group_call,) = [call for call in calls if call[0] == "group"]
    np.testing.assert_array_equal(np.sort(group_call[1]), np.sort(np.r_[train_rows, other_target_rows]))
    np.testing.assert_array_equal(np.sort(group_call[2]), test_rows)
    assert subject_call[3] == group_call[3] == seed
    assert fold[PROBE_COLUMNS].tolist() == [0.3, 0.25, 0.6]
    assert run.summary["probe"] == {"subject_acc": 0.3, "subject_chance": 0.25, "group_bacc": 0.6}

    calls.clear()  # one validation subject: no subject probe (NaN), the group probe still runs
    one_val = apply_overrides(cfg, ["protocol.n_val_subjects=1"])
    run = run_within_database(source, "si_eeg", "valence", one_val, folds=["7"], progress=False)
    assert [call[0] for call in calls] == ["group"]
    assert np.isnan(run.folds.loc[0, "probe_subject_acc"]) and np.isnan(run.summary["probe"]["subject_acc"])
    assert run.summary["probe"]["group_bacc"] == 0.6


def test_gradient_registry_cleanup_is_skipped_without_the_private_api(spy, monkeypatch, tiny_config,
                                                                      small_featureset):
    from tensorflow.python.framework import ops

    monkeypatch.delattr(ops, "_gradient_registry")
    assert protocols._gradient_registry() is None
    run = run_within_database(small_featureset, "si_eeg", "valence", tiny_config, folds=["S01"], progress=False)
    assert len(run.predictions) == 12


# ------------------------------------------------------------- training
@pytest.mark.parametrize("name", WITHIN_METHODS)
def test_within_database_run(tiny_config, small_featureset, name, tmp_path):
    fs = small_featureset
    run = run_within_database(fs, name, "valence", tiny_config, folds=["S02", "S05"], tag="tiny", progress=False)
    assert_valid_run(run, fs, "valence", ["S02", "S05"], ["S02", "S05"], tiny_config)
    summary, folds = run.summary, run.folds
    assert (summary["protocol"], summary["method"], summary["tag"]) == ("within", name, "tiny")
    assert summary["train_dataset"] == summary["test_dataset"] == fs.dataset
    assert folds["test_subjects"].tolist() == [["S02"], ["S05"]]
    assert folds["n_train"].tolist() == [48, 48] and folds["n_val"].tolist() == [12, 12]
    assert folds["n_outgroup"].tolist() == ([0, 0] if name == "cnn" else [12, 12])
    assert folds["n_target"].tolist() == [0, 0]
    loaded = load_run(save_run(run, tmp_path / "run"))
    pd.testing.assert_frame_equal(loaded.predictions, run.predictions, check_dtype=False)


@pytest.mark.parametrize(
    ("name", "folds", "fold_names"),
    [
        ("si_eeg", ["B01", "B03"], ["B01", "B03"]),  # group adversary: one model per target subject
        ("dann", None, ["all"]),  # one model for the whole target database
        ("sida_eeg", ["B02", "B03"], ["B02", "B03"]),
        ("sidann", None, ["all"]),
    ],
)
def test_cross_database_run(tiny_config, source_target_featuresets, name, folds, fold_names):
    source, target = source_target_featuresets
    run = run_cross_database(source, target, name, "arousal", tiny_config, folds=folds, progress=False)
    assert_valid_run(run, target, "arousal", fold_names, folds or target.subject_ids, tiny_config)
    summary, spec = run.summary, get_method(name)
    assert (summary["protocol"], summary["method"], summary["tag"]) == ("cross", name, None)
    assert (summary["train_dataset"], summary["test_dataset"]) == ("src", "tgt")
    assert (run.folds["n_train"] == 48).all() and (run.folds["n_val"] == 12).all()
    assert (run.folds["n_outgroup"] == (12 if spec.has("group") else 0)).all()
    assert (run.folds["n_target"] == (36 if spec.adversaries else 0)).all()


@pytest.mark.parametrize("protocol", ["within", "cross"])
def test_test_and_target_labels_never_reach_training(op_determinism, tiny_config, small_featureset,
                                                     source_target_featuresets, protocol):
    """Permuting the ratings of the test subject (within) or of the target database (cross) changes the
    scores but not one predicted probability: those labels reach neither training nor model selection."""
    rng = np.random.default_rng(11)
    if protocol == "within":
        fs = small_featureset
        shuffled = dataclasses.replace(fs, valence=fs.valence.copy())
        test = fs.subjects == "S03"
        shuffled.valence[test] = rng.permutation(fs.valence[test])

        def run(data):
            return run_within_database(data, "si_eeg", "valence", tiny_config, folds=["S03"], progress=False)

    else:
        source, fs = source_target_featuresets
        shuffled = dataclasses.replace(fs, valence=rng.permutation(fs.valence))

        def run(data):
            return run_cross_database(source, data, "sida_eeg", "valence", tiny_config, folds=["B02"], progress=False)

    original, permuted = run(fs), run(shuffled)
    assert not np.array_equal(permuted.predictions["y_true"], original.predictions["y_true"])  # labels did change
    np.testing.assert_array_equal(permuted.predictions["p1"], original.predictions["p1"])
    (fold,) = original.histories
    pd.testing.assert_frame_equal(
        permuted.histories[fold].drop(columns="seconds"), original.histories[fold].drop(columns="seconds")
    )


def record_steps(monkeypatch) -> list[tuple[str, list[np.ndarray]]]:
    """Record (method name, arguments) of every train-step call of the real trainer."""
    from sieeg.training import trainer as trainer_module

    calls = []
    compile_step = trainer_module.AdversarialTrainer._train_step

    def recording(self, *layout):
        step = compile_step(self, *layout)

        def call(*args):
            calls.append((self.method.name, [np.asarray(a) for a in args]))
            return step(*args)

        return call

    monkeypatch.setattr(trainer_module.AdversarialTrainer, "_train_step", recording)
    return calls


def test_folds_release_their_gradient_registry_entries(monkeypatch, tiny_config, small_featureset):
    """Each traced train step adds tf.custom_gradient entries (GRL, optimiser) to TensorFlow's global
    gradient registry, which pin the step's graph for the rest of the process (about 18 MiB per
    paper-size fold). _fit_fold removes the entries of its fold."""
    from sieeg.training import trainer as trainer_module

    registry = protocols._gradient_registry()
    if registry is None:
        pytest.skip("TensorFlow's private gradient registry is not available")

    def count() -> int:
        return sum(str(key).startswith("CustomGradient-") for key in registry)

    during, predict_proba = [], trainer_module.AdversarialTrainer.predict_proba

    def counting(self, x, batch_size=512):
        during.append(count())
        return predict_proba(self, x, batch_size)

    monkeypatch.setattr(trainer_module.AdversarialTrainer, "predict_proba", counting)
    before = count()
    cfg = apply_overrides(tiny_config, ["train.epochs=1"])
    run = run_within_database(small_featureset, "si_eeg", "valence", cfg, folds=["S01", "S02", "S03"], progress=False)
    assert len(run.folds) == 3
    assert during == [before + 2] * 3  # the GRL and the optimiser register one entry each per fold ...
    assert count() == before  # ... and none is left behind


def test_within_sinn_weighs_the_test_subject_like_one_training_subject(monkeypatch, tiny_config, small_featureset):
    """The single test subject fills half of every identity batch (D13). Its rows weigh
    1 / n_train_subjects, so over an epoch it carries the identity weight of one training subject
    (D15) instead of half of the identity loss."""
    calls = record_steps(monkeypatch)
    cfg = apply_overrides(tiny_config, ["train.epochs=1"])
    run = run_within_database(small_featureset, "sinn", "valence", cfg, folds=["S03"], progress=False)
    weights = np.zeros(len(small_featureset.subject_ids))
    for _, (_, y, _, subjects, _, _, alpha) in calls:
        np.add.at(weights, subjects, np.r_[np.ones(len(y)), np.full(len(y), alpha)])
    ids = small_featureset.subject_ids
    test, val = ids.index("S03"), [ids.index(s) for s in run.folds.loc[0, "val_subjects"]]
    train = [k for k in range(len(ids)) if k != test and k not in val]
    assert len(train) == 4 and calls[0][1][6] == np.float32(1 / 4)
    np.testing.assert_allclose(weights[train], 12.0)  # every training trial once per epoch
    assert weights[test] == pytest.approx(12.0)  # 48 pool rows x 1/4: like one training subject
    assert not weights[val].any()


def test_si_eeg_across_databases_is_no_dann_on_the_test_subject(monkeypatch, tiny_config, source_target_featuresets):
    """Eq. 3 / D14: SI-EEG's group head sees the whole target pool, and the target subjects that are
    not tested are in-group (0). With only S and the test subject's trials, SI-EEG across databases
    was exactly a DANN (lambda 0.2) on that subject: same predictions, same histories."""
    source, target = source_target_featuresets
    calls = record_steps(monkeypatch)
    cfg = apply_overrides(tiny_config, ["train.epochs=1"])
    si_eeg = run_cross_database(source, target, "si_eeg", "valence", cfg, folds=["B01"], progress=False)  # seed 0
    dann = run_cross_database(source, target.select_subjects(["B01"]), get_method("dann", lambda_domain=0.2), "valence",
                              cfg, progress=False)  # one model, seed 0
    assert si_eeg.folds.loc[0, "n_target"] == len(target)
    assert not np.allclose(si_eeg.predictions["p1"], dann.predictions["p1"])

    subject_of = {image.tobytes(): s for image, s in zip(images_from_config(target, cfg), target.subjects)}
    labels = {s: set() for s in target.subject_ids}
    for name, (x, y, *_, group_labels, _) in calls:
        if name == "si_eeg":  # one group label per batch row: blocks S, O, T
            n = len(y)
            for image, label in zip(x[2 * n :], group_labels[2 * n :]):
                labels[subject_of[image.tobytes()]].add(float(label))
    assert labels == {"B01": {1.0}, "B02": {0.0}, "B03": {0.0}}


def test_subject_probe_after_training(op_determinism, tiny_config, small_featureset, tmp_path):
    """protocol.subject_probe adds the probe scores (fractions) of every fold and their means; the probe
    runs after prediction, so the predictions do not change."""
    cfg = apply_overrides(tiny_config, ["protocol.n_val_subjects=2", "train.epochs=1"])
    folds = ["S02", "S05"]
    off = run_within_database(small_featureset, "si_eeg", "valence", cfg, folds=folds, progress=False)
    probed = apply_overrides(cfg, ["protocol.subject_probe=true"])
    on = run_within_database(small_featureset, "si_eeg", "valence", probed, folds=folds, progress=False)
    assert_valid_run(on, small_featureset, "valence", folds, folds, probed)
    pd.testing.assert_frame_equal(on.predictions, off.predictions)
    scores = on.folds[PROBE_COLUMNS]
    assert scores.notna().all().all() and ((scores >= 0) & (scores <= 1)).all().all()
    assert (scores["probe_subject_chance"] == 0.5).all()  # 2 validation subjects x 12 trials
    means = dict(zip(["subject_acc", "subject_chance", "group_bacc"], scores.mean()))
    assert on.summary["probe"] == pytest.approx(means)
    assert off.summary["probe"] is None and off.folds[PROBE_COLUMNS].isna().all().all()
    assert load_run(save_run(on, tmp_path / "run")).summary["probe"] == on.summary["probe"]


@pytest.mark.slow
def test_parallel_folds_match_sequential_folds(tiny_config, small_featureset):
    def run(jobs: int) -> RunResult:
        cfg = apply_overrides(tiny_config, [f"protocol.jobs={jobs}"])
        return run_within_database(small_featureset, "si_eeg", "valence", cfg, folds=["S01", "S04"], progress=False)

    sequential, parallel = run(1), run(2)
    # Same splits, seeds, data and fold order. Workers run cpu_count // 2 intra-op threads, so
    # float32 sums may round differently (about 1e-7); predicted classes must not change.
    np.testing.assert_array_equal(parallel.predictions["y_pred"], sequential.predictions["y_pred"])
    pd.testing.assert_frame_equal(parallel.predictions, sequential.predictions, check_exact=False, atol=1e-5)
    pd.testing.assert_frame_equal(
        parallel.folds.drop(columns="seconds"), sequential.folds.drop(columns="seconds"), check_exact=False, atol=1e-5
    )
    assert list(parallel.histories) == list(sequential.histories) == ["S01", "S04"]
