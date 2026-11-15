"""Command-line interface (docs/DESIGN.md Sec. 10).

The fast tests replace the protocol runners by recorders, so they check what the CLI passes
on and writes without training. The slow test runs the whole workflow on synthetic data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sieeg import __version__
from sieeg.cli import build_parser, main
from sieeg.config import load_config
from sieeg.data.synthetic import make_synthetic_featureset, make_synthetic_trials
from sieeg.evaluation.metrics import summarize_predictions
from sieeg.evaluation.report import SUMMARY_KEYS, RunResult, load_run
from sieeg.features.extract import extract_features
from sieeg.features.featureset import FeatureSet

REPO = Path(__file__).resolve().parents[1]
SMOKE = str(REPO / "configs" / "smoke.yaml")
#: The fixture's 5 subjects leave room for a test subject, 2 validation subjects and 2 training subjects.
N_VAL = ["--set", "protocol.n_val_subjects=2"]


@pytest.fixture(autouse=True)
def _restore_log_levels():
    """main() sets the sieeg and tensorflow logger levels; keep that from leaking into other tests."""
    loggers = [logging.getLogger(name) for name in ("sieeg", "tensorflow")]
    levels = [logger.level for logger in loggers]
    yield
    for logger, level in zip(loggers, levels):
        logger.setLevel(level)


@pytest.fixture
def features(tmp_path) -> Path:
    """Saved synthetic features: dataset "syn", subjects S01-S05 with 8 trials each."""
    path = tmp_path / "syn.npz"
    make_synthetic_featureset(n_subjects=5, n_trials=8, dataset="syn", seed=0).save(path)
    return path


class _Calls(list):
    """Recorded protocol calls; each of ``hooks`` is called with (protocol, label) while a fake run trains."""

    def __init__(self) -> None:
        super().__init__()
        self.hooks: list = []


@pytest.fixture
def recorder(monkeypatch):
    """Replace run_within_database / run_cross_database by recorders that return a fake run at once.

    The fake SI-EEG run predicts every test trial correctly and the other methods always
    predict class 1, so paired tests across subjects have something to compare. With
    ``protocol.subject_probe`` the fake summary holds probe scores. ``calls`` gets a ``hook``
    entry point: a function called with (protocol, label) while a run "trains".
    """
    from sieeg.evaluation import protocols

    calls = _Calls()

    def fake_run(protocol, train_fs, test_fs, method, label, cfg, folds, tag) -> RunResult:
        for hook in calls.hooks:
            hook(protocol, label)
        test = test_fs.subset(test_fs.has_rating(label))
        if folds is not None:
            test = test.select_subjects(folds)
        y = test.labels(label, cfg.protocol.label_threshold)
        y_pred = y if method.name == "si_eeg" else np.ones_like(y)
        predictions = pd.DataFrame(
            {"fold": test.subjects, "dataset": test.dataset, "subject": test.subjects, "trial": test.trials,
             "rating": test.ratings(label), "y_true": y, "y_pred": y_pred, "p1": 0.75}
        )
        probe = {"subject_acc": 0.4, "subject_chance": 0.25, "group_bacc": 0.55} if cfg.protocol.subject_probe else None
        summary = {
            "protocol": protocol, "method": method.name, "tag": tag, "label": label,
            "train_dataset": train_fs.dataset, "test_dataset": test_fs.dataset,
            "lambdas": {a.kind: a.lam for a in method.adversaries}, "seed": cfg.seed,
            "n_folds": len(test.subject_ids), **summarize_predictions(predictions), "probe": probe,
            "created": "2026-01-01T00:00:00+00:00", "sieeg_version": __version__,
        }
        histories = {s: pd.DataFrame({"epoch": [1], "loss_cls": [0.69]}) for s in test.subject_ids}
        return RunResult(summary, predictions, pd.DataFrame({"fold": test.subject_ids}), histories)

    def within(fs, method, label, cfg, *, folds=None, tag=None, progress=True):
        calls.append({"fs": fs, "method": method, "label": label, "cfg": cfg, "folds": folds, "tag": tag})
        return fake_run("within", fs, fs, method, label, cfg, folds, tag)

    def cross(source, target, method, label, cfg, *, folds=None, tag=None, progress=True):
        calls.append({"source": source, "target": target, "method": method, "label": label, "cfg": cfg,
                      "folds": folds, "tag": tag})
        return fake_run("cross", source, target, method, label, cfg, folds, tag)

    monkeypatch.setattr(protocols, "run_within_database", within)
    monkeypatch.setattr(protocols, "run_cross_database", cross)
    return calls


def _subprocess_env() -> dict[str, str]:
    src = str(REPO / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    env.pop("TF_CPP_MIN_LOG_LEVEL", None)
    return env


# ------------------------------------------------------------------- parsing
def test_parser_reads_lists_flags_and_defaults():
    parser = build_parser()
    args = parser.parse_args(
        ["within", "f.npz", "--method", "cnn, SI-EEG,cnn", "--label", "both",
         "--exclude-subjects", "3,7", "--exclude-subjects", "12", "--set", "train.batch_size=8", "--set", "seed=2"]
    )
    assert (args.command, args.file, args.method, args.label) == ("within", Path("f.npz"), ["cnn", "si_eeg"], "both")
    assert args.exclude_subjects == ["3", "7", "12"]
    assert args.overrides == ["train.batch_size=8", "seed=2"]
    assert (args.out, args.tag, args.overwrite, args.folds, args.config) == (Path("runs"), None, False, None, None)
    assert (args.epochs, args.seed, args.jobs, args.lambda_subject, args.probe) == (None, None, None, None, False)
    assert (args.verbose, args.quiet) == (False, False)
    assert not hasattr(args, "lambda_domain")

    args = parser.parse_args(
        ["cross", "a.npz", "b.npz", "--method", "sida_eeg", "--label", "arousal", "--folds", "S1,S2",
         "--exclude-source", "9", "--exclude-target", "4", "--lambda-subject", "0.3", "--lambda-domain", "0.05",
         "--epochs", "5", "--seed", "1", "--jobs", "2", "--tag", "ls0.3-ld0.05", "--overwrite"]
    )
    assert (args.source, args.target, args.folds) == (Path("a.npz"), Path("b.npz"), ["S1", "S2"])
    assert (args.exclude_source, args.exclude_target) == (["9"], ["4"])
    assert (args.lambda_subject, args.lambda_domain, args.epochs, args.seed, args.jobs) == (0.3, 0.05, 5, 1, 2)
    assert (args.tag, args.overwrite) == ("ls0.3-ld0.05", True)

    args = parser.parse_args(["report", "runs", "old"])
    assert (args.dirs, args.reference, args.metric, args.source, args.out) == (
        [Path("runs"), Path("old")], ["sida_eeg"], "f1", "pooled", None
    )
    args = parser.parse_args(["bias", "a.npz"])
    assert (args.other, args.n_splits, args.n_estimators, args.seed, args.no_balance, args.jobs, args.out) == (
        None, 10, 500, 0, False, -1, None
    )
    args = parser.parse_args(["extract", "synthetic", "--out", "x.npz"])
    assert (args.kind, args.n_subjects, args.n_trials, args.dataset, args.domain_shift, args.seed, args.raw) == (
        "synthetic", 10, 20, "synthetic", 0.0, 0, False
    )
    assert args.overwrite is False
    args = parser.parse_args(["extract", "mahnob", "/data", "--out", "m.npz", "--subjects", "1,2"])
    assert (args.root, args.subjects, args.pad_sec, args.config, args.overrides) == (
        Path("/data"), ["1", "2"], 30.0, None, []
    )
    args = parser.parse_args(
        ["extract", "manifest", "m.csv", "--dataset", "daief", "--out", "d.npz", "--ch-names", "Fp1, Fz",
         "--sfreq", "512", "--scale", "1e-6"]
    )
    assert (args.dataset, args.ch_names, args.sfreq, args.scale) == ("daief", ["Fp1", "Fz"], 512.0, 1e-6)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["within", "f.npz", "--method", "svm", "--label", "valence"],  # unknown method
        ["within", "f.npz", "--method", "cnn", "--label", "dominance"],
        ["within", "f.npz", "--label", "valence"],  # --method is required
        ["within", "f.npz", "--method", "cnn", "--label", "valence", "--tag", "a/b"],
        ["within", "f.npz", "--method", "cnn", "--label", "valence", "--epochs", "0"],
        ["within", "f.npz", "--method", "cnn", "--label", "valence", "--lambda-domain", "0.1"],  # cross only
        ["extract", "manifest", "m.csv", "--out", "d.npz"],  # --dataset is required
        ["info", "f.npz", "-v", "-q"],
        ["-v", "info", "f.npz", "-q"],  # -v before the command and -q after it
    ],
)
def test_parser_rejects_invalid_arguments(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "usage: sieeg" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "verbose", "quiet"),
    [
        (["-v", "info", "f.npz"], True, False),
        (["info", "f.npz", "-v"], True, False),
        (["--quiet", "info", "f.npz"], False, True),
        (["info", "-q", "f.npz"], False, True),
        (["-q", "info", "f.npz", "-q"], False, True),
        (["info", "f.npz"], False, False),
    ],
)
def test_verbosity_flags_go_before_or_after_the_command(argv, verbose, quiet):
    args = build_parser().parse_args(argv)
    assert (args.verbose, args.quiet) == (verbose, quiet)


def test_verbose_before_the_command_shows_the_traceback(tmp_path, capsys):
    missing = str(tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError):
        main(["-v", "info", missing])
    assert main(["-q", "info", missing]) == 1
    assert logging.getLogger("sieeg").level == logging.WARNING
    assert capsys.readouterr().err.startswith("sieeg info: error:")


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert all(command in out for command in ("extract", "info", "bias", "within", "cross", "report"))
    with pytest.raises(SystemExit):
        main(["extract", "--help"])
    out = capsys.readouterr().out
    assert all(kind in out for kind in ("mahnob", "manifest", "synthetic"))


def test_module_entry_point_and_info_do_not_import_tensorflow(features):
    """``python -m sieeg`` works, sets TF_CPP_MIN_LOG_LEVEL, and ``info`` needs neither TensorFlow nor scikit-learn."""
    env = _subprocess_env()
    help_text = subprocess.run(
        [sys.executable, "-m", "sieeg", "--help"], env=env, check=True, capture_output=True, text=True
    ).stdout
    assert "within" in help_text
    code = (
        "import os, sys; from sieeg.cli import main; "
        f"assert main(['info', {str(features)!r}]) == 0; "
        "assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '2'; "
        "heavy = {'tensorflow', 'keras', 'sklearn', 'mne'} & set(sys.modules); assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)


# ------------------------------------------------------------------- info / extract
def test_info_summarises_a_feature_file(features, capsys):
    assert main(["info", str(features)]) == 0
    out = capsys.readouterr().out
    assert "dataset     syn" in out
    assert "trials      40 from 5 subjects (8 per subject)" in out
    assert "channels    32: Fp1 AF3" in out
    assert "gamma 30-45 Hz" in out
    fs = FeatureSet.load(features)
    n_high = int((fs.valence > 5).sum())
    assert f"valence     40 rated, {n_high} high (> 5)" in out
    assert "generator   make_synthetic_featureset (seed 0)" in out
    table = out.split("\n\n")[1].splitlines()
    assert table[0].split() == ["subject", "trials", "valence>5", "arousal>5"]
    assert [row.split()[:2] for row in table[1:]] == [[f"S0{i}", "8"] for i in range(1, 6)]


def test_info_reports_a_missing_file(tmp_path, capsys):
    assert main(["info", str(tmp_path / "missing.npz")]) == 1
    assert capsys.readouterr().err.startswith("sieeg info: error:")


def test_extract_synthetic_features_appends_npz(tmp_path, capsys):
    assert main(["extract", "synthetic", "--out", str(tmp_path / "a"), "--n-subjects", "3", "--n-trials", "4",
                 "--dataset", "syn_a", "--domain-shift", "0.5", "--seed", "2"]) == 0
    path = tmp_path / "a.npz"
    assert capsys.readouterr().out.strip().endswith(f"-> {path}")
    fs = FeatureSet.load(path)
    assert (fs.dataset, len(fs), fs.subject_ids) == ("syn_a", 12, ["S01", "S02", "S03"])
    expected = make_synthetic_featureset(3, 4, dataset="syn_a", domain_shift=0.5, seed=2)
    np.testing.assert_array_equal(fs.band_power, expected.band_power)


def test_extract_refuses_to_overwrite_a_feature_file_without_overwrite(tmp_path, capsys):
    out = tmp_path / "big.npz"
    assert main(["extract", "synthetic", "--out", str(out), "--n-subjects", "4", "--n-trials", "3"]) == 0
    capsys.readouterr()
    for target in (out, tmp_path / "big"):  # without the suffix, FeatureSet.save would also write big.npz
        assert main(["extract", "synthetic", "--out", str(target), "--n-subjects", "2", "--n-trials", "2"]) == 1
        assert capsys.readouterr().err.strip() == f"sieeg extract: error: {out} exists; pass --overwrite to replace it"
    assert len(FeatureSet.load(out)) == 12  # untouched
    assert main(["extract", "synthetic", "--out", str(out), "--n-subjects", "2", "--n-trials", "2", "--overwrite"]) == 0
    assert len(FeatureSet.load(out)) == 4


def test_extract_refuses_before_reading_any_trial(tmp_path, monkeypatch):
    from sieeg.data import mahnob

    def unexpected(*args, **kwargs):
        raise AssertionError("the loader ran although the output exists")

    monkeypatch.setattr(mahnob, "iter_mahnob_trials", unexpected)
    (tmp_path / "m.npz").write_bytes(b"")
    assert main(["extract", "mahnob", str(tmp_path), "--out", str(tmp_path / "m.npz")]) == 1


def test_extract_synthetic_raw_runs_the_extraction_pipeline(tmp_path):
    out = tmp_path / "raw.npz"
    overrides = ["features.welch_segment_sec=10"]
    assert main(["extract", "synthetic", "--raw", "--out", str(out), "--n-subjects", "2", "--n-trials", "2",
                 "--config", SMOKE, "--set", overrides[0]]) == 0
    fs = FeatureSet.load(out)
    assert fs.meta["feature_config"]["resample_sfreq"] == 128.0  # from smoke.yaml
    assert fs.meta["feature_config"]["welch_segment_sec"] == 10.0  # from --set
    expected = extract_features(make_synthetic_trials(2, 2), load_config(SMOKE, overrides).features)
    np.testing.assert_allclose(fs.band_power, expected.band_power)


def test_extract_manifest_uses_the_array_metadata_options(tmp_path):
    trials = list(make_synthetic_trials(n_subjects=2, n_trials=2, seed=3))
    rows = []
    for trial in trials:
        name = f"{trial.subject}_{trial.trial}.npy"
        np.save(tmp_path / name, trial.data * 1e6)  # microvolts, no metadata in the file
        rows.append({"subject": trial.subject, "trial": trial.trial, "path": name,
                     "valence": trial.valence, "arousal": trial.arousal})
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    out = tmp_path / "daief.npz"
    assert main(["extract", "manifest", str(tmp_path / "manifest.csv"), "--dataset", "daief", "--out", str(out),
                 "--sfreq", "128", "--ch-names", ",".join(trials[0].ch_names), "--scale", "1e-6",
                 "--config", SMOKE]) == 0
    fs = FeatureSet.load(out)
    expected = extract_features(trials, load_config(SMOKE).features)
    assert (fs.dataset, fs.subject_ids, fs.trials.tolist()) == ("daief", ["S01", "S02"], ["T01", "T02"] * 2)
    np.testing.assert_allclose(fs.band_power, expected.band_power, rtol=1e-5)


def test_extract_manifest_records_the_rows_the_loader_skips(tmp_path, capsys):
    """A manifest row whose file is missing is dropped by the loader, but still listed in the feature file."""
    trials = list(make_synthetic_trials(n_subjects=2, n_trials=1, seed=3))
    rows = []
    for trial in trials:
        np.savez(tmp_path / f"{trial.subject}.npz", data=trial.data, sfreq=trial.sfreq, ch_names=trial.ch_names)
        rows.append({"subject": trial.subject, "trial": trial.trial, "path": f"{trial.subject}.npz",
                     "valence": trial.valence, "arousal": trial.arousal})
    rows.append({"subject": "S03", "trial": "T01", "path": "S3.npz", "valence": 7, "arousal": 3})  # typo
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    out = tmp_path / "daief.npz"
    assert main(["extract", "manifest", str(tmp_path / "manifest.csv"), "--dataset", "daief", "--out", str(out),
                 "--config", SMOKE]) == 0
    assert ", 1 trials skipped (see sieeg info) -> " in capsys.readouterr().out
    fs = FeatureSet.load(out)
    assert (len(fs), fs.meta["n_input_trials"]) == (2, 3)
    [skipped] = fs.meta["skipped"]
    assert (skipped["subject"], skipped["trial"]) == ("S03", "T01") and "S3.npz" in skipped["reason"]


def test_extract_mahnob_passes_subjects_padding_and_configured_channels(tmp_path, monkeypatch):
    from sieeg.data import mahnob

    seen = {}
    no_bdf = {"subject": "2", "trial": "14", "reason": "rated session without a BDF file"}

    def fake_iter_mahnob_trials(root, **kwargs):
        seen.update(root=root, **kwargs)
        yield from make_synthetic_trials(n_subjects=1, n_trials=2, dataset="mahnob")
        kwargs["skipped"].append(no_bdf)  # loaders complete the list once they are exhausted

    monkeypatch.setattr(mahnob, "iter_mahnob_trials", fake_iter_mahnob_trials)
    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    assert main(["extract", "mahnob", str(tmp_path / "MAHNOB-HCI"), "--out", str(tmp_path / "m.npz"),
                 "--subjects", "1,2", "--pad-sec", "25", "--config", SMOKE,
                 "--set", f"features.channels=[{', '.join(channels)}]"]) == 0
    assert seen["root"] == tmp_path / "MAHNOB-HCI"
    assert (seen["subjects"], seen["pad_sec"], list(seen["channels"])) == (["1", "2"], 25.0, channels)
    fs = FeatureSet.load(tmp_path / "m.npz")
    assert (fs.dataset, fs.ch_names, len(fs)) == ("mahnob", channels, 2)
    assert (fs.meta["skipped"], fs.meta["n_input_trials"]) == ([no_bdf], 3)


# ------------------------------------------------------------------- bias
def test_bias_prints_and_saves_the_experiment_1_table(tmp_path, capsys):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    make_synthetic_featureset(n_subjects=4, n_trials=10, dataset="a", seed=0).save(a)
    make_synthetic_featureset(n_subjects=3, n_trials=10, dataset="b", seed=1, domain_shift=1.0).save(b)
    out = tmp_path / "results" / "bias.csv"
    assert main(["bias", str(a), "--other", str(b), "--exclude-subjects", "S04", "--n-splits", "3",
                 "--n-estimators", "10", "--jobs", "1", "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert list(zip(table["task"], table["cv"], table["normalization"])) == [
        ("subject", "trial", "none"), ("subject", "trial", "subject"), ("domain", "trial", "none"),
        ("domain", "trial", "subject"), ("domain", "subject", "none"), ("domain", "subject", "subject"),
    ]
    assert list(table.columns[:8]) == [
        "task", "normalization", "cv", "accuracy_pct", "accuracy_std_pct", "balanced_accuracy_pct",
        "chance_uniform_pct", "chance_majority_pct",
    ]
    assert (table.loc[0, "n_classes"], table.loc[0, "n_samples"]) == (3, 30)  # S04 left out
    assert json.loads(table.loc[2, "n_per_domain"]) == {"a": 30, "b": 30}
    assert table.loc[0, "chance_uniform_pct"] == pytest.approx(100 / 3)  # percent, as printed
    assert table.loc[2, "chance_uniform_pct"] == 50.0 and pd.isna(table.loc[0, "balanced_accuracy_pct"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7
    assert lines[0].startswith("subject identification, a, normalization none: accuracy ")
    for line, row in zip(lines[:6], table.to_dict("records")):
        assert f"accuracy {row['accuracy_pct']:.1f} % +- {row['accuracy_std_pct']:.1f}" in line
    assert lines[2].startswith(
        "database identification (trial-level folds), a (30 trials) vs b (30 trials), normalization none"
    )
    assert lines[4].startswith("database identification (subject-grouped folds), a (30 trials) vs b (30 trials)")
    assert lines[6].startswith("Database identification: trial-level folds follow the paper")


# ------------------------------------------------------------------- within / cross
def test_within_saves_one_run_directory_per_method_and_label(features, tmp_path, recorder, capsys):
    config = tmp_path / "exp.yaml"
    config.write_text("seed: 3\ntrain:\n  epochs: 10\n  batch_size: 8\n", encoding="utf-8")
    out = tmp_path / "runs"
    argv = ["within", str(features), "--method", "cnn,si-eeg", "--label", "both", "--folds", "S01,S02",
            "--exclude-subjects", "S05", "--out", str(out), "--tag", "t1", "--config", str(config),
            "--set", "train.epochs=20", "--set", "protocol.n_val_subjects=2", "--epochs", "30",
            "--lambda-subject", "0.3", "--jobs", "2"]
    assert main(argv) == 0

    assert [(c["method"].name, c["label"]) for c in recorder] == [
        ("cnn", "valence"), ("cnn", "arousal"), ("si_eeg", "valence"), ("si_eeg", "arousal")
    ]
    call = recorder[2]
    assert [(a.kind, a.lam) for a in call["method"].adversaries] == [("group", 0.3)]
    assert (call["folds"], call["tag"], call["fs"].subject_ids) == (["S01", "S02"], "t1", ["S01", "S02", "S03", "S04"])
    cfg = call["cfg"]  # YAML < --set < flags
    assert (cfg.seed, cfg.train.epochs, cfg.train.batch_size, cfg.protocol.n_val_subjects) == (3, 30, 8, 2)
    assert (cfg.protocol.lambda_subject, cfg.protocol.jobs) == (0.3, 2)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    for method in ("cnn", "si_eeg"):
        for label in ("valence", "arousal"):
            run_dir = out / "within-syn" / f"{method}-t1" / label / "seed3"
            assert sum(line.endswith(f"-> {run_dir}") for line in lines) == 1
            run = load_run(run_dir)
            assert set(SUMMARY_KEYS) <= set(run.summary)
            assert (run.summary["method"], run.summary["label"], run.summary["tag"]) == (method, label, "t1")
            saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            assert (saved["seed"], saved["train"]["epochs"], saved["protocol"]["lambda_subject"]) == (3, 30, 0.3)
            assert load_config(run_dir / "config.yaml") == cfg
            command = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
            assert command["argv"] == argv and command["command"].startswith("sieeg within ")
            assert command["arguments"]["exclude_subjects"] == ["S05"]
            assert sorted(p.name for p in (run_dir / "history").iterdir()) == ["S01.csv", "S02.csv"]
            assert sorted(p.name for p in run_dir.iterdir()) == [
                "command.json", "config.yaml", "features.json", "folds.csv", "history", "predictions.csv",
                "summary.json",
            ]  # the .running claim is gone


def test_within_refuses_to_overwrite_a_run_without_overwrite(features, tmp_path, recorder, capsys):
    argv = ["within", str(features), "--method", "cnn", "--label", "valence", "--out", str(tmp_path / "runs"), *N_VAL]
    assert main(argv) == 0
    run_dir = tmp_path / "runs" / "within-syn" / "cnn" / "valence" / "seed0"
    stale = run_dir / "history" / "old_fold.csv"
    stale.write_text("epoch\n1\n", encoding="utf-8")
    capsys.readouterr()

    assert main([*argv, "--method", "si_eeg,cnn"]) == 1  # checked before anything trains
    err = capsys.readouterr().err
    assert str(run_dir) in err and "--overwrite" in err
    assert len(recorder) == 1

    assert main([*argv, "--overwrite"]) == 0
    assert len(recorder) == 2 and not stale.exists() and (run_dir / "history" / "S01.csv").exists()
    assert main([*argv, "--seed", "1"]) == 0  # another seed is another run directory
    assert (tmp_path / "runs" / "within-syn" / "cnn" / "valence" / "seed1" / "summary.json").exists()


def test_result_line_of_a_single_test_subject_has_no_spread(features, tmp_path, recorder, capsys):
    assert main(["within", str(features), "--method", "cnn", "--label", "valence", "--folds", "S03",
                 "--out", str(tmp_path / "runs"), *N_VAL]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("within syn cnn valence: CR ") and "+-" not in line and "nan" not in line
    assert ", 1 subject, 8 trials) -> " in line  # the SD over one subject is undefined, so it is left out


def test_within_rejects_methods_that_need_a_target_database(features, tmp_path, recorder, capsys):
    assert main(["within", str(features), "--method", "cnn,sidann", "--label", "valence",
                 "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "sidann needs a second, different target database (domain adversary): sieeg cross SOURCE TARGET" in err
    assert recorder == [] and not any(tmp_path.glob("within-*"))


def test_within_checks_folds_and_validation_of_every_label_before_training(tmp_path, recorder, capsys):
    """S05 has valence but no arousal ratings: --label both --folds S05 fails before valence trains."""
    fs = make_synthetic_featureset(n_subjects=5, n_trials=4, dataset="syn", seed=0)
    fs = dataclasses.replace(fs, arousal=np.where(fs.subjects == "S05", np.nan, fs.arousal))
    path = tmp_path / "syn.npz"
    fs.save(path)
    out = tmp_path / "runs"
    base = ["within", str(path), "--method", "cnn,si_eeg", "--label", "both", "--out", str(out), *N_VAL]

    assert main([*base, "--folds", "S02,S05"]) == 1
    err = capsys.readouterr().err.strip()
    assert err == (
        "sieeg within: error: arousal: --folds S05 is not rated (rated syn subjects: S01, S02, S03, S04); "
        "nothing was trained"
    )
    assert main([*base, "--folds", "S01,S03,S9", "--exclude-subjects", "S03"]) == 1
    err = capsys.readouterr().err
    assert (
        "valence: --folds S03 is excluded, S9 is not a subject of syn (rated syn subjects: S01, S02, S04, S05)"
    ) in err
    assert main([*base, "--label", "arousal", "--set", "protocol.n_val_subjects=3"]) == 1  # 4 rated subjects
    assert "arousal: protocol.n_val_subjects=3 leaves no training subject: syn has 4 rated subjects" in (
        capsys.readouterr().err
    )
    assert recorder == [] and not out.exists()
    assert main([*base, "--folds", "S02"]) == 0 and len(recorder) == 4


def test_cross_names_runs_after_the_target_and_applies_its_options(tmp_path, recorder, capsys):
    source, target = tmp_path / "src.npz", tmp_path / "tgt.npz"
    make_synthetic_featureset(n_subjects=4, n_trials=6, dataset="src", seed=1).save(source)
    make_synthetic_featureset(n_subjects=3, n_trials=6, dataset="tgt", seed=2, subject_prefix="T").save(target)
    out = tmp_path / "runs"
    assert main(["cross", str(source), str(target), "--method", "sida_eeg,cnn", "--label", "arousal",
                 "--exclude-source", "S04", "--exclude-target", "T03", "--folds", "T01",
                 "--lambda-subject", "0.2", "--lambda-domain", "0.05", "--seed", "1", "--out", str(out),
                 "--set", "protocol.n_val_subjects=1"]) == 0
    assert [c["method"].name for c in recorder] == ["sida_eeg", "cnn"]
    call = recorder[0]
    assert (call["source"].subject_ids, call["target"].subject_ids) == (["S01", "S02", "S03"], ["T01", "T02"])
    assert {a.kind: a.lam for a in call["method"].adversaries} == {"domain": 0.05, "group": 0.2}
    assert (call["label"], call["folds"], call["cfg"].seed, call["cfg"].protocol.lambda_domain) == (
        "arousal", ["T01"], 1, 0.05
    )
    for method in ("sida_eeg", "cnn"):
        summary = json.loads((out / "cross-tgt" / method / "arousal" / "seed1" / "summary.json").read_text())
        assert (summary["train_dataset"], summary["test_dataset"]) == ("src", "tgt")
    assert len(capsys.readouterr().out.splitlines()) == 2


# ------------------------------------------------------------------- report
def test_report_prints_and_writes_results_and_paired_tests(features, tmp_path, recorder, capsys):
    runs = tmp_path / "runs"
    assert main(["within", str(features), "--method", "cnn,sinn,si_eeg", "--label", "both", "--out", str(runs),
                 *N_VAL]) == 0
    capsys.readouterr()
    report = tmp_path / "results" / "table1.md"
    assert main(["report", str(runs), "--reference", "si-eeg", "--metric", "cr", "--out", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert capsys.readouterr().out == text
    results, tests = text.split("## Paired tests")
    assert "A row averages the runs with the same protocol, databases, method, tag and lambdas" in results
    assert "Runs that differ only in seed" not in results
    assert (
        "| protocol | test_dataset | train_dataset | method | tag | lambdas | valence_cr | valence_f1 | arousal_cr "
        "| arousal_f1 | n_runs | n_subjects |"
    ) in results
    rows = [line.split(" | ")[3] for line in results.splitlines() if line.startswith("| within |")]
    assert rows == ["cnn", "sinn", "si_eeg"]
    assert '| within | syn | syn | si_eeg |  | {"group": 0.2} | 100.0 | 100.0 | 100.0 | 100.0 | 1 | 5 |' in results
    # the fake si_eeg is always right
    assert '| within | syn | syn |  | valence | si_eeg | {"group": 0.2} | cnn | {} | 5 | 100 |' in tests
    assert sum(line.startswith("| within |") for line in tests.splitlines()) == 4  # 2 labels x (cnn, sinn)
    assert "Subject probes" not in text  # none of the runs measured them

    assert main(["report", str(runs), "--reference", "dann"]) == 0
    assert "No comparison: no protocol, databases, tag and label have runs of dann" in capsys.readouterr().out
    assert main(["report", str(tmp_path / "results")]) == 1  # no summary.json below it
    assert "no run directories" in capsys.readouterr().err


def test_report_compares_tagged_runs_with_like_tagged_runs_only(features, tmp_path, recorder, capsys):
    runs = tmp_path / "runs"
    assert main(["within", str(features), "--method", "cnn", "--label", "valence", "--out", str(runs), *N_VAL]) == 0
    assert main(["within", str(features), "--method", "si_eeg", "--label", "valence", "--tag", "v2",
                 "--out", str(runs), *N_VAL]) == 0
    capsys.readouterr()
    assert main(["report", str(runs), "--reference", "si_eeg"]) == 0
    out = capsys.readouterr().out
    assert "| within | syn | syn | si_eeg | v2 |" in out
    assert "(runs with different --tag are not compared)" in out


def test_report_lists_the_subject_probes_of_runs_trained_with_probe(features, tmp_path, recorder, capsys):
    runs = tmp_path / "runs"
    assert main(["within", str(features), "--method", "si_eeg", "--label", "valence", "--out", str(runs),
                 "--probe", *N_VAL]) == 0
    assert recorder[0]["cfg"].protocol.subject_probe is True
    line = capsys.readouterr().out
    assert "; probes: subject 0.40 (chance 0.25), group 0.55) -> " in line
    run_dir = runs / "within-syn" / "si_eeg" / "valence" / "seed0"
    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["protocol"]["subject_probe"] is True
    assert main(["report", str(runs), "--reference", "si_eeg"]) == 0
    probes = capsys.readouterr().out.split("## Subject probes (not in the paper)")[1]
    assert "0.5 = indistinguishable" in probes
    assert (
        "| protocol | test_dataset | train_dataset | method | tag | lambdas | label | n_runs | probe_subject_acc "
        "| probe_subject_chance | probe_group_bacc |"
    ) in probes
    assert '| within | syn | syn | si_eeg |  | {"group": 0.2} | valence | 1 | 0.400 | 0.250 | 0.550 |' in probes


# ------------------------------------------------------------------- run directories
def test_run_directories_are_claimed_while_training(features, tmp_path, recorder, capsys):
    out = tmp_path / "runs"
    claims = []

    def look(protocol, label):
        claims.append({p.parents[2].name: json.loads(p.read_text()) for p in out.rglob(".running")})

    recorder.hooks.append(look)
    argv = ["within", str(features), "--method", "cnn,si_eeg", "--label", "valence", "--out", str(out), *N_VAL]
    assert main(argv) == 0
    # every run directory is claimed before anything trains; a saved run gives its claim up
    assert [sorted(claim) for claim in claims] == [["cnn", "si_eeg"], ["si_eeg"]]
    holder = claims[0]["si_eeg"]
    assert (holder["pid"], holder["host"], holder["argv"]) == (os.getpid(), socket.gethostname(), ["sieeg", *argv])
    assert not list(out.rglob(".running"))


def test_a_claimed_run_directory_is_refused_unless_overwrite(features, tmp_path, recorder, capsys):
    out = tmp_path / "runs"
    run_dir = out / "within-syn" / "cnn" / "valence" / "seed0"
    run_dir.mkdir(parents=True)
    marker = run_dir / ".running"
    other = {"pid": 4242, "host": "node7", "argv": ["sieeg", "within"], "started": "2026-11-15T09:00:00+00:00"}
    marker.write_text(json.dumps(other) + "\n", encoding="utf-8")
    argv = ["within", str(features), "--method", "si_eeg,cnn", "--label", "valence", "--out", str(out), *N_VAL]

    assert main(argv) == 1
    err = capsys.readouterr().err
    assert f"{run_dir} is being written by another sieeg process (pid 4242 on node7 since" in err
    assert f"delete {marker} or pass --overwrite" in err
    assert recorder == [] and json.loads(marker.read_text()) == other  # nothing trained, the claim is kept
    assert not (out / "within-syn" / "si_eeg").exists()  # the claim this command made first was undone

    assert main([*argv, "--overwrite"]) == 0  # takes the claim over
    assert len(recorder) == 2 and not marker.exists() and (run_dir / "summary.json").exists()


def test_a_run_saved_by_another_process_during_training_is_kept(features, tmp_path, recorder, capsys):
    out = tmp_path / "runs"
    run_dir = out / "within-syn" / "cnn" / "valence" / "seed0"

    def other_process(protocol, label):
        (run_dir / "summary.json").write_text('{"other": true}\n', encoding="utf-8")

    recorder.hooks.append(other_process)
    argv = ["within", str(features), "--method", "cnn", "--label", "valence", "--out", str(out), *N_VAL]
    assert main(argv) == 1
    assert "another process saved a run in" in capsys.readouterr().err
    assert sorted(p.name for p in run_dir.iterdir()) == ["summary.json"]  # untouched, and no .running left
    assert json.loads((run_dir / "summary.json").read_text()) == {"other": True}


def test_a_failed_run_releases_its_claims_and_leaves_no_empty_directories(features, tmp_path, recorder, capsys):
    out = tmp_path / "runs"

    def fail(protocol, label):
        if label == "arousal":
            raise ValueError("training failed")

    recorder.hooks.append(fail)
    assert main(["within", str(features), "--method", "cnn", "--label", "both", "--out", str(out), *N_VAL]) == 1
    assert capsys.readouterr().err.strip() == "sieeg within: error: training failed"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_dir()) == [
        "within-syn", "within-syn/cnn", "within-syn/cnn/valence", "within-syn/cnn/valence/seed0",
        "within-syn/cnn/valence/seed0/history",
    ]  # the saved valence run; the arousal directory was removed
    assert not list(out.rglob(".running"))


def test_features_json_records_the_input_feature_files(features, tmp_path, recorder):
    assert main(["within", str(features), "--method", "cnn", "--label", "valence", "--exclude-subjects", "S05",
                 "--out", str(tmp_path / "runs"), *N_VAL]) == 0
    record = json.loads((tmp_path / "runs/within-syn/cnn/valence/seed0/features.json").read_text())["file"]
    fs = FeatureSet.load(features)
    assert record["path"] == str(features.resolve())
    assert record["sha256"] == hashlib.sha256(features.read_bytes()).hexdigest()
    assert (record["dataset"], record["excluded_subjects"], record["n_trials"]) == ("syn", ["S05"], 32)
    assert record["subjects"] == ["S01", "S02", "S03", "S04"]
    assert (record["ch_names"], record["bands"]) == (fs.ch_names, [list(b) for b in fs.bands])
    assert record["meta"] == fs.meta

    source, target = tmp_path / "src.npz", tmp_path / "tgt.npz"
    make_synthetic_featureset(n_subjects=3, n_trials=4, dataset="src", seed=1).save(source)
    make_synthetic_featureset(n_subjects=2, n_trials=4, dataset="tgt", seed=2, subject_prefix="T").save(target)
    assert main(["cross", str(source), str(target), "--method", "dann", "--label", "valence",
                 "--out", str(tmp_path / "runs"), "--set", "protocol.n_val_subjects=1"]) == 0
    records = json.loads((tmp_path / "runs/cross-tgt/dann/valence/seed0/features.json").read_text())
    assert [records[role]["dataset"] for role in ("source", "target")] == ["src", "tgt"]
    assert records["target"]["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


# ------------------------------------------------------------------- configuration errors
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--set", "train.epochs=0"], "train.epochs must be >= 1, got 0"),
        (["--set", "train.epochs=1O"], "train.epochs must be an integer, got '1O'"),
        (["--set", "train=5"], "section 'train' must be a mapping of keys to values, got 5"),
        (["--set", "train.batch_size=0"], "train.batch_size must be >= 1, got 0"),
        (["--set", "protocol.normalization=zscore"], "protocol.normalization must be one of: subject, none"),
        (["--config", "BAD"], "invalid YAML at line 3, column 2"),
    ],
)
def test_invalid_configuration_is_a_one_line_error(args, message, features, tmp_path, recorder, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("train:\n  epochs: 2\n batch_size: 8\n", encoding="utf-8")
    args = [str(bad) if arg == "BAD" else arg for arg in args]
    assert main(["within", str(features), "--method", "cnn", "--label", "valence", "--out", str(tmp_path / "runs"),
                 *args]) == 1
    err = capsys.readouterr().err
    assert err.startswith("sieeg within: error: ") and message in err and len(err.splitlines()) == 1
    assert recorder == []


def test_type_errors_are_one_line_errors(features, tmp_path, recorder, capsys):
    def broken(protocol, label):
        raise TypeError("can only concatenate str (not \"int\") to str")

    recorder.hooks.append(broken)
    assert main(["within", str(features), "--method", "cnn", "--label", "valence", "--out", str(tmp_path), *N_VAL]) == 1
    assert capsys.readouterr().err == 'sieeg within: error: can only concatenate str (not "int") to str\n'


# ------------------------------------------------------------------- end to end
@pytest.mark.slow
def test_end_to_end_on_synthetic_data(tmp_path, capsys):
    """extract (--raw) -> info -> bias -> within -> cross -> report with configs/smoke.yaml and 1 epoch."""
    source, target, runs = tmp_path / "src.npz", tmp_path / "tgt.npz", tmp_path / "runs"
    assert main(["extract", "synthetic", "--raw", "--n-subjects", "5", "--n-trials", "6", "--dataset", "src",
                 "--config", SMOKE, "--out", str(source)]) == 0
    assert main(["extract", "synthetic", "--n-subjects", "3", "--n-trials", "6", "--dataset", "tgt",
                 "--domain-shift", "0.5", "--seed", "1", "--out", str(target)]) == 0
    assert main(["info", str(source)]) == 0
    assert main(["bias", str(source), "--other", str(target), "--n-splits", "2", "--n-estimators", "5",
                 "--jobs", "1", "--out", str(tmp_path / "bias.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "bias.csv")) == 6  # subject x 2, database x 2 cv x 2 normalisations

    train = ["--config", SMOKE, "--set", "train.epochs=1", "--out", str(runs)]
    assert main(["within", str(source), "--method", "cnn,si_eeg", "--label", "valence",
                 "--folds", "S01,S02", *train]) == 0
    assert main(["cross", str(source), str(target), "--method", "sida_eeg", "--label", "arousal",
                 "--folds", "S01", *train]) == 0
    expected = {
        runs / "within-src" / "cnn" / "valence" / "seed0": 2,
        runs / "within-src" / "si_eeg" / "valence" / "seed0": 2,
        runs / "cross-tgt" / "sida_eeg" / "arousal" / "seed0": 1,
    }
    for run_dir, n_folds in expected.items():
        assert {"summary.json", "predictions.csv", "folds.csv", "config.yaml", "command.json", "features.json"} <= {
            p.name for p in run_dir.iterdir()
        }
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["n_folds"] == n_folds and 0 <= summary["pooled"]["f1"] <= 100
        assert load_config(run_dir / "config.yaml").train.epochs == 1

    table = tmp_path / "table1.md"
    assert main(["report", str(runs), "--reference", "si_eeg,sida_eeg", "--out", str(table)]) == 0
    text = table.read_text(encoding="utf-8")
    for row in ("| within | src | src | cnn |", "| within | src | src | si_eeg |", "| cross | tgt | src | sida_eeg |"):
        assert row in text
    assert '| within | src | src |  | valence | si_eeg | {"group": 0.2} | cnn | {} | 2 |' in text
    out = capsys.readouterr().out
    assert all(f"-> {run_dir}" in out for run_dir in expected)
