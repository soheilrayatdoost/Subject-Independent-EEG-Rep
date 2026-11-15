"""Run directories, Table 1 / significance tables and Markdown output (docs/DESIGN.md 8.3)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sieeg import __version__
from sieeg.config import load_config
from sieeg.evaluation.metrics import per_subject_metrics, summarize_predictions
from sieeg.evaluation.report import (
    PROBE_COLUMNS,
    RESULT_COLUMNS,
    SUMMARY_KEYS,
    RunResult,
    find_runs,
    load_run,
    probe_table,
    results_table,
    save_run,
    significance_table,
    to_markdown,
)
from sieeg.evaluation.stats import COMPARISON_FIELDS, paired_comparison
from sieeg.training.methods import get_method

RUN_KEY = ["protocol", "test_dataset", "train_dataset", "method", "tag", "lambdas"]
TABLE_COLUMNS = [*RUN_KEY, *RESULT_COLUMNS, "n_runs", "n_subjects"]


def _make_run(
    method: str,
    label: str,
    correct: list[int],
    *,
    n_trials: int | list[int] = 10,
    protocol: str = "within",
    test_dataset: str = "daief",
    train_dataset: str | None = None,
    tag: str | None = None,
    seed: int = 0,
    lambda_subject: float | None = None,
    first_subject: int = 1,
    probe: dict | None = None,
) -> RunResult:
    """Fabricated run: test subject i ("01", "02", ...) gets ``correct[i]`` of its ``n_trials[i]`` trials right.

    ``first_subject`` numbers the test subjects from another id; ``train_dataset`` defaults to the
    test dataset within a database and to "daief" across databases.
    """
    subjects = [f"{first_subject + i:02d}" for i in range(len(correct))]
    sizes = [n_trials] * len(correct) if isinstance(n_trials, int) else n_trials
    rows = []
    for subject, n_correct, n in zip(subjects, correct, sizes):
        for t in range(n):
            y_true = t % 2
            y_pred = y_true if t < n_correct else 1 - y_true
            rows.append(
                {"fold": subject, "dataset": test_dataset, "subject": subject, "trial": f"T{t + 1}",
                 "rating": 7.0 if y_true else 3.0, "y_true": y_true, "y_pred": y_pred, "p1": 0.75 if y_pred else 0.25}
            )
    predictions = pd.DataFrame(rows)
    per_subject = per_subject_metrics(predictions)
    folds = pd.DataFrame(
        {"fold": subjects, "test_subjects": [[s] for s in subjects],
         "val_subjects": [[o for o in subjects if o != s][:2] for s in subjects],
         "n_train": 100, "n_val": 20, "n_test": sizes, "n_outgroup": sizes, "n_target": 0, "best_epoch": 2,
         "best_val": np.nan, "cr": per_subject["cr"], "f1": per_subject["f1"], "seconds": 1.5}
    )
    if train_dataset is None:
        train_dataset = test_dataset if protocol == "within" else "daief"
    summary = {
        "protocol": protocol, "method": method, "tag": tag, "label": label,
        "train_dataset": train_dataset, "test_dataset": test_dataset,
        "lambdas": {a.kind: a.lam for a in get_method(method, lambda_subject=lambda_subject).adversaries},
        "seed": seed, "n_folds": len(subjects), **summarize_predictions(predictions), "probe": probe,
        "created": "2026-11-15T12:00:00", "sieeg_version": __version__,
    }
    histories = {s: pd.DataFrame({"epoch": [1, 2], "loss_cls": [0.7, 0.6], "val_f1": [50.0, 55.0]}) for s in subjects}
    return RunResult(summary=summary, predictions=predictions, folds=folds, histories=histories)


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-standard JSON constant {name}")


# ------------------------------------------------------------------------ run io
def test_save_and_load_run_round_trip(tmp_path, tiny_config):
    run = _make_run("si_eeg", "valence", [7, 5, 9])
    assert set(run.summary) == set(SUMMARY_KEYS)
    run_dir = tmp_path / "runs" / "within-daief" / "si_eeg" / "valence" / "seed0"
    assert save_run(run, run_dir, config=tiny_config) == run_dir

    files = ["config.yaml", "folds.csv", "history", "predictions.csv", "summary.json"]
    assert sorted(p.name for p in run_dir.iterdir()) == files
    assert sorted(p.name for p in (run_dir / "history").iterdir()) == ["01.csv", "02.csv", "03.csv"]
    pd.testing.assert_frame_equal(pd.read_csv(run_dir / "history" / "02.csv"), run.histories["02"])
    assert load_config(run_dir / "config.yaml") == tiny_config

    loaded = load_run(run_dir)
    assert loaded.summary == run.summary
    assert loaded.histories == {}
    assert (loaded.run_dir, run.run_dir) == (run_dir, None)  # the tables name it in their warnings
    pd.testing.assert_frame_equal(loaded.predictions, run.predictions)  # subject "01" stays a string
    joined = run.folds.assign(test_subjects=["01", "02", "03"], val_subjects=["02;03", "01;03", "01;02"])
    pd.testing.assert_frame_equal(loaded.folds, joined)


def test_save_run_writes_strict_json_and_safe_history_names(tmp_path, caplog):
    run = _make_run("cnn", "arousal", [6])  # a single subject: NaN standard deviations
    run.summary["seed"] = np.int64(3)
    run.summary["n_folds"] = np.array(1)
    run.summary["lambdas"] = {"group": np.float32(0.25)}
    del run.summary["created"]
    run.histories = {"mahnob:1/a": run.histories["01"]}
    with caplog.at_level(logging.WARNING, logger="sieeg.evaluation.report"):
        run_dir = save_run(run, tmp_path / "run")
    assert "created" in caplog.text  # incomplete summaries are flagged

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"), parse_constant=_reject_constant)
    assert summary["seed"] == 3 and summary["n_folds"] == 1 and summary["lambdas"] == {"group": 0.25}
    assert summary["subject_mean"]["cr_std"] is None and summary["subject_mean"]["cr"] == 60.0
    assert not (run_dir / "config.yaml").exists()
    assert [p.name for p in (run_dir / "history").iterdir()] == ["mahnob_1_a.csv"]


def test_load_run_keeps_ids_that_look_like_numbers_or_missing_values(tmp_path):
    run = _make_run("cnn", "valence", [3, 4])
    run.predictions["subject"] = run.predictions["subject"].map({"01": "NA", "02": "007"})
    loaded = load_run(save_run(run, tmp_path / "run"))
    assert loaded.predictions["subject"].unique().tolist() == ["NA", "007"]
    assert loaded.folds["best_val"].isna().all()


def test_find_runs_in_nested_directories(tmp_path):
    run = _make_run("cnn", "valence", [5, 6])
    runs = tmp_path / "runs"
    for rel in ("within-daief/cnn/valence/seed10", "within-daief/cnn/valence/seed2",
                "cross-mahnob/dann/arousal/seed0"):
        save_run(run, runs / rel)
    (runs / "within-daief" / "cnn" / "arousal" / "seed0").mkdir(parents=True)  # no summary.json: not a run
    save_run(run, tmp_path / "single")

    assert find_runs([runs, tmp_path / "single"]) == [
        runs / "cross-mahnob/dann/arousal/seed0",
        runs / "within-daief/cnn/valence/seed2",
        runs / "within-daief/cnn/valence/seed10",
        tmp_path / "single",
    ]
    assert find_runs(tmp_path / "single") == [tmp_path / "single"]
    assert find_runs([runs / "within-daief", runs]) == find_runs(runs)  # overlapping roots: no duplicates
    with pytest.raises(FileNotFoundError):
        find_runs([tmp_path / "missing"])


def test_find_runs_ignores_running_claims(tmp_path):
    """A ``.running`` claim neither makes a directory a run nor hides a complete run that is being replaced."""
    runs = tmp_path / "runs"
    training = runs / "within-daief" / "cnn" / "valence" / "seed0"
    training.mkdir(parents=True)
    (training / ".running").write_text('{"pid": 1}\n', encoding="utf-8")
    replaced = save_run(_make_run("cnn", "arousal", [5, 6]), runs / "within-daief" / "cnn" / "arousal" / "seed0")
    (replaced / ".running").write_text('{"pid": 2}\n', encoding="utf-8")
    assert find_runs(runs) == [replaced]


# ------------------------------------------------------------------------ tables
def _table1_runs() -> list[RunResult]:
    return [
        _make_run("sida_eeg", "valence", [7, 6, 8], protocol="cross", test_dataset="mahnob"),
        _make_run("si_eeg", "valence", [8, 7, 9]),
        _make_run("cnn", "arousal", [5, 6, 6]),
        _make_run("cnn", "valence", [6, 6, 7], n_trials=[10, 10, 20]),  # pooled 19/40, subject mean (60+60+35)/3
    ]


def test_results_table_layout_order_and_values():
    runs = _table1_runs()
    table = results_table(runs)
    assert list(table.columns) == TABLE_COLUMNS
    assert table[RUN_KEY].values.tolist() == [
        ["within", "daief", "daief", "cnn", "", "{}"],
        ["within", "daief", "daief", "si_eeg", "", '{"group": 0.2}'],
        ["cross", "mahnob", "daief", "sida_eeg", "", '{"domain": 0.1, "group": 0.15}'],
    ]
    assert table.loc[0, "valence_cr"] == pytest.approx(47.5)
    assert table.loc[0, "valence_f1"] == pytest.approx(runs[3].summary["pooled"]["f1"])
    assert table.loc[0, "arousal_cr"] == pytest.approx(100 * 17 / 30)
    assert table.loc[2, "valence_f1"] == pytest.approx(runs[0].summary["pooled"]["f1"])
    assert table.loc[1, ["arousal_cr", "arousal_f1"]].isna().all()
    assert table[["n_runs", "n_subjects"]].values.tolist() == [[1, 3], [1, 3], [1, 3]]

    by_subject = results_table(runs, source="subject_mean")
    assert by_subject.loc[0, "valence_cr"] == pytest.approx(155 / 3)
    assert by_subject.loc[0, "valence_f1"] == pytest.approx(runs[3].summary["subject_mean"]["f1"])
    with pytest.raises(ValueError, match="source"):
        results_table(runs, source="median")


def test_results_table_averages_seeds_keeps_tags_apart_and_reads_directories(tmp_path):
    runs = [
        _make_run("cnn", "valence", [6, 8], seed=0),  # 70 %
        _make_run("cnn", "valence", [8, 8], seed=1),  # 80 %
        _make_run("cnn", "valence", [2, 2], tag="v2"),  # 20 %
    ]
    table = results_table(runs)
    assert table["tag"].tolist() == ["", "v2"]
    assert table["valence_cr"].tolist() == pytest.approx([75.0, 20.0])
    assert table[["n_runs", "n_subjects"]].values.tolist() == [[2, 2], [1, 2]]
    paths = [save_run(run, tmp_path / str(i)) for i, run in enumerate(runs)]
    pd.testing.assert_frame_equal(results_table(paths), table)
    assert results_table([]).columns.tolist() == TABLE_COLUMNS


def test_results_table_averages_only_runs_with_the_same_settings_and_databases(caplog):
    """Lambdas and the source database are part of a row; runs that differ in them are never averaged."""
    cross = {"protocol": "cross", "test_dataset": "mahnob"}
    runs = [
        _make_run("si_eeg", "valence", [9, 9], seed=0),  # lambda 0.2 (default): 90 %
        _make_run("si_eeg", "valence", [5, 5], seed=1, lambda_subject=0.5),  # 50 %
        _make_run("cnn", "valence", [8, 8], **cross),  # trained on daief: 80 %
        _make_run("cnn", "valence", [4, 4], train_dataset="deap", **cross),  # 40 %
    ]
    with caplog.at_level(logging.INFO, logger="sieeg.evaluation.report"):
        table = results_table(runs)
    assert table[["train_dataset", "method", "lambdas", "valence_cr", "n_runs"]].values.tolist() == [
        ["daief", "si_eeg", '{"group": 0.2}', 90.0, 1],
        ["daief", "si_eeg", '{"group": 0.5}', 50.0, 1],
        ["daief", "cnn", "{}", 80.0, 1],
        ["deap", "cnn", "{}", 40.0, 1],
    ]
    assert "Averaging" not in caplog.text and not caplog.records


def test_results_table_warns_about_repeated_seeds_and_different_test_subjects(tmp_path, caplog):
    """Shards of one seed saved under two roots, or a --folds check next to a full run, are named in warnings."""
    shards = [
        save_run(_make_run("cnn", "valence", [6, 8, 7]), tmp_path / "shard1" / "seed0"),  # subjects 01-03
        save_run(_make_run("cnn", "valence", [9, 9, 9], first_subject=4), tmp_path / "shard2" / "seed0"),  # 04-06
        save_run(_make_run("cnn", "arousal", [5, 5, 5, 5, 5, 5]), tmp_path / "full" / "seed0"),  # 01-06
    ]
    with caplog.at_level(logging.INFO, logger="sieeg.evaluation.report"):
        table = results_table(shards)
    assert table[["valence_cr", "arousal_cr", "n_runs", "n_subjects"]].values.tolist() == [[80.0, 50.0, 2, 6]]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    seeds, subjects = warnings
    assert "same label and seed" in seeds and str(shards[0]) in seeds and str(shards[1]) in seeds
    assert str(shards[2]) not in seeds  # arousal seed 0 is another label, not a repeat
    assert "test different subjects" in subjects and f"{shards[0]} (3 subjects)" in subjects
    assert f"{shards[2]} (6 subjects)" in subjects
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert info == ["Averaging 2 runs of within daief cnn, valence"]  # no claim that they are seeds

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sieeg.evaluation.report"):
        results_table([_make_run("cnn", "valence", [6, 8], seed=s) for s in (0, 1)])  # seeds of one experiment
    assert not caplog.records


def _f1_by_subject(run: RunResult) -> pd.Series:
    return per_subject_metrics(run.predictions).set_index("subject")["f1"]


def test_significance_table_compares_reference_with_every_other_method():
    runs = [  # no subject scores the same twice, so every Wilcoxon test is exact
        _make_run("cnn", "valence", [6, 5, 5, 3, 4]),
        _make_run("si_eeg", "valence", [9, 8, 9, 7, 8]),
        _make_run("sinn", "valence", [7, 7, 6, 4, 3]),
        _make_run("si_eeg", "arousal", [7, 8, 7, 9, 6]),
        _make_run("cnn", "arousal", [5, 5, 6, 7]),  # 4 of the 5 subjects
        _make_run("cnn", "valence", [6, 6], protocol="cross", test_dataset="mahnob"),  # no reference: skipped
    ]
    table = significance_table(runs, reference="SI-EEG")
    key = ["protocol", "test_dataset", "train_dataset", "tag", "label", "reference", "reference_lambdas", "method",
           "method_lambdas"]
    assert list(table.columns) == [*key, *COMPARISON_FIELDS]
    assert table[["train_dataset", "reference_lambdas", "method_lambdas"]].values.tolist() == [
        ["daief", '{"group": 0.2}', "{}"], ["daief", '{"group": 0.2}', '{"identity": 0.2}'],
        ["daief", '{"group": 0.2}', "{}"],
    ]
    assert table[["protocol", "label", "reference", "method"]].values.tolist() == [
        ["within", "valence", "si_eeg", "cnn"],
        ["within", "valence", "si_eeg", "sinn"],
        ["within", "arousal", "si_eeg", "cnn"],
    ]
    for row, (ref, other) in zip(table.to_dict("records"), [(1, 0), (1, 2), (3, 4)]):
        expected = paired_comparison(_f1_by_subject(runs[ref]), _f1_by_subject(runs[other]))
        assert {k: row[k] for k in COMPARISON_FIELDS} == pytest.approx(expected, nan_ok=True)
    assert table["n"].tolist() == [5, 5, 4]
    assert (table["mean_diff"] > 0).all()  # the reference (a) scores higher here

    by_cr = significance_table(runs, reference="si_eeg", metric="cr")
    assert by_cr.loc[0, "mean_a"] == pytest.approx(82.0) and by_cr.loc[0, "mean_b"] == pytest.approx(46.0)
    with pytest.raises(ValueError, match="metric"):
        significance_table(runs, reference="si_eeg", metric="auc")


def test_significance_table_averages_seeds_per_subject_and_skips_small_overlaps():
    cross = {"protocol": "cross", "test_dataset": "mahnob"}
    runs = [
        _make_run("sida_eeg", "valence", [8, 6, 7], seed=0, **cross),
        _make_run("sida_eeg", "valence", [6, 8, 9], seed=1, **cross),  # per subject: 70, 70, 80 %
        _make_run("dann", "valence", [5, 6, 6], **cross),
        _make_run("cnn", "valence", [5], **cross),  # one common subject: no paired test
    ]
    table = significance_table(runs, reference="sida_eeg", metric="cr")
    assert table["method"].tolist() == ["dann"]
    assert table.loc[0, "mean_a"] == pytest.approx((70 + 70 + 80) / 3)
    assert table.loc[0, "mean_b"] == pytest.approx((50 + 60 + 60) / 3)


def test_significance_table_keeps_tags_lambdas_and_source_databases_apart():
    """Tagged variants are compared with like-tagged runs only; other lambdas or sources are separate entries."""
    runs = [
        _make_run("si_eeg", "valence", [9, 8, 9]),
        _make_run("si_eeg", "valence", [5, 5, 6], seed=1, lambda_subject=0.5),  # untagged other lambda
        _make_run("cnn", "valence", [6, 5, 5]),
        _make_run("si_eeg", "valence", [7, 7, 7], tag="v2"),  # no untagged baseline in its group
        _make_run("sida_eeg", "valence", [8, 7, 9], protocol="cross", test_dataset="mahnob"),
        _make_run("dann", "valence", [6, 6, 6], protocol="cross", test_dataset="mahnob", train_dataset="deap"),
    ]
    table = significance_table(runs, reference="si_eeg", metric="cr")
    assert table[["tag", "reference_lambdas", "method", "mean_a", "mean_b"]].values.tolist() == [
        ["", '{"group": 0.2}', "cnn", pytest.approx(260 / 3), pytest.approx(160 / 3)],
        ["", '{"group": 0.5}', "cnn", pytest.approx(160 / 3), pytest.approx(160 / 3)],
    ]
    assert significance_table(runs, reference="sida_eeg").empty  # dann was trained on another source database


# ---------------------------------------------------------------------- markdown
def test_to_markdown_formats_without_tabulate(monkeypatch):
    monkeypatch.setitem(sys.modules, "tabulate", None)  # importing tabulate would now fail
    df = pd.DataFrame({"method": ["cnn", "si|eeg"], "n": [3, 10], "cr": [68.04, np.nan], "tag": [None, "v2"]})
    assert to_markdown(df) == (
        "| method | n | cr | tag |\n"
        "|---|---:|---:|---|\n"
        "| cnn | 3 | 68.0 |  |\n"
        "| si\\|eeg | 10 |  | v2 |"
    )
    assert to_markdown(df, floatfmt=".3f").splitlines()[2] == "| cnn | 3 | 68.040 |  |"

    lines = to_markdown(results_table(_table1_runs())).splitlines()
    assert lines[0] == (
        "| protocol | test_dataset | train_dataset | method | tag | lambdas | valence_cr | valence_f1 | arousal_cr "
        "| arousal_f1 | n_runs | n_subjects |"
    )
    assert lines[1] == "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"
    assert lines[2].startswith("| within | daief | daief | cnn |  | {} | 47.5 |") and lines[2].endswith("| 1 | 3 |")
    assert len(lines) == 5


def test_probe_table_averages_the_subject_probes_of_probed_runs(tmp_path):
    probes = [
        {"subject_acc": 0.4, "subject_chance": 0.2, "group_bacc": 0.6},
        {"subject_acc": 0.2, "subject_chance": 0.2, "group_bacc": None},  # NaN, saved as null
    ]
    runs = [
        _make_run("si_eeg", "valence", [8, 7], seed=0, probe=probes[0]),
        load_run(save_run(_make_run("si_eeg", "valence", [8, 7], seed=1, probe=probes[1]), tmp_path / "run")),
        _make_run("cnn", "valence", [5, 6], probe={"subject_acc": 0.9, "subject_chance": 0.2, "group_bacc": 0.8}),
        _make_run("cnn", "arousal", [5, 6]),  # trained without probes
    ]
    table = probe_table(runs)
    assert list(table.columns) == [*RUN_KEY, "label", "n_runs", *PROBE_COLUMNS]
    assert table[["method", "label", "n_runs", *PROBE_COLUMNS]].values.tolist() == [
        ["cnn", "valence", 1, 0.9, 0.2, 0.8],
        ["si_eeg", "valence", 2, pytest.approx(0.3), 0.2, 0.6],
    ]
    assert probe_table(runs[3:]).empty and list(probe_table([]).columns) == list(table.columns)


def test_evaluation_and_bias_modules_do_not_import_tensorflow():
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(src), os.environ.get("PYTHONPATH")]))}
    code = (
        "import sys, sieeg.evaluation.report, sieeg.evaluation.metrics, sieeg.evaluation.stats, sieeg.analysis.bias; "
        "assert 'tensorflow' not in sys.modules and 'keras' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
