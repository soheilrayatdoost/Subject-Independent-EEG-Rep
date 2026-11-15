"""Experiment configuration: YAML round trip, overrides and validation (sieeg.config)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from sieeg.config import (
    CHOICES,
    MINIMUMS,
    PAPER_BANDS,
    ExperimentConfig,
    apply_overrides,
    config_from_dict,
    config_to_dict,
    dump_config,
    load_config,
    save_config,
    validate_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ------------------------------------------------------------------ round trip
def test_paper_yaml_holds_the_defaults_and_smoke_yaml_loads():
    assert load_config(CONFIGS / "paper.yaml") == ExperimentConfig()
    smoke = load_config(CONFIGS / "smoke.yaml")
    assert (smoke.features.resample_sfreq, smoke.model.conv_blocks, smoke.train.epochs) == (
        128.0, ((8, 8), (16, 16)), 3
    )


def test_save_and_load_round_trip(tmp_path):
    cfg = apply_overrides(
        ExperimentConfig(),
        ["seed=7", "model.conv_blocks=[[4], [8, 8]]", "features.reference=null", "train.patience=3",
         "train.class_weight=balanced", "protocol.lambda_subject=0.35", "protocol.subject_probe=true"],
    )
    save_config(cfg, tmp_path / "sub" / "config.yaml")
    assert load_config(tmp_path / "sub" / "config.yaml") == cfg
    assert config_from_dict(config_to_dict(cfg)) == cfg
    assert "conv_blocks:\n  - [4]\n  - [8, 8]" in dump_config(cfg)
    assert load_config() == ExperimentConfig()


def test_empty_file_and_empty_sections_keep_the_defaults(tmp_path):
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    (tmp_path / "sections.yaml").write_text("train:\nprotocol: {}\n", encoding="utf-8")
    assert load_config(tmp_path / "empty.yaml") == load_config(tmp_path / "sections.yaml") == ExperimentConfig()


# ------------------------------------------------------------------- overrides
def test_overrides_take_precedence_over_the_file_and_convert_types(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("seed: 3\ntrain:\n  epochs: 10\n  batch_size: 8\n", encoding="utf-8")
    cfg = load_config(path, ["train.epochs=20", "train.learning_rate=1e-3", "features.welch_segment_sec=10",
                             "features.channels=[Fz, Cz]", "features.bands=[[theta, 4, 8]]", "train.patience=null"])
    assert (cfg.seed, cfg.train.epochs, cfg.train.batch_size) == (3, 20, 8)
    assert cfg.train.learning_rate == 0.001  # PyYAML reads 1e-3 as a string; a number field converts it
    assert isinstance(cfg.features.welch_segment_sec, float) and cfg.features.welch_segment_sec == 10.0
    assert cfg.features.channels == ("Fz", "Cz") and cfg.features.bands == (("theta", 4.0, 8.0),)
    assert cfg.train.patience is None and cfg.features.bands != PAPER_BANDS
    assert apply_overrides(cfg, ["train.epochs=5", "train.epochs=6"]).train.epochs == 6  # the last one wins
    assert apply_overrides(cfg, None) == cfg


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        ("train.epoch=3", KeyError, "Unknown config key 'train.epoch'"),
        ("trian.epochs=3", KeyError, "Unknown config section 'trian'"),
        ("train.epochs.x=3", KeyError, "Unknown config section 'epochs'"),
        ("train.epochs", ValueError, "must look like section.key=value"),
    ],
)
def test_unknown_keys_and_malformed_overrides_are_rejected(override, error, message):
    with pytest.raises(error, match=message):
        apply_overrides(ExperimentConfig(), [override])


def test_unknown_keys_in_a_file_name_the_file_and_key(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("train:\n  epoch: 3\n", encoding="utf-8")
    with pytest.raises(KeyError, match=r"exp\.yaml: Unknown config key 'train\.epoch' for TrainConfig"):
        load_config(path)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("train.epochs=1O", "train.epochs must be an integer, got '1O'"),
        ("train.epochs=2.5", "train.epochs must be an integer, got 2.5"),
        ("train.epochs=true", "train.epochs must be an integer, got True"),
        ("train.epochs=0", "train.epochs must be >= 1, got 0"),
        ("train.batch_size=0", "train.batch_size must be >= 1, got 0"),
        ("protocol.n_val_subjects=-1", "protocol.n_val_subjects must be >= 0, got -1"),
        ("protocol.jobs=0", "protocol.jobs must be >= 1, got 0"),
        ("seed=x", "seed must be an integer, got 'x'"),
        ("train.learning_rate=fast", "train.learning_rate must be a number, got 'fast'"),
        ("protocol.lambda_subject=abc", "protocol.lambda_subject must be a number or null, got 'abc'"),
        ("train.patience=2.5", "train.patience must be an integer or null, got 2.5"),
        ("protocol.outgroup_includes_val=1", "protocol.outgroup_includes_val must be true or false, got 1"),
        ("features.montage=1020", "features.montage must be a string, got 1020"),
        ("features.channels=Fp1", "features.channels must be a list, got 'Fp1'"),
        ("features.channels=[Fp1, 3]", r"features.channels\[1\] must be a string, got 3"),
        ("features.bands=[[gamma, 30]]", r"features.bands\[0\] must be a list of 3 items"),
        ("model.conv_blocks=[[16, x]]", r"model.conv_blocks\[0\]\[1\] must be an integer, got 'x'"),
        ("train=5", "section 'train' must be a mapping of keys to values, got 5"),
        ("train.optimizer=sgd", "train.optimizer must be one of: nadam_legacy, nadam, adam; got 'sgd'"),
        ("train.lambda_schedule=linear", "train.lambda_schedule must be one of: constant, dann"),
        ("train.selection_metric=auc", "train.selection_metric must be one of: f1, accuracy, loss"),
        ("train.class_weight=none", "train.class_weight must be one of: null, balanced; got 'none'"),
        ("features.reference=Cz", "features.reference must be one of: average, null"),
        ("features.interpolation=nearest", "features.interpolation must be one of: cubic, linear"),
        ("model.padding=causal", "model.padding must be one of: same, valid"),
        ("protocol.normalization=zscore", "protocol.normalization must be one of: subject, none"),
        ("protocol.cross_per_subject=sometimes", "protocol.cross_per_subject must be one of: auto, always, never"),
    ],
)
def test_invalid_values_name_their_key(override, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(ExperimentConfig(), [override])


def test_valid_choices_and_minimums_are_accepted():
    for key, choices in CHOICES.items():
        for choice in choices:
            value = "null" if choice is None else choice
            assert apply_overrides(ExperimentConfig(), [f"{key}={value}"]) is not None
    overrides = [f"{key}={minimum}" for key, minimum in MINIMUMS.items()]
    cfg = apply_overrides(ExperimentConfig(), overrides)
    assert (cfg.train.epochs, cfg.train.batch_size, cfg.protocol.n_val_subjects, cfg.protocol.jobs) == (1, 1, 0, 1)


def test_choices_match_the_modules_that_use_them():
    from sieeg.evaluation.protocols import CROSS_MODES
    from sieeg.features import topomap
    from sieeg.training.schedules import lambda_factor

    assert set(CHOICES["protocol.cross_per_subject"]) == set(CROSS_MODES)
    assert set(CHOICES["features.interpolation"]) == set(topomap._INTERPOLATORS)
    for schedule in CHOICES["train.lambda_schedule"]:
        lambda_factor(0.5, schedule)  # accepted


def test_validate_config_checks_a_config_built_in_python():
    cfg = ExperimentConfig()
    assert validate_config(cfg) is cfg
    with pytest.raises(ValueError, match="train.epochs must be >= 1, got 0"):
        validate_config(dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, epochs=0)))
    with pytest.raises(ValueError, match="protocol.jobs must be an integer, got '2'"):
        validate_config(dataclasses.replace(cfg, protocol=dataclasses.replace(cfg.protocol, jobs="2")))


# ------------------------------------------------------------------ YAML errors
def test_yaml_syntax_errors_name_the_file_line_and_column(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  epochs: 2\n batch_size: 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.yaml: invalid YAML at line 3, column 2: ") as exc:
        load_config(path)
    assert "\n" not in str(exc.value)
    with pytest.raises(ValueError, match=r"Override 'train.epochs=\[1' is not valid YAML \(line 1, column 3: "):
        apply_overrides(ExperimentConfig(), ["train.epochs=[1"])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- 1\n- 2\n", r"the configuration must be a mapping of keys to values, got \[1, 2\]"),
        ("train: 5\n", "section 'train' must be a mapping of keys to values, got 5"),
        ("train:\n  epochs: x\n", "train.epochs must be an integer, got 'x'"),
    ],
)
def test_invalid_files_name_the_file(tmp_path, text, message):
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"exp\.yaml: {message}"):
        load_config(path)


def test_a_missing_file_is_an_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
