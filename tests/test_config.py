"""Experiment configuration: YAML round trip, overrides and validation (sieeg.config)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from sieeg.config import (
    CHOICES,
    FRACTIONS,
    MINIMUMS,
    PAPER_BANDS,
    POSITIVE,
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
#: Overrides of list keys whose bound holds for every number in the list.
LIST_FORMS = {"model.conv_blocks": "[[{}]]", "model.dense_units": "[{}]"}


def _set(key: str, number: float) -> ExperimentConfig:
    """The defaults with ``key`` (or every number of a list key) set to ``number``."""
    return apply_overrides(ExperimentConfig(), [f"{key}={LIST_FORMS.get(key, '{}').format(number)}"])


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
        ("train.learning_rate=fast", "train.learning_rate must be a finite number, got 'fast'"),
        ("protocol.lambda_subject=abc", "protocol.lambda_subject must be a finite number or null, got 'abc'"),
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


@pytest.mark.parametrize(
    ("override", "message"),
    [
        # NaN would label every trial class 0 (rating > nan is False) and the run would still finish.
        ("protocol.label_threshold=.nan", "protocol.label_threshold must be a finite number, got nan"),
        ("protocol.label_threshold=nan", "protocol.label_threshold must be a finite number, got 'nan'"),
        ("train.learning_rate=.inf", "train.learning_rate must be a finite number, got inf"),
        ("train.learning_rate=-Infinity", "train.learning_rate must be a finite number, got '-Infinity'"),
        pytest.param(  # an integer beyond the float range
            "train.beta_2=1" + "0" * 400, "train.beta_2 must be a finite number, got 10{20}", id="train.beta_2=10**400"
        ),
        ("features.resample_sfreq=.nan", "features.resample_sfreq must be a finite number or null, got nan"),
        ("protocol.lambda_subject=-.inf", "protocol.lambda_subject must be a finite number or null, got -inf"),
        ("features.bands=[[gamma, 30, .inf]]", r"features.bands\[0\]\[2\] must be a finite number, got inf"),
        ("features.bands=[[gamma, NaN, 45]]", r"features.bands\[0\]\[1\] must be a finite number, got 'NaN'"),
    ],
)
def test_non_finite_numbers_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(ExperimentConfig(), [override])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("features.welch_overlap=1", r"features.welch_overlap must be in \[0, 1\), got 1.0"),  # Welch rejects it
        ("features.welch_overlap=-0.5", r"features.welch_overlap must be in \[0, 1\), got -0.5"),  # skips samples
        ("train.beta_2=1", r"train.beta_2 must be in \[0, 1\), got 1.0"),  # LegacyNadam: NaN weights
        ("train.beta_1=1.0", r"train.beta_1 must be in \[0, 1\), got 1.0"),  # Adam: NaN weights
        ("model.dropout=1", r"model.dropout must be in \[0, 1\), got 1.0"),
        ("train.learning_rate=0", "train.learning_rate must be > 0, got 0.0"),
        ("train.epsilon=0", "train.epsilon must be > 0, got 0.0"),  # 0 / 0 for a weight without gradient
        ("features.welch_segment_sec=0", "features.welch_segment_sec must be > 0, got 0.0"),
        ("features.resample_sfreq=-128", "features.resample_sfreq must be > 0, got -128.0"),
        ("features.h_freq=0", "features.h_freq must be > 0, got 0.0"),  # MNE would low-pass the signal away
        ("features.l_freq=-1", "features.l_freq must be >= 0, got -1.0"),
        ("train.schedule_decay=-1", "train.schedule_decay must be >= 0, got -1.0"),
        ("model.kernel_size=0", "model.kernel_size must be >= 1, got 0"),
        ("model.conv_blocks=[[16, 0]]", r"model.conv_blocks\[0\]\[1\] must be >= 1, got 0"),
        ("model.dense_units=[256, 0]", r"model.dense_units\[1\] must be >= 1, got 0"),
        ("model.classifier_hidden=-8", "model.classifier_hidden must be >= 0, got -8"),
        ("model.discriminator_hidden=-1", "model.discriminator_hidden must be >= 0, got -1"),
        ("protocol.lambda_subject=-0.2", "protocol.lambda_subject must be >= 0, got -0.2"),
        ("protocol.lambda_domain=-0.1", "protocol.lambda_domain must be >= 0, got -0.1"),
        ("seed=-1", "seed must be >= 0, got -1"),
        ("features.bands=[[alpha, 12, 8]]", r"features.bands\[0\] must have 0 <= low < high, got \('alpha', 12.0, 8"),
        ("features.bands=[[delta, 1, 4], [alpha, 8, 8]]", r"features.bands\[1\] must have 0 <= low < high, got \('al"),
        ("features.bands=[[delta, -1, 4]]", r"features.bands\[0\] must have 0 <= low < high, got \('delta', -1.0, 4"),
        ("features.grid_size=3", "features.grid_size must be >= 4, got 3: smaller images leave the encoder"),
        ("features.grid_size=0", "features.grid_size must be >= 4, got 0"),
    ],
)
def test_out_of_range_values_name_their_key(override, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(ExperimentConfig(), [override])


@pytest.mark.parametrize(
    ("key", "accepted", "rejected"),
    [pytest.param(key, [minimum], [minimum - 1], id=key) for key, minimum in MINIMUMS.items()]
    + [pytest.param(key, [1e-9], [0, -1], id=key) for key in POSITIVE]
    + [pytest.param(key, [0, 0.999], [1, -0.1], id=key) for key in FRACTIONS],
)
def test_range_bounds(key, accepted, rejected):
    for number in accepted:
        _set(key, number)
    for number in rejected:
        with pytest.raises(ValueError, match=rf"^{re.escape(key)}(\[0\])* must be "):
            _set(key, number)


def test_valid_choices_and_minimums_are_accepted():
    for key, choices in CHOICES.items():
        for choice in choices:
            value = "null" if choice is None else choice
            assert apply_overrides(ExperimentConfig(), [f"{key}={value}"]) is not None
    overrides = [f"{key}={LIST_FORMS.get(key, '{}').format(minimum)}" for key, minimum in MINIMUMS.items()]
    cfg = apply_overrides(ExperimentConfig(), overrides)
    assert (cfg.train.epochs, cfg.train.batch_size, cfg.protocol.n_val_subjects, cfg.protocol.jobs) == (1, 1, 0, 1)
    assert (cfg.seed, cfg.model.conv_blocks, cfg.model.dense_units, cfg.model.kernel_size) == (0, ((1,),), (1,), 1)
    edges = apply_overrides(ExperimentConfig(), ["features.bands=[[dc, 0, 0.5]]", "protocol.label_threshold=-1e6"])
    assert edges.features.bands == (("dc", 0.0, 0.5),) and edges.protocol.label_threshold == -1e6
    # Zero means something for these keys (no adversary, no high-pass, a linear read-out), whatever MINIMUMS says.
    zeros = ["protocol.lambda_subject", "protocol.lambda_domain", "features.l_freq", "model.classifier_hidden",
             "model.discriminator_hidden", "train.schedule_decay"]
    cfg = apply_overrides(ExperimentConfig(), [f"{key}=0" for key in zeros])
    assert (cfg.protocol.lambda_subject, cfg.protocol.lambda_domain, cfg.features.l_freq) == (0, 0, 0)


@pytest.mark.parametrize(
    "model",
    [
        [],  # the paper's encoder: two blocks, 'same' padding -> grid_size >= 4
        ["model.padding=valid"],
        ["model.conv_blocks=[[4], [], [8, 8]]", "model.padding=valid", "model.kernel_size=2"],
        ["model.conv_blocks=[]"],
    ],
)
def test_smallest_grid_size_leaves_the_encoder_a_feature_map(model):
    from sieeg.models.networks import build_encoder

    cfg = apply_overrides(ExperimentConfig(), model)

    def flat_size(grid_size: int) -> int:
        """Features left after the convolution blocks for grid_size x grid_size images."""
        try:
            encoder = build_encoder((grid_size, grid_size, 1), cfg.model)
        except ValueError:  # Keras: negative output size
            return 0
        return encoder.get_layer("encoder_flatten").output.shape[-1]

    def accepted(grid_size: int) -> bool:
        try:
            apply_overrides(cfg, [f"features.grid_size={grid_size}"])
        except ValueError as err:
            assert str(err).startswith("features.grid_size must be >= ")
            return False
        return True

    smallest = next(grid_size for grid_size in range(64) if accepted(grid_size))
    assert flat_size(smallest) > 0 and accepted(smallest + 1)
    # One pixel less leaves an empty feature map: the encoder's output would not depend on the image.
    assert flat_size(smallest - 1) == 0
    with pytest.raises(ValueError, match=f"features.grid_size must be >= {smallest}, got {smallest - 1}"):
        apply_overrides(cfg, [f"features.grid_size={smallest - 1}"])


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
    with pytest.raises(ValueError, match=r"train.beta_2 must be in \[0, 1\), got 1.0"):
        validate_config(dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, beta_2=1.0)))
    with pytest.raises(ValueError, match="train.beta_1 must be a finite number, got nan"):
        validate_config(dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, beta_1=float("nan"))))
    with pytest.raises(ValueError, match=r"model.conv_blocks\[1\]\[0\] must be >= 1, got 0"):  # a list is checked too
        validate_config(dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, conv_blocks=[[8], [0]])))


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        # A string would reach Welch (segment_sec * sfreq) or the label comparison unconverted.
        ("features", "welch_segment_sec", "15", "features.welch_segment_sec must be a finite number, got '15'"),
        ("features", "resample_sfreq", "128", "features.resample_sfreq must be a finite number or null, got '128'"),
        ("features", "bands", (("delta", "1", 4.0),), r"features.bands\[0\]\[1\] must be a finite number, got '1'"),
        ("protocol", "label_threshold", "5", "protocol.label_threshold must be a finite number, got '5'"),
    ],
)
def test_a_config_built_in_python_cannot_give_numbers_as_strings(section, field, value, message):
    cfg = ExperimentConfig()
    built = dataclasses.replace(cfg, **{section: dataclasses.replace(getattr(cfg, section), **{field: value})})
    with pytest.raises(ValueError, match=message):
        validate_config(built)


def test_a_config_built_in_python_may_use_integers_and_lists():
    cfg = ExperimentConfig()
    features = dataclasses.replace(cfg.features, welch_segment_sec=15, channels=["Fz", "Cz"], bands=[["theta", 4, 8]])
    built = dataclasses.replace(cfg, features=features, model=dataclasses.replace(cfg.model, conv_blocks=[[8], [16]]))
    assert validate_config(built) is built


def test_a_yaml_file_may_give_numbers_as_integers_or_strings(tmp_path):
    path = tmp_path / "exp.yaml"
    text = "features:\n  welch_segment_sec: 15\n  welch_overlap: '0.25'\n  bands: [[theta, 4, 8]]\n"
    path.write_text(text + "train:\n  epsilon: 1e-8\n", encoding="utf-8")  # 1e-8: a string to PyYAML
    cfg = load_config(path)
    assert isinstance(cfg.features.welch_segment_sec, float) and cfg.features.welch_segment_sec == 15.0
    assert (cfg.features.welch_overlap, cfg.train.epsilon) == (0.25, 1e-8)
    assert cfg.features.bands == (("theta", 4.0, 8.0),)


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
        ("protocol:\n  label_threshold: .nan\n", "protocol.label_threshold must be a finite number, got nan"),
        ("features:\n  welch_overlap: 1.0\n", r"features.welch_overlap must be in \[0, 1\), got 1.0"),
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
