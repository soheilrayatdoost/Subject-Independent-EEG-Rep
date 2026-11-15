"""Experiment configuration.

Defaults follow the ICASSP 2021 paper wherever it is specific; every other
default is marked "not in paper" and documented in ``docs/DESIGN.md``.
Configs are plain dataclasses that round-trip through YAML::

    cfg = load_config("configs/paper.yaml", overrides=["train.epochs=5"])
    save_config(cfg, "runs/x/config.yaml")

Values read from YAML or ``--set`` overrides are checked against the field types, ranges
and choices (:func:`validate_config`); an invalid value or a YAML syntax error is a
``ValueError`` that names the key or the file, and an unknown key is a ``KeyError``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: The 32 EEG channels of the BioSemi-32 layout, in MAHNOB-HCI file order.
#: Both databases contain these channels, so they are the default common montage.
BIOSEMI32_CHANNELS: tuple[str, ...] = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7",
    "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
)

#: Frequency bands of the paper (Sec. 4). The paper gives gamma as ">30 Hz";
#: the 45 Hz upper edge (not in paper) keeps it clear of 50 Hz mains noise.
PAPER_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("slow_alpha", 8.0, 10.0),
    ("alpha", 8.0, 12.0),
    ("beta", 12.0, 30.0),
    ("gamma", 30.0, 45.0),
)


@dataclass
class FeatureConfig:
    """Raw EEG -> log band power -> topographic images."""

    channels: tuple[str, ...] = BIOSEMI32_CHANNELS
    bands: tuple[tuple[str, float, float], ...] = PAPER_BANDS
    resample_sfreq: float | None = 256.0  # not in paper: common rate across databases
    reference: str | None = "average"  # not in paper: BioSemi data is reference-free
    l_freq: float | None = None  # optional band-pass before Welch (not in paper)
    h_freq: float | None = None
    welch_segment_sec: float = 15.0  # paper: 15 s windows ...
    welch_overlap: float = 0.5  # ... with 50% overlap (Welch)
    min_duration_sec: float = 5.0  # trials shorter than this are skipped
    montage: str = "standard_1005"  # electrode positions for the topographic maps
    grid_size: int = 32  # paper: 32x32 images
    interpolation: str = "cubic"  # 'cubic' (Clough-Tocher, as Bashivan et al.) or 'linear'


@dataclass
class ModelConfig:
    """Encoder / heads. Paper: 2x conv16, maxpool, 2x conv32, maxpool, FC256, FC128."""

    conv_blocks: tuple[tuple[int, ...], ...] = ((16, 16), (32, 32))
    kernel_size: int = 3  # not in paper
    padding: str = "same"  # not in paper
    conv_activation: str = "relu"  # paper
    dense_units: tuple[int, ...] = (256, 128)  # paper
    dense_activation: str = "tanh"  # paper
    dropout: float = 0.0  # not in paper
    classifier_hidden: int = 64  # paper text: FC64 for discriminator and emotion classifier; 0 = linear read-out
    classifier_activation: str = "tanh"  # paper: tanh for fully connected layers
    discriminator_hidden: int = 64  # paper: FC64 ...
    discriminator_activation: str = "elu"  # ... with elu


@dataclass
class TrainConfig:
    """Optimisation. Paper: Nadam(lr=0.002, b1=0.9, b2=0.999), 100 epochs."""

    epochs: int = 100
    batch_size: int = 32  # not in paper (Keras default)
    # 'nadam_legacy' reproduces Keras-2 Nadam (schedule_decay=0.004), the optimiser behind the
    # paper's lr=0.002 default; Keras>=2.11/3 'nadam' dropped schedule_decay. Also: 'adam'.
    optimizer: str = "nadam_legacy"
    learning_rate: float = 0.002
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7  # Keras default
    schedule_decay: float = 0.004  # Keras-2 Nadam momentum schedule (only used by 'nadam_legacy')
    lambda_schedule: str = "constant"  # 'constant' or 'dann' (2/(1+exp(-10p))-1 ramp)
    selection_metric: str = "f1"  # model selection on validation subjects: 'f1' | 'accuracy' | 'loss'
    patience: int | None = None  # early stopping; None trains all epochs as in the paper
    class_weight: str | None = None  # None or 'balanced' (not in paper)
    jit_compile: bool = False  # XLA-compile the train step
    verbose: int = 0


@dataclass
class ProtocolConfig:
    """Evaluation protocol (paper Sec. 5.3)."""

    label_threshold: float = 5.0  # rating > 5 -> high/positive; 5 counts as low/negative
    normalization: str = "subject"  # per-participant z-score ('subject') or 'none'
    n_val_subjects: int = 5  # paper: validation set = data from five subjects
    outgroup_includes_val: bool = False  # add validation subjects to the out-group pool
    lambda_subject: float | None = None  # None -> method default (SI-EEG 0.2, SIDA-EEG 0.15)
    lambda_domain: float | None = None  # None -> method default (0.1)
    cross_per_subject: str = "auto"  # cross-db: one model per target subject: 'auto' | 'always' | 'never'
    jobs: int = 1  # folds trained in parallel processes
    subject_probe: bool = False  # not in paper: linear probes of subject information in z (analysis/probe.py)


@dataclass
class ExperimentConfig:
    seed: int = 0
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)


# ------------------------------------------------------------------- validation

#: Allowed values of the keys that select an option (``None`` is YAML ``null``).
CHOICES: dict[str, tuple[str | None, ...]] = {
    "features.reference": ("average", None),
    "features.interpolation": ("cubic", "linear"),
    "model.padding": ("same", "valid"),
    "train.optimizer": ("nadam_legacy", "nadam", "adam"),
    "train.lambda_schedule": ("constant", "dann"),
    "train.selection_metric": ("f1", "accuracy", "loss"),
    "train.class_weight": (None, "balanced"),
    "protocol.normalization": ("subject", "none"),
    "protocol.cross_per_subject": ("auto", "always", "never"),
}
#: Smallest allowed value of integer keys.
MINIMUMS: dict[str, int] = {"train.epochs": 1, "train.batch_size": 1, "protocol.n_val_subjects": 0, "protocol.jobs": 1}

_SCALARS = {bool: "true or false", int: "an integer", float: "a number", str: "a string"}


def _convert(value: Any, hint: Any, key: str, *, optional: bool = False) -> Any:
    """``value`` checked against the field type ``hint``; lists become tuples and integers floats.

    A field annotated ``X | None`` also accepts ``None``. A number field accepts a numeric
    string, because PyYAML reads ``1e-3`` (no decimal point) as a string.

    Raises:
        ValueError: naming ``key`` if ``value`` does not have the field's type.
    """
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):  # X | None
        if value is None and type(None) in args:
            return None
        (inner,) = [arg for arg in args if arg is not type(None)]
        return _convert(value, inner, key, optional=True)
    null = " or null" if optional else ""
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a list{null}, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(item, args[0], f"{key}[{i}]") for i, item in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{key} must be a list of {len(args)} items, got {value!r}")
        return tuple(_convert(item, arg, f"{key}[{i}]") for i, (item, arg) in enumerate(zip(value, args)))
    if hint is bool or isinstance(value, bool):  # bool is a subclass of int
        if hint is bool and isinstance(value, bool):
            return value
    elif hint is float and isinstance(value, (int, float)):
        return float(value)
    elif hint is float and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    elif isinstance(value, hint):
        return value
    raise ValueError(f"{key} must be {_SCALARS[hint]}{null}, got {value!r}")


def _leaves(cfg: Any, prefix: str = "") -> Iterator[tuple[str, Any, Any]]:
    """``(section.key, value, type hint)`` of every non-section field of ``cfg``."""
    hints = typing.get_type_hints(type(cfg))
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if dataclasses.is_dataclass(value):
            yield from _leaves(value, f"{prefix}{f.name}.")
        else:
            yield f"{prefix}{f.name}", value, hints[f.name]


def validate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Check every value of ``cfg``: its field type, :data:`MINIMUMS` and :data:`CHOICES`.

    Returns ``cfg`` unchanged. Raises ``ValueError`` naming the first invalid key, e.g.
    ``train.epochs must be >= 1, got 0``.
    """
    for key, value, hint in _leaves(cfg):
        _convert(value, hint, key)
        if key in MINIMUMS and value < MINIMUMS[key]:
            raise ValueError(f"{key} must be >= {MINIMUMS[key]}, got {value!r}")
        if key in CHOICES and value not in CHOICES[key]:
            allowed = ", ".join("null" if choice is None else choice for choice in CHOICES[key])
            raise ValueError(f"{key} must be one of: {allowed}; got {value!r}")
    return cfg


# --------------------------------------------------------------------------- io


def _to_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_list(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_list(v) for k, v in value.items()}
    return value


def _from_dict(cls: type, data: Any, prefix: str = "") -> Any:
    """``cls`` built from a mapping (``None`` = defaults); every value is type-checked by :func:`_convert`."""
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        where = f"section '{prefix[:-1]}'" if prefix else "the configuration"
        raise ValueError(f"{where} must be a mapping of keys to values, got {data!r}")
    defaults = cls()
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{prefix}{key}' for {cls.__name__}; valid keys: {sorted(names)}")
        default = getattr(defaults, key)
        if dataclasses.is_dataclass(default):
            kwargs[key] = _from_dict(type(default), value, f"{prefix}{key}.")
        else:
            kwargs[key] = _convert(value, hints[key], f"{prefix}{key}")
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any] | None) -> ExperimentConfig:
    """An :class:`ExperimentConfig` from nested dicts (missing keys keep their defaults), validated.

    Raises:
        KeyError: for an unknown key.
        ValueError: for a section that is not a mapping, or a value of the wrong type, out of
            range or not among the choices (:func:`validate_config`).
    """
    return validate_config(_from_dict(ExperimentConfig, data))


def config_to_dict(cfg: Any) -> dict[str, Any]:
    return _to_list(dataclasses.asdict(cfg))


def _yaml_problem(err: yaml.YAMLError) -> str:
    """A YAML error on one line: ``line L, column C: problem``."""
    mark, problem = getattr(err, "problem_mark", None), getattr(err, "problem", None)
    if mark is not None and problem:
        return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"
    return " ".join(str(err).split())


def apply_overrides(cfg: ExperimentConfig, overrides: Iterable[str] | None) -> ExperimentConfig:
    """Apply ``section.key=value`` overrides; values are parsed as YAML scalars/lists.

    The result is validated like a configuration file (:func:`config_from_dict`).
    """
    data = config_to_dict(cfg)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise KeyError(f"Unknown config section '{k}' in override '{item}'")
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"Unknown config key '{dotted}' in override '{item}'")
        try:
            node[keys[-1]] = yaml.safe_load(raw)
        except yaml.YAMLError as err:
            raise ValueError(f"Override '{item}' is not valid YAML ({_yaml_problem(err)})") from err
    return config_from_dict(data)


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> ExperimentConfig:
    """The defaults, updated by the YAML file ``path`` (if given) and then by ``overrides``.

    Raises:
        OSError: if ``path`` cannot be read.
        KeyError, ValueError: as :func:`config_from_dict`, prefixed with ``path`` for errors in
            the file; a YAML syntax error is a ``ValueError`` naming the file, line and column.
    """
    cfg = ExperimentConfig()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError as err:
                raise ValueError(f"{path}: invalid YAML at {_yaml_problem(err)}") from err
        try:
            cfg = config_from_dict(data)
        except KeyError as err:
            raise KeyError(f"{path}: {err.args[0]}") from err
        except ValueError as err:
            raise ValueError(f"{path}: {err}") from err
    return apply_overrides(cfg, overrides)


class _Dumper(yaml.SafeDumper):
    """Writes lists of scalars in flow style (``[a, b]``) to keep config files short."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flow = all(not isinstance(v, (list, dict)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def dump_config(cfg: ExperimentConfig) -> str:
    return yaml.dump(config_to_dict(cfg), Dumper=_Dumper, sort_keys=False, width=100)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dump_config(cfg), encoding="utf-8")
