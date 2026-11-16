"""Command-line interface: ``sieeg`` or ``python -m sieeg`` (docs/DESIGN.md Sec. 10).

::

    sieeg extract mahnob ROOT --out FILE                  MAHNOB-HCI sessions -> features (paper Sec. 3-4)
    sieeg extract manifest CSV --dataset NAME --out FILE  trials listed in a CSV manifest (DAI-EF, ...)
    sieeg extract synthetic --out FILE [--raw]            synthetic EEG for smoke tests
    sieeg info FILE                                       what a feature file holds
    sieeg bias FILE [--other FILE2]                       Experiment 1: subject / database identification
    sieeg within FILE --method M --label L                Experiment 2: leave-one-subject-out in one database
    sieeg cross SOURCE TARGET --method M --label L        Experiment 2: labeled SOURCE -> unlabeled TARGET
    sieeg report DIR [DIR ...]                            Table 1 and paired tests from run directories

Every (method, label) pair of ``within`` and ``cross`` is saved in its own run directory
``<out>/<protocol>-<test_dataset>/<method>[-<tag>]/<label>/seed<seed>/``, which the command
claims with a ``.running`` file while it trains. Configuration precedence: built-in defaults
(the paper's values) < ``--config`` < ``--set`` < convenience flags such as ``--epochs``.
scikit-learn and TensorFlow are imported inside the commands that use them, so
``sieeg --help`` and ``sieeg info`` start quickly.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow's C++ logging; must precede its import

import argparse
import contextlib
import hashlib
import json
import logging
import math
import re
import shlex
import shutil
import socket
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import __version__
from .config import ExperimentConfig, ProtocolConfig, load_config, validate_config
from .features.featureset import LABELS, FeatureSet
from .training.methods import MethodSpec, available_methods, get_method
from .utils import natural_sorted

if TYPE_CHECKING:
    import pandas as pd

    from .data.trial import Trial
    from .evaluation.report import RunResult

logger = logging.getLogger(__name__)

_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+=-]*")
_CLAIM = ".running"  # claims a run directory while a command trains it (docs/DESIGN.md Sec. 10)
_EXAMPLES = """\
examples:
  sieeg extract mahnob /data/MAHNOB-HCI --out features/mahnob.npz
  sieeg extract manifest daief.csv --dataset daief --out features/daief.npz
  sieeg within features/daief.npz --method cnn,sinn,si_eeg --label both
  sieeg cross features/daief.npz features/mahnob.npz --method dann,sida_eeg --label both
  sieeg report runs --out results/table1.md
"""
#: Experiment 1 columns that hold fractions; ``sieeg bias`` prints and writes them in percent.
_BIAS_PERCENT = ("accuracy", "accuracy_std", "balanced_accuracy", "chance_uniform", "chance_majority")
_BIAS_NOTE = (
    "Database identification: trial-level folds follow the paper but also reward recognising the participants "
    "(each belongs to one database); subject-grouped folds keep every participant in one fold and measure the "
    "database bias itself (chance 50 %)."
)


# ------------------------------------------------------------------ argument types
def _comma_list(text: str) -> list[str]:
    """argparse type: ``"a, b,c"`` -> ``["a", "b", "c"]``."""
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list, got {text!r}")
    return items


def _method_list(text: str) -> list[str]:
    """argparse type: comma-separated method names, spelled as :func:`get_method` does (``si-eeg`` -> ``si_eeg``)."""
    names: list[str] = []
    for item in _comma_list(text):
        name = item.lower().replace("-", "_")
        if name not in available_methods():
            raise argparse.ArgumentTypeError(f"unknown method {item!r}; choose from {', '.join(available_methods())}")
        if name not in names:
            names.append(name)
    return names


def _positive_int(text: str) -> int:
    """argparse type: an integer >= 1."""
    if not text.strip().isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return int(text)


def _tag(text: str) -> str:
    """argparse type: a run tag, which becomes part of a directory name."""
    if not _TAG.fullmatch(text):
        raise argparse.ArgumentTypeError(
            f"tag {text!r} may only contain letters, digits and . _ + = - (and must start with a letter or digit)"
        )
    return text


# ------------------------------------------------------------------------ parser
def _add_verbosity(parser: argparse.ArgumentParser, *, default: Any) -> None:
    """``-v`` / ``-q``, accepted before the command (top-level parser) and after it.

    The commands' copies default to ``argparse.SUPPRESS``: an absent flag then leaves the
    top-level value alone instead of resetting it to False.
    """
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", default=default, help="debug logging; show tracebacks of errors"
    )
    verbosity.add_argument("-q", "--quiet", action="store_true", default=default, help="log warnings and errors only")


def _add_feature_output(parser: argparse.ArgumentParser) -> None:
    """``--out`` and ``--overwrite`` of the ``extract`` commands."""
    parser.add_argument("--out", type=Path, required=True, metavar="FILE", help="feature file to write (.npz)")
    parser.add_argument("--overwrite", action="store_true", help="replace FILE if it exists")


def _add_run_arguments(parser: argparse.ArgumentParser, *, folds_help: str) -> None:
    """Options shared by ``within`` and ``cross``."""
    parser.add_argument(
        "--method", type=_method_list, required=True, metavar="M[,M2..]",
        help=f"one or more of: {', '.join(available_methods())}",
    )
    parser.add_argument(
        "--label", choices=(*LABELS, "both"), required=True,
        help="emotion dimension; ratings above protocol.label_threshold (paper: 5) form the high/positive class",
    )
    parser.add_argument("--folds", type=_comma_list, metavar="S1,S2", help=folds_help)
    parser.add_argument(
        "--out", type=Path, default=Path("runs"), metavar="DIR", help="root of the run directories (default: runs)"
    )
    parser.add_argument("--tag", type=_tag, help="suffix of the method directory, e.g. to keep a lambda sweep apart")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace runs that already have a summary.json, and take over run directories claimed by another "
        "sieeg process (.running)",
    )
    parser.add_argument("--epochs", type=_positive_int, help="training epochs (train.epochs; paper: 100)")
    parser.add_argument("--seed", type=int, help="seed of the splits, initialisation and batches (seed; default: 0)")
    parser.add_argument("--jobs", type=_positive_int, help="folds trained in parallel processes (protocol.jobs)")
    parser.add_argument(
        "--lambda-subject", type=float, metavar="X",
        help="GRL weight of the subject adversary (protocol.lambda_subject; default: 0.2, SIDA-EEG/SIDANN 0.15)",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="after each fold, measure the subject information left in its representation with linear probes "
        "(protocol.subject_probe; not in the paper)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The ``sieeg`` argument parser (docs/DESIGN.md Sec. 10)."""
    parser = argparse.ArgumentParser(
        prog="sieeg",
        description="Subject-invariant EEG representation learning for emotion recognition:\n"
        "SI-EEG and SIDA-EEG (ICASSP 2021) and the baselines EEG CNN, SINN, DANN and SIDANN.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_verbosity(parser, default=False)
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_verbosity(common, default=argparse.SUPPRESS)

    config = argparse.ArgumentParser(add_help=False)
    config.add_argument(
        "--config", type=Path, metavar="YAML",
        help="configuration file (default: the paper's values, as in configs/paper.yaml)",
    )
    config.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override one configuration value, e.g. --set train.batch_size=64 (repeatable)",
    )

    # extract ---------------------------------------------------------------
    extract = commands.add_parser(
        "extract", help="raw EEG -> per-trial log band-power feature file (.npz)",
        description="Welch PSD over each stimulus and log10 power in six bands (paper Sec. 4).",
    )
    kinds = extract.add_subparsers(dest="kind", required=True)

    p = kinds.add_parser(
        "mahnob", parents=[common, config], help="MAHNOB-HCI rated emotion sessions",
        description="Read <ROOT>/Sessions/<id>/session.xml and the session's BDF; crop each trial to the stimulus.",
    )
    p.add_argument("root", type=Path, metavar="ROOT", help="MAHNOB-HCI folder that contains Sessions/ (or Sessions/)")
    _add_feature_output(p)
    p.add_argument("--subjects", type=_comma_list, metavar="S1,S2", help="only these subject ids (default: all)")
    p.add_argument(
        "--pad-sec", type=float, default=30.0, help="seconds recorded before and after each stimulus (default: 30)"
    )
    p.set_defaults(func=_extract_mahnob)

    p = kinds.add_parser(
        "manifest", parents=[common, config], help="trials listed in a CSV manifest (DAI-EF or any dataset)",
        description="One row per trial. Required columns: subject, trial, path, valence, arousal. Optional: tmin, "
        "tmax (seconds), sfreq, ch_names (';'-separated), array_key, transpose, scale (to volts).",
    )
    p.add_argument("csv", type=Path, metavar="CSV", help="manifest file; relative paths are relative to its folder")
    p.add_argument("--dataset", required=True, help="dataset name stored in the features, e.g. daief")
    _add_feature_output(p)
    p.add_argument("--sfreq", type=float, help="sampling rate (Hz) of array files that do not state it")
    p.add_argument(
        "--ch-names", type=_comma_list, metavar="A,B,..", help="channel names of array files that do not state them"
    )
    p.add_argument("--scale", type=float, help="multiplier to volts where the manifest has none, e.g. 1e-6 for uV")
    p.set_defaults(func=_extract_manifest)

    p = kinds.add_parser(
        "synthetic", parents=[common, config], help="synthetic EEG with subject, emotion and database effects",
        description="Synthetic per-trial band power (or, with --raw, raw signals run through the real pipeline).",
    )
    _add_feature_output(p)
    p.add_argument("--n-subjects", type=_positive_int, default=10, help="default: 10")
    p.add_argument("--n-trials", type=_positive_int, default=20, help="trials per subject (default: 20)")
    p.add_argument("--dataset", default="synthetic", help="dataset name (default: synthetic)")
    p.add_argument(
        "--domain-shift", type=float, default=0.0, help="database effect: perturbs the emotion patterns (default: 0)"
    )
    p.add_argument("--seed", type=int, default=0, help="generator seed (default: 0)")
    p.add_argument(
        "--raw", action="store_true",
        help="render raw multichannel EEG and extract its features with the configured pipeline",
    )
    p.set_defaults(func=_extract_synthetic)

    # info ------------------------------------------------------------------
    p = commands.add_parser("info", parents=[common], help="summarise a feature file")
    p.add_argument("file", type=Path, metavar="FILE")
    p.set_defaults(func=_info)

    # bias ------------------------------------------------------------------
    p = commands.add_parser(
        "bias", parents=[common], help="Experiment 1: random-forest subject and database identification",
        description="Cross-validated random forests on the flattened log band power identify the subject (FILE) "
        "and, with --other, the database (FILE vs FILE2), with trial-level folds (as in the paper) and with "
        "subject-grouped folds, without and with per-subject z-scoring fitted inside each fold (paper Sec. 6). "
        "Scores are printed and written in percent.",
    )
    p.add_argument("file", type=Path, metavar="FILE")
    p.add_argument("--other", type=Path, metavar="FILE2", help="second database for database identification")
    p.add_argument(
        "--exclude-subjects", type=_comma_list, action="extend", metavar="S1,S2", help="subjects of FILE to leave out"
    )
    p.add_argument(
        "--n-splits", type=int, default=10,
        help="cross-validation folds (default: 10; subject-grouped database identification uses at most as many "
        "as the smaller database has subjects)",
    )
    p.add_argument("--n-estimators", type=int, default=500, help="trees per forest (default: 500)")
    p.add_argument("--seed", type=int, default=0, help="default: 0")
    p.add_argument(
        "--no-balance", action="store_true",
        help="keep every trial of both databases (chance is then the majority share, not 50 %%)",
    )
    p.add_argument("--jobs", type=int, default=-1, help="CPU cores of the forests (default: -1, all)")
    p.add_argument("--out", type=Path, metavar="CSV", help="also write the table as CSV (scores in percent, *_pct)")
    p.set_defaults(func=_bias)

    # within / cross --------------------------------------------------------
    p = commands.add_parser(
        "within", parents=[common, config], help="Experiment 2: leave-one-subject-out within one database",
        description="Each subject is the test subject once; 5 (protocol.n_val_subjects) of the others select the "
        "best epoch, the rest train. The test subject's unlabeled EEG is SI-EEG's out-group (Eq. 1).",
    )
    p.add_argument("file", type=Path, metavar="FILE")
    _add_run_arguments(p, folds_help="test subjects to run (default: all)")
    p.add_argument(
        "--exclude-subjects", type=_comma_list, action="extend", metavar="S1,S2", help="subjects to leave out entirely"
    )
    p.set_defaults(func=_within)

    p = commands.add_parser(
        "cross", parents=[common, config], help="Experiment 2: train on SOURCE (labeled), test on TARGET (unlabeled)",
        description="SOURCE and TARGET are two different databases. Validation subjects come from SOURCE; TARGET "
        "emotion labels only score the predictions. Methods with the in-group/out-group adversary train one model "
        "per target subject.",
    )
    p.add_argument("source", type=Path, metavar="SOURCE")
    p.add_argument("target", type=Path, metavar="TARGET")
    _add_run_arguments(p, folds_help="target subjects to test (default: all)")
    p.add_argument(
        "--exclude-source", type=_comma_list, action="extend", metavar="S1,S2", help="SOURCE subjects to leave out"
    )
    p.add_argument(
        "--exclude-target", type=_comma_list, action="extend", metavar="S1,S2", help="TARGET subjects to leave out"
    )
    p.add_argument(
        "--lambda-domain", type=float, metavar="X",
        help="GRL weight of the database adversary (protocol.lambda_domain; default: 0.1)",
    )
    p.set_defaults(func=_cross)

    # report ----------------------------------------------------------------
    p = commands.add_parser(
        "report", parents=[common], help="Table 1 and paired tests across subjects from run directories"
    )
    p.add_argument("dirs", type=Path, nargs="+", metavar="DIR", help="run directories or folders that contain them")
    p.add_argument(
        "--reference", type=_method_list, default=["sida_eeg"], metavar="M[,M2..]",
        help="method(s) compared by paired tests with every other method of the same tag (default: sida_eeg)",
    )
    p.add_argument("--metric", choices=("f1", "cr"), default="f1", help="per-subject score of the tests (default: f1)")
    p.add_argument(
        "--source", choices=("pooled", "subject_mean"), default="pooled",
        help="results table: scores pooled over test trials or averaged over test subjects (default: pooled)",
    )
    p.add_argument("--out", type=Path, metavar="FILE.md", help="also write the report to this Markdown file")
    p.set_defaults(func=_report)
    return parser


# ----------------------------------------------------------------------- helpers
def _run_config(args: argparse.Namespace) -> ExperimentConfig:
    """``--config`` (default: the paper's values), then the ``--set`` overrides, then the convenience flags."""
    cfg = load_config(args.config, args.overrides)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.jobs is not None:
        cfg.protocol.jobs = args.jobs
    if args.lambda_subject is not None:
        cfg.protocol.lambda_subject = args.lambda_subject
    if getattr(args, "lambda_domain", None) is not None:
        cfg.protocol.lambda_domain = args.lambda_domain
    if args.probe:
        cfg.protocol.subject_probe = True
    return validate_config(cfg)  # the flags bypass load_config's checks (e.g. --lambda-subject nan)


def _load_features(path: Path, exclude: Sequence[str] | None) -> FeatureSet:
    """A feature file without the ``exclude`` subjects; ids it does not contain are named in a warning."""
    fs = FeatureSet.load(path)
    if exclude:
        unknown = natural_sorted(set(exclude) - set(fs.subject_ids))
        if unknown:
            logger.warning("%s has no subjects %s to exclude; its subjects are %s", path, unknown, fs.subject_ids)
        fs = fs.drop_subjects(exclude)
        logger.info("%s: %d subjects after excluding %s", path, len(fs.subject_ids), natural_sorted(exclude))
    return fs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_record(path: Path, fs: FeatureSet, excluded: Sequence[str] | None) -> dict[str, Any]:
    """Provenance of an input feature file for ``<run>/features.json`` (docs/DESIGN.md Sec. 10).

    The ``features`` section of a run's ``config.yaml`` is the command's configuration, not the
    one the file was extracted with. This records what the run read: the file and its SHA-256,
    the subjects left after ``excluded``, the channels and bands, and the file's metadata
    (``meta["feature_config"]`` holds the extraction settings).
    """
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "dataset": fs.dataset,
        "excluded_subjects": natural_sorted(excluded or ()),
        "subjects": fs.subject_ids,
        "n_trials": len(fs),
        "ch_names": list(fs.ch_names),
        "bands": [list(band) for band in fs.bands],
        "meta": fs.meta,
    }


def _rated_subjects(fs: FeatureSet, label: str) -> list[str]:
    """Subjects of ``fs`` with at least one ``label`` rating, in natural order."""
    return natural_sorted(fs.subjects[fs.has_rating(label)])


def _fold_problems(
    folds: Sequence[str] | None, fs: FeatureSet, rated: list[str], excluded: Sequence[str] | None
) -> list[str]:
    """Why ``--folds`` subjects cannot be tested: excluded, not in ``fs``, or not rated (empty if they can)."""
    if not folds:
        return []
    excluded, present = set(excluded or ()), set(fs.subjects.tolist())
    reasons = []
    for subject in natural_sorted(folds):
        if subject in rated:
            continue
        if subject in excluded:
            reasons.append(f"{subject} is excluded")
        elif subject not in present:
            reasons.append(f"{subject} is not a subject of {fs.dataset}")
        else:
            reasons.append(f"{subject} is not rated")
    if not reasons:
        return []
    return [f"--folds {', '.join(reasons)} (rated {fs.dataset} subjects: {', '.join(rated)})"]


def _feature_file(out: Path) -> Path:
    """The file :meth:`FeatureSet.save` writes for ``--out`` (it appends ``.npz`` when missing)."""
    return out if out.suffix == ".npz" else out.with_name(out.name + ".npz")


def _new_feature_file(args: argparse.Namespace) -> Path:
    """The ``extract`` output file, checked before any data is read: an existing file needs ``--overwrite``."""
    path = _feature_file(args.out)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    return path


def _save_features(fs: FeatureSet, path: Path) -> int:
    path = fs.save(path)
    skipped = len(fs.meta.get("skipped", []))
    note = f", {skipped} trials skipped (see sieeg info)" if skipped else ""
    print(
        f"{fs.dataset}: {len(fs)} trials of {len(fs.subject_ids)} subjects, "
        f"{fs.n_channels} channels x {fs.n_bands} bands{note} -> {path}"
    )
    return 0


def _extract(
    trials: Iterable[Trial], cfg: ExperimentConfig, path: Path, loader_skipped: list[dict[str, str]] | None = None
) -> int:
    """Extract and save; ``loader_skipped`` is the list the loader of ``trials`` records its dropped trials in."""
    from .features.extract import extract_features

    return _save_features(extract_features(trials, cfg.features, loader_skipped=loader_skipped), path)


def _run_dir(out: Path, protocol: str, test_dataset: str, method: str, tag: str | None, label: str, seed: int) -> Path:
    """``<out>/<protocol>-<test_dataset>/<method>[-<tag>]/<label>/seed<seed>`` (docs/DESIGN.md Sec. 10)."""
    return out / f"{protocol}-{test_dataset}" / (f"{method}-{tag}" if tag else method) / label / f"seed{seed}"


def _command_record(args: argparse.Namespace) -> dict[str, Any]:
    """How a run was launched: command line, parsed arguments, working directory and version."""
    return {
        "command": shlex.join(["sieeg", *args.argv]),
        "argv": args.argv,
        "arguments": {key: value for key, value in vars(args).items() if key not in ("func", "argv")},
        "cwd": str(Path.cwd()),
        "sieeg_version": __version__,
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _claim_holder(marker: Path) -> str:
    """Who holds a ``.running`` file, e.g. ``pid 123 on host-a since 2026-11-15T10:00:00+01:00``."""
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        return f"pid {info['pid']} on {info['host']} since {info['started']}"
    except (OSError, ValueError, KeyError, TypeError):
        return "an unknown process"


def _claim(run_dir: Path, args: argparse.Namespace) -> str:
    """Claim ``run_dir`` for this process: create ``run_dir/.running`` exclusively (``O_EXCL``).

    The file holds the pid, host, command line and start time; its text is returned for
    :func:`_release`.

    Raises:
        FileExistsError: if another sieeg process holds the directory, unless ``--overwrite``
            (which takes the claim over, with a warning).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / _CLAIM
    claim = json.dumps(
        {
            "pid": os.getpid(), "host": socket.gethostname(), "argv": ["sieeg", *args.argv],
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    ) + "\n"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        holder = _claim_holder(marker)
        if not args.overwrite:
            raise FileExistsError(
                f"{run_dir} is being written by another sieeg process ({holder}); "
                f"if that process has stopped, delete {marker} or pass --overwrite"
            ) from None
        logger.warning("Taking over %s from %s (--overwrite)", run_dir, holder)
        marker.write_text(claim, encoding="utf-8")
        return claim
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(claim)
    return claim


def _release(run_dir: Path, claim: str, out: Path) -> None:
    """Remove this process's ``.running`` file, then the directories below ``out`` that it left empty."""
    marker = run_dir / _CLAIM
    with contextlib.suppress(OSError):
        if marker.read_text(encoding="utf-8") == claim:  # not if another process took the claim over
            marker.unlink()
    parts = run_dir.relative_to(out).parts
    for depth in range(len(parts), 0, -1):
        try:
            out.joinpath(*parts[:depth]).rmdir()  # only an empty directory: a failed run leaves nothing behind
        except OSError:
            break


def _fraction(value: float | None) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.2f}"


def _result_line(summary: dict[str, Any], run_dir: Path) -> str:
    """One line per run; the spread over subjects is left out for a single test subject (it is undefined)."""
    pooled, subjects, n = summary["pooled"], summary["subject_mean"], summary["n_subjects"]
    spread = f" +- {subjects['f1_std']:.1f}" if n > 1 else ""
    probe = summary.get("probe")
    probes = (
        f"; probes: subject {_fraction(probe['subject_acc'])} (chance {_fraction(probe['subject_chance'])}), "
        f"group {_fraction(probe['group_bacc'])}"
        if probe else ""
    )
    return (
        f"{summary['protocol']} {summary['test_dataset']} {summary['method']} {summary['label']}: "
        f"CR {pooled['cr']:.1f}, F1 {pooled['f1']:.1f} (subject mean F1 {subjects['f1']:.1f}{spread}, "
        f"{n} subject{'' if n == 1 else 's'}, {summary['n_trials']} trials{probes}) -> {run_dir}"
    )


def _run_protocol(
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    protocol: str,
    test_dataset: str,
    train: Callable[[MethodSpec, str], RunResult],
    *,
    check: Callable[[str], list[str]],
    features: dict[str, Any],
) -> int:
    """Train and save every (method, label) pair of ``args``, one run directory each (docs/DESIGN.md Sec. 10).

    Nothing trains unless every run can start: a method with a domain adversary is refused
    within a database, ``check(label)`` lists what stops a label (``--folds`` subjects without
    a rating, too few subjects for the validation draw), an existing ``summary.json`` needs
    ``--overwrite``, and every run directory is claimed with a ``.running`` file
    (:func:`_claim`), so two processes never write one directory. After training, a run
    checks ``summary.json`` again, then writes ``command.json``, ``features.json``
    (``features``: the input feature files) and :func:`save_run`'s files, ``summary.json``
    last, and releases its claim. After an error, the remaining claims are released too.
    """
    from .evaluation.report import save_run

    specs = [
        get_method(name, lambda_subject=cfg.protocol.lambda_subject, lambda_domain=cfg.protocol.lambda_domain)
        for name in args.method
    ]
    if protocol == "within":
        cross_only = [spec.name for spec in specs if spec.requires_target_domain]
        if cross_only:
            raise ValueError(
                f"{', '.join(cross_only)} {'needs' if len(cross_only) == 1 else 'need'} a second, different target "
                "database (domain adversary): sieeg cross SOURCE TARGET"
            )
    labels = list(LABELS) if args.label == "both" else [args.label]
    problems = [f"{label}: {problem}" for label in labels for problem in check(label)]
    if problems:
        raise ValueError(f"{'; '.join(problems)}; nothing was trained")
    runs = [
        (spec, label, _run_dir(args.out, protocol, test_dataset, spec.name, args.tag, label, cfg.seed))
        for spec in specs
        for label in labels
    ]
    existing = [str(run_dir) for _, _, run_dir in runs if (run_dir / "summary.json").exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"run(s) already exist: {', '.join(existing)}; pass --overwrite, or choose another --tag or --out"
        )
    claims: dict[Path, str] = {}
    try:
        for _, _, run_dir in runs:
            claims[run_dir] = _claim(run_dir, args)
        logger.info("%d run(s): %s", len(runs), ", ".join(f"{spec.name}/{label}" for spec, label, _ in runs))
        for spec, label, run_dir in runs:
            result = train(spec, label)
            summary = run_dir / "summary.json"
            if summary.exists() and not args.overwrite:
                raise FileExistsError(
                    f"another process saved a run in {run_dir} while this one trained; this run was not saved "
                    "(pass --overwrite to replace it)"
                )
            summary.unlink(missing_ok=True)  # a replaced run is incomplete until its new summary.json
            shutil.rmtree(run_dir / "history", ignore_errors=True)  # fold histories of a replaced run
            _write_json(run_dir / "command.json", _command_record(args))
            _write_json(run_dir / "features.json", features)
            save_run(result, run_dir, config=cfg)  # writes summary.json last: the run is complete
            _release(run_dir, claims.pop(run_dir), args.out)
            print(_result_line(result.summary, run_dir), flush=True)
    finally:
        for run_dir, claim in claims.items():
            _release(run_dir, claim, args.out)
    return 0


def _describe(fs: FeatureSet, path: Path) -> str:
    """``sieeg info`` text: sizes, montage, bands, power range, label balance and trials per subject."""
    threshold = ProtocolConfig().label_threshold
    per_subject = [int(np.sum(fs.subjects == s)) for s in fs.subject_ids]
    low, high = min(per_subject), max(per_subject)
    finite = fs.band_power[np.isfinite(fs.band_power)]
    lines = [
        f"file        {path}",
        f"dataset     {fs.dataset}",
        f"trials      {len(fs)} from {len(fs.subject_ids)} subjects "
        f"({low if low == high else f'{low}-{high}'} per subject)",
        f"channels    {fs.n_channels}: {' '.join(fs.ch_names)}",
        "bands       " + ", ".join(f"{name} {lo:g}-{hi:g} Hz" for name, lo, hi in fs.bands),
        f"log10 power mean {finite.mean():.2f}, min {finite.min():.2f}, max {finite.max():.2f}",
    ]
    for label in LABELS:
        ratings = fs.ratings(label)
        rated = np.isfinite(ratings)
        n_high = int(np.sum(ratings[rated] > threshold))
        share = f", {n_high} high (> {threshold:g}) = {100 * n_high / rated.sum():.1f} %" if rated.any() else ""
        lines.append(f"{label:<12}{int(rated.sum())} rated{share}")
    meta = fs.meta
    if meta.get("feature_config"):
        features = meta["feature_config"]
        rate = f"{features['resample_sfreq']:g} Hz" if features.get("resample_sfreq") else "native rate"
        lines.append(
            f"extraction  sieeg {meta.get('sieeg_version', '?')}: {features.get('reference') or 'no'} reference, "
            f"{rate}, Welch {features['welch_segment_sec']:g} s segments with "
            f"{100 * features['welch_overlap']:g} % overlap"
        )
    elif meta.get("generator"):
        lines.append(f"generator   {meta['generator']} (seed {meta.get('seed')})")
    skipped = meta.get("skipped", [])
    if skipped:
        lines.append(f"skipped     {len(skipped)} of {meta.get('n_input_trials', '?')} input trials:")
        lines += [f"  subject {s['subject']} trial {s['trial']}: {s['reason']}" for s in skipped]

    width = max(len("subject"), *map(len, fs.subject_ids)) + 2
    lines += ["", f"{'subject':<{width}}{'trials':>8}" + "".join(f"{f'{lab}>{threshold:g}':>12}" for lab in LABELS)]
    for subject, n in zip(fs.subject_ids, per_subject):
        rows = fs.subjects == subject
        counts = "".join(f"{int(np.sum(fs.ratings(label)[rows] > threshold)):>12}" for label in LABELS)
        lines.append(f"{subject:<{width}}{n:>8}{counts}")
    return "\n".join(lines)


def _bias_line(row: dict[str, Any], dataset: str) -> str:
    """One row of the Experiment 1 table, in percent."""
    score = f"accuracy {100 * row['accuracy']:.1f} % +- {100 * row['accuracy_std']:.1f}"
    if row["task"] == "subject":
        return (
            f"subject identification, {dataset}, normalization {row['normalization']}: {score} "
            f"(chance {100 * row['chance_uniform']:.1f} %, majority {100 * row['chance_majority']:.1f} %; "
            f"{row['n_classes']} subjects, {row['n_samples']} trials)"
        )
    folds = "subject-grouped folds" if row.get("cv") == "subject" else "trial-level folds"
    domains = " vs ".join(f"{name} ({n} trials)" for name, n in row["n_per_domain"].items())
    return (
        f"database identification ({folds}), {domains}, normalization {row['normalization']}: {score} "
        f"(balanced accuracy {100 * row['balanced_accuracy']:.1f} %, majority {100 * row['chance_majority']:.1f} %)"
    )


def _bias_csv(table: pd.DataFrame) -> pd.DataFrame:
    """The Experiment 1 table as ``bias --out`` writes it: scores and chance levels in percent (``*_pct``)."""
    out = table.copy()
    for column in _BIAS_PERCENT:
        if column in out:
            out[column] = 100 * out[column]
    if "n_per_domain" in out:
        out["n_per_domain"] = out["n_per_domain"].map(lambda v: json.dumps(v) if isinstance(v, dict) else v)
    first = [column for column in ("task", "normalization", "cv", *_BIAS_PERCENT) if column in out]
    out = out[first + [column for column in out.columns if column not in first]]
    return out.rename(columns={column: f"{column}_pct" for column in _BIAS_PERCENT})


# ---------------------------------------------------------------------- commands
def _extract_mahnob(args: argparse.Namespace) -> int:
    from .data.mahnob import iter_mahnob_trials

    cfg = load_config(args.config, args.overrides)
    path = _new_feature_file(args)
    skipped: list[dict[str, str]] = []  # rated sessions the loader drops (no BDF, unreadable)
    trials = iter_mahnob_trials(
        args.root, channels=cfg.features.channels, subjects=args.subjects, pad_sec=args.pad_sec, skipped=skipped
    )
    return _extract(trials, cfg, path, loader_skipped=skipped)


def _extract_manifest(args: argparse.Namespace) -> int:
    from .data.manifest import iter_manifest_trials

    cfg = load_config(args.config, args.overrides)
    path = _new_feature_file(args)
    skipped: list[dict[str, str]] = []  # manifest rows the loader drops (missing file, crop outside the recording)
    trials = iter_manifest_trials(
        args.csv, dataset=args.dataset, sfreq=args.sfreq, ch_names=args.ch_names, scale=args.scale, skipped=skipped
    )
    return _extract(trials, cfg, path, loader_skipped=skipped)


def _extract_synthetic(args: argparse.Namespace) -> int:
    from .data.synthetic import make_synthetic_featureset, make_synthetic_trials

    cfg = load_config(args.config, args.overrides)
    path = _new_feature_file(args)
    kwargs = {
        "dataset": args.dataset,
        "channels": cfg.features.channels,
        "bands": cfg.features.bands,
        "domain_shift": args.domain_shift,
        "seed": args.seed,
    }
    if args.raw:
        return _extract(make_synthetic_trials(args.n_subjects, args.n_trials, **kwargs), cfg, path)
    return _save_features(make_synthetic_featureset(args.n_subjects, args.n_trials, **kwargs), path)


def _info(args: argparse.Namespace) -> int:
    print(_describe(FeatureSet.load(args.file), args.file))
    return 0


def _bias(args: argparse.Namespace) -> int:
    from .analysis.bias import run_bias_analysis

    fs = _load_features(args.file, args.exclude_subjects)
    other = FeatureSet.load(args.other) if args.other is not None else None
    table = run_bias_analysis(
        fs, other, balance=not args.no_balance, n_splits=args.n_splits, n_estimators=args.n_estimators,
        seed=args.seed, n_jobs=args.jobs,
    )
    rows = table.to_dict("records")
    for row in rows:
        print(_bias_line(row, fs.dataset))
    if any(row["task"] == "domain" for row in rows):
        print(_BIAS_NOTE)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _bias_csv(table).to_csv(args.out, index=False)
        logger.info("Wrote %s", args.out)
    return 0


def _within(args: argparse.Namespace) -> int:
    from .evaluation.protocols import run_within_database

    cfg = _run_config(args)
    fs = _load_features(args.file, args.exclude_subjects)
    features = {"file": _feature_record(args.file, fs, args.exclude_subjects)}

    def check(label: str) -> list[str]:
        rated = _rated_subjects(fs, label)
        if not rated:
            return [f"no trial of {fs.dataset} is rated"]
        problems = _fold_problems(args.folds, fs, rated, args.exclude_subjects)
        n_val = cfg.protocol.n_val_subjects
        if n_val > len(rated) - 2:  # each fold needs a test subject and at least one training subject
            problems.append(
                f"protocol.n_val_subjects={n_val} leaves no training subject: {fs.dataset} has {len(rated)} "
                "rated subjects, one of them the test subject"
            )
        return problems

    def train(spec: MethodSpec, label: str) -> RunResult:
        return run_within_database(fs, spec, label, cfg, folds=args.folds, tag=args.tag)

    return _run_protocol(args, cfg, "within", fs.dataset, train, check=check, features=features)


def _cross(args: argparse.Namespace) -> int:
    from .evaluation.protocols import run_cross_database

    cfg = _run_config(args)
    source = _load_features(args.source, args.exclude_source)
    target = _load_features(args.target, args.exclude_target)
    features = {
        "source": _feature_record(args.source, source, args.exclude_source),
        "target": _feature_record(args.target, target, args.exclude_target),
    }

    def check(label: str) -> list[str]:
        problems = []
        rated_target = _rated_subjects(target, label)
        if rated_target:
            problems += _fold_problems(args.folds, target, rated_target, args.exclude_target)
        else:
            problems.append(f"no trial of {target.dataset} is rated")
        rated_source, n_val = _rated_subjects(source, label), cfg.protocol.n_val_subjects
        if not rated_source:
            problems.append(f"no trial of {source.dataset} is rated")
        elif n_val > len(rated_source) - 1:
            problems.append(
                f"protocol.n_val_subjects={n_val} leaves no training subject among the {len(rated_source)} rated "
                f"{source.dataset} subjects"
            )
        return problems

    def train(spec: MethodSpec, label: str) -> RunResult:
        return run_cross_database(source, target, spec, label, cfg, folds=args.folds, tag=args.tag)

    return _run_protocol(args, cfg, "cross", target.dataset, train, check=check, features=features)


def _report(args: argparse.Namespace) -> int:
    import pandas as pd

    from .evaluation.report import (
        find_runs,
        load_run,
        probe_table,
        results_table,
        significance_table,
        to_markdown,
    )

    run_dirs = find_runs(args.dirs)
    if not run_dirs:
        raise FileNotFoundError(f"no run directories (with a summary.json) under {', '.join(map(str, args.dirs))}")
    runs = [load_run(run_dir) for run_dir in run_dirs]
    tests = [significance_table(runs, reference=reference, metric=args.metric) for reference in args.reference]
    tests = pd.concat([t for t in tests if len(t)] or tests[:1], ignore_index=True)
    scores = "pooled over all test trials" if args.source == "pooled" else "averaged over test subjects"
    parts = [
        "# Emotion recognition results",
        f"{len(runs)} runs. Classification rate (CR) and weighted F1 in %, {scores}. A row averages the runs with "
        "the same protocol, databases, method, tag and lambdas, normally different seeds (`n_runs`: the most runs "
        "behind one score); `n_subjects` counts the test subjects of its runs.",
        to_markdown(results_table(runs, source=args.source)),
        f"## Paired tests across test subjects ({args.metric.upper()})",
    ]
    if len(tests):
        parts += [
            "Per-subject scores of the reference (`mean_a`) and the other method (`mean_b`) on their common test "
            "subjects, for the same protocol, databases, tag and label; `mean_diff` > 0 means the reference scores "
            "higher. `p_ttest`: paired t-test, `p_wilcoxon`: Wilcoxon signed-rank test; a blank test is undefined "
            "for these scores (every subject has the same difference).",
            to_markdown(tests, floatfmt=".3g"),
        ]
    else:
        parts.append(
            f"No comparison: no protocol, databases, tag and label have runs of {' or '.join(args.reference)} "
            "and of another method on at least two common test subjects (runs with different --tag are not "
            "compared)."
        )
    probes = probe_table(runs)
    if len(probes):
        parts += [
            "## Subject probes (not in the paper)",
            "Linear probes of each fold's representation, as fractions: `probe_subject_acc` decodes the subject "
            "of validation trials (chance `probe_subject_chance`), `probe_group_bacc` separates the test subject "
            "from the in-group (balanced accuracy; 0.5 = indistinguishable). Lower means less subject information.",
            to_markdown(probes, floatfmt=".3f"),
        ]
    text = "\n\n".join(parts) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        logger.info("Wrote %s", args.out)
    return 0


# -------------------------------------------------------------------------- main
def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    """Log to stderr: sieeg at INFO (DEBUG with -v, WARNING with -q), other libraries at WARNING.

    TensorFlow's Python logger is raised to ERROR unless ``-v``: every fold builds a new
    trainer, and TensorFlow warns each time that the same train step is traced again.
    """
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("sieeg").setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    logging.getLogger("tensorflow").setLevel(logging.WARNING if verbose else logging.ERROR)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``sieeg`` with ``argv`` (default: ``sys.argv[1:]``) and return the exit status.

    Invalid arguments exit with status 2 (argparse); ``-v`` / ``-q`` go before or after the
    command. Missing files, invalid configuration values and existing runs (OSError, KeyError,
    TypeError, ValueError) print a one-line error and return 1; with ``-v`` the exception
    propagates.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose and args.quiet:  # one before the command and one after it
        parser.error("argument -q/--quiet: not allowed with argument -v/--verbose")
    args.argv = argv
    _configure_logging(verbose=args.verbose, quiet=args.quiet)
    try:
        return args.func(args)
    except (OSError, KeyError, TypeError, ValueError) as err:
        if args.verbose:
            raise
        message = err.args[0] if isinstance(err, KeyError) and err.args else err
        print(f"sieeg {args.command}: error: {' '.join(str(message).split())}", file=sys.stderr)
        return 1
