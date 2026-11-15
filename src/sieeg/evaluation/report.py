"""Run directories and result tables (paper Table 1; docs/DESIGN.md 8.3).

One run (protocol x test dataset x method x label x seed) is saved as::

    summary.json        protocol, method, tag, label, datasets, lambdas, seed, counts,
                        pooled / subject_mean scores, subject-probe means (null unless
                        protocol.subject_probe), created, sieeg_version
    predictions.csv     one row per test trial: fold, dataset, subject, trial, rating, y_true, y_pred, p1
    folds.csv           one row per fold: subjects, set sizes, best epoch, scores, seconds,
                        probe_subject_acc, probe_subject_chance, probe_group_bacc
    history/<fold>.csv  per-epoch training history of each fold
    config.yaml         the experiment configuration (optional)

(``sieeg within`` / ``cross`` add ``command.json`` and ``features.json``.)
:func:`results_table` lays runs out like Table 1, :func:`significance_table` runs the paired
tests across subjects behind the paper's significance claim (Sec. 6), and
:func:`probe_table` summarises the optional subject probes. Runs are only averaged when they
share the protocol, both databases, method, tag and lambdas (normally different seeds).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, save_config
from ..features.featureset import LABELS
from ..training.methods import available_methods
from ..utils import natural_key
from .metrics import per_subject_metrics
from .stats import COMPARISON_FIELDS, paired_comparison

logger = logging.getLogger(__name__)

#: Keys of ``summary.json`` (docs/DESIGN.md 8.3).
SUMMARY_KEYS = (
    "protocol", "method", "tag", "label", "train_dataset", "test_dataset", "lambdas", "seed",
    "n_folds", "n_subjects", "n_trials", "pooled", "subject_mean", "probe", "created", "sieeg_version",
)
#: Score columns of :func:`results_table`.
RESULT_COLUMNS = ("valence_cr", "valence_f1", "arousal_cr", "arousal_f1")
#: Score columns of :func:`probe_table` (fractions; the fold columns of the same name, averaged).
PROBE_COLUMNS = ("probe_subject_acc", "probe_subject_chance", "probe_group_bacc")

_SOURCES = ("pooled", "subject_mean")
_PROTOCOL_ORDER = ("within", "cross")  # as in Table 1
#: A results-table row: only runs with the same key are averaged. lambdas = sorted JSON of summary["lambdas"].
_RUN_KEY = ("protocol", "test_dataset", "train_dataset", "method", "tag", "lambdas")
#: A group of paired tests; inside it, the entries are (method, lambdas).
_GROUP_KEY = ("protocol", "test_dataset", "train_dataset", "tag", "label")
_PROBE_KEYS = ("subject_acc", "subject_chance", "group_bacc")  # keys of summary["probe"] (protocols.PROBE_KEYS)
#: Identifier columns of predictions.csv / folds.csv, read back as strings (subject "01" stays "01").
_ID_COLUMNS = ("fold", "dataset", "subject", "trial", "test_subjects", "val_subjects")


@dataclass
class RunResult:
    """Everything one run produces (see the module docstring for the files).

    ``run_dir`` is the directory a run was loaded from (:func:`load_run`), None for a run in
    memory; the tables name it in their warnings.
    """

    summary: dict[str, Any]
    predictions: pd.DataFrame
    folds: pd.DataFrame
    histories: dict[str, pd.DataFrame]
    run_dir: Path | None = None


# ------------------------------------------------------------------------ run io
def _jsonable(value: Any) -> Any:
    """``value`` with numpy scalars / arrays as Python objects and NaN / inf as ``None``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _join_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` whose list-like cells (e.g. ``val_subjects``) are ``;``-joined strings."""
    out = df.copy()
    for column in out.columns[out.dtypes == object]:
        out[column] = out[column].map(
            lambda v: ";".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else v
        )
    return out


def _file_name(fold: str) -> str:
    """``fold`` as a safe file name (subject ids may contain ``/`` or ``:``)."""
    return re.sub(r"[^\w.-]", "_", str(fold))


def _read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False keeps ids such as "NA" or "None"; only empty cells are missing.
    return pd.read_csv(path, dtype=dict.fromkeys(_ID_COLUMNS, str), keep_default_na=False, na_values=[""])


def save_run(result: RunResult, out_dir: str | Path, config: ExperimentConfig | None = None) -> Path:
    """Write ``result`` (and ``config`` as ``config.yaml``) to ``out_dir``; returns the directory.

    ``summary.json`` is written last, so a directory that has one holds a complete run
    (:func:`find_runs`). It is strict JSON: NaN and inf are written as ``null``. List-valued
    fold cells are joined with ``;`` and fold names are made file-name safe for
    ``history/<fold>.csv``. Existing files are overwritten.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing = [key for key in SUMMARY_KEYS if key not in result.summary]
    if missing:
        logger.warning("Summary of run %s lacks %s", out, missing)
    result.predictions.to_csv(out / "predictions.csv", index=False)
    _join_lists(result.folds).to_csv(out / "folds.csv", index=False)
    if result.histories:
        (out / "history").mkdir(exist_ok=True)
    for fold, history in result.histories.items():
        history.to_csv(out / "history" / f"{_file_name(fold)}.csv", index=False)
    if config is not None:
        save_config(config, out / "config.yaml")
    text = json.dumps(_jsonable(result.summary), indent=2, allow_nan=False)
    (out / "summary.json").write_text(text + "\n", encoding="utf-8")
    return out


def load_run(run_dir: str | Path) -> RunResult:
    """Read a run written by :func:`save_run`; training histories are not loaded (``{}``)."""
    run_dir = Path(run_dir)
    return RunResult(
        summary=json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
        predictions=_read_csv(run_dir / "predictions.csv"),
        folds=_read_csv(run_dir / "folds.csv"),
        histories={},
        run_dir=run_dir,
    )


def find_runs(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Every directory at or below ``paths`` that contains ``summary.json``, naturally sorted.

    Other files do not matter: a directory that only holds the ``.running`` claim of a run
    still training is not a run, and a complete run that is being overwritten still is.

    Raises:
        FileNotFoundError: if one of ``paths`` does not exist.
    """
    roots = [paths] if isinstance(paths, (str, Path)) else list(paths)
    found: set[Path] = set()
    for root in map(Path, roots):
        if not root.exists():
            raise FileNotFoundError(f"No such file or directory: {root}")
        found.update(path.parent for path in root.rglob("summary.json") if path.is_file())
    return sorted(found, key=lambda path: natural_key(str(path)))


# ------------------------------------------------------------------------ tables
def _as_runs(runs: Iterable[RunResult | str | Path]) -> list[RunResult]:
    return [run if isinstance(run, RunResult) else load_run(run) for run in runs]


def _rank(value: str, order: Sequence[str]) -> tuple:
    """Sort key: the values in ``order`` first (in that order), then the others naturally."""
    return (order.index(value), ()) if value in order else (len(order), natural_key(value))


def _lambdas(summary: dict[str, Any]) -> str:
    """``summary["lambdas"]`` as sorted JSON, e.g. ``{"domain": 0.1, "group": 0.15}`` (``{}`` without adversary)."""
    return json.dumps(summary.get("lambdas") or {}, sort_keys=True)


def _run_key(summary: dict[str, Any]) -> tuple[str, ...]:
    """The results-table row of a run (:data:`_RUN_KEY`); an untagged run has tag ``""``."""
    return (
        summary["protocol"], summary["test_dataset"], summary.get("train_dataset") or "", summary["method"],
        summary.get("tag") or "", _lambdas(summary),
    )


def _run_order(key: tuple[str, ...]) -> tuple:
    protocol, test_dataset, train_dataset, method, tag, lambdas = key
    return (
        _rank(protocol, _PROTOCOL_ORDER), natural_key(test_dataset), natural_key(train_dataset),
        _rank(method, available_methods()), natural_key(tag), natural_key(lambdas),
    )


def _group_order(group: tuple[str, ...]) -> tuple:
    protocol, test_dataset, train_dataset, tag, label = group
    return (
        _rank(protocol, _PROTOCOL_ORDER), natural_key(test_dataset), natural_key(train_dataset), natural_key(tag),
        _rank(label, LABELS),
    )


def _row_name(key: tuple[str, ...]) -> str:
    """A results-table row for log messages, e.g. ``cross daief -> mahnob sida_eeg (tag v2) {"domain": 0.1, ...}``."""
    protocol, test_dataset, train_dataset, method, tag, lambdas = key
    data = test_dataset if train_dataset in ("", test_dataset) else f"{train_dataset} -> {test_dataset}"
    return f"{protocol} {data} {method}" + (f" (tag {tag})" if tag else "") + (f" {lambdas}" if lambdas != "{}" else "")


def _run_name(run: RunResult, index: int) -> str:
    """The run's directory, or its position and seed for a run in memory."""
    if run.run_dir is not None:
        return str(run.run_dir)
    return f"run {index + 1} ({run.summary['label']}, seed {run.summary.get('seed')})"


def _test_subjects(run: RunResult) -> frozenset[str]:
    return frozenset(run.predictions["subject"].astype(str))


def _check_row(key: tuple[str, ...], members: list[tuple[str, RunResult]]) -> None:
    """Warn when the runs of one results-table row are not simply different seeds of one experiment."""
    by_seed: dict[tuple[str, Any], list[str]] = {}
    for name, run in members:
        by_seed.setdefault((run.summary["label"], run.summary.get("seed")), []).append(name)
    repeated = [names for names in by_seed.values() if len(names) > 1]
    if repeated:
        logger.warning(
            "%s: runs of the same label and seed are averaged as if their seeds differed: %s",
            _row_name(key), "; ".join(", ".join(names) for names in repeated),
        )
    subjects = [(name, _test_subjects(run)) for name, run in members]
    if len({tested for _, tested in subjects}) > 1:
        logger.warning(
            "%s: its runs test different subjects, so its scores do not cover the same subjects: %s",
            _row_name(key), ", ".join(f"{name} ({len(tested)} subjects)" for name, tested in subjects),
        )


def results_table(runs: Iterable[RunResult | str | Path], *, source: str = "pooled") -> pd.DataFrame:
    """Table 1 layout: one row per (protocol, test_dataset, train_dataset, method, tag, lambdas).

    Columns: that key (``lambdas`` as sorted JSON, an untagged run has tag ``""``), CR and F1
    per label, ``n_runs`` (the most runs averaged into one score of the row) and ``n_subjects``
    (the distinct test subjects of the row's runs). ``source`` selects the ``"pooled"`` scores
    (all test trials) or the ``"subject_mean"``. Only runs of the same row and label are
    averaged, which are normally different seeds: a warning names the run directories when two
    of them have the same seed, or when the row's runs test different subjects (e.g. a
    ``--folds`` check next to a full run). A label without runs is NaN. Rows follow Table 1:
    within- before cross-database, then test dataset, training dataset, methods in registry
    order (EEG CNN first), tag and lambdas. ``runs`` are :class:`RunResult` objects or run
    directories.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {_SOURCES}, got {source!r}")
    rows: dict[tuple[str, ...], list[tuple[str, RunResult]]] = {}
    for i, run in enumerate(_as_runs(runs)):
        rows.setdefault(_run_key(run.summary), []).append((_run_name(run, i), run))
    records = []
    for key in sorted(rows, key=_run_order):
        members = rows[key]
        _check_row(key, members)
        record: dict[str, Any] = dict(zip(_RUN_KEY, key))
        n_runs = 0
        for label in LABELS:
            cell = [run.summary[source] for _, run in members if run.summary["label"] == label]
            if not cell:
                continue
            n_runs = max(n_runs, len(cell))
            if len(cell) > 1:
                logger.info("Averaging %d runs of %s, %s", len(cell), _row_name(key), label)
            for metric in ("cr", "f1"):
                record[f"{label}_{metric}"] = float(np.mean([scores[metric] for scores in cell]))
        record["n_runs"] = n_runs
        record["n_subjects"] = len(frozenset().union(*(_test_subjects(run) for _, run in members)))
        records.append(record)
    return pd.DataFrame(records, columns=[*_RUN_KEY, *RESULT_COLUMNS, "n_runs", "n_subjects"])


def significance_table(
    runs: Iterable[RunResult | str | Path], *, reference: str, metric: str = "f1"
) -> pd.DataFrame:
    """Paired tests of the ``reference`` method against every other method (paper Sec. 6).

    Per-subject ``metric`` scores (``"cr"`` or ``"f1"``) are computed from each run's
    predictions. Runs are grouped by (protocol, test_dataset, train_dataset, tag, label), so a
    tagged variant is only compared with runs of the same tag. Inside a group, the runs of one
    (method, lambdas) are averaged per subject (normally different seeds), and every reference
    entry is compared with every entry of another method on their common test subjects.
    Groups without the reference, and pairs with fewer than 2 common subjects, are skipped.
    ``runs`` are :class:`RunResult` objects or run directories.

    Columns: protocol, test_dataset, train_dataset, tag, label, reference, reference_lambdas,
    method, method_lambdas (sorted JSON), then the
    :func:`~sieeg.evaluation.stats.paired_comparison` fields with ``a`` = reference and
    ``b`` = method, so ``mean_diff > 0`` means the reference scores higher.
    """
    if metric not in ("cr", "f1"):
        raise ValueError(f"metric must be 'cr' or 'f1', got {metric!r}")
    reference = reference.lower().replace("-", "_")  # method names as get_method() spells them
    groups: dict[tuple[str, ...], dict[tuple[str, str], list[pd.Series]]] = {}
    for run in _as_runs(runs):
        summary = run.summary
        group = (
            summary["protocol"], summary["test_dataset"], summary.get("train_dataset") or "",
            summary.get("tag") or "", summary["label"],
        )
        scores = per_subject_metrics(run.predictions).set_index("subject")[metric]
        groups.setdefault(group, {}).setdefault((summary["method"], _lambdas(summary)), []).append(scores)

    def order(entry: tuple[str, str]) -> tuple:
        return _rank(entry[0], available_methods()), natural_key(entry[1])

    rows = []
    for group in sorted(groups, key=_group_order):
        entries = {entry: pd.concat(seeds).groupby(level=0).mean() for entry, seeds in groups[group].items()}
        name = "/".join(filter(None, group))
        references = sorted((entry for entry in entries if entry[0] == reference), key=order)
        if not references:
            logger.info("No %s run for %s; no comparison", reference, name)
            continue
        for ref in references:
            for other in sorted((entry for entry in entries if entry[0] != reference), key=order):
                n_common = len(entries[ref].index.intersection(entries[other].index))
                if n_common < 2:
                    logger.warning("%s vs %s on %s: %d common subject(s); skipped", reference, other[0], name, n_common)
                    continue
                comparison = paired_comparison(entries[ref], entries[other])
                rows.append(
                    {
                        **dict(zip(_GROUP_KEY, group)), "reference": reference, "reference_lambdas": ref[1],
                        "method": other[0], "method_lambdas": other[1], **comparison,
                    }
                )
    columns = [*_GROUP_KEY, "reference", "reference_lambdas", "method", "method_lambdas", *COMPARISON_FIELDS]
    return pd.DataFrame(rows, columns=columns)


def probe_table(runs: Iterable[RunResult | str | Path]) -> pd.DataFrame:
    """Subject-probe scores per results-table row and label (``protocol.subject_probe``; not in the paper).

    Only runs whose ``summary["probe"]`` holds scores count (an empty table if none does).
    Columns: the row key of :func:`results_table`, ``label``, ``n_runs`` and
    :data:`PROBE_COLUMNS`: the means over the runs of their means over folds, as fractions
    (NaN where no fold had a defined probe). ``probe_subject_acc`` decodes the validation
    subjects from the representation (chance ``probe_subject_chance``) and ``probe_group_bacc``
    separates the test subject from the in-group (0.5 = indistinguishable); lower means less
    subject information.
    """
    cells: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for run in _as_runs(runs):
        probe = run.summary.get("probe")
        if isinstance(probe, dict):
            cells.setdefault((*_run_key(run.summary), run.summary["label"]), []).append(probe)
    records = []
    for cell in sorted(cells, key=lambda c: (_run_order(c[:-1]), _rank(c[-1], LABELS))):
        record: dict[str, Any] = {**dict(zip(_RUN_KEY, cell[:-1])), "label": cell[-1], "n_runs": len(cells[cell])}
        for key, column in zip(_PROBE_KEYS, PROBE_COLUMNS):
            values = [probe.get(key) for probe in cells[cell]]
            finite = [float(v) for v in values if v is not None and math.isfinite(v)]
            record[column] = float(np.mean(finite)) if finite else math.nan
        records.append(record)
    return pd.DataFrame(records, columns=[*_RUN_KEY, "label", "n_runs", *PROBE_COLUMNS])


# ---------------------------------------------------------------------- markdown
def _markdown_cell(value: Any, floatfmt: str) -> str:
    if isinstance(value, (float, np.floating)):
        text = "" if math.isnan(value) else format(value, floatfmt)
    elif value is None or value is pd.NA:
        text = ""
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_row(values: Iterable[Any], floatfmt: str) -> str:
    return "| " + " | ".join(_markdown_cell(v, floatfmt) for v in values) + " |"


def to_markdown(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    """GitHub Markdown table of ``df``'s columns (the index is not printed).

    Floats use ``floatfmt``, missing values are blank and numeric columns are
    right-aligned. Pure Python, so the optional ``tabulate`` package is not needed.
    """
    align = ["---:" if pd.api.types.is_numeric_dtype(dtype) else "---" for dtype in df.dtypes]
    lines = [_markdown_row(df.columns, floatfmt), "|" + "|".join(align) + "|"]
    lines.extend(_markdown_row(row, floatfmt) for row in df.itertuples(index=False, name=None))
    return "\n".join(lines)
