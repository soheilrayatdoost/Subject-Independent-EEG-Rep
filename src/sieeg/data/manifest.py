"""Generic CSV-manifest loader, used for DAI-EF (paper Sec. 3) or any other dataset.

One row per trial::

    subject,trial,path,valence,arousal,tmin,tmax
    P01,1,raw/P01.bdf,7,3,12.0,72.0
    P01,2,raw/P01.bdf,2,6,95.5,155.5
    P02,1,arrays/P02_t1.npy,5,5,,

Required columns: ``subject, trial, path, valence, arousal``. Optional columns:

* ``tmin``, ``tmax``: crop ``[tmin, tmax)`` in seconds from the start of the file;
* ``sfreq``, ``ch_names`` (``;``-separated): sampling rate and channel names of array files;
* ``array_key``: the variable that holds the EEG in a .npz / .mat file;
* ``transpose``: true if the array is ``(n_times, n_channels)``;
* ``scale``: multiplier to volts, e.g. ``1e-6`` for microvolts.

Column names are exact, because an optional column that is not found silently falls back
to its default (the whole file, no scaling): a known column in another case or spelling
(``Tmin``, ``Scale``, ``array key``) or a common alias (``onset``, ``fs``, ``channels`` ...)
is an error naming the expected column. Other columns are ignored with one warning.
Whitespace around header names and text cells (``subject``, ``trial``, ``path``, ...) is
removed. An empty rating cell means "not rated"; any other non-numeric rating is an error.

Relative paths are relative to the manifest's folder. Several rows may share one file.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.io

from .trial import Trial

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("subject", "trial", "path", "valence", "arousal")
OPTIONAL_COLUMNS = ("tmin", "tmax", "sfreq", "ch_names", "array_key", "transpose", "scale")
#: Other names people give the manifest columns; such a column is rejected with the expected name.
COLUMN_ALIASES = {
    "onset": "tmin", "start": "tmin",
    "offset": "tmax", "end": "tmax", "stop": "tmax",
    "fs": "sfreq", "srate": "sfreq", "sampling_rate": "sfreq",
    "arraykey": "array_key", "key": "array_key",
    "channels": "ch_names",
}
ARRAY_SUFFIXES = (".npy", ".npz", ".mat")
_TEXT_COLUMNS = ("subject", "trial", "path", "ch_names", "array_key")  # read as strings ("01" stays "01")
_RATING_COLUMNS = ("valence", "arousal")
_FILE_EXTRAS = ("sfreq", "ch_names")  # metadata stored next to the EEG in .npz / .mat files
_TRUE, _FALSE = ("1", "true", "yes"), ("0", "false", "no")


def _check_columns(columns: Sequence[str], source: object) -> None:
    """Reject duplicate, misspelled and aliased column names; log the other unknown ones once."""
    duplicates = sorted({c for c in columns if list(columns).count(c) > 1})
    if duplicates:
        raise ValueError(f"Manifest {source}: duplicate column(s) {duplicates}")
    known = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    wrong, ignored = [], []
    for name in columns:
        if name in known:
            continue
        key = re.sub(r"[\s\-]+", "_", name.strip().lower())
        expected = key if key in known else COLUMN_ALIASES.get(key)
        if expected is None:
            ignored.append(name)
        else:
            wrong.append(f"{name!r} (expected {expected!r})")
    if wrong:
        raise ValueError(
            f"Manifest {source}: unrecognised column name(s) {', '.join(wrong)}; names are case-sensitive. "
            f"Required: {list(REQUIRED_COLUMNS)}, optional: {list(OPTIONAL_COLUMNS)}"
        )
    if ignored:
        logger.warning("Manifest %s: ignored columns: %s", source, ", ".join(repr(c) for c in ignored))


def _strip_text(value: Any) -> Any:
    """A text cell without surrounding whitespace; a blank one becomes NaN."""
    return (value.strip() or np.nan) if isinstance(value, str) else value


def _parse_ratings(values: pd.Series, column: str, source: object) -> pd.Series:
    """Ratings as floats: an empty cell is NaN (not rated), any other non-numeric cell raises."""
    parsed, bad = [], []
    for i, value in enumerate(values):
        value = _strip_text(value)
        if pd.isna(value):
            parsed.append(np.nan)
            continue
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            bad.append(i)
    if bad:
        cells = [values.iloc[i] for i in bad]
        raise ValueError(
            f"Manifest {source}: non-numeric {column} {cells} in rows {bad} (0-based); "
            "leave the cell empty for a missing rating"
        )
    return pd.Series(parsed, index=values.index, dtype=np.float64)


def _check_manifest(df: pd.DataFrame, source: object) -> pd.DataFrame:
    """Validated copy of a manifest: column names, stripped header and text cells, numeric ratings."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _check_columns(list(df.columns), source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {source} lacks required column(s) {missing}; required: {list(REQUIRED_COLUMNS)}")
    for column in _TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(_strip_text)
    blank = df[["subject", "trial", "path"]].isna().any(axis=1).to_numpy()
    if blank.any():
        rows = np.flatnonzero(blank).tolist()
        raise ValueError(f"Manifest {source}: rows {rows} (0-based) lack a subject, trial or path")
    for column in _RATING_COLUMNS:
        df[column] = _parse_ratings(df[column], column, source)
    return df


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read and validate a manifest CSV, keeping row order; ``path`` values become absolute.

    Raises:
        ValueError: for a missing required column, a misspelled or aliased known column, a
            row without subject, trial or path, or a non-numeric rating.
    """
    path = Path(path)
    header = pd.read_csv(path, nrows=0, skipinitialspace=True).columns  # text columns by their stripped names
    text = {c: str for c in header if str(c).strip() in _TEXT_COLUMNS}
    df = _check_manifest(pd.read_csv(path, dtype=text, skipinitialspace=True), path)
    base = path.resolve().parent
    df["path"] = [str(base / p) for p in df["path"]]  # an absolute p replaces base
    return df


# ------------------------------------------------------------------ array files


def _load_variables(path: Path) -> dict[str, Any]:
    """Every variable of a .npz / .mat file; the array of a .npy file is the variable ``"data"``."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return {"data": np.load(path, allow_pickle=False)}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if suffix == ".mat":
        return {k: v for k, v in scipy.io.loadmat(path, squeeze_me=True).items() if not k.startswith("__")}
    raise ValueError(f"{path}: not an array file; expected one of {ARRAY_SUFFIXES}")


def _numeric_matrix(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 2 and np.issubdtype(value.dtype, np.number)


def _select_array(
    variables: Mapping[str, Any], path: Path, array_key: str | None
) -> tuple[str, np.ndarray, dict[str, Any]]:
    """The rules of :func:`load_array_file`, applied to the loaded variables of ``path``; also returns the key."""
    if path.suffix.lower() == ".npy":
        array_key = None  # a .npy file holds a single unnamed array
    if array_key is not None:
        if array_key not in variables:
            raise KeyError(f"{path}: no variable {array_key!r}; found {sorted(variables)}")
        key = array_key
    elif "data" in variables:
        key = "data"
    else:
        matrices = [k for k, v in variables.items() if k not in _FILE_EXTRAS and _numeric_matrix(v)]
        if len(matrices) != 1:
            raise ValueError(
                f"{path}: cannot tell which variable holds the EEG (2-D arrays: {matrices}); set array_key"
            )
        key = matrices[0]
    array = np.asarray(variables[key])
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D EEG array in {key!r}, got shape {array.shape}")
    extras: dict[str, Any] = {}
    if "sfreq" in variables:
        extras["sfreq"] = float(np.asarray(variables["sfreq"]).item())
    if "ch_names" in variables:  # MATLAB char matrices pad names with spaces
        extras["ch_names"] = [str(c).strip() for c in np.atleast_1d(variables["ch_names"])]
    return key, array, extras


def load_array_file(path: str | Path, *, array_key: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """The EEG matrix of a .npy / .npz / .mat file and the ``sfreq`` / ``ch_names`` stored with it.

    In .npz and .mat files the matrix is ``array_key``, else ``"data"``, else the only
    2-D numeric array (``ValueError`` if there is not exactly one); ``array_key`` is
    ignored for .npy. .mat files are read with ``scipy.io.loadmat(squeeze_me=True)``,
    so MATLAB scalars and vectors do not count as 2-D.
    """
    path = Path(path)
    _, array, extras = _select_array(_load_variables(path), path, array_key)
    return array, extras


# ------------------------------------------------------------------ trials


def _read_raw(path: Path) -> mne.io.BaseRaw:
    import mne

    return mne.io.read_raw(path, preload=True, verbose="error")


def _open(path: Path, reader: Callable[[Path], mne.io.BaseRaw] | None) -> dict[str, Any]:
    """The variables of an array file; an MNE-readable file becomes its EEG ``data`` (volts),
    ``sfreq`` and ``ch_names``."""
    if path.suffix.lower() in ARRAY_SUFFIXES:
        return _load_variables(path)
    raw = (reader or _read_raw)(path)
    eeg = [i for i, kind in enumerate(raw.get_channel_types()) if kind == "eeg"]
    if not eeg:
        raise ValueError(f"{path}: no EEG channels")
    return {
        "data": raw.get_data(picks=eeg),
        "sfreq": float(raw.info["sfreq"]),
        "ch_names": [raw.ch_names[i] for i in eeg],
    }


def _cell(row: Mapping[str, Any], name: str) -> Any:
    """Value of an optional column; ``None`` if the column is absent or the cell is empty."""
    value = row.get(name)
    return None if value is None or pd.isna(value) else value


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text not in _TRUE + _FALSE:
            raise ValueError(f"transpose must be one of {_TRUE + _FALSE}, got {value!r}")
        return text in _TRUE
    return bool(value)


def _split_names(text: Any) -> list[str] | None:
    return None if text is None else [n.strip() for n in str(text).split(";") if n.strip()]


def _row_trial(
    row: Mapping[str, Any],
    variables: Mapping[str, Any],
    path: Path,
    *,
    index: int,
    dataset: str,
    sfreq: float | None,
    ch_names: Sequence[str] | None,
    scale: float | None,
) -> tuple[Trial, tuple[str, str | None, int, int]]:
    """The row's trial, and the samples it reads: ``(path, array variable, start, stop)``."""
    key = None
    if path.suffix.lower() in ARRAY_SUFFIXES:
        key, data, extras = _select_array(variables, path, _cell(row, "array_key"))
        data = data.T if _as_bool(_cell(row, "transpose")) else data
        fs = _first(_cell(row, "sfreq"), extras.get("sfreq"), sfreq)
        names = _first(_split_names(_cell(row, "ch_names")), extras.get("ch_names"), ch_names)
        if fs is None:
            raise ValueError("unknown sampling rate: add an sfreq column, store sfreq in the file or pass sfreq=")
        if names is None:
            raise ValueError(
                "unknown channel names: add a ch_names column, store ch_names in the file or pass ch_names="
            )
        if len(names) != data.shape[0]:
            raise ValueError(
                f"the array has {data.shape[0]} rows but there are {len(names)} channel names "
                "(set transpose if it is (n_times, n_channels))"
            )
    else:  # MNE-readable: the file defines the sampling rate and the channels
        data, fs, names = variables["data"], variables["sfreq"], variables["ch_names"]
    fs = float(fs)
    tmin, tmax = _cell(row, "tmin"), _cell(row, "tmax")
    start = 0 if tmin is None else int(round(float(tmin) * fs))
    stop = data.shape[1] if tmax is None else int(round(float(tmax) * fs))
    if not 0 <= start < stop <= data.shape[1]:
        raise ValueError(
            f"tmin={tmin}, tmax={tmax} s give samples [{start}, {stop}), which is empty or outside "
            f"the recording's {data.shape[1]} samples"
        )
    factor = float(_first(_cell(row, "scale"), scale, 1.0))
    trial = Trial(
        dataset=dataset,
        subject=row["subject"],
        trial=row["trial"],
        data=np.asarray(data[:, start:stop], dtype=np.float64) * factor,  # a copy: never a view of the cache
        sfreq=fs,
        ch_names=list(names),
        valence=row["valence"],
        arousal=row["arousal"],
        meta={"path": str(path), "row": index, "onset_sec": start / fs, "offset_sec": stop / fs, "scale": factor},
    )
    return trial, (str(path), key, start, stop)


def iter_manifest_trials(
    manifest: str | Path | pd.DataFrame,
    *,
    dataset: str = "daief",
    sfreq: float | None = None,
    ch_names: Sequence[str] | None = None,
    scale: float | None = None,
    reader: Callable[[Path], mne.io.BaseRaw] | None = None,
    on_error: str = "warn",
    skipped: list[dict[str, str]] | None = None,
) -> Iterator[Trial]:
    """Yield one :class:`Trial` per manifest row, in row order.

    ``manifest`` is a CSV path or a frame from :func:`read_manifest` (a frame is validated
    the same way, on a copy). MNE-readable files (.bdf .edf .gdf .fif .set .vhdr .cnt ...)
    are opened with ``reader(path)`` (default ``mne.io.read_raw``) and keep only their EEG
    channels; their sampling rate and channel names come from the file. For .npy / .npz /
    .mat files they come from the row, else the file, else the ``sfreq`` / ``ch_names``
    keywords. The data are multiplied by the row's ``scale``, else the keyword, else 1.0.
    The most recently opened file is cached, so consecutive rows that share a file read it
    once. ``on_error="warn"`` logs and skips a failing row; ``"raise"`` re-raises the error.
    Every skipped row is appended to ``skipped`` (if given) as ``{"subject", "trial",
    "reason"}``, for ``extract_features(..., loader_skipped=skipped)``; the list is complete
    once the iterator is exhausted. Rows that read the same samples as an earlier row (same
    file, array variable and crop, e.g. forgotten ``tmin`` / ``tmax``) are reported in one
    warning at the end.
    """
    if on_error not in ("warn", "raise"):
        raise ValueError(f"on_error must be 'warn' or 'raise', got {on_error!r}")
    if isinstance(manifest, pd.DataFrame):
        df = _check_manifest(manifest, "(DataFrame)")
    else:
        df = read_manifest(manifest)
    cached: tuple[Path, dict[str, Any]] | None = None
    first_reader: dict[tuple[str, str | None, int, int], int] = {}
    repeats: list[tuple[int, int]] = []  # (row, earlier row with the same samples)
    n_loaded = 0
    for index, row in enumerate(df.to_dict("records")):
        try:
            path = Path(row["path"])
            if cached is None or cached[0] != path:
                cached = None  # release the previous file before reading the next one
                cached = (path, _open(path, reader))
            trial, samples = _row_trial(
                row, cached[1], path, index=index, dataset=dataset, sfreq=sfreq, ch_names=ch_names, scale=scale
            )
        except Exception as err:
            if on_error == "raise":
                raise
            logger.warning(
                "Skipping manifest row %d (subject %s, trial %s): %s", index, row["subject"], row["trial"], err
            )
            if skipped is not None:
                reason = f"{type(err).__name__}: {err}"
                skipped.append({"subject": str(row["subject"]), "trial": str(row["trial"]), "reason": reason})
            continue
        earlier = first_reader.setdefault(samples, index)
        if earlier != index:
            repeats.append((index, earlier))
        n_loaded += 1
        yield trial
    if repeats:
        examples = ", ".join(f"row {i} = row {j}" for i, j in repeats[:5])
        logger.warning(
            "%d manifest row(s) read the same samples (file, array variable and crop) as an earlier row, so their "
            "features are identical: %s%s. Check the tmin / tmax columns",
            len(repeats), examples, ", ..." if len(repeats) > 5 else "",
        )
    logger.info("Loaded %d of %d manifest rows", n_loaded, len(df))
