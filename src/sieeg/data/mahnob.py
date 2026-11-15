"""MAHNOB-HCI loader (paper Sec. 3): rated emotion sessions as stimulus-locked trials.

Layout of the database (MAHNOB-HCI manual)::

    <root>/Sessions/<sessionId>/session.xml
    <root>/Sessions/<sessionId>/Part_<p>_S_Trial<k>_emotion.bdf   (absent when physiology is missing)

Only rated emotion sessions carry the self-assessments ``feltVlnc`` / ``feltArsl``
(SAM, 1-9) on the root ``<session>`` element; neutral clips and the tagging
experiment have none. The BDF holds 47 channels: 32 EEG channels
(:data:`~sieeg.config.BIOSEMI32_CHANNELS`), peripheral signals and ``Status``.
Every file keeps 30 s before and after the stimulus, and rising edges of bit
value 16 in ``Status`` mark stimulus onset and offset.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import BIOSEMI32_CHANNELS
from ..utils import natural_key, natural_sorted
from .trial import Trial

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)

DATASET = "mahnob"
STATUS_CHANNEL = "Status"
TRIGGER_MASK = 0xFFFF  # BioSemi trigger bits; bits 16-23 of Status are system flags
STIMULUS_BIT = 16  # rising edges of this Status value mark stimulus onset and offset


def parse_session_xml(path: str | Path) -> dict[str, Any] | None:
    """Read one ``session.xml``; ``None`` unless it is a rated emotion session.

    Rated sessions have ``feltVlnc`` and ``feltArsl`` on the root element. The session
    id is the folder name, the subject id comes from ``<subject id="...">``. Ratings
    are floats; the other values are the attribute strings (``None`` when absent).
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    valence, arousal = root.get("feltVlnc"), root.get("feltArsl")
    if valence is None or arousal is None:
        return None
    subject = root.find("subject")
    if subject is None or not subject.get("id", "").strip():
        raise ValueError(f"{path}: rated session without <subject id=...>")
    try:
        ratings = float(valence), float(arousal)
    except ValueError as err:
        raise ValueError(f"{path}: non-numeric rating ({err})") from err
    return {
        "session": path.parent.name,
        "subject": subject.get("id").strip(),
        "valence": ratings[0],
        "arousal": ratings[1],
        "media_file": root.get("mediaFile"),
        "exp_type": root.get("expType"),
        "cut_nr": root.get("cutNr"),
        "xml": path,
    }


def _find_bdf(folder: Path) -> Path | None:
    """The folder's ``*_emotion.bdf``, else its only ``*.bdf``; ``None`` if there is none."""
    bdfs = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".bdf" and p.is_file())
    candidates = [p for p in bdfs if p.name.lower().endswith("_emotion.bdf")] or bdfs
    if len(candidates) > 1:
        raise ValueError(f"{folder}: cannot choose between BDF files {[p.name for p in candidates]}")
    return candidates[0] if candidates else None


def _session_order(session: Mapping[str, Any]) -> tuple:
    """Natural (subject, session) order; the raw ids break ties such as "1" / "01"."""
    return natural_key(session["subject"]), session["subject"], natural_key(session["session"]), session["session"]


def find_rated_sessions(
    root: str | Path, *, skipped: list[dict[str, str]] | None = None
) -> list[dict[str, Any]]:
    """Rated emotion sessions that have a BDF, sorted by (subject, session) in natural order.

    ``root`` is the database folder or its ``Sessions`` folder. Each dict from
    :func:`parse_session_xml` gains ``"bdf"`` (a :class:`~pathlib.Path`). Rated sessions
    without a BDF (missing physiology) are skipped with one summary warning and appended to
    ``skipped`` (if given) as ``{"subject", "trial" (the session id), "reason"}``.
    """
    root = Path(root)
    sessions_dir = root / "Sessions" if (root / "Sessions").is_dir() else root
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"MAHNOB-HCI sessions folder not found: {sessions_dir}")
    sessions, no_bdf = [], []
    for xml in sessions_dir.glob("*/session.xml"):
        info = parse_session_xml(xml)
        if info is None:
            continue
        bdf = _find_bdf(xml.parent)
        if bdf is None:
            no_bdf.append(info)
        else:
            sessions.append({**info, "bdf": bdf})
    if no_bdf:
        no_bdf.sort(key=_session_order)
        logger.warning(
            "Skipped %d rated MAHNOB-HCI sessions without a BDF file (subjects %s)",
            len(no_bdf),
            ", ".join(natural_sorted(s["subject"] for s in no_bdf)),
        )
        logger.debug("Sessions without a BDF: %s", natural_sorted(s["session"] for s in no_bdf))
        if skipped is not None:
            skipped.extend(
                {"subject": s["subject"], "trial": s["session"], "reason": "rated session without a BDF file"}
                for s in no_bdf
            )
    if not sessions:
        logger.warning("No rated MAHNOB-HCI sessions with a BDF file under %s", sessions_dir)
    sessions.sort(key=_session_order)
    return sessions


def stimulus_window(
    status: np.ndarray | None,
    sfreq: float,
    n_times: int,
    *,
    pad_sec: float = 30.0,
    tol_sec: float = 2.0,
) -> tuple[int, int, str]:
    """Stimulus samples ``[start, stop)`` and the method that found them.

    ``"status"``: the first two 0->1 transitions of ``status & 0xFFFF & 16``, used only
    if the onset lies within ``tol_sec`` of ``pad_sec`` and the offset within
    ``2 * tol_sec`` of ``pad_sec`` before the end. Otherwise (no ``status``, missing or
    implausible pulses) ``"fixed"``: ``pad_sec`` is cropped from both ends.
    """
    if status is not None:
        status = np.asarray(status)
        if status.shape != (n_times,):
            raise ValueError(f"status must have shape ({n_times},), got {status.shape}")
        high = (np.rint(status).astype(np.int64) & TRIGGER_MASK & STIMULUS_BIT) != 0
        edges = np.flatnonzero(high[1:] & ~high[:-1]) + 1
        if len(edges) >= 2:
            start, stop = int(edges[0]), int(edges[1])
            onset_ok = abs(start / sfreq - pad_sec) <= tol_sec
            offset_ok = abs((n_times - stop) / sfreq - pad_sec) <= 2 * tol_sec
            if onset_ok and offset_ok:
                return start, stop, "status"
    pad = int(round(pad_sec * sfreq))
    start, stop = pad, n_times - pad
    if stop <= start:
        raise ValueError(
            f"{n_times} samples at {sfreq:g} Hz leave no stimulus after removing {pad_sec:g} s at each end"
        )
    return start, stop, "fixed"


def _read_bdf(path: Path) -> mne.io.BaseRaw:
    import mne

    return mne.io.read_raw_bdf(path, preload=True, verbose="error")


def load_session(
    session: Mapping[str, Any],
    *,
    channels: Sequence[str] = BIOSEMI32_CHANNELS,
    pad_sec: float = 30.0,
    reader: Callable[[Path], mne.io.BaseRaw] | None = None,
) -> Trial:
    """Load one session from :func:`find_rated_sessions` as a stimulus-locked trial in volts.

    ``channels`` are looked up case-insensitively and returned in the given order and
    spelling. ``reader(path)`` returns an MNE raw (default: ``mne.io.read_raw_bdf``).
    """
    bdf = Path(session["bdf"])
    raw = (reader or _read_bdf)(bdf)
    lookup = {name.lower(): name for name in raw.ch_names}
    missing = [c for c in channels if c.lower() not in lookup]
    if missing:
        raise KeyError(f"{bdf}: channels not found: {missing}")
    sfreq = float(raw.info["sfreq"])
    status_name = lookup.get(STATUS_CHANNEL.lower())
    status = raw.get_data(picks=[status_name])[0] if status_name else None
    start, stop, method = stimulus_window(status, sfreq, raw.n_times, pad_sec=pad_sec)
    if method == "fixed":
        logger.warning(
            "MAHNOB-HCI session %s: no plausible stimulus pulses in Status; cropping %g s at each end",
            session["session"],
            pad_sec,
        )
    data = raw.get_data(picks=[lookup[c.lower()] for c in channels], start=start, stop=stop)
    return Trial(
        dataset=DATASET,
        subject=session["subject"],
        trial=session["session"],
        data=data,
        sfreq=sfreq,
        ch_names=list(channels),
        valence=session["valence"],
        arousal=session["arousal"],
        meta={
            "bdf": str(bdf),
            "media_file": session.get("media_file"),
            "crop": method,
            "onset_sec": start / sfreq,
            "offset_sec": stop / sfreq,
            "sfreq_file": sfreq,
        },
    )


def iter_mahnob_trials(
    root: str | Path,
    *,
    channels: Sequence[str] = BIOSEMI32_CHANNELS,
    subjects: Iterable[str | int] | None = None,
    pad_sec: float = 30.0,
    reader: Callable[[Path], mne.io.BaseRaw] | None = None,
    on_error: str = "warn",
    skipped: list[dict[str, str]] | None = None,
) -> Iterator[Trial]:
    """Yield every rated MAHNOB-HCI emotion trial under ``root`` (paper Sec. 3).

    ``subjects`` keeps only those subject ids. ``on_error="warn"`` logs and skips a
    session that fails to load; ``"raise"`` re-raises the error. Rated sessions of the
    selected subjects that are dropped, those without a BDF and those that fail to load, are
    appended to ``skipped`` (if given) as ``{"subject", "trial" (the session id), "reason"}``,
    for ``extract_features(..., loader_skipped=skipped)``; the list is complete once the
    iterator is exhausted.
    """
    if on_error not in ("warn", "raise"):
        raise ValueError(f"on_error must be 'warn' or 'raise', got {on_error!r}")
    no_bdf: list[dict[str, str]] = []
    sessions = find_rated_sessions(root, skipped=no_bdf)
    if subjects is not None:
        wanted = {subjects} if isinstance(subjects, str) else {str(s) for s in subjects}  # "12" is one subject
        absent = wanted - {s["subject"] for s in sessions}
        if absent:
            logger.warning("No rated MAHNOB-HCI sessions for subjects %s", ", ".join(natural_sorted(absent)))
        sessions = [s for s in sessions if s["subject"] in wanted]
        no_bdf = [s for s in no_bdf if s["subject"] in wanted]
    if skipped is not None:
        skipped.extend(no_bdf)
    n_loaded = 0
    for session in sessions:
        try:
            trial = load_session(session, channels=channels, pad_sec=pad_sec, reader=reader)
        except Exception as err:
            if on_error == "raise":
                raise
            logger.warning(
                "Skipping MAHNOB-HCI session %s (subject %s): %s", session["session"], session["subject"], err
            )
            if skipped is not None:
                reason = f"{type(err).__name__}: {err}"
                skipped.append({"subject": session["subject"], "trial": session["session"], "reason": reason})
            continue
        n_loaded += 1
        yield trial
    logger.info("Loaded %d of %d rated MAHNOB-HCI sessions", n_loaded, len(sessions))
