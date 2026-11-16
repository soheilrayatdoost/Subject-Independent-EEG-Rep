"""MAHNOB-HCI loader on a fake ``Sessions/`` tree (no real data needed).

The injected reader returns an ``mne.io.RawArray`` shaped like a MAHNOB BDF: the 32
EEG channels (here reversed and in mixed case, to exercise the by-name lookup), extra
channels that MNE also types as EEG, and a ``Status`` channel with BioSemi system bits
plus 16-pulses at stimulus onset and offset.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import mne
import numpy as np
import pytest

from sieeg.config import BIOSEMI32_CHANNELS
from sieeg.data.mahnob import (
    find_rated_sessions,
    iter_mahnob_trials,
    load_session,
    parse_session_xml,
    stimulus_window,
)

SFREQ = 128.0
PAD = int(30 * SFREQ)  # 30 s of padding before and after the stimulus
N_TIMES = 2 * PAD + int(10 * SFREQ)  # 10 s stimulus
FILE_EEG = [c.upper() if i % 2 else c.lower() for i, c in enumerate(BIOSEMI32_CHANNELS)][::-1]
LOADABLE = [("1", "2"), ("1", "10"), ("2", "22"), ("10", "130")]  # (subject, session), natural order

SESSION_XML = """<?xml version="1.0" encoding="utf-8"?>
<session {attrs}>
  <subject id="{subject}" />
  <track type="Physiological" id="1" />
</session>
"""


def write_session(sessions: Path, session: str, *, subject: str, exp_type: str = "1", ratings=None, bdf=None) -> Path:
    folder = sessions / session
    folder.mkdir(parents=True)
    attrs = {"cutNr": "3", "cutLenSec": "10.0", "expType": exp_type, "isStim": "1", "mediaFile": "52.avi"}
    if ratings is not None:
        attrs.update(feltEmo="4", feltArsl=str(ratings[1]), feltVlnc=str(ratings[0]), feltCtrl="5", feltPred="6")
    text = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    (folder / "session.xml").write_text(SESSION_XML.format(attrs=text, subject=subject), encoding="utf-8")
    if bdf is not None:
        (folder / bdf).write_bytes(b"")
    return folder


@pytest.fixture
def mahnob_root(tmp_path: Path) -> Path:
    """Four loadable rated sessions, plus a neutral, a tagging and a BDF-less rated session."""
    sessions = tmp_path / "mahnob" / "Sessions"
    write_session(sessions, "10", subject="1", ratings=(2, 8), bdf="Part_1_S_Trial5_emotion.bdf")
    write_session(sessions, "2", subject="1", ratings=(7, 3), bdf="Part_1_S_Trial1_emotion.bdf")
    write_session(sessions, "4", subject="1", bdf="Part_1_N_Trial2_emotion.bdf")  # neutral clip, unrated
    write_session(sessions, "1000", subject="1", exp_type="2", bdf="Part_1_Trial1_tagging.bdf")
    write_session(sessions, "130", subject="10", ratings=(5, 5), bdf="Part_10_S_Trial1_emotion.bdf")
    write_session(sessions, "22", subject="2", ratings=(9, 1), bdf="Part_2_S_Trial1_emotion.bdf")
    write_session(sessions, "24", subject="2", ratings=(3, 6))  # physiology missing: no BDF
    return tmp_path / "mahnob"


def eeg_signal(n_times: int = N_TIMES) -> np.ndarray:
    """Volts; row k (channel BIOSEMI32_CHANNELS[k]) is a k-specific offset plus a ramp."""
    k = np.arange(len(BIOSEMI32_CHANNELS))[:, None]
    return (k + 1) * 1e-6 + 1e-9 * np.arange(n_times)[None, :]


def status_signal(onsets, n_times: int = N_TIMES, *, base: int = 0x1E0000, width: int = 5) -> np.ndarray:
    """BioSemi system bits everywhere, plus 16-pulses starting at ``onsets``."""
    status = np.full(n_times, float(base))
    for t in onsets:
        status[t : t + width] += 16
    return status


def make_raw(onsets=(PAD, N_TIMES - PAD)) -> mne.io.RawArray:
    canonical = [c.lower() for c in BIOSEMI32_CHANNELS]
    eeg = eeg_signal()[[canonical.index(c.lower()) for c in FILE_EEG]]
    extra = np.vstack([np.full(N_TIMES, 1.0), np.full(N_TIMES, 33.0), status_signal(onsets)])
    names = FILE_EEG + ["GSR1", "Temp", "Status"]
    info = mne.create_info(names, SFREQ, ["eeg"] * (len(names) - 1) + ["stim"])  # MNE types GSR/Temp as EEG
    return mne.io.RawArray(np.vstack([eeg, extra]), info, verbose="error")


class FakeReader:
    """Stands in for ``mne.io.read_raw_bdf``; records the files it reads."""

    def __init__(self, fail: set[str] = frozenset()):
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, path: Path) -> mne.io.RawArray:
        self.calls.append(Path(path).name)
        if Path(path).name in self.fail:
            raise OSError(f"corrupt file {Path(path).name}")
        return make_raw()


# ------------------------------------------------------------------ session.xml


def test_parse_session_xml_rated(mahnob_root: Path) -> None:
    xml = mahnob_root / "Sessions" / "2" / "session.xml"
    assert parse_session_xml(xml) == {
        "session": "2",
        "subject": "1",
        "valence": 7.0,
        "arousal": 3.0,
        "media_file": "52.avi",
        "exp_type": "1",
        "cut_nr": "3",
        "xml": xml,
    }


@pytest.mark.parametrize("session", ["4", "1000"], ids=["neutral", "tagging"])
def test_parse_session_xml_unrated(mahnob_root: Path, session: str) -> None:
    assert parse_session_xml(mahnob_root / "Sessions" / session / "session.xml") is None


def test_parse_session_xml_rated_without_subject(tmp_path: Path) -> None:
    xml = tmp_path / "7" / "session.xml"
    xml.parent.mkdir()
    xml.write_text('<session expType="1" feltVlnc="5" feltArsl="6"></session>', encoding="utf-8")
    with pytest.raises(ValueError, match="subject"):
        parse_session_xml(xml)


# ------------------------------------------------------------------ discovery


@pytest.mark.parametrize("sub", ["", "Sessions"], ids=["database-root", "sessions-folder"])
def test_find_rated_sessions(mahnob_root: Path, sub: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sieeg.data.mahnob"):
        sessions = find_rated_sessions(mahnob_root / sub)
    assert [(s["subject"], s["session"]) for s in sessions] == LOADABLE
    assert sessions[0]["bdf"] == mahnob_root / "Sessions" / "2" / "Part_1_S_Trial1_emotion.bdf"
    assert sessions[0]["valence"] == 7.0 and sessions[0]["arousal"] == 3.0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and warnings[0].startswith("Skipped 1 rated")


def test_find_rated_sessions_bdf_choice(tmp_path: Path) -> None:
    folder = write_session(tmp_path / "Sessions", "2", subject="1", ratings=(7, 3), bdf="Part_1_S_Trial1_emotion.bdf")
    (folder / "notes.bdf").write_bytes(b"")
    assert find_rated_sessions(tmp_path)[0]["bdf"].name == "Part_1_S_Trial1_emotion.bdf"
    (folder / "Part_1_S_Trial1_emotion.bdf").unlink()
    assert find_rated_sessions(tmp_path)[0]["bdf"].name == "notes.bdf"  # the single BDF
    (folder / "other.bdf").write_bytes(b"")
    with pytest.raises(ValueError, match="cannot choose"):
        find_rated_sessions(tmp_path)
    with pytest.raises(FileNotFoundError):
        find_rated_sessions(tmp_path / "missing")


# ------------------------------------------------------------------ stimulus window


def test_stimulus_window_status_pulses() -> None:
    assert stimulus_window(status_signal([PAD, N_TIMES - PAD]), SFREQ, N_TIMES) == (PAD, N_TIMES - PAD, "status")


def test_stimulus_window_masks_system_bits() -> None:
    # Negative 24-bit values (bit 23 set), a trigger already high at sample 0 (no 0->1
    # transition there) and an unrelated trigger code must all be ignored.
    status = status_signal([PAD + 64, N_TIMES - PAD - 128], base=-(1 << 23) + 0x10000)
    status[:3] += 16
    status[1000:1010] += 32
    assert stimulus_window(status, SFREQ, N_TIMES) == (PAD + 64, N_TIMES - PAD - 128, "status")


@pytest.mark.parametrize(
    ("onset", "tail", "method"),
    [
        (PAD + 256, PAD, "status"),  # onset 2 s late: at the tolerance
        (PAD + 257, PAD, "fixed"),
        (PAD - 256, PAD, "status"),  # onset 2 s early
        (PAD, PAD + 512, "status"),  # offset 4 s before the expected one: at 2 * tolerance
        (PAD, PAD + 513, "fixed"),
        (PAD, PAD - 513, "fixed"),
    ],
)
def test_stimulus_window_tolerances(onset: int, tail: int, method: str) -> None:
    start, stop, found = stimulus_window(status_signal([onset, N_TIMES - tail]), SFREQ, N_TIMES)
    assert found == method
    assert (start, stop) == ((onset, N_TIMES - tail) if method == "status" else (PAD, N_TIMES - PAD))


@pytest.mark.parametrize("onsets", [[], [PAD], [5 * 128, 10 * 128]], ids=["none", "one", "implausible"])
def test_stimulus_window_fixed_fallback(onsets: list[int]) -> None:
    assert stimulus_window(status_signal(onsets), SFREQ, N_TIMES) == (PAD, N_TIMES - PAD, "fixed")


def test_stimulus_window_edge_cases() -> None:
    assert stimulus_window(None, SFREQ, N_TIMES) == (PAD, N_TIMES - PAD, "fixed")
    assert stimulus_window(None, SFREQ, N_TIMES, pad_sec=20.0) == (2560, N_TIMES - 2560, "fixed")
    with pytest.raises(ValueError, match="no stimulus"):
        stimulus_window(None, SFREQ, 2 * PAD)
    with pytest.raises(ValueError, match="shape"):
        stimulus_window(np.zeros(10), SFREQ, N_TIMES)


# ------------------------------------------------------------------ loading


@pytest.mark.parametrize(
    ("onset", "stop"), [(PAD, N_TIMES - PAD), (PAD + 64, N_TIMES - PAD - 128)], ids=["at-30s", "shifted"]
)
def test_load_session_status_crop(mahnob_root: Path, onset: int, stop: int) -> None:
    session = find_rated_sessions(mahnob_root)[0]
    trial = load_session(session, reader=lambda path: make_raw([onset, stop]))
    assert (trial.dataset, trial.subject, trial.trial) == ("mahnob", "1", "2")
    assert (trial.valence, trial.arousal, trial.sfreq) == (7.0, 3.0, SFREQ)
    assert trial.ch_names == list(BIOSEMI32_CHANNELS)
    np.testing.assert_array_equal(trial.data, eeg_signal()[:, onset:stop])  # volts, file order undone
    assert trial.meta == {
        "bdf": str(session["bdf"]),
        "media_file": "52.avi",
        "crop": "status",
        "onset_sec": onset / SFREQ,
        "offset_sec": stop / SFREQ,
        "sfreq_file": SFREQ,
    }


@pytest.mark.parametrize("onsets", [[], [5 * 128, 10 * 128]], ids=["no-pulses", "implausible"])
def test_load_session_fixed_fallback(mahnob_root: Path, onsets: list[int], caplog: pytest.LogCaptureFixture) -> None:
    session = find_rated_sessions(mahnob_root)[0]
    with caplog.at_level(logging.WARNING, logger="sieeg.data.mahnob"):
        trial = load_session(session, reader=lambda path: make_raw(onsets))
    assert trial.meta["crop"] == "fixed" and "no plausible stimulus pulses" in caplog.text
    assert (trial.meta["onset_sec"], trial.meta["offset_sec"]) == (30.0, 40.0)
    np.testing.assert_array_equal(trial.data, eeg_signal()[:, PAD : N_TIMES - PAD])
    shorter_pad = load_session(session, pad_sec=25.0, reader=lambda path: make_raw(onsets))
    np.testing.assert_array_equal(shorter_pad.data, eeg_signal()[:, 3200 : N_TIMES - 3200])


def test_load_session_channel_lookup(mahnob_root: Path) -> None:
    session = find_rated_sessions(mahnob_root)[0]
    trial = load_session(session, channels=("Cz", "fp1"), reader=lambda path: make_raw())
    assert trial.ch_names == ["Cz", "fp1"]
    rows = [BIOSEMI32_CHANNELS.index("Cz"), BIOSEMI32_CHANNELS.index("Fp1")]
    np.testing.assert_array_equal(trial.data, eeg_signal()[rows, PAD : N_TIMES - PAD])
    with pytest.raises(KeyError, match="Iz"):
        load_session(session, channels=("Cz", "Iz"), reader=lambda path: make_raw())


def test_load_session_default_reader(mahnob_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def read_raw_bdf(path, **kwargs):
        calls.append((Path(path), kwargs))
        return make_raw()

    monkeypatch.setattr(mne.io, "read_raw_bdf", read_raw_bdf)
    session = find_rated_sessions(mahnob_root)[0]
    load_session(session)
    assert calls == [(session["bdf"], {"preload": True, "verbose": "error"})]


# ------------------------------------------------------------------ iteration


def test_iter_mahnob_trials(mahnob_root: Path) -> None:
    trials = list(iter_mahnob_trials(mahnob_root, reader=FakeReader()))
    assert [(t.subject, t.trial) for t in trials] == LOADABLE
    assert [(t.valence, t.arousal) for t in trials] == [(7.0, 3.0), (2.0, 8.0), (9.0, 1.0), (5.0, 5.0)]
    assert all(t.data.shape == (32, N_TIMES - 2 * PAD) and t.meta["crop"] == "status" for t in trials)


def test_iter_mahnob_trials_subjects(mahnob_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    reader = FakeReader()
    with caplog.at_level(logging.WARNING, logger="sieeg.data.mahnob"):
        trials = list(iter_mahnob_trials(mahnob_root, subjects=[10, "2", "99"], reader=reader))
    assert [(t.subject, t.trial) for t in trials] == [("2", "22"), ("10", "130")]
    assert reader.calls == ["Part_2_S_Trial1_emotion.bdf", "Part_10_S_Trial1_emotion.bdf"]
    assert "No rated MAHNOB-HCI sessions for subjects 99" in caplog.text
    only = list(iter_mahnob_trials(mahnob_root, subjects="10", reader=reader))  # one id, not "1" and "0"
    assert [t.subject for t in only] == ["10"]


def test_iter_mahnob_trials_on_error(mahnob_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    reader = FakeReader(fail={"Part_1_S_Trial5_emotion.bdf"})
    with caplog.at_level(logging.WARNING, logger="sieeg.data.mahnob"):
        trials = list(iter_mahnob_trials(mahnob_root, reader=reader))
    assert [t.trial for t in trials] == ["2", "22", "130"]
    assert "Skipping MAHNOB-HCI session 10 (subject 1): corrupt file" in caplog.text
    with pytest.raises(OSError, match="corrupt file"):
        list(iter_mahnob_trials(mahnob_root, reader=reader, on_error="raise"))
    with pytest.raises(ValueError, match="on_error"):
        list(iter_mahnob_trials(mahnob_root, reader=reader, on_error="ignore"))


def test_find_rated_sessions_reports_sessions_without_bdf(mahnob_root: Path) -> None:
    skipped: list[dict] = []
    find_rated_sessions(mahnob_root, skipped=skipped)
    assert skipped == [{"subject": "2", "trial": "24", "reason": "rated session without a BDF file"}]


def test_iter_mahnob_trials_reports_skipped_sessions(mahnob_root: Path) -> None:
    skipped: list[dict] = []
    trials = iter_mahnob_trials(mahnob_root, reader=FakeReader(fail={"Part_1_S_Trial5_emotion.bdf"}), skipped=skipped)
    assert skipped == []  # filled lazily, while the sessions are read
    assert [t.trial for t in trials] == ["2", "22", "130"]
    assert skipped == [
        {"subject": "2", "trial": "24", "reason": "rated session without a BDF file"},
        {"subject": "1", "trial": "10", "reason": "OSError: corrupt file Part_1_S_Trial5_emotion.bdf"},
    ]
    only = []
    assert len(list(iter_mahnob_trials(mahnob_root, subjects=["10"], reader=FakeReader(), skipped=only))) == 1
    assert only == []  # subject 2's session without a BDF was not asked for


def test_unreadable_session_metadata_follows_on_error(mahnob_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    """One bad session.xml must not abort the extraction of every other session with on_error="warn"."""
    sessions = mahnob_root / "Sessions"
    (sessions / "30").mkdir()
    (sessions / "30" / "session.xml").write_text('<session feltVlnc="7" feltArsl="3"><subject id="3"/>', "utf-8")
    write_session(sessions, "40", subject=" ", ratings=(6, 4), bdf="Part_4_S_Trial1_emotion.bdf")  # no subject id
    write_session(sessions, "50", subject="5", ratings=(6, 4), bdf="Part_5_S_Trial1_emotion.bdf")
    (sessions / "50" / "Part_5_S_Trial2_emotion.bdf").write_bytes(b"")  # two candidate BDF files

    with pytest.raises((ET.ParseError, ValueError)):
        list(iter_mahnob_trials(mahnob_root, reader=FakeReader(), on_error="raise"))
    with pytest.raises((ET.ParseError, ValueError)):
        find_rated_sessions(mahnob_root)  # the default is on_error="raise", as before

    skipped: list[dict] = []
    with caplog.at_level(logging.WARNING, logger="sieeg.data.mahnob"):
        trials = list(iter_mahnob_trials(mahnob_root, reader=FakeReader(), skipped=skipped))
    assert [t.trial for t in trials] == [session for _, session in LOADABLE]
    assert [(s["subject"], s["trial"]) for s in skipped] == [("unknown", "30"), ("unknown", "40"), ("5", "50"), ("2", "24")]
    assert skipped[0]["reason"].startswith("ParseError: ")
    assert skipped[1]["reason"].startswith("ValueError: ") and "without <subject id" in skipped[1]["reason"]
    assert "cannot choose between BDF files" in skipped[2]["reason"]
    assert "Skipping MAHNOB-HCI session 30: cannot read its metadata" in caplog.text

    only: list[dict] = []
    assert [t.trial for t in iter_mahnob_trials(mahnob_root, subjects=["10"], reader=FakeReader(), skipped=only)] == ["130"]
    assert [s["trial"] for s in only] == ["30", "40"]  # their subject is unknown, so they are always listed
    with pytest.raises(FileNotFoundError):  # a missing root is not a per-session error
        list(iter_mahnob_trials(mahnob_root / "missing", on_error="warn"))


def test_channel_names_colliding_ignoring_case_fail_the_session(mahnob_root: Path) -> None:
    """The case-insensitive lookup would keep the later of "CZ" and "cz", silently loading the wrong row."""

    def reader(path: Path) -> mne.io.RawArray:
        raw = make_raw()
        if path.name == "Part_1_S_Trial5_emotion.bdf":
            raw.rename_channels({"GSR1": "cz"})  # the file's EEG channel is spelled "CZ"
        return raw

    session = next(s for s in find_rated_sessions(mahnob_root) if s["session"] == "10")
    message = r"Part_1_S_Trial5_emotion\.bdf: channel names must be unique ignoring case, got \['CZ', 'cz'\]"
    with pytest.raises(ValueError, match=message):
        load_session(session, reader=reader)
    skipped: list[dict] = []
    trials = list(iter_mahnob_trials(mahnob_root, reader=reader, skipped=skipped))
    assert [t.trial for t in trials] == ["2", "22", "130"]
    assert (skipped[-1]["subject"], skipped[-1]["trial"]) == ("1", "10")
    assert skipped[-1]["reason"].startswith("ValueError: ") and "['CZ', 'cz']" in skipped[-1]["reason"]
    with pytest.raises(ValueError, match=message):
        list(iter_mahnob_trials(mahnob_root, reader=reader, on_error="raise"))


def test_colliding_channels_are_rejected_before_any_session_is_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"channels must be unique ignoring case, got \['Cz', 'CZ'\]"):
        list(iter_mahnob_trials(tmp_path / "missing", channels=["Cz", "CZ"]))
