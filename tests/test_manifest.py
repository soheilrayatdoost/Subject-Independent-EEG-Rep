"""CSV-manifest loader: array files (.npy / .npz / .mat) and MNE-readable files, cropping,
metadata and scale precedence, path resolution and the one-file cache."""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import scipy.io

from sieeg.data.manifest import REQUIRED_COLUMNS, iter_manifest_trials, load_array_file, read_manifest

SFREQ = 100.0
NAMES = ["Fp1", "AF3", "F3"]


def signal(n_ch: int = 3, n_times: int = 500) -> np.ndarray:
    """Distinct values at every (channel, sample)."""
    return np.arange(n_ch * n_times, dtype=float).reshape(n_ch, n_times) * 1e-6


def row(path: str, trial: str = "1", **optional) -> dict:
    return {"subject": "S01", "trial": trial, "path": path, "valence": 7, "arousal": 3, **optional}


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_fif(path: Path, data: np.ndarray) -> None:
    """EEG plus a stim and a misc channel, as an MNE-readable file."""
    info = mne.create_info(NAMES + ["STI 014", "GSR"], SFREQ, ["eeg"] * len(NAMES) + ["stim", "misc"])
    extra = np.vstack([np.zeros(data.shape[1]), np.ones(data.shape[1])])
    raw = mne.io.RawArray(np.vstack([data, extra]), info, verbose="error")
    raw.save(path, fmt="double", overwrite=True, verbose="error")


# ------------------------------------------------------------------ read_manifest


def test_read_manifest_paths_order_and_text_columns(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / "b.npy"
    csv = write_csv(
        tmp_path / "meta" / "manifest.csv",
        [row("../arrays/a.npy", trial="02") | {"subject": "10"}, row(str(elsewhere)) | {"subject": "02"}],
    )
    df = read_manifest(csv)
    assert df["subject"].tolist() == ["10", "02"] and df["trial"].tolist() == ["02", "1"]  # order, no int casts
    assert Path(df["path"][0]).resolve() == (tmp_path / "arrays" / "a.npy").resolve()
    assert Path(df["path"][1]) == elsewhere


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_read_manifest_missing_column(tmp_path: Path, column: str) -> None:
    record = row("a.npy")
    del record[column]
    with pytest.raises(ValueError) as excinfo:
        read_manifest(write_csv(tmp_path / "m.csv", [record]))
    assert f"column(s) ['{column}']" in str(excinfo.value)


def test_read_manifest_blank_subject(tmp_path: Path) -> None:
    csv = write_csv(tmp_path / "m.csv", [row("a.npy"), row("a.npy", trial="2") | {"subject": None}])
    with pytest.raises(ValueError, match=r"rows \[1\]"):
        read_manifest(csv)


# ------------------------------------------------------------------ array files


def test_load_array_file_selection(tmp_path: Path) -> None:
    data = signal()
    np.save(tmp_path / "x.npy", data)
    array, extras = load_array_file(tmp_path / "x.npy")
    np.testing.assert_array_equal(array, data)
    assert extras == {}
    np.savez(tmp_path / "keyed.npz", data=data, other=2 * data)
    np.testing.assert_array_equal(load_array_file(tmp_path / "keyed.npz")[0], data)  # "data" first
    np.testing.assert_array_equal(load_array_file(tmp_path / "keyed.npz", array_key="other")[0], 2 * data)
    np.savez(tmp_path / "only.npz", eeg=data, sfreq=np.array(SFREQ), labels=np.arange(3))
    array, extras = load_array_file(tmp_path / "only.npz")  # the only 2-D array
    np.testing.assert_array_equal(array, data)
    assert extras == {"sfreq": SFREQ}
    np.savez(tmp_path / "two.npz", a=data, b=data)
    with pytest.raises(ValueError, match="array_key"):
        load_array_file(tmp_path / "two.npz")
    with pytest.raises(KeyError, match="'c'"):
        load_array_file(tmp_path / "two.npz", array_key="c")
    np.save(tmp_path / "flat.npy", data[0])
    with pytest.raises(ValueError, match="2-D"):
        load_array_file(tmp_path / "flat.npy")


@pytest.mark.parametrize("flag", [True, 1, "yes"])
def test_npy_transpose(tmp_path: Path, flag: object) -> None:
    data = signal()
    np.save(tmp_path / "t.npy", data.T)  # stored as (n_times, n_channels)
    meta = {"sfreq": SFREQ, "ch_names": ";".join(NAMES)}
    csv = write_csv(tmp_path / "m.csv", [row("t.npy", transpose=flag, **meta)])
    (trial,) = iter_manifest_trials(csv, on_error="raise")
    np.testing.assert_array_equal(trial.data, data)
    assert (trial.dataset, trial.subject, trial.trial) == ("daief", "S01", "1")
    assert (trial.sfreq, trial.ch_names, trial.valence, trial.arousal) == (SFREQ, NAMES, 7.0, 3.0)
    unflagged = write_csv(tmp_path / "m2.csv", [row("t.npy", **meta)])
    with pytest.raises(ValueError, match="transpose"):
        list(iter_manifest_trials(unflagged, on_error="raise"))


def test_npz_stored_metadata_and_precedence(tmp_path: Path) -> None:
    data = signal()
    np.savez(tmp_path / "s.npz", data=data, sfreq=256.0, ch_names=np.array(NAMES))
    np.save(tmp_path / "bare.npy", data)
    rows = [row("s.npz"), row("s.npz", trial="2", sfreq=128.0, ch_names="Cz; Pz; Oz"), row("bare.npy", trial="3")]
    csv = write_csv(tmp_path / "m.csv", rows)
    trials = iter_manifest_trials(csv, sfreq=64.0, ch_names=["x", "y", "z"], on_error="raise")
    from_file, from_row, from_keyword = trials
    assert (from_file.sfreq, from_file.ch_names) == (256.0, NAMES)  # file beats keyword
    assert (from_row.sfreq, from_row.ch_names) == (128.0, ["Cz", "Pz", "Oz"])  # row beats file
    assert (from_keyword.sfreq, from_keyword.ch_names) == (64.0, ["x", "y", "z"])
    np.testing.assert_array_equal(from_file.data, data)
    bare = write_csv(tmp_path / "bare.csv", [row("bare.npy")])
    with pytest.raises(ValueError, match="sampling rate"):
        list(iter_manifest_trials(bare, ch_names=NAMES, on_error="raise"))
    with pytest.raises(ValueError, match="channel names"):
        list(iter_manifest_trials(bare, sfreq=SFREQ, on_error="raise"))


def test_mat_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = signal()
    # MATLAB stores every variable as >= 2-D: the scalar and the vector must not count as EEG candidates.
    variables = {"eeg": data, "sfreq": SFREQ, "ch_names": ["Fp1", "Oz", "AF3"], "subject": 7, "events": np.arange(4)}
    scipy.io.savemat(tmp_path / "s.mat", variables)
    array, extras = load_array_file(tmp_path / "s.mat")
    np.testing.assert_array_equal(array, data)
    assert extras == {"sfreq": SFREQ, "ch_names": ["Fp1", "Oz", "AF3"]}  # char-matrix padding stripped
    scipy.io.savemat(tmp_path / "cell.mat", {"data": data, "ch_names": np.array(NAMES, dtype=object)})
    assert load_array_file(tmp_path / "cell.mat")[1] == {"ch_names": NAMES}  # a cell array of names
    scipy.io.savemat(tmp_path / "two.mat", {"eeg": data, "eog": data[:2]})
    with pytest.raises(ValueError, match="array_key"):
        load_array_file(tmp_path / "two.mat")

    opened = []
    loadmat = scipy.io.loadmat
    monkeypatch.setattr(scipy.io, "loadmat", lambda path, **kw: opened.append(Path(path).name) or loadmat(path, **kw))
    meta = {"sfreq": SFREQ, "ch_names": "a;b"}
    rows = [
        row("s.mat"),
        row("two.mat", trial="2", array_key="eog", **meta),
        row("two.mat", trial="3", array_key="eeg"),
    ]
    csv = write_csv(tmp_path / "m.csv", rows)
    first, eog, eeg = iter_manifest_trials(csv, ch_names=NAMES, sfreq=SFREQ, on_error="raise")
    assert first.ch_names == ["Fp1", "Oz", "AF3"]
    np.testing.assert_array_equal(eog.data, data[:2])
    np.testing.assert_array_equal(eeg.data, data)  # another variable of the cached file
    assert opened == ["s.mat", "two.mat"]


# ------------------------------------------------------------------ MNE files


def test_mne_file_matches_array_file(tmp_path: Path) -> None:
    data = signal()
    write_fif(tmp_path / "rec_raw.fif", data)
    np.save(tmp_path / "rec.npy", data)
    rows = [
        row("rec_raw.fif", tmin=1.0, tmax=3.0),
        row("rec.npy", trial="2", tmin=1.0, tmax=3.0, sfreq=SFREQ, ch_names=";".join(NAMES)),
    ]
    from_mne, from_array = iter_manifest_trials(write_csv(tmp_path / "m.csv", rows), on_error="raise")
    assert from_mne.ch_names == NAMES and from_mne.sfreq == SFREQ  # stim and misc channels dropped
    np.testing.assert_array_equal(from_array.data, data[:, 100:300])
    np.testing.assert_allclose(from_mne.data, from_array.data, rtol=1e-12, atol=0)


def test_rows_sharing_a_file_read_it_once(tmp_path: Path) -> None:
    data = signal()
    calls = []

    def reader(path: Path) -> mne.io.RawArray:
        calls.append(path.name)
        info = mne.create_info(NAMES, SFREQ, "eeg")
        return mne.io.RawArray(data * (2.0 if path.name == "b.edf" else 1.0), info, verbose="error")

    rows = [
        row("a.edf", tmin=0.0, tmax=1.0),
        row("a.edf", trial="2", tmin=1.0, tmax=2.0),
        row("b.edf", trial="3"),
        row("a.edf", trial="4", tmin=4.0),
    ]
    trials = list(iter_manifest_trials(write_csv(tmp_path / "m.csv", rows), reader=reader, on_error="raise"))
    assert calls == ["a.edf", "b.edf", "a.edf"]  # only the most recent file is cached
    np.testing.assert_array_equal(trials[0].data, data[:, :100])
    np.testing.assert_array_equal(trials[1].data, data[:, 100:200])
    np.testing.assert_array_equal(trials[2].data, 2.0 * data)
    np.testing.assert_array_equal(trials[3].data, data[:, 400:])
    assert [t.meta["row"] for t in trials] == [0, 1, 2, 3]


# ------------------------------------------------------------------ cropping, scale, paths, errors


def test_tmin_tmax_crop(tmp_path: Path) -> None:
    data = signal()
    np.save(tmp_path / "a.npy", data)
    rows = [row("a.npy", tmin=1.0, tmax=2.5), row("a.npy", trial="2", tmin=4.0), row("a.npy", trial="3", tmax=0.5)]
    a, b, c = iter_manifest_trials(write_csv(tmp_path / "m.csv", rows), sfreq=SFREQ, ch_names=NAMES, on_error="raise")
    np.testing.assert_array_equal(a.data, data[:, 100:250])
    np.testing.assert_array_equal(b.data, data[:, 400:])
    np.testing.assert_array_equal(c.data, data[:, :50])
    assert (a.meta["onset_sec"], a.meta["offset_sec"]) == (1.0, 2.5)
    bad = [row("a.npy", tmin=2.0, tmax=6.0), row("a.npy", trial="2", tmin=2.0, tmax=2.0)]  # past the end; empty
    bad_csv = write_csv(tmp_path / "bad.csv", bad)
    assert list(iter_manifest_trials(bad_csv, sfreq=SFREQ, ch_names=NAMES)) == []
    with pytest.raises(ValueError, match="outside"):
        list(iter_manifest_trials(bad_csv, sfreq=SFREQ, ch_names=NAMES, on_error="raise"))


def test_scale_precedence(tmp_path: Path) -> None:
    data = signal()
    np.save(tmp_path / "a.npy", data)
    csv = write_csv(tmp_path / "m.csv", [row("a.npy", scale=1e-6), row("a.npy", trial="2")])
    keyword = list(iter_manifest_trials(csv, sfreq=SFREQ, ch_names=NAMES, scale=1e-3, on_error="raise"))
    default = list(iter_manifest_trials(csv, sfreq=SFREQ, ch_names=NAMES, on_error="raise"))
    assert [t.meta["scale"] for t in keyword] == [1e-6, 1e-3]  # row > keyword
    assert [t.meta["scale"] for t in default] == [1e-6, 1.0]  # keyword > 1.0
    np.testing.assert_allclose(keyword[0].data, data * 1e-6)
    np.testing.assert_allclose(keyword[1].data, data * 1e-3)  # the cached array was not scaled in place
    np.testing.assert_array_equal(default[1].data, data)


def test_relative_paths_and_dataframe_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = signal()
    (tmp_path / "arrays").mkdir()
    np.save(tmp_path / "arrays" / "a.npy", data)
    meta = {"sfreq": SFREQ, "ch_names": ";".join(NAMES)}
    csv = write_csv(tmp_path / "meta" / "m.csv", [row("../arrays/a.npy", **meta) | {"valence": None}])
    monkeypatch.chdir(tmp_path)  # the path must resolve against the manifest, not the working directory
    (trial,) = iter_manifest_trials(csv, on_error="raise")
    np.testing.assert_array_equal(trial.data, data)
    assert np.isnan(trial.valence) and trial.arousal == 3.0
    (same,) = iter_manifest_trials(read_manifest(csv), dataset="other", on_error="raise")
    assert same.dataset == "other"
    np.testing.assert_array_equal(same.data, data)
    with pytest.raises(ValueError, match="arousal"):
        list(iter_manifest_trials(pd.DataFrame([row("a.npy")]).drop(columns="arousal")))


def test_on_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    np.save(tmp_path / "a.npy", signal())
    csv = write_csv(tmp_path / "m.csv", [row("missing.npy"), row("a.npy", trial="2")])
    with caplog.at_level(logging.WARNING, logger="sieeg.data.manifest"):
        trials = list(iter_manifest_trials(csv, sfreq=SFREQ, ch_names=NAMES))
    assert [t.trial for t in trials] == ["2"]
    assert "Skipping manifest row 0 (subject S01, trial 1)" in caplog.text
    with pytest.raises(FileNotFoundError):
        list(iter_manifest_trials(csv, sfreq=SFREQ, ch_names=NAMES, on_error="raise"))
    with pytest.raises(ValueError, match="on_error"):
        list(iter_manifest_trials(csv, on_error="skip"))


# ------------------------------------------------------------------ column names, ids, ratings


@pytest.mark.parametrize(
    ("header", "expected"),
    [("Tmin", "tmin"), ("TMAX", "tmax"), ("onset", "tmin"), ("Start", "tmin"), ("offset", "tmax"), ("end", "tmax"),
     ("stop", "tmax"), ("fs", "sfreq"), ("srate", "sfreq"), ("Sampling Rate", "sfreq"), ("arrayKey", "array_key"),
     ("key", "array_key"), ("channels", "ch_names"), ("Scale", "scale"), ("Transpose", "transpose"),
     ("Subject", "subject")],
)
def test_misspelled_or_aliased_column_raises(tmp_path: Path, header: str, expected: str) -> None:
    record = row("a.npy") | {header: "1"}
    if expected == "subject":
        del record["subject"]
    with pytest.raises(ValueError, match=f"{header!r} \\(expected {expected!r}\\)"):
        read_manifest(write_csv(tmp_path / "m.csv", [record]))


def test_header_whitespace_is_stripped(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", signal())
    csv = tmp_path / "m.csv"
    csv.write_text("subject ,trial, path,valence,arousal,tmin ,tmax \n01,1,a.npy,7,3,1.0,2.5\n", encoding="utf-8")
    df = read_manifest(csv)
    assert list(df.columns) == [*REQUIRED_COLUMNS, "tmin", "tmax"]
    assert df["subject"].tolist() == ["01"]  # still read as text through a padded header
    (trial,) = iter_manifest_trials(csv, sfreq=SFREQ, ch_names=NAMES, on_error="raise")
    np.testing.assert_array_equal(trial.data, signal()[:, 100:250])  # the crop was not lost
    (tmp_path / "dup.csv").write_text("subject,trial,path,valence,arousal,tmin,tmin \nS01,1,a.npy,7,3,1,2\n")
    with pytest.raises(ValueError, match=r"duplicate column\(s\) \['tmin'\]"):
        read_manifest(tmp_path / "dup.csv")


def test_unknown_columns_are_kept_and_logged_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv = write_csv(tmp_path / "m.csv", [row("a.npy", notes="calm", video=3), row("a.npy", trial="2", notes="x")])
    with caplog.at_level(logging.WARNING, logger="sieeg.data.manifest"):
        df = read_manifest(csv)
    assert df["notes"].tolist() == ["calm", "x"]
    [record] = caplog.records
    assert record.getMessage() == f"Manifest {csv}: ignored columns: 'notes', 'video'"


def test_ids_and_paths_are_stripped(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", signal())
    csv = tmp_path / "m.csv"
    csv.write_text("subject,trial,path,valence,arousal\nS01 ,1 , a.npy ,7,3\nS01,2,a.npy,2,6\n", encoding="utf-8")
    df = read_manifest(csv)
    assert df["subject"].tolist() == ["S01", "S01"] and df["trial"].tolist() == ["1", "2"]
    assert Path(df["path"][0]) == (tmp_path / "a.npy").resolve()
    frame = pd.DataFrame([row(str(tmp_path / "a.npy")) | {"subject": " S01 "}])
    (trial,) = iter_manifest_trials(frame, sfreq=SFREQ, ch_names=NAMES, on_error="raise")
    assert trial.subject == "S01" and frame["subject"][0] == " S01 "  # the caller's frame is not modified
    blank = write_csv(tmp_path / "blank.csv", [row("a.npy") | {"subject": "  "}])
    with pytest.raises(ValueError, match=r"rows \[0\]"):
        read_manifest(blank)


def test_malformed_rating_raises_and_empty_means_unrated(tmp_path: Path) -> None:
    csv = tmp_path / "m.csv"
    csv.write_text("subject,trial,path,valence,arousal\nS01,1,a.npy,7,3\nS01,2,a.npy,,6\nS01,3,a.npy,7.5.,6\n")
    with pytest.raises(ValueError, match=r"non-numeric valence \['7.5.'\] in rows \[2\]"):
        read_manifest(csv)
    csv.write_text("subject,trial,path,valence,arousal\nS01,1,a.npy,7,3\nS01,2,a.npy, ,6\nS01,3,a.npy,n/a,-\n")
    with pytest.raises(ValueError, match=r"non-numeric arousal \['-'\] in rows \[2\]"):
        read_manifest(csv)
    csv.write_text("subject,trial,path,valence,arousal\nS01,1,a.npy,7,3\nS01,2,a.npy, ,6\nS01,3,a.npy,n/a,5.5\n")
    df = read_manifest(csv)
    np.testing.assert_array_equal(df["valence"], [7.0, np.nan, np.nan])
    np.testing.assert_array_equal(df["arousal"], [3.0, 6.0, 5.5])
    with pytest.raises(ValueError, match="non-numeric valence"):
        list(iter_manifest_trials(pd.DataFrame([row("a.npy") | {"valence": "seven"}])))


def test_rows_reading_the_same_samples_are_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    np.save(tmp_path / "a.npy", signal())
    np.savez(tmp_path / "b.npz", data=signal(), eog=signal(2))
    rows = [
        row("a.npy", tmin=0.0, tmax=2.0),
        row("a.npy", trial="2", tmin=2.0, tmax=4.0),
        row("b.npz", trial="3"),  # the default variable, data
        row("b.npz", trial="4", array_key="eog", ch_names="x;y"),  # another variable of the file
        row("a.npy", trial="5", tmin=0.0, tmax=2.0, scale=1e-6),  # = row 0 (scale does not change the samples)
        row("b.npz", trial="6", array_key="data"),  # = row 2
    ]
    csv = write_csv(tmp_path / "m.csv", rows)
    with caplog.at_level(logging.WARNING, logger="sieeg.data.manifest"):
        trials = list(iter_manifest_trials(csv, sfreq=SFREQ, ch_names=NAMES, on_error="raise"))
    assert len(trials) == 6
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert record.getMessage().startswith("2 manifest row(s) read the same samples")
    assert "row 4 = row 0, row 5 = row 2" in record.getMessage()


def test_skipped_rows_are_reported_to_the_caller(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", signal())
    rows = [row("a.npy", tmax=1.0), row("missing.npy", trial="2"), row("a.npy", trial="3", tmin=2.0, tmax=9.0)]
    skipped: list[dict] = []
    trials = iter_manifest_trials(write_csv(tmp_path / "m.csv", rows), sfreq=SFREQ, ch_names=NAMES, skipped=skipped)
    assert skipped == []  # filled lazily, while the rows are read
    assert [t.trial for t in trials] == ["1"]
    assert [(s["subject"], s["trial"]) for s in skipped] == [("S01", "2"), ("S01", "3")]
    assert skipped[0]["reason"].startswith("FileNotFoundError: ") and "missing.npy" in skipped[0]["reason"]
    assert skipped[1]["reason"].startswith("ValueError: ") and "outside the recording" in skipped[1]["reason"]
