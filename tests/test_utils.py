"""Natural ordering of subject ids and dataset-qualified ids (sieeg.utils)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sieeg.utils import check_channel_names, check_dataset_name, natural_key, natural_sorted, qualify

SRC = str(Path(__file__).resolve().parents[1] / "src")
#: Ids that tie under natural_key ("1" / "01", "P1" / "p1"): the raw text must break the tie.
TIES = ["10", "1", "01", "2", "P1", "p1", "S1", "s01", "02"]
TIES_SORTED = ["01", "1", "02", "2", "10", "P1", "p1", "S1", "s01"]


def test_natural_key_orders_numbers_numerically_and_text_case_insensitively():
    assert sorted(["S10", "S2", "S1"], key=natural_key) == ["S1", "S2", "S10"]
    assert natural_key("P07b") == natural_key("p7B")
    assert natural_key("2") < natural_key("10") < natural_key("a")  # numbers before text


def test_natural_sorting_treats_only_decimal_digits_as_numbers():
    """"²" is a digit to str.isdigit() but not to the regex split or to int(): it must sort as text."""
    assert natural_key("²") == ((1, "²"),)
    assert natural_key("S²2") == ((1, "s²"), (0, 2), (1, ""))
    assert natural_sorted(["2", "²", "S1"]) == ["2", "S1", "²"]
    assert natural_sorted(["①", "10", "9"]) == ["9", "10", "①"]
    assert natural_key("S٣") == natural_key("S3")  # decimal digits of any script still count


def test_natural_sorted_deduplicates_and_returns_strings():
    assert natural_sorted([10, "2", "S2", "S10", "2", 1]) == ["1", "2", "10", "S2", "S10"]
    assert natural_sorted([]) == []


def test_natural_sorted_breaks_ties_by_the_raw_text():
    assert natural_sorted(TIES) == TIES_SORTED
    assert natural_sorted(reversed(TIES)) == TIES_SORTED


def test_natural_sorted_is_independent_of_the_string_hash_seed():
    """Set iteration order follows PYTHONHASHSEED; ids equal under natural_key must not inherit it."""
    code = (
        "import numpy as np; from sieeg.utils import natural_sorted; from sieeg.features.featureset import FeatureSet; "
        f"ids = {TIES!r}; "
        "fs = FeatureSet('x', np.zeros((len(ids), 1, 1)), ids, ids, [5] * len(ids), [5] * len(ids), ['Cz'], "
        "[('alpha', 8, 12)]); "
        "print(natural_sorted(ids), fs.subject_ids)"
    )
    outputs = set()
    for seed in ("0", "1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": SRC}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        outputs.add(result.stdout.strip())
    assert outputs == {f"{TIES_SORTED} {TIES_SORTED}"}


def test_qualify():
    assert qualify("mahnob", "12") == "mahnob:12"
    assert qualify("daief", "12") != qualify("mahnob", "12")


def test_qualified_ids_cannot_collide_because_dataset_names_have_no_colon():
    """("a", "b:c") and ("a:b", "c") would both give "a:b:c": only subject ids may contain ':'."""
    assert qualify("a", "b:c") == "a:b:c"
    assert qualify("a", "b:c").split(":", 1) == ["a", "b:c"]  # the first colon ends the dataset name
    with pytest.raises(ValueError, match=r"Dataset name 'a:b' contains ':'.*use e\.g\. 'a-b'"):
        qualify("a:b", "c")
    assert check_dataset_name("dai-ef") == "dai-ef"
    assert check_dataset_name(2021) == "2021"  # Trial and FeatureSet store the name as a string
    with pytest.raises(ValueError, match=r"Dataset name ':' contains ':'"):
        check_dataset_name(":")


def test_check_channel_names():
    assert check_channel_names(("Fp1", "Cz", 7)) == ["Fp1", "Cz", "7"]
    assert check_channel_names([]) == []
    with pytest.raises(ValueError, match=r"^Channel names must be unique ignoring case, got \['Cz', 'CZ'\]$"):
        check_channel_names(["Cz", "Fp1", "CZ"])
    with pytest.raises(
        ValueError,
        match=r"^x\.bdf: channel names must be unique ignoring case, got \['O1', 'O1', 'o1'\], \['Cz', 'cz'\]$",
    ):
        check_channel_names(["O1", "Cz", "O1", "cz", "o1", "Fz"], "x.bdf: channel names")
    # str.lower(), the folding every channel lookup uses; casefold() would merge these two
    assert check_channel_names(["STRASSE", "straße"]) == ["STRASSE", "straße"]
