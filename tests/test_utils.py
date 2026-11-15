"""Natural ordering of subject ids and dataset-qualified ids (sieeg.utils)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sieeg.utils import natural_key, natural_sorted, qualify

SRC = str(Path(__file__).resolve().parents[1] / "src")
#: Ids that tie under natural_key ("1" / "01", "P1" / "p1"): the raw text must break the tie.
TIES = ["10", "1", "01", "2", "P1", "p1", "S1", "s01", "02"]
TIES_SORTED = ["01", "1", "02", "2", "10", "P1", "p1", "S1", "s01"]


def test_natural_key_orders_numbers_numerically_and_text_case_insensitively():
    assert sorted(["S10", "S2", "S1"], key=natural_key) == ["S1", "S2", "S10"]
    assert natural_key("P07b") == natural_key("p7B")
    assert natural_key("2") < natural_key("10") < natural_key("a")  # numbers before text


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
