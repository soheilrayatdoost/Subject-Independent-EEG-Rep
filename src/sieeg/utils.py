"""Small helpers shared across the package."""

from __future__ import annotations

import re
from collections.abc import Iterable

_NUM_RE = re.compile(r"(\d+)")


def natural_key(text: str) -> tuple:
    """Sort key that orders embedded integers numerically ("S2" < "S10")."""
    parts = _NUM_RE.split(str(text))
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower()) for p in parts)


def natural_sorted(items: Iterable[str]) -> list[str]:
    """Return ``items`` as strings, de-duplicated and naturally sorted.

    Ids that compare equal under :func:`natural_key` (``"1"`` and ``"01"``) are ordered by
    their raw text, so the order never depends on Python's string-hash randomisation.
    """
    return sorted({str(i) for i in items}, key=lambda s: (natural_key(s), s))


def qualify(dataset: str, subject: str) -> str:
    """Globally unique subject id across datasets, e.g. ``"mahnob:12"``."""
    return f"{dataset}:{subject}"
