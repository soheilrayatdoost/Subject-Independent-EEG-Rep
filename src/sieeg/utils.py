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


def check_dataset_name(dataset: str) -> str:
    """``dataset`` as a string, if it can name a database: it must not contain ``":"``.

    ``":"`` ends the dataset name in the ids of :func:`qualify`. Subject ids may contain it, so
    the ids are unambiguous only if dataset names never do: ``qualify("a:b", "c")`` would equal
    ``qualify("a", "b:c")``.

    Raises:
        ValueError: naming the dataset.
    """
    name = str(dataset)
    if ":" in name:
        raise ValueError(
            f"Dataset name {name!r} contains ':', which separates dataset and subject in qualified subject ids; "
            f"use e.g. {name.replace(':', '-')!r}"
        )
    return name


def qualify(dataset: str, subject: str) -> str:
    """Globally unique subject id across datasets, e.g. ``"mahnob:12"``.

    Dataset names contain no ``":"`` (:func:`check_dataset_name`) while subject ids may, so the id
    splits back into ``(dataset, subject)`` at its first colon and distinct pairs never collide.

    Raises:
        ValueError: if ``dataset`` contains ``":"``.
    """
    return f"{check_dataset_name(dataset)}:{subject}"
