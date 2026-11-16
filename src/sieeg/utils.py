"""Small helpers shared across the package."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_NUM_RE = re.compile(r"(\d+)")


def natural_key(text: str) -> tuple:
    """Sort key that orders embedded integers numerically ("S2" < "S10").

    Only runs of decimal digits (what ``_NUM_RE`` splits out and ``int()`` reads) are numbers;
    other digit characters such as "²" sort as text.
    """
    parts = _NUM_RE.split(str(text))
    return tuple((0, int(p)) if p.isdecimal() else (1, p.lower()) for p in parts)


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


def check_channel_names(names: Iterable[object], what: str = "Channel names") -> list[str]:
    """``names`` as a list of strings, if no two of them are equal ignoring case.

    Channels are looked up by ``name.lower()`` throughout the package (``preprocess``,
    ``FeatureSet.pick_channels``, the loaders), so names that differ only in case would share
    one lookup entry and the later row would silently stand in for the earlier one. The check
    folds case exactly as those lookups do (``str.lower()``, not ``str.casefold()``).

    Raises:
        TypeError: if ``names`` is a single string (it would be read letter by letter).
        ValueError: starting with ``what`` and listing every group of colliding names.
    """
    if isinstance(names, str):
        raise TypeError(f"{what} must be a list of names, got the string {names!r}")
    names = [str(name) for name in names]
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name.lower(), []).append(name)
    clashes = [group for group in groups.values() if len(group) > 1]
    if clashes:
        raise ValueError(f"{what} must be unique ignoring case, got {', '.join(map(str, clashes))}")
    return names


def check_band_names(bands: Iterable[Sequence[object]], what: str = "Band names") -> None:
    """Raise ``ValueError`` (starting with ``what``) if two ``(name, low, high)`` bands share a name."""
    names = [str(band[0]) for band in bands]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{what} must not repeat a band name, got {repeated}")


def qualify(dataset: str, subject: str) -> str:
    """Globally unique subject id across datasets, e.g. ``"mahnob:12"``.

    Dataset names contain no ``":"`` (:func:`check_dataset_name`) while subject ids may, so the id
    splits back into ``(dataset, subject)`` at its first colon and distinct pairs never collide.

    Raises:
        ValueError: if ``dataset`` contains ``":"``.
    """
    return f"{check_dataset_name(dataset)}:{subject}"
