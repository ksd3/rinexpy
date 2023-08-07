"""Shared type aliases for rinexpy.

These type aliases are kept in their own module so that they can be imported
from anywhere in the package without provoking a circular import. Module
authors are encouraged to use these aliases instead of re-spelling the union
types each time, both for consistency and to give the type checker a stable
identity to attach errors to.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO, TextIOWrapper
from pathlib import Path
from typing import IO, TypeAlias

#: Anything we accept as an input "RINEX file": a filesystem path (string or
#: ``pathlib.Path``) or an in-memory text stream. ``IO[str]`` is the protocol
#: from ``typing`` that covers both ``StringIO`` and ``TextIOWrapper``.
FileLike: TypeAlias = str | Path | StringIO | TextIOWrapper | IO[str]

#: A pair ``(start, stop)`` of inclusive datetime bounds, or ``None`` for "no
#: limit". The pair may be passed as ISO 8601 strings; the API normalizes them
#: to ``datetime`` objects via :func:`rinexpy._time.normalize_tlim`.
TimeLimit: TypeAlias = tuple[datetime, datetime] | tuple[str, str] | None

#: A choice of GNSS systems to keep, expressed as a set of single-letter codes
#: (``G``, ``R``, ``E``, ``C``, ``J``, ``S``, ``I``). ``None`` means "all".
SystemSelection: TypeAlias = set[str] | frozenset[str] | list[str] | tuple[str, ...] | str | None

#: A choice of measurement types to keep, e.g. ``["L1C", "C1C"]``. Strings
#: are matched as prefixes against the observation labels in the header. A
#: bare string is treated as a one-element list. ``None`` means "all".
MeasSelection: TypeAlias = list[str] | tuple[str, ...] | str | None

__all__ = [
    "FileLike",
    "MeasSelection",
    "SystemSelection",
    "TimeLimit",
]
