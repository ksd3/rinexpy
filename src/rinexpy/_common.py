"""Small shared utilities for the reader modules.

Helpers in this module are deliberately kept very narrow — anything that is
RINEX-version-specific lives in the corresponding reader module.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
    import psutil

    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


def fortran_float(s: str) -> float:
    """Parse a Fortran-style scientific float (``"1.23D+04"``) as a Python float.

    RINEX files emit floats in Fortran's ``D``-exponent form. This is a hot
    path; the implementation deliberately avoids ``str.replace`` when no
    ``D`` is present.

    Parameters
    ----------
    s:
        The text to parse. Leading/trailing whitespace is tolerated by
        ``float`` itself.

    Returns
    -------
    float
    """
    if "D" in s:
        s = s.replace("D", "E")
    return float(s)


def check_unique_times(times: np.ndarray) -> bool:
    """Log an error and return ``False`` if ``times`` contains duplicates.

    Parameters
    ----------
    times:
        Array-like of timestamps (any dtype that ``numpy.unique`` accepts).

    Returns
    -------
    bool
        ``True`` if all timestamps are unique, ``False`` otherwise.
    """
    n_unique = np.unique(times).size
    n_total = times.size
    if n_unique != n_total:
        log.error("only %d/%d timestamps are unique", n_unique, n_total)
        return False
    return True


def check_ram(needed_bytes: int, *, source: object | None = None) -> None:
    """Raise ``RuntimeError`` if ``needed_bytes`` would exhaust available RAM.

    Parameters
    ----------
    needed_bytes:
        Estimated allocation size in bytes.
    source:
        Optional object to mention in the error message (typically a Path).

    Notes
    -----
    Runs only if ``psutil`` is importable; otherwise this is a no-op so we
    don't burden minimal installs.
    """
    if not _HAVE_PSUTIL:
        return
    available = psutil.virtual_memory().available
    # The 0.5 factor accounts for the temporary array copy NumPy makes when
    # an xarray.Dataset is constructed from a NumPy buffer.
    if needed_bytes > 0.5 * available:
        msg = (
            f"need {needed_bytes / 1e9:.2f} GB RAM but only {available / 1e9:.2f} GB available; "
            "try fast=False to halve memory usage"
        )
        if source is not None:
            msg = f"{source}: {msg}"
        raise RuntimeError(msg)


# Map RINEX file-type letter to the human-readable time system name. RINEX 4
# table A2 page 7.
_TIME_SYSTEMS: dict[str, str] = {
    "G": "GPS",
    "R": "GLO",
    "E": "GAL",
    "J": "QZS",
    "C": "BDT",
    "I": "IRN",
}


def determine_time_system(header: Mapping[Hashable, Any]) -> str:
    """Determine the time system used in an OBS file from its header dict.

    Parameters
    ----------
    header:
        The parsed header mapping (as returned by ``obsheader2`` or
        ``obsheader3``).

    Returns
    -------
    str
        One of ``GPS``, ``GLO``, ``GAL``, ``QZS``, ``BDT``, ``IRN``. For mixed
        files (file-type letter ``M``) the string is read from cols 48-51 of
        the ``TIME OF FIRST OBS`` header field.

    Raises
    ------
    ValueError
        If the file-type letter is unknown.
    """
    try:
        file_type = header["RINEX VERSION / TYPE"][40]
    except (KeyError, IndexError):
        file_type = header["systems"]

    if file_type in _TIME_SYSTEMS:
        return _TIME_SYSTEMS[file_type]
    if file_type == "M":
        return header["TIME OF FIRST OBS"][48:51].strip()
    raise ValueError(f"unknown file type {file_type!r}")


def globber(path, glob: str | Iterable[str]) -> list:
    """List files in ``path`` matching one or more glob patterns.

    Parameters
    ----------
    path:
        A directory or a single file. If a file, it is returned as a 1-list.
    glob:
        A single glob pattern, or an iterable of patterns. Patterns are
        matched independently and the union is returned (not deduplicated by
        intent — duplicates are removed only by file identity).

    Returns
    -------
    list[Path]
        Files matching the patterns, in arbitrary order.
    """
    from pathlib import Path

    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    patterns = [glob] if isinstance(glob, str) else list(glob)
    out: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for f in p.glob(pat):
            if f.is_file() and f not in seen:
                out.append(f)
                seen.add(f)
    return out


def warn(msg: str) -> None:
    """Issue a ``UserWarning`` with the given message."""
    warnings.warn(msg, UserWarning, stacklevel=2)


__all__ = [
    "check_ram",
    "check_unique_times",
    "determine_time_system",
    "fortran_float",
    "globber",
    "warn",
]
