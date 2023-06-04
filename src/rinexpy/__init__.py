"""rinexpy — modern, fast RINEX 2/3 / CRINEX / SP3 reader for Python.

Public API:

- :func:`load`  — auto-detect format and dispatch to the right reader
- :func:`rinexnav` / :func:`rinexobs` — version-agnostic readers
- :func:`load_sp3` — SP3-a/c/d ephemeris reader
- :func:`batch_convert` — convert a directory of RINEX files to NetCDF
- :func:`gettime` — extract just the timestamp axis
- :func:`rinexheader`, :func:`rinexinfo` — header inspection
- :func:`keplerian2ecef` — Keplerian orbital elements to ECEF position
- :func:`to_datetime` — convert xarray time coords to plain ``datetime``

The version-specific entry points (``rinexnav2``, ``rinexnav3``,
``rinexobs2``, ``rinexobs3``, ``obsheader2`` etc.) remain importable for
parity with georinex, but the high-level :func:`load` / :func:`rinexnav` /
:func:`rinexobs` are the recommended entry points.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._common import globber
from ._time import to_datetime
from .headers import (
    navheader2,
    navheader3,
    obsheader2,
    obsheader3,
    rinexheader,
    rinexinfo,
)

__all__ = [
    "__version__",
    "globber",
    "navheader2",
    "navheader3",
    "obsheader2",
    "obsheader3",
    "rinexheader",
    "rinexinfo",
    "to_datetime",
]
