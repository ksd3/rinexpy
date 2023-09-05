"""rinexpy - modern, fast RINEX 2/3 / CRINEX / SP3 reader for Python.

Public API:

- :func:`load`  - auto-detect format and dispatch to the right reader
- :func:`rinexnav` / :func:`rinexobs` - version-agnostic readers
- :func:`load_sp3` - SP3-a/c/d ephemeris reader
- :func:`batch_convert` - convert a directory of RINEX files to NetCDF
- :func:`gettime` - extract just the timestamp axis
- :func:`rinexheader`, :func:`rinexinfo` - header inspection
- :func:`keplerian2ecef` - Keplerian orbital elements to ECEF position
- :func:`to_datetime` - convert xarray time coords to plain ``datetime``

The version-specific entry points (``rinexnav2``, ``rinexnav3``,
``rinexobs2``, ``rinexobs3``, ``obsheader2`` etc.) remain importable for
parity with georinex, but the high-level :func:`load` / :func:`rinexnav` /
:func:`rinexobs` are the recommended entry points.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._common import globber
from ._io import opener
from ._time import to_datetime
from .api import batch_convert, gettime, load, rinexnav, rinexobs
from .headers import (
    navheader2,
    navheader3,
    obsheader2,
    obsheader3,
    rinexheader,
    rinexinfo,
)
from .keplerian import keplerian2ecef
from .nav2 import navtime2, rinexnav2
from .nav3 import navtime3, rinexnav3
from .obs2 import obstime2, rinexobs2
from .obs3 import obstime3, rinexobs3
from .sp3 import load_sp3
from .streaming import iter_obs3_epochs

# Plotting helpers are an optional extra; do NOT import at top level so
# the bare install stays matplotlib-free. Users import directly:
#     from rinexpy.plots import timeseries

__all__ = [
    "__version__",
    "batch_convert",
    "gettime",
    "globber",
    "iter_obs3_epochs",
    "keplerian2ecef",
    "load",
    "load_sp3",
    "navheader2",
    "navheader3",
    "navtime2",
    "navtime3",
    "obsheader2",
    "obsheader3",
    "obstime2",
    "obstime3",
    "opener",
    "rinexheader",
    "rinexinfo",
    "rinexnav",
    "rinexnav2",
    "rinexnav3",
    "rinexobs",
    "rinexobs2",
    "rinexobs3",
    "to_datetime",
]
