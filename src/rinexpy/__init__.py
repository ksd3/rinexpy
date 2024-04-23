"""rinexpy - modern, fast RINEX 2/3 / CRINEX / SP3 reader for Python.

Public API:

- :func:`load` - auto-detect format and dispatch to the right reader
- :func:`rinexnav` / :func:`rinexobs` - version-agnostic readers
- :func:`load_sp3`, :func:`load_clk`, :func:`load_ionex`, :func:`load_antex`
- :func:`batch_convert` - parallel directory-of-files NetCDF conversion
- :func:`iter_obs3_epochs` - per-epoch streaming for files larger than RAM
- :func:`gettime` - extract just the timestamp axis
- :func:`rinexheader`, :func:`rinexinfo` - header inspection
- :func:`keplerian2ecef` - Keplerian orbital elements to ECEF position
- :func:`interpolate_sp3` - Lagrange interpolation of SP3 ephemerides
- :func:`to_datetime` - convert xarray time coords to plain ``datetime``
- :func:`to_rinex_obs` - write a parsed dataset back to RINEX
- :func:`spp_solve` - single-point positioning least-squares solver
- :mod:`rinexpy.tools` - validate / concat / diff helpers
- :mod:`rinexpy.geodesy` - ECEF/LLA, az/el, DOP, Klobuchar
- :mod:`rinexpy.gpstime` - GPS week and leap-second utilities
- :mod:`rinexpy.rtcm3` - RTCM3 streaming-feed decoder
- :mod:`rinexpy.nmea` - NMEA-0183 sentence decoder
- :mod:`rinexpy.ubx` - u-blox UBX binary decoder
- :mod:`rinexpy.sbf` - Septentrio SBF binary decoder
- :mod:`rinexpy.novatel` - NovAtel OEM binary decoder
- :mod:`rinexpy.binex` - UNAVCO BINEX archival decoder
- :mod:`rinexpy.rtcm2` - legacy RTCM SC-104 v2.x DGPS decoder
- :mod:`rinexpy.beidou` - BeiDou D1/D2 raw subframe decoder
- :mod:`rinexpy.plots` - matplotlib timeseries / skyplot / map plots
- :mod:`rinexpy.asyncio` - asyncio-friendly load wrappers

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
from .antex import apply_antex_pcv, find_antenna, load_antex
from .api import batch_convert, gettime, load, rinexnav, rinexobs
from .clk import interpolate_clk, load_clk
from .gpt2w import gpt2w, load_gpt2w_grid
from .headers import (
    navheader2,
    navheader3,
    obsheader2,
    obsheader3,
    rinexheader,
    rinexinfo,
)
from .interp import interpolate_sp3
from .ionex import interp_tec, load_ionex, slant_tec
from .keplerian import keplerian2ecef
from .lambda_ar import lambda_resolve
from .multifreq import lambda_dual_freq
from .nav2 import navtime2, rinexnav2
from .nav3 import navtime3, rinexnav3
from .obs2 import obstime2, rinexobs2
from .obs3 import obstime3, rinexobs3
from .positioning import spp_solve
from .sp3 import load_sp3
from .streaming import iter_obs3_epochs
from .writer import to_rinex_obs

__all__ = [
    "__version__",
    "apply_antex_pcv",
    "batch_convert",
    "find_antenna",
    "gettime",
    "globber",
    "gpt2w",
    "interp_tec",
    "interpolate_clk",
    "interpolate_sp3",
    "iter_obs3_epochs",
    "keplerian2ecef",
    "lambda_dual_freq",
    "lambda_resolve",
    "load",
    "load_antex",
    "load_clk",
    "load_gpt2w_grid",
    "load_ionex",
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
    "slant_tec",
    "spp_solve",
    "to_datetime",
    "to_rinex_obs",
]
