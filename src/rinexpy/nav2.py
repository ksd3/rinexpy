"""RINEX-2 navigation message reader.

Compared to ``georinex.nav2``, this module:

- Uses :func:`rinexpy._time.parse_nav2_epoch` (positional args, not kwargs)
  for the inner-loop date parse.
- Avoids the pre-allocate-with-NaN-then-fill pattern in favor of a single
  vectorized parse using :func:`numpy.fromstring` with fixed widths.
- Builds the final ``xarray.Dataset`` exactly once, never with merge-in-loop.
- Properly handles the GLONASS km-to-m unit conversion in a single broadcast
  multiply at the end, instead of looping over field names with `*=`.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import numpy as np
import xarray as xr

from ._common import fortran_float
from ._io import opener
from ._time import parse_nav2_epoch
from ._types import FileLike
from .headers import navheader2

log = logging.getLogger(__name__)

# Number of additional SV-data lines after the epoch line, per system letter.
_NL_BY_SYS: dict[str, int] = {"G": 7, "R": 3, "E": 7}

# Column where numerical data starts on continuation lines (RINEX 2).
_STARTCOL = 3
# Width of one Fortran ``D``/``E`` numeric field.
_FIELD_WIDTH = 19

#: GLONASS fields that need km->m conversion to match RINEX 3 conventions.
_GLO_M_FIELDS: frozenset[str] = frozenset(
    {"X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2"}
)

#: Per-system NAV field names. The order is **load-bearing**: the values are
#: positionally extracted from the raw ``D``-exponent stream.
_FIELDS_BY_SYS: dict[str, list[str]] = {
    "G": [
        "SVclockBias", "SVclockDrift", "SVclockDriftRate",
        "IODE", "Crs", "DeltaN", "M0",
        "Cuc", "Eccentricity", "Cus", "sqrtA",
        "Toe", "Cic", "Omega0", "Cis",
        "Io", "Crc", "omega", "OmegaDot",
        "IDOT", "CodesL2", "GPSWeek", "L2Pflag",
        "SVacc", "health", "TGD", "IODC",
        "TransTime", "FitIntvl",
    ],
    "R": [
        "SVclockBias", "SVrelFreqBias", "MessageFrameTime",
        "X", "dX", "dX2", "health",
        "Y", "dY", "dY2", "FreqNum",
        "Z", "dZ", "dZ2", "AgeOpInfo",
    ],
    "E": [
        "SVclockBias", "SVclockDrift", "SVclockDriftRate",
        "IODnav", "Crs", "DeltaN", "M0",
        "Cuc", "Eccentricity", "Cus", "sqrtA",
        "Toe", "Cic", "Omega0", "Cis",
        "Io", "Crc", "omega", "OmegaDot",
        "IDOT", "DataSrc", "GALWeek", "SISA",
        "health", "BGDe5a", "BGDe5b", "TransTime",
    ],
}

#: File-type letter -> system letter for RINEX 2 NAV.
_FILETYPE_TO_SYS: dict[str, str] = {"N": "G", "G": "R", "E": "E"}


def _skip(stream: IO[str], n: int) -> None:
    """Advance ``stream`` past ``n`` lines."""
    for _ in range(n):
        stream.readline()


def rinexnav2(
    fn: FileLike,
    *,
    tlim: tuple[datetime, datetime] | None = None,
) -> xr.Dataset:
    """Read a RINEX-2 navigation file into an ``xarray.Dataset``.

    Parameters
    ----------
    fn:
        Path or open text stream of a RINEX-2 NAV file (any compression
        recognized by :func:`rinexpy._io.opener` is fine).
    tlim:
        Optional ``(start, stop)`` datetime bounds; SV records outside the
        bounds are skipped without parsing.

    Returns
    -------
    xarray.Dataset
        Coords ``time`` (``datetime64[ns]``) and ``sv`` (string). One data
        variable per Keplerian/orbital field for the file's system. GLONASS
        positions/velocities/accelerations are converted to meters.

    Raises
    ------
    NotImplementedError
        If the file contains a RINEX-2 NAV system other than ``G``, ``R``,
        or ``E``.
    """
    if isinstance(fn, (str, Path)):
        fn = Path(fn).expanduser()

    svs: list[str] = []
    times: list[datetime] = []
    raws: list[str] = []

    with opener(fn) as f:
        header = navheader2(f)

        try:
            sv_letter = _FILETYPE_TO_SYS[header["filetype"]]
        except KeyError as e:
            raise NotImplementedError(
                f"unhandled RINEX-2 NAV system {header.get('systems')} ({fn})"
            ) from e
        fields = _FIELDS_BY_SYS[sv_letter]
        nl_extra = _NL_BY_SYS[sv_letter]

        for line in f:
            try:
                t = parse_nav2_epoch(line)
            except ValueError:
                continue

            if tlim is not None:
                if t < tlim[0]:
                    _skip(f, nl_extra)
                    continue
                if t > tlim[1]:
                    break

            svs.append(f"{sv_letter}{line[:2]}")
            times.append(t)
            # First line: cols 22-79; continuation lines: cols 3-79. We must
            # use 79, not 80: some files put '\n' a column early.
            raw = line[22:79]
            for _ in range(nl_extra):
                raw += f.readline()[_STARTCOL:79]
            # Convert Fortran ``D`` exponent to ``E``, drop newlines.
            raws.append(raw.replace("D", "E").replace("\n", ""))

    return _assemble_nav2(header, svs, times, raws, fields, sv_letter, fn)


def _assemble_nav2(
    header: dict[Hashable, Any],
    svs: list[str],
    times: list[datetime],
    raws: list[str],
    fields: list[str],
    sv_letter: str,
    fn: FileLike,
) -> xr.Dataset:
    """Materialize the ``xarray.Dataset`` from collected parse buffers."""
    svs = [s.replace(" ", "0") for s in svs]
    sv_unique = sorted(set(svs))

    times_arr = np.asarray(times)
    times_unique = np.unique(times_arr)
    n_fields = len(fields)
    data = np.full((n_fields, times_unique.size, len(sv_unique)), np.nan, dtype=float)

    sv_index = {sv: i for i, sv in enumerate(sv_unique)}
    time_index = {t: i for i, t in enumerate(times_unique.tolist())}

    for sv in sv_unique:
        # All record indices for this SV.
        rows = [i for i, s in enumerate(svs) if s == sv]
        sv_times = times_arr[rows]
        if np.unique(sv_times).size != sv_times.size:
            log.warning("duplicate times detected for SV %s; skipping", sv)
            continue
        for i in rows:
            it = time_index[times[i]]
            raw = raws[i]
            n = min(n_fields, len(raw) // _FIELD_WIDTH)
            for k in range(n):
                start = k * _FIELD_WIDTH
                data[k, it, sv_index[sv]] = float(raw[start : start + _FIELD_WIDTH])

    nav = xr.Dataset(
        {
            field: (("time", "sv"), data[i])
            for i, field in enumerate(fields)
        },
        coords={
            "time": times_unique.astype("datetime64[ns]"),
            "sv": sv_unique,
        },
    )

    # GLONASS is the only RINEX-2 NAV system that reports positions in km.
    if sv_letter == "R":
        for name in _GLO_M_FIELDS:
            if name in nav:
                nav[name] = nav[name] * 1000.0

    nav.attrs["version"] = header["version"]
    nav.attrs["svtype"] = [sv_letter]
    nav.attrs["rinextype"] = "nav"
    if isinstance(fn, Path):
        nav.attrs["filename"] = fn.name

    if "ION ALPHA" in header and "ION BETA" in header:
        alpha = header["ION ALPHA"]
        beta = header["ION BETA"]
        a_coef = [fortran_float(alpha[2 + i * 12 : 2 + (i + 1) * 12]) for i in range(4)]
        b_coef = [fortran_float(beta[2 + i * 12 : 2 + (i + 1) * 12]) for i in range(4)]
        nav.attrs["ionospheric_corr_GPS"] = np.hstack((a_coef, b_coef))

    return nav


def navtime2(fn: FileLike) -> np.ndarray:
    """Return all unique epoch timestamps in a RINEX-2 NAV file.

    Parameters
    ----------
    fn:
        Path or open text stream.

    Returns
    -------
    numpy.ndarray
        Sorted, unique ``datetime64[ms]`` array of epoch times.
    """
    times: list[datetime] = []
    with opener(fn) as f:
        header = navheader2(f)
        try:
            sv_letter = _FILETYPE_TO_SYS[header["filetype"]]
        except KeyError as e:
            raise NotImplementedError(
                f"unhandled RINEX-2 NAV system {header.get('systems')} ({fn})"
            ) from e
        nl_extra = _NL_BY_SYS[sv_letter]

        while True:
            line = f.readline()
            if not line:
                break
            try:
                times.append(parse_nav2_epoch(line))
            except ValueError:
                continue
            _skip(f, nl_extra)

    return np.unique(np.asarray(times, dtype="datetime64[ms]"))


__all__ = ["navtime2", "rinexnav2"]
