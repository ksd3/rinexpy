"""Date/time helpers tuned for RINEX line parsing.

The bulk of a RINEX read is consumed by *one* hot operation: parsing an
ASCII timestamp into a Python ``datetime``. The implementations here are
deliberately small and avoid keyword-argument calls (which are measurably
slower than positional calls inside CPython's ``datetime``).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import xarray as xr
from dateutil.parser import parse as _date_parse

from ._types import TimeLimit


def parse_obs2_epoch(line: str) -> datetime:
    """Parse a RINEX-2 OBS epoch line into a ``datetime``.

    Parameters
    ----------
    line:
        Full text of the epoch header line (cols 1-32+).

    Returns
    -------
    datetime
        Naive datetime in the file's time system.

    Raises
    ------
    ValueError
        If the year, month, or day fields fail to parse, or the epoch flag
        is not one of the valid values (0 = OK, 1 = power-failure, 5 = ext.
        event, 6 = cycle-slip).

    Notes
    -----
    Two-digit RINEX-2 years are interpreted with the canonical pivot of 1980:
    ``00..79`` map to ``2000..2079`` and ``80..99`` map to ``1980..1999``.
    """
    year = int(line[1:3])
    year += 2000 if year < 80 else 1900

    # Microseconds — only the slice 16-26 holds the seconds field, but RINEX
    # 2 actually allows up to ``%11.7f`` so we float-parse and modulo it.
    try:
        usec = int(float(line[16:26]) % 1 * 1_000_000)
    except ValueError:
        usec = 0

    epoch = datetime(
        year,
        int(line[4:6]),
        int(line[7:9]),
        int(line[10:12]),
        int(line[13:15]),
        int(line[16:18]),
        usec,
    )

    flag = int(line[28])
    if flag not in {0, 1, 5, 6}:
        raise ValueError(f"{epoch}: epoch flag {flag}")
    return epoch


def parse_obs3_epoch(line: str) -> datetime:
    """Parse a RINEX-3 OBS epoch line (``"> YYYY MM DD ..."``) into a ``datetime``.

    Parameters
    ----------
    line:
        Full text of the epoch header line (must start with ``"> "``).

    Returns
    -------
    datetime
        Naive datetime in the file's time system.

    Raises
    ------
    ValueError
        If the line does not start with ``"> "`` (RINEX 3 OBS epoch marker).
    """
    if not line.startswith("> "):
        raise ValueError("RINEX 3 epoch line must begin with '> '")
    try:
        usec = int(float(line[19:29]) % 1 * 1_000_000)
    except ValueError:
        usec = 0
    return datetime(
        int(line[2:6]),
        int(line[7:9]),
        int(line[10:12]),
        int(line[13:15]),
        int(line[16:18]),
        int(line[19:21]),
        usec,
    )


def parse_nav2_epoch(line: str) -> datetime:
    """Parse a RINEX-2 NAV epoch line into a ``datetime``.

    Parameters
    ----------
    line:
        Full text of the epoch header line (cols 1-22+).

    Returns
    -------
    datetime
        Naive datetime in the file's time system.

    Raises
    ------
    ValueError
        If any of the integer fields fail to parse, or the year is in an
        ambiguous out-of-range region.
    """
    year = int(line[3:5])
    if 80 <= year <= 99:
        year += 1900
    elif year < 80:
        year += 2000
    else:
        raise ValueError(f"unknown year format {year}")

    try:
        usec = int(float(line[17:22]) % 1 * 1_000_000)
    except ValueError:
        usec = 0

    return datetime(
        year,
        int(line[6:8]),
        int(line[9:11]),
        int(line[12:14]),
        int(line[15:17]),
        int(float(line[17:20])),
        usec,
    )


def parse_nav3_epoch(line: str) -> datetime:
    """Parse a RINEX-3 NAV epoch line into a ``datetime``.

    Parameters
    ----------
    line:
        Full text of the epoch header line (cols 4-23+).

    Returns
    -------
    datetime
    """
    return datetime(
        int(line[4:8]),
        int(line[9:11]),
        int(line[12:14]),
        int(line[15:17]),
        int(line[18:20]),
        int(line[21:23]),
    )


def parse_header_epoch(field: str) -> datetime:
    """Parse a ``TIME OF FIRST/LAST OBS`` header field into a ``datetime``.

    The field is fixed-width but real-world files frequently abuse the
    decimal-point alignment for the seconds, so we tolerate that by
    float-parsing and discarding obviously invalid values.
    """
    try:
        second = int(float(field[30:36]))
    except ValueError:
        second = 0
    if not 0 <= second <= 59:
        second = 0
    try:
        usec = int(float(field[30:43]) % 1 * 1_000_000)
    except ValueError:
        usec = 0
    if not 0 <= usec <= 999_999:
        usec = 0

    return datetime(
        int(field[:6]),
        int(field[6:12]),
        int(field[12:18]),
        int(field[18:24]),
        int(field[24:30]),
        second,
        usec,
    )


def normalize_tlim(tlim: TimeLimit) -> tuple[datetime, datetime] | None:
    """Coerce a ``tlim`` argument into ``(datetime, datetime)`` or ``None``.

    Parameters
    ----------
    tlim:
        Either ``None``, a 2-tuple of ``datetime`` instances, or a 2-tuple of
        ISO-8601 strings.

    Returns
    -------
    tuple[datetime, datetime] | None
        Normalized bounds, or ``None`` if the input was ``None``.

    Raises
    ------
    ValueError
        If the bounds are out of order, or the input has the wrong shape.
    """
    if tlim is None:
        return None
    if len(tlim) != 2:
        raise ValueError(f"time bounds must be a 2-tuple, got length {len(tlim)}")

    a, b = tlim
    if isinstance(a, str):
        a = _date_parse(a)
    if isinstance(b, str):
        b = _date_parse(b)
    assert isinstance(a, datetime) and isinstance(b, datetime)

    if b < a:
        raise ValueError("stop time must be ≥ start time")
    return a, b


def normalize_interval(interval: float | int | timedelta | None) -> timedelta | None:
    """Coerce an ``interval`` argument into a ``timedelta`` or ``None``.

    Parameters
    ----------
    interval:
        Either ``None`` (no decimation), a non-negative number of seconds, or
        a ``timedelta`` instance.

    Returns
    -------
    timedelta | None

    Raises
    ------
    ValueError
        If a numeric ``interval`` is negative.
    TypeError
        If ``interval`` is not one of the accepted types.
    """
    if interval is None:
        return None
    if isinstance(interval, timedelta):
        return interval
    if isinstance(interval, (int, float)):
        if interval < 0:
            raise ValueError("time interval must be non-negative")
        return timedelta(seconds=float(interval))
    raise TypeError(
        f"interval must be float, int, datetime.timedelta, or None; got {type(interval).__name__}"
    )


def to_datetime(times):
    """Convert an ``xarray`` time coord (or anything else) to ``datetime``.

    Parameters
    ----------
    times:
        Anything supporting ``.values.astype("datetime64[us]")``. If not an
        ``xarray.DataArray`` (or similar), it is returned unchanged.

    Returns
    -------
    datetime | numpy.ndarray[datetime]
        A scalar ``datetime`` if the input squeezes down to one element,
        otherwise an object-dtype NumPy array of ``datetime`` instances.
    """
    if not isinstance(times, xr.DataArray):
        return times
    arr = times.values.astype("datetime64[us]").astype(datetime)
    if not isinstance(arr, datetime):
        arr = arr.squeeze()[()]
    return arr


__all__ = [
    "normalize_interval",
    "normalize_tlim",
    "parse_header_epoch",
    "parse_nav2_epoch",
    "parse_nav3_epoch",
    "parse_obs2_epoch",
    "parse_obs3_epoch",
    "to_datetime",
]
