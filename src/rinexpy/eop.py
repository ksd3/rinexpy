"""IERS Earth Orientation Parameter (EOP) C04 reader.

The IERS EOP 14 / 20 C04 series gives daily values (at 0h UTC) of:

- Polar motion ``x``, ``y`` in arcseconds.
- ``UT1 - UTC`` in seconds.
- Length of day (``LOD``) in seconds.
- Celestial pole offsets ``dX``, ``dY`` in arcseconds.

Plus per-row formal uncertainties (the suffix ``_err`` variables).

These are needed for ECEF <-> ECI transforms whenever a downstream
caller wants to position a satellite in an inertial frame (orbit
integration, ITRF realisations, etc.).

The file format is fixed-width per the spec but whitespace-tokenised
parsing is more robust to small layout drift across versions; we use
that here. Source: https://hpiers.obspm.fr/iers/eop/eopc04/
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

from ._io import opener
from ._types import FileLike


def load_eop(fn: FileLike) -> xr.Dataset:
    """Read an IERS EOP C04 file into an ``xarray.Dataset``.

    Parameters
    ----------
    fn:
        Path or stream to an ``eopc04*`` file (any opener-recognised
        compression is fine).

    Returns
    -------
    xarray.Dataset
        ``time`` (``datetime64[ns]``) coordinate plus 12 data variables:
        ``x``, ``y`` (arcsec), ``ut1_utc`` (s), ``lod`` (s), ``dx``, ``dy``
        (arcsec), and the matching ``_err`` formal uncertainties.

    Raises
    ------
    ValueError
        If no valid data rows are found.
    """
    rows: list[tuple] = []
    with opener(fn) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
            except ValueError:
                continue
            if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
                continue
            try:
                t = datetime(year, month, day)
                # parts[3] is MJD, useful as a sanity check but we use the
                # year/month/day for the time coord.
                values = tuple(float(parts[i]) for i in range(4, 16))
            except (ValueError, IndexError):
                continue
            rows.append((t, *values))

    if not rows:
        raise ValueError("no valid EOP rows in file")

    times = np.array([r[0] for r in rows], dtype="datetime64[ns]")
    arr = np.asarray([r[1:] for r in rows], dtype=float)
    return xr.Dataset(
        {
            "x": (("time",), arr[:, 0]),
            "y": (("time",), arr[:, 1]),
            "ut1_utc": (("time",), arr[:, 2]),
            "lod": (("time",), arr[:, 3]),
            "dx": (("time",), arr[:, 4]),
            "dy": (("time",), arr[:, 5]),
            "x_err": (("time",), arr[:, 6]),
            "y_err": (("time",), arr[:, 7]),
            "ut1_utc_err": (("time",), arr[:, 8]),
            "lod_err": (("time",), arr[:, 9]),
            "dx_err": (("time",), arr[:, 10]),
            "dy_err": (("time",), arr[:, 11]),
        },
        coords={"time": times},
        attrs={
            "rinextype": "eop",
            "format": "IERS EOP C04",
            "x_units": "arcsec",
            "y_units": "arcsec",
            "ut1_utc_units": "s",
            "lod_units": "s",
            "dx_units": "arcsec",
            "dy_units": "arcsec",
        },
    )


def interp_eop(eop: xr.Dataset, epoch: datetime | np.datetime64) -> dict[str, Any]:
    """Linearly interpolate EOP parameters at an arbitrary epoch.

    Parameters
    ----------
    eop:
        Dataset from :func:`load_eop`.
    epoch:
        Query time (``datetime`` or ``numpy.datetime64``).

    Returns
    -------
    dict
        ``{"x": ..., "y": ..., "ut1_utc": ..., "lod": ..., "dx": ..., "dy": ...}``
        with all values in their native units (arcsec or seconds).

    Raises
    ------
    ValueError
        If ``epoch`` is outside the dataset's time range.
    """
    t_ns = (
        epoch if isinstance(epoch, np.datetime64) else np.datetime64(epoch, "ns")
    )
    times = eop.time.values.astype("datetime64[ns]")
    if t_ns < times[0] or t_ns > times[-1]:
        raise ValueError(
            f"epoch {t_ns} outside EOP range [{times[0]}, {times[-1]}]"
        )
    # np.interp wants float x-axis.
    t_float = t_ns.astype("int64").astype(float)
    src = times.astype("int64").astype(float)
    out: dict[str, Any] = {}
    for name in ("x", "y", "ut1_utc", "lod", "dx", "dy"):
        out[name] = float(np.interp(t_float, src, eop[name].values))
    return out


__all__ = ["interp_eop", "load_eop"]
