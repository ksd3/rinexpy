"""RINEX MET (meteorological observation) file reader.

RINEX MET (.YYm, where YY is the 2-digit year) carries surface
meteorological measurements co-located with a GNSS receiver:
pressure, temperature, humidity, wet/dry temperature, dew-point,
zenith wet/total delay, etc. The format is identical between
RINEX 2.x and RINEX 3.x: a small ASCII header followed by epoch
records of fixed-width measurement fields.

The reader returns an :class:`xarray.Dataset` with the same shape
shape contract as the OBS readers: one ``time`` dimension, and one
data variable per meteorological quantity. Missing observations are
``NaN``.

Reference: RINEX 3.04 §10 (Met). Layouts also covered by
RINEX 2.11 §10.
"""

from __future__ import annotations

from datetime import datetime
from typing import IO

import numpy as np
import xarray as xr

from ._io import opener
from ._types import FileLike


# RINEX MET observation codes per section 10.1 of the spec. Values are
# 7.1 floats; missing entries are left as NaN.
_MET_CODE_DESCRIPTIONS: dict[str, str] = {
    "PR": "barometric pressure (mbar)",
    "TD": "dry temperature (deg C)",
    "HR": "relative humidity (%)",
    "ZW": "wet zenith path delay (mm)",
    "ZD": "dry zenith path delay (mm)",
    "ZT": "total zenith path delay (mm)",
    "WD": "wind azimuth (deg from north)",
    "WS": "wind speed (m/s)",
    "RI": "rain increment since last (1/10 mm)",
    "HI": "hail indicator (non-zero = hailing)",
}


def _parse_header(stream: IO[str]) -> dict:
    """Read the RINEX MET header and return ``(obs_codes, hdr_meta)``.

    Stops after the ``END OF HEADER`` line, leaving ``stream``
    positioned at the first epoch record.
    """
    codes: list[str] = []
    version: float | None = None
    marker_name = ""
    marker_number = ""
    interval: float | None = None
    n_codes = 0
    for line in stream:
        label = line[60:80].rstrip()
        head = line[:60]
        if label == "END OF HEADER":
            break
        if label == "RINEX VERSION / TYPE":
            try:
                version = float(head[:9])
            except ValueError:
                version = 2.11
        elif label == "MARKER NAME":
            marker_name = head.strip()
        elif label == "MARKER NUMBER":
            marker_number = head.strip()
        elif label == "INTERVAL":
            try:
                interval = float(head[:10])
            except ValueError:
                pass
        elif label == "# / TYPES OF OBSERV":
            # First line has N then up to 9 codes; continuation lines
            # carry up to 9 more codes (6 chars per code, 4-char field).
            if not codes:
                try:
                    n_codes = int(head[:6])
                except ValueError:
                    n_codes = 0
            for c in head[6:].split():
                if len(codes) < n_codes:
                    codes.append(c)
    return {
        "obs_codes": codes,
        "version": version,
        "marker_name": marker_name,
        "marker_number": marker_number,
        "interval_s": interval,
    }


def _parse_epoch_line(line: str, codes: list[str]) -> tuple[datetime, dict[str, float]]:
    """Parse one epoch line + values.

    RINEX MET epoch format:
       YY MM DD HH MM SS  V1 V2 V3 ...
    where the time is 2-digit year through seconds (cols 1-18) and each
    value is F7.1 (7-char wide field) starting at col 18.
    """
    yy = int(line[1:4])
    mm = int(line[4:7])
    dd = int(line[7:10])
    hh = int(line[10:13])
    mi = int(line[13:16])
    ss = int(line[16:19])
    if yy < 80:
        year = 2000 + yy
    elif yy < 100:
        year = 1900 + yy
    else:
        year = yy
    t = datetime(year, mm, dd, hh, mi, ss)
    values: dict[str, float] = {}
    # Each value lives in a 7-char field, right-justified, starting at
    # column 19 (after the time). Missing values are spaces / blanks
    # and decode to NaN.
    for i, code in enumerate(codes):
        col_start = 19 + i * 7
        col_end = col_start + 7
        field = line[col_start:col_end]
        try:
            values[code] = float(field)
        except ValueError:
            values[code] = float("nan")
    return t, values


def load_met(fn: FileLike) -> xr.Dataset:
    """Read a RINEX 2.x / 3.x meteorological observation file.

    Parameters
    ----------
    fn:
        Path, ``str``, or open text stream to a ``.YYm`` (or ``.met``)
        meteorological observation file. Transparent decompression
        applies via :func:`rinexpy._io.opener`.

    Returns
    -------
    xarray.Dataset
        Dataset with a ``time`` coordinate and one data variable per
        observation code declared in the header. Missing observations
        come back as ``NaN``.
    """
    with opener(fn) as stream:
        hdr = _parse_header(stream)
        codes = hdr["obs_codes"]
        times: list[datetime] = []
        per_code: dict[str, list[float]] = {c: [] for c in codes}
        for line in stream:
            if not line.strip():
                continue
            try:
                t, vals = _parse_epoch_line(line, codes)
            except ValueError:
                continue
            times.append(t)
            for c in codes:
                per_code[c].append(vals.get(c, float("nan")))

    time_arr = np.array(times, dtype="datetime64[ns]")
    data_vars = {c: (("time",), np.asarray(per_code[c], dtype=float)) for c in codes}
    ds = xr.Dataset(data_vars, coords={"time": time_arr})
    ds.attrs["rinex_version"] = hdr["version"]
    ds.attrs["marker_name"] = hdr["marker_name"]
    ds.attrs["marker_number"] = hdr["marker_number"]
    if hdr["interval_s"] is not None:
        ds.attrs["interval_s"] = hdr["interval_s"]
    ds.attrs["obs_codes"] = codes
    ds.attrs["filetype"] = "met"
    return ds


__all__ = ["load_met"]
