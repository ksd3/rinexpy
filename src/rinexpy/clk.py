"""RINEX clock (.clk) file reader.

Reference: ftp://igs.org/pub/data/format/rinex_clock304.txt

The data section consists of one record per line:

    <type> <name>  <YYYY MM DD HH MM SS.SSSSSS> <N> <values...>

where ``type`` is one of ``AR`` (receiver), ``AS`` (satellite), ``CR``
(combined receiver), ``DR`` (discontinuity receiver). ``N`` is the
number of values that follow (typically 1 for the bias, sometimes 2 for
bias+drift, 3 for bias+drift+accel). The ``values...`` are 19-byte
Fortran D-exponent floats. We currently emit just the bias.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ._common import fortran_float
from ._io import opener
from ._types import FileLike

log = logging.getLogger(__name__)


def load_clk(fn: FileLike) -> xr.Dataset:
    """Read a RINEX clock (.clk) file into an ``xarray.Dataset``.

    Parameters
    ----------
    fn:
        Path or open text stream of a ``.clk`` file (any opener-recognized
        compression is fine).

    Returns
    -------
    xarray.Dataset
        Coords: ``time`` (``datetime64[ns]``), ``sv`` (string).
        Data variables: ``bias`` (seconds, ``float64``).
        Attribute ``stations`` lists any ``AR``/``CR`` receiver labels seen.
    """
    sv_records: dict[str, list[tuple[datetime, float]]] = {}
    stations: set[str] = set()
    header: dict[Hashable, Any] = {"rinextype": "clk"}

    with opener(fn) as f:
        in_header = True
        for line in f:
            if in_header:
                if "END OF HEADER" in line:
                    in_header = False
                else:
                    label = line[60:].strip()
                    if label:
                        header.setdefault(label, line[:60])
                continue

            if len(line) < 40:
                continue
            # Whitespace-tokenized parsing: IGS final clocks use a more
            # compact column layout than the RINEX CLOCK spec example, and
            # different ACs vary slightly. Splitting on whitespace handles
            # every flavor seen in practice.
            parts = line.split()
            if len(parts) < 10:
                continue
            rec_type, name = parts[0], parts[1]
            try:
                t = datetime(
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                    int(parts[6]),
                    int(float(parts[7])),
                )
                # parts[8] is the N value count, not used here.
                bias = fortran_float(parts[9])
            except (ValueError, IndexError) as e:
                log.debug("skip clk line %r: %s", line[:80], e)
                continue

            if rec_type == "AS":
                sv_records.setdefault(name, []).append((t, bias))
            elif rec_type in ("AR", "CR"):
                stations.add(name)

    if not sv_records:
        ds = xr.Dataset(coords={"time": [], "sv": []})
        ds.attrs.update(header)
        ds.attrs["stations"] = sorted(stations)
        return ds

    all_times = sorted({t for recs in sv_records.values() for t, _ in recs})
    sv_labels = sorted(sv_records)
    t_idx = {t: i for i, t in enumerate(all_times)}
    s_idx = {s: i for i, s in enumerate(sv_labels)}

    bias = np.full((len(all_times), len(sv_labels)), np.nan, dtype=float)
    for sv, recs in sv_records.items():
        for t, b in recs:
            bias[t_idx[t], s_idx[sv]] = b

    ds = xr.Dataset(
        {"bias": (("time", "sv"), bias)},
        coords={
            "time": np.array(all_times, dtype="datetime64[ns]"),
            "sv": sv_labels,
        },
    )
    ds.attrs.update(header)
    ds.attrs["stations"] = sorted(stations)
    if isinstance(fn, Path):
        ds.attrs["filename"] = fn.name
    return ds


def interpolate_clk(
    ds: xr.Dataset,
    sv: str,
    epoch: datetime,
) -> float:
    """Linearly interpolate the clock bias of one SV at an arbitrary epoch.

    Parameters
    ----------
    ds:
        Clock dataset from :func:`load_clk`.
    sv:
        Satellite label (e.g. ``"G07"``).
    epoch:
        Time at which to interpolate.

    Returns
    -------
    float
        Clock bias in seconds. Returns NaN outside the dataset's time
        range or when the requested SV is absent.

    Notes
    -----
    Linear interpolation is the IGS-recommended method for 5-minute
    clock files (the products themselves are smoother than that to
    sub-ns level). For 30-second clocks, linear is exact for nearby
    queries.
    """
    if sv not in ds.sv.values:
        return float("nan")
    target = np.datetime64(epoch, "ns")
    times = ds.time.values
    if target < times[0] or target > times[-1]:
        return float("nan")
    after = int(np.searchsorted(times, target))
    before = max(0, after - 1)
    if after >= times.size:
        after = times.size - 1
    series = ds.bias.sel(sv=sv).values
    if before == after:
        return float(series[before])
    span = (times[after] - times[before]).astype("timedelta64[ns]").astype(float)
    if span == 0:
        return float(series[before])
    w = (target - times[before]).astype("timedelta64[ns]").astype(float) / span
    return float(series[before] * (1 - w) + series[after] * w)


__all__ = ["interpolate_clk", "load_clk"]
