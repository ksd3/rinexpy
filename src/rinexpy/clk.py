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
            rec_type = line[:2]
            name = line[3:13].strip()
            try:
                y = int(line[8:13])  # spec puts date in cols 8-37
            except ValueError:
                # Some files use a longer name column; fall back to split.
                parts = line.split()
                if len(parts) < 9:
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
                    bias = fortran_float(parts[9])
                except (ValueError, IndexError) as e:
                    log.debug("skip clk line %r: %s", line[:80], e)
                    continue
            else:
                try:
                    t = datetime(
                        y,
                        int(line[14:16]),
                        int(line[17:19]),
                        int(line[20:22]),
                        int(line[23:25]),
                        int(float(line[26:34])),
                    )
                    # N value count at col 34-37 (we don't actually need it).
                    bias = fortran_float(line[40:59])
                except ValueError as e:
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


__all__ = ["load_clk"]
