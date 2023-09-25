"""IONEX (.inx, .i) reader for global ionospheric maps.

Reference: https://files.igs.org/pub/data/format/ionex1.pdf

The data section is a sequence of MAPs, each tagged ``START OF TEC MAP``
through ``END OF TEC MAP``. A MAP holds an exponent and a 2-D grid of
TEC values (units: 0.1 TECU after applying the exponent). The grid is
laid out latitude-major: one ``LAT/LON1/LON2/DLON/H`` record per
latitude row, followed by the values (16 per record, 5 chars each).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from ._io import opener
from ._types import FileLike

log = logging.getLogger(__name__)


def load_ionex(fn: FileLike) -> xr.Dataset:
    """Read an IONEX file into an ``xarray.Dataset``.

    Parameters
    ----------
    fn:
        Path or open text stream of an ``.inx`` / ``.i`` file.

    Returns
    -------
    xarray.Dataset
        Coords: ``time`` (``datetime64[ns]``), ``lat`` (degrees),
        ``lon`` (degrees). Data variable ``tec`` (units: TECU,
        ``1 TECU = 1e16 electrons/m^2``).
    """
    with opener(fn) as f:
        header = _parse_ionex_header(f)
        maps = list(_iter_tec_maps(f, header))

    if not maps:
        ds = xr.Dataset(coords={"time": [], "lat": [], "lon": []})
        ds.attrs.update(header)
        return ds

    times = np.array([m[0] for m in maps], dtype="datetime64[ns]")
    grid = np.stack([m[1] for m in maps])
    lat = header["lat_axis"]
    lon = header["lon_axis"]

    ds = xr.Dataset(
        {"tec": (("time", "lat", "lon"), grid)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    for k in ("EXPONENT", "BASE RADIUS", "MAP DIMENSION"):
        if k in header:
            ds.attrs[k] = header[k]
    if isinstance(fn, Path):
        ds.attrs["filename"] = fn.name
    return ds


def _parse_ionex_header(f) -> dict:
    """Read up to ``END OF HEADER``; return the lat/lon axes and metadata."""
    hdr: dict = {}
    for line in f:
        label = line[60:].strip()
        if label == "END OF HEADER":
            break
        if label == "EPOCH OF FIRST MAP":
            hdr["epoch_first"] = _parse_epoch(line)
        elif label == "EPOCH OF LAST MAP":
            hdr["epoch_last"] = _parse_epoch(line)
        elif label == "INTERVAL":
            hdr["interval"] = int(line[:6])
        elif label == "# OF MAPS IN FILE":
            hdr["n_maps"] = int(line[:6])
        elif label == "EXPONENT":
            hdr["EXPONENT"] = int(line[:6])
        elif label == "LAT1 / LAT2 / DLAT":
            hdr["lat1"] = float(line[2:8])
            hdr["lat2"] = float(line[8:14])
            hdr["dlat"] = float(line[14:20])
        elif label == "LON1 / LON2 / DLON":
            hdr["lon1"] = float(line[2:8])
            hdr["lon2"] = float(line[8:14])
            hdr["dlon"] = float(line[14:20])
        elif label == "MAP DIMENSION":
            hdr["MAP DIMENSION"] = int(line[:6])
        elif label == "BASE RADIUS":
            hdr["BASE RADIUS"] = float(line[:8])

    if "lat1" not in hdr or "lon1" not in hdr:
        raise ValueError("IONEX header missing lat/lon axis records")

    hdr["lat_axis"] = _build_axis(hdr["lat1"], hdr["lat2"], hdr["dlat"])
    hdr["lon_axis"] = _build_axis(hdr["lon1"], hdr["lon2"], hdr["dlon"])
    hdr.setdefault("EXPONENT", -1)
    return hdr


def _build_axis(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive evenly-spaced axis from ``start`` to ``stop`` with ``step``.

    Handles the IONEX convention where ``lat1=87.5, lat2=-87.5, dlat=-2.5``
    yields a descending axis.
    """
    n = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, n)


def _parse_epoch(line: str) -> datetime:
    """Parse an IONEX epoch line (``yyyy mm dd hh mm ss`` in the first 36 cols)."""
    return datetime(
        int(line[:6]),
        int(line[6:12]),
        int(line[12:18]),
        int(line[18:24]),
        int(line[24:30]),
        int(line[30:36]),
    )


def _iter_tec_maps(f, header):
    """Yield ``(time, grid)`` tuples for each ``START OF TEC MAP`` block.

    ``grid`` is a 2-D ``(n_lat, n_lon)`` float array in TECU. The
    ``EXPONENT`` from the header (or any per-map override) is applied.
    """
    n_lat = header["lat_axis"].size
    n_lon = header["lon_axis"].size
    exponent = header["EXPONENT"]
    cur_grid: np.ndarray | None = None
    cur_time: datetime | None = None
    cur_lat_idx = 0
    cur_lon_idx = 0
    in_map = False

    for line in f:
        label = line[60:].strip()
        if label == "START OF TEC MAP":
            cur_grid = np.full((n_lat, n_lon), np.nan)
            cur_time = None
            cur_lat_idx = -1
            cur_lon_idx = 0
            in_map = True
        elif label == "END OF TEC MAP":
            if cur_grid is not None and cur_time is not None:
                yield cur_time, cur_grid * 10.0**exponent
            in_map = False
            cur_grid = None
        elif label == "END OF FILE":
            return
        elif not in_map:
            continue
        elif label == "EPOCH OF CURRENT MAP":
            cur_time = _parse_epoch(line)
        elif label == "LAT/LON1/LON2/DLON/H":
            cur_lat_idx += 1
            cur_lon_idx = 0
        else:
            # Data line: 16 values of 5 chars each.
            assert cur_grid is not None
            for off in range(0, 80, 5):
                chunk = line[off : off + 5]
                if not chunk.strip():
                    continue
                try:
                    v = int(chunk)
                except ValueError:
                    continue
                if v == 9999:  # IONEX missing-value sentinel
                    continue
                if cur_lon_idx < n_lon and 0 <= cur_lat_idx < n_lat:
                    cur_grid[cur_lat_idx, cur_lon_idx] = v
                cur_lon_idx += 1


__all__ = ["load_ionex"]
