"""GPT2w empirical surface-meteorology and VMF1 coefficient model.

GPT2w (Boehm et al., 2014) provides global empirical surface pressure,
temperature, water-vapor pressure, and the Vienna Mapping Function 1
hydrostatic / wet ``a`` coefficients on a regular latitude/longitude
grid. Each grid cell stores a mean plus four seasonal coefficients
(cos/sin of annual + semi-annual) for each quantity, so the model
can be evaluated at any day-of-year.

We do **not** ship the ~2 MB grid file. Download ``gpt2_5w.grd``
(5x5 deg resolution) or ``gpt2_1wA.grd`` (1x1 deg) from the VMF Data Server:

    https://vmf.geo.tuwien.ac.at/codes/

Then load it once at startup with :func:`load_gpt2w_grid` and pass the
returned dict to :func:`gpt2w` for each query.

The output ``a_h`` / ``a_w`` coefficients feed the VMF1 mapping
function (``rinexpy.geodesy.niell_mapping`` is a coarser substitute
that doesn't need a grid).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from . import _native


def load_gpt2w_grid(path: Path | str) -> dict:
    """Read a GPT2w ``.grd`` text file into an in-memory grid dict.

    Parameters
    ----------
    path:
        Path to ``gpt2_5w.grd`` (5x5 deg) or ``gpt2_1wA.grd`` (1x1 deg).

    Returns
    -------
    dict
        ``{"resolution_deg": float, "lat": ndarray, "lon": ndarray,
        "data": ndarray}`` where ``data`` has shape ``(n_lat, n_lon, 44)``
        — the 44 columns are exactly the rows of the official format
        (pressure mean + 4 coefs; temperature mean + 4 coefs; ... ).

    Raises
    ------
    ValueError
        If the file's first row doesn't have at least 23 numbers.
    """
    path = Path(path).expanduser()
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows or len(rows[0]) < 23:
        raise ValueError(f"{path}: not a recognisable GPT2w grid")

    arr = np.asarray(rows)
    # GPT2w convention: first 2 columns are lat (descending) and lon
    # (ascending), then 42-44 numerical columns.
    lats = np.unique(arr[:, 0])[::-1]
    lons = np.unique(arr[:, 1])
    res = float(np.abs(np.diff(lats)[0]))
    n_lat = lats.size
    n_lon = lons.size
    data = np.zeros((n_lat, n_lon, arr.shape[1] - 2))
    for row in arr:
        i = int(np.argmin(np.abs(lats - row[0])))
        j = int(np.argmin(np.abs(lons - row[1])))
        data[i, j] = row[2:]
    return {
        "resolution_deg": res,
        "lat": lats,
        "lon": lons,
        "data": data,
    }


def _seasonal(mean: float, a1: float, b1: float, a2: float, b2: float, doy: float) -> float:
    """Combine GPT2w seasonal coefs into a single value at ``doy``.

    Used by the per-quantity expansions inside :func:`gpt2w`.
    """
    cosa = np.cos((doy - 1) / 365.25 * 2 * np.pi)
    sina = np.sin((doy - 1) / 365.25 * 2 * np.pi)
    cosb = np.cos((doy - 1) / 365.25 * 4 * np.pi)
    sinb = np.sin((doy - 1) / 365.25 * 4 * np.pi)
    return mean + a1 * cosa + b1 * sina + a2 * cosb + b2 * sinb


def gpt2w(
    grid: dict,
    lat_deg: float,
    lon_deg: float,
    epoch: datetime | float,
    altitude_m: float = 0.0,
) -> dict:
    """Evaluate GPT2w at ``(lat, lon, doy)``.

    Parameters
    ----------
    grid:
        Output of :func:`load_gpt2w_grid`.
    lat_deg:
        Geodetic latitude in degrees.
    lon_deg:
        Longitude in degrees, ``[-180, 180]`` or ``[0, 360]``.
    epoch:
        ``datetime`` (used for day-of-year) or a float day-of-year.
    altitude_m:
        Receiver altitude in meters above sea level. Used for the
        hydrostatic-pressure altitude correction.

    Returns
    -------
    dict
        Keys: ``pressure_hpa``, ``temperature_k``, ``e_hpa`` (water
        vapor partial pressure), ``a_h``, ``a_w`` (VMF1 mapping coefs),
        ``T_lapse``, ``undulation_m`` (geoid - ellipsoid).
    """
    if isinstance(epoch, datetime):
        doy = epoch.timetuple().tm_yday
    else:
        doy = float(epoch)

    lon = lon_deg % 360.0
    lat = float(lat_deg)
    lat_axis = grid["lat"]
    lon_axis = grid["lon"]
    data = grid["data"]
    res = grid["resolution_deg"]

    # Bilinear-interp indices.
    i0 = int(np.clip(np.argmin(np.abs(lat_axis - lat)), 0, lat_axis.size - 1))
    j0 = int(np.clip(np.argmin(np.abs(lon_axis - lon)), 0, lon_axis.size - 1))
    # Use the four nearest grid cells.
    i1 = min(i0 + 1, lat_axis.size - 1)
    j1 = min(j0 + 1, lon_axis.size - 1)
    if lat_axis[i0] < lat:
        i0, i1 = max(0, i0 - 1), i0
    if lon_axis[j0] > lon:
        j0, j1 = max(0, j0 - 1), j0

    # Bilinear weights.
    w_lat = (
        0.0
        if lat_axis[i0] == lat_axis[i1]
        else (lat - lat_axis[i0]) / (lat_axis[i1] - lat_axis[i0])
    )
    w_lon = (
        0.0
        if lon_axis[j0] == lon_axis[j1]
        else (lon - lon_axis[j0]) / (lon_axis[j1] - lon_axis[j0])
    )
    _ = res  # kept for callers that want to inspect

    if _native.have_gpt2w_eval():
        # Pack the 4 corner cells into a flat 176-float buffer
        # (4 corners x 44 columns each). Pad columns 42..43 with the
        # neighbour's data if the grid has only 42 columns to satisfy
        # the kernel's 44-column expectation.
        cell_cols = data.shape[2]
        if cell_cols >= 44:
            buf = np.ascontiguousarray(
                np.stack([
                    data[i0, j0, :44], data[i0, j1, :44],
                    data[i1, j0, :44], data[i1, j1, :44],
                ]).ravel(),
                dtype=float,
            )
        else:
            # Pad with zeros so the layout matches; columns 42..43
            # (water-vapour-decrease and friends) are read but only the
            # ones the kernel uses are propagated to the output.
            padded = np.zeros((4, 44), dtype=float)
            padded[0, :cell_cols] = data[i0, j0]
            padded[1, :cell_cols] = data[i0, j1]
            padded[2, :cell_cols] = data[i1, j0]
            padded[3, :cell_cols] = data[i1, j1]
            buf = np.ascontiguousarray(padded.ravel(), dtype=float)
        result = _native.gpt2w_eval_cell(
            buf, w_lat, w_lon, doy, altitude_m,
        )
        return {
            "pressure_hpa": float(result[0]),
            "temperature_k": float(result[1]),
            "e_hpa": float(result[2]),
            "a_h": float(result[3]),
            "a_w": float(result[4]),
            "T_lapse": float(result[5]),
            "undulation_m": float(result[6]),
        }

    def at(corner_i: int, corner_j: int) -> dict:
        cell = data[corner_i, corner_j]
        # Layout per the GPT2w grid file (5x5 deg, 44 numbers):
        #  0    : undulation (m)
        #  1    : reference altitude (m)
        #  2- 6: P mean, a1, b1, a2, b2
        #  7-11: T mean, a1, b1, a2, b2
        # 12-16: q (specific humidity) mean, ...
        # 17-21: dT (T lapse) mean, ...
        # 22-26: Tm (mean temp) mean, ...
        # 27-31: lambda (water vapor decrease factor) mean, ...
        # 32-36: a_h mean, ...
        # 37-41: a_w mean, ...
        return {
            "undu": cell[0],
            "ref_h": cell[1],
            "p": _seasonal(*cell[2:7], doy),
            "t": _seasonal(*cell[7:12], doy),
            "q": _seasonal(*cell[12:17], doy),
            "dt": _seasonal(*cell[17:22], doy),
            "tm": _seasonal(*cell[22:27], doy),
            "lam": _seasonal(*cell[27:32], doy),
            "a_h": _seasonal(*cell[32:37], doy),
            "a_w": _seasonal(*cell[37:42], doy),
        }

    c00 = at(i0, j0)
    c01 = at(i0, j1)
    c10 = at(i1, j0)
    c11 = at(i1, j1)

    def interp(key: str) -> float:
        v00, v01, v10, v11 = c00[key], c01[key], c10[key], c11[key]
        v0 = v00 * (1 - w_lon) + v01 * w_lon
        v1 = v10 * (1 - w_lon) + v11 * w_lon
        return float(v0 * (1 - w_lat) + v1 * w_lat)

    p0 = interp("p")
    t0 = interp("t")
    dt = interp("dt")
    q = interp("q")
    lam = interp("lam")
    a_h = interp("a_h")
    a_w = interp("a_w")
    undu = interp("undu")
    ref_h = interp("ref_h")

    # Altitude reduction (Boehm 2007 §3): adjust pressure using lapse and
    # virtual temperature at the reference altitude.
    dh = altitude_m - ref_h
    t_h = t0 + dt * dh
    e0 = q * p0 / (0.622 + 0.378 * q)
    p_h = p0 * (1 - 0.0000226 * dh) ** 5.225
    # Approximate water vapor decrease.
    e_h = e0 * (p_h / p0) ** (lam + 1.0)

    return {
        "pressure_hpa": float(p_h),
        "temperature_k": float(t_h),
        "e_hpa": float(e_h),
        "a_h": float(a_h),
        "a_w": float(a_w),
        "T_lapse": float(dt),
        "undulation_m": float(undu),
    }


__all__ = ["gpt2w", "load_gpt2w_grid"]
