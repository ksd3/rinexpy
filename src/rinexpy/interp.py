"""SP3 satellite-position interpolation.

Lagrange interpolation is the IGS-recommended method for SP3 ephemeris
(typical 15-minute spacing). Order-9 or 10 are common (8 nearest-
neighbor epochs around the target time, plus the target's two bracketing
epochs). Higher orders are unstable near the file ends.
"""

from __future__ import annotations

from datetime import datetime
from typing import overload

import numpy as np
import xarray as xr

from . import _native


def interpolate_sp3(
    sp3: xr.Dataset,
    times: datetime | np.ndarray | xr.DataArray,
    *,
    order: int = 10,
) -> xr.Dataset:
    """Interpolate SP3 satellite positions to arbitrary epoch(s).

    Parameters
    ----------
    sp3:
        SP3 dataset as returned by :func:`rinexpy.load_sp3`. Must contain
        a ``position`` data variable with dims ``(time, sv, ECEF)``.
    times:
        A single ``datetime`` (returns scalar interp), or any 1-D
        ndarray-like of ``datetime64`` / ``datetime`` (returns batched).
    order:
        Lagrange polynomial order. The function picks the ``order + 1``
        nearest source epochs around each query. Default 10 (matches
        IGS convention).

    Returns
    -------
    xarray.Dataset
        Coords ``time`` and ``sv``. One data variable ``position`` with
        dims ``(time, sv, ECEF)`` interpolated to the requested epochs.
    """
    src_t = sp3.time.values.astype("datetime64[ns]").astype("int64")
    if isinstance(times, datetime):
        query = np.array([np.datetime64(times, "ns").astype("int64")])
        scalar = True
    elif isinstance(times, xr.DataArray):
        query = times.values.astype("datetime64[ns]").astype("int64")
        scalar = False
    else:
        query = np.asarray(times).astype("datetime64[ns]").astype("int64")
        scalar = query.ndim == 0
        if scalar:
            query = query.reshape(1)

    pos = sp3.position.values  # (n_t, n_sv, 3)
    n_sv = pos.shape[1]

    span = order + 1
    n_src = src_t.size
    if span > n_src:
        span = n_src

    if _native.have_interpolate_sp3() and query.size > 0:
        out = np.asarray(
            _native.interpolate_sp3_lagrange(
                np.ascontiguousarray(src_t, dtype=np.int64),
                np.ascontiguousarray(pos, dtype=np.float64),
                np.ascontiguousarray(query, dtype=np.int64),
                span,
            ),
            dtype=float,
        )
    else:
        out = np.full((query.size, n_sv, 3), np.nan)
        for q_idx, q in enumerate(query):
            # Pick `span` source epochs centered around q.
            idx = int(np.searchsorted(src_t, q))
            lo = max(0, idx - span // 2)
            hi = lo + span
            if hi > n_src:
                hi = n_src
                lo = max(0, hi - span)
            sub_t = src_t[lo:hi].astype(float)
            sub_p = pos[lo:hi]  # (span, n_sv, 3)
            out[q_idx] = _lagrange(sub_t, sub_p, float(q))

    times_arr = query.view("datetime64[ns]")
    if scalar:
        result = xr.Dataset(
            {"position": (("sv", "ECEF"), out[0])},
            coords={"sv": sp3.sv, "ECEF": sp3.ECEF},
        )
        result.attrs["time"] = str(times_arr[0])
    else:
        result = xr.Dataset(
            {"position": (("time", "sv", "ECEF"), out)},
            coords={"time": times_arr, "sv": sp3.sv, "ECEF": sp3.ECEF},
        )
    return result


def _lagrange(xs: np.ndarray, ys: np.ndarray, x: float) -> np.ndarray:
    """Evaluate Lagrange interpolation at ``x`` for nodes ``(xs, ys)``.

    Parameters
    ----------
    xs:
        Node times, ``(n,)`` float64.
    ys:
        Node values, ``(n, ...)`` of any trailing shape.
    x:
        Query time.

    Returns
    -------
    ndarray
        ``ys[...].sum(axis=0)`` weighted by Lagrange basis polynomials.
    """
    n = xs.size
    # Compute weights w_i = prod_{j != i} (x - xs[j]) / (xs[i] - xs[j]).
    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weights[i] *= (x - xs[j]) / (xs[i] - xs[j])
    # ys may have trailing shape; broadcast weights against the leading axis.
    return np.tensordot(weights, ys, axes=(0, 0))


__all__ = ["interpolate_sp3"]


# silence "unused" warnings on overload decorator for older mypy
_ = overload
