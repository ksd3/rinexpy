"""Tests for SP3 Lagrange interpolation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

import rinexpy as rp
from rinexpy.interp import interpolate_sp3

from .conftest import fixture


def test_interp_at_source_epoch_recovers_input():
    """Interpolating at an existing epoch returns the original values."""
    sp3 = rp.load_sp3(fixture("igs19362.sp3c"))
    t0 = sp3.time.values[5]
    result = interpolate_sp3(sp3, t0.astype("datetime64[us]").astype(datetime))
    truth = sp3.position.isel(time=5)
    np.testing.assert_allclose(result.position.values, truth.values, rtol=1e-6, atol=1e-3)


def test_interp_midpoint_on_orbit():
    """Interpolated position halfway between two source epochs lies on the
    orbit shell — i.e. its radius is similar to the source radii.

    (Lagrange-10 can overshoot a per-component bounding box; the
    physically meaningful invariant is that the satellite stays on its
    ~26000 km orbit.)
    """
    sp3 = rp.load_sp3(fixture("igs19362.sp3c"))
    t0 = sp3.time.values[10].astype("datetime64[us]").astype(datetime)
    t1 = sp3.time.values[11].astype("datetime64[us]").astype(datetime)
    midpoint_dt = t0 + (t1 - t0) / 2

    result = interpolate_sp3(sp3, midpoint_dt)
    p_mid = result.position.values
    p0 = sp3.position.isel(time=10).values
    # GPS positions are stored in km in SP3; orbit radius ~26600 km.
    finite = np.isfinite(p_mid).all(axis=-1) & np.isfinite(p0).all(axis=-1)
    r_mid = np.sqrt((p_mid[finite] ** 2).sum(axis=-1))
    r_src = np.sqrt((p0[finite] ** 2).sum(axis=-1))
    assert np.allclose(r_mid, r_src, rtol=1e-3)


def test_interp_batch():
    """Interpolating an array of times returns a Dataset with that time axis."""
    sp3 = rp.load_sp3(fixture("igs19362.sp3c"))
    t0 = sp3.time.values[5].astype("datetime64[us]").astype(datetime)
    queries = [t0 + timedelta(seconds=s) for s in (0, 60, 120, 180)]
    result = interpolate_sp3(sp3, np.array(queries, dtype="datetime64[ns]"))
    assert result.time.size == 4
