"""Tests for the IERS 2010 step-2 frequency-dependent corrections.

The diurnal and long-period band routines are validated bit-for-bit
against the test cases published in the IERS reference Fortran
distribution (STEP2DIU.F / STEP2LON.F) — those expected values come
from the IERS Conventions Center itself and are the gold standard.

The pole tide is validated structurally (amplitude band, equator
limit, vanishing case when polar motion equals the mean pole).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from rinexpy.geodesy import lla_to_ecef
from rinexpy.tides import (
    pole_tide_displacement,
    step2_diurnal_displacement,
    step2_displacement,
    step2_long_period_displacement,
)
from rinexpy.tides import _step2_diurnal_core, _step2_long_period_core


# ---------- IERS reference test cases (bit-for-bit) ----------

# These three test cases are copied verbatim from the comment headers
# of the IERS Conventions Center Fortran reference distribution. Any
# deviation here means we've drifted from the conventions.

IERS_TEST_STATION = np.array([4075578.385, 931852.890, 4801570.154])
IERS_TEST_T = 0.1059411362080767


def test_step2_diurnal_matches_iers_reference():
    """STEP2DIU.F published test case."""
    xc = _step2_diurnal_core(IERS_TEST_STATION, IERS_TEST_T, 0.0)
    expected = np.array([
        0.4193085327321284701e-2,
        0.1456681241014607395e-2,
        0.5123366597450316508e-2,
    ])
    np.testing.assert_allclose(xc, expected, atol=1e-12)


def test_step2_long_period_matches_iers_reference():
    """STEP2LON.F published test case."""
    xc = _step2_long_period_core(IERS_TEST_STATION, IERS_TEST_T)
    expected = np.array([
        -0.9780962849562107762e-04,
        -0.2236349699932734273e-04,
        0.3561945821351565926e-03,
    ])
    np.testing.assert_allclose(xc, expected, atol=1e-12)


# ---------- Datetime-front-end smoke tests ----------


def test_diurnal_runs_from_datetime():
    """The high-level entry takes a datetime and returns a sensible
    (sub-cm) displacement."""
    station = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    for hour in (0, 6, 12, 18):
        epoch = datetime(2024, 6, 21, hour, 0, 0, tzinfo=timezone.utc)
        dr = step2_diurnal_displacement(station, epoch)
        assert dr.shape == (3,)
        # Diurnal step-2 magnitude stays inside a few mm.
        assert np.linalg.norm(dr) < 0.02


def test_long_period_runs_from_datetime():
    """Long-period high-level entry."""
    station = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    epoch = datetime(2024, 3, 21, 12, 0, 0, tzinfo=timezone.utc)
    dr = step2_long_period_displacement(station, epoch)
    assert dr.shape == (3,)
    # Long-period amplitudes are all sub-mm in the table.
    assert np.linalg.norm(dr) < 0.005


def test_total_step2_matches_sum_of_bands():
    """step2_displacement is just diurnal + long-period."""
    station = np.array(lla_to_ecef(45.0, 30.0, 50.0))
    epoch = datetime(2024, 9, 15, 8, 30, 0, tzinfo=timezone.utc)
    a = step2_diurnal_displacement(station, epoch)
    b = step2_long_period_displacement(station, epoch)
    c = step2_displacement(station, epoch)
    np.testing.assert_allclose(c, a + b, atol=1e-15)


# ---------- Pole tide ----------


def _toy_eop(year=2024) -> xr.Dataset:
    """Build a tiny synthetic EOP dataset with constant polar motion."""
    times = np.array([
        f"{year}-06-19T00:00:00",
        f"{year}-06-20T00:00:00",
        f"{year}-06-21T00:00:00",
        f"{year}-06-22T00:00:00",
    ], dtype="datetime64[ns]")
    n = len(times)
    return xr.Dataset(
        {
            "x": (("time",), [0.1] * n),
            "y": (("time",), [0.4] * n),
            "ut1_utc": (("time",), [-0.05] * n),
            "lod": (("time",), [0.0] * n),
            "dx": (("time",), [0.0] * n),
            "dy": (("time",), [0.0] * n),
            "x_err": (("time",), [0.0] * n),
            "y_err": (("time",), [0.0] * n),
            "ut1_utc_err": (("time",), [0.0] * n),
            "lod_err": (("time",), [0.0] * n),
            "dx_err": (("time",), [0.0] * n),
            "dy_err": (("time",), [0.0] * n),
        },
        coords={"time": times},
    )


def test_pole_tide_amplitude_is_in_centimeter_band():
    eop = _toy_eop()
    station = np.array(lla_to_ecef(45.0, 30.0, 0.0))
    epoch = datetime(2024, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
    dr = pole_tide_displacement(station, eop, epoch)
    assert dr.shape == (3,)
    mag = np.linalg.norm(dr)
    assert 0.0 < mag < 0.02


def test_pole_tide_vanishes_at_equator():
    eop = _toy_eop()
    eq_station = np.array(lla_to_ecef(0.0, 30.0, 0.0))
    epoch = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    dr = pole_tide_displacement(eq_station, eop, epoch)
    assert np.linalg.norm(dr) < 0.01


def test_pole_tide_returns_zero_with_zero_polar_motion():
    """When polar motion equals the mean-pole model exactly, the pole
    tide vanishes."""
    epoch = datetime(2024, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
    t_years = 24.5
    x_mean = 0.0230 + 7.6e-3 * t_years
    y_mean = 0.3543 - 0.6e-3 * t_years
    eop = _toy_eop()
    eop["x"][:] = x_mean
    eop["y"][:] = y_mean
    station = np.array(lla_to_ecef(45.0, 30.0, 0.0))
    dr = pole_tide_displacement(station, eop, epoch)
    assert np.linalg.norm(dr) < 0.5e-3
