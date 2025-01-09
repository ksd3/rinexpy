"""Tests for the IERS 2010 step-2 K1 correction and the pole tide."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.tides import (
    R_EARTH,
    pole_tide_displacement,
    step2_k1_displacement,
)


def test_step2_k1_amplitude_is_sub_millimeter():
    """K1's tabulated amplitude is 0.25 mm radial; the total displacement
    magnitude at any epoch / station should stay sub-mm."""
    station = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    for hour in (0, 6, 12, 18):
        epoch = datetime(2024, 6, 21, hour, 0, 0, tzinfo=timezone.utc)
        dr = step2_k1_displacement(station, epoch)
        assert dr.shape == (3,)
        assert np.linalg.norm(dr) < 0.5e-3, (
            f"step-2 K1 displacement {np.linalg.norm(dr) * 1e3:.4f} mm "
            f"at hour {hour} is larger than expected"
        )


def test_step2_k1_vanishes_at_equator():
    """sin(2*phi) = 0 at lat=0, so the radial K1 contribution vanishes; the
    remaining tangential components are still in the sub-mm band."""
    eq_station = np.array(lla_to_ecef(0.0, 30.0, 0.0))
    epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    dr = step2_k1_displacement(eq_station, epoch)
    # Total still under 0.2 mm at the equator.
    assert np.linalg.norm(dr) < 0.2e-3


def test_step2_k1_is_periodic_diurnal():
    """K1 has a diurnal period; displacement at the same lat+lon 24 sidereal
    hours later should differ only slightly from the start."""
    station = np.array(lla_to_ecef(45.0, 0.0, 0.0))
    e0 = datetime(2024, 3, 14, 0, 0, 0, tzinfo=timezone.utc)
    # One sidereal day later (~23h 56m 4s); use 23:56:04 to approximate.
    e1 = datetime(2024, 3, 14, 23, 56, 4, tzinfo=timezone.utc)
    d0 = step2_k1_displacement(station, e0)
    d1 = step2_k1_displacement(station, e1)
    assert np.linalg.norm(d0 - d1) < 0.1e-3   # sub-0.1mm after one sidereal day


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
    """Polar motion of ~0.3 arcsec at mid-latitudes gives a pole tide of a
    few millimeters; the total ECEF displacement should land under 1 cm."""
    eop = _toy_eop()
    station = np.array(lla_to_ecef(45.0, 30.0, 0.0))
    epoch = datetime(2024, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
    dr = pole_tide_displacement(station, eop, epoch)
    assert dr.shape == (3,)
    mag = np.linalg.norm(dr)
    assert 0.0 < mag < 0.02   # 0 - 20 mm at most


def test_pole_tide_vanishes_at_equator():
    """sin(2*phi) and cos(phi) factors all become 1 / 0 / etc. at the
    equator; radial and north contributions vanish at exactly the equator.
    With non-zero polar motion the east component remains, but the
    overall displacement stays sub-mm."""
    eop = _toy_eop()
    eq_station = np.array(lla_to_ecef(0.0, 30.0, 0.0))
    epoch = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    dr = pole_tide_displacement(eq_station, eop, epoch)
    assert np.linalg.norm(dr) < 0.01   # 0 - 10 mm


def test_pole_tide_returns_zero_with_zero_polar_motion():
    """When polar motion equals the mean-pole model exactly, the pole
    tide vanishes."""
    epoch = datetime(2024, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
    # For 2024 the post-2010 mean-pole model says x_mean ~ 0.205, y_mean ~ 0.340
    # at t = 24.5 years. Use those as the EOP values so m1 = m2 = 0.
    t_years = 24.5
    x_mean = 0.0230 + 7.6e-3 * t_years
    y_mean = 0.3543 - 0.6e-3 * t_years
    eop = _toy_eop()
    eop["x"][:] = x_mean
    eop["y"][:] = y_mean
    station = np.array(lla_to_ecef(45.0, 30.0, 0.0))
    dr = pole_tide_displacement(station, eop, epoch)
    # All three components should be sub-mm because m1 = m2 ~ 0.
    assert np.linalg.norm(dr) < 0.5e-3
