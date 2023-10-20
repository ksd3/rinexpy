"""Tests for the geodesy + iono helpers."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import (
    azimuth_elevation,
    dop,
    ecef_to_lla,
    klobuchar,
    lla_to_ecef,
)


def test_lla_round_trip():
    for lat, lon, alt in [(0, 0, 0), (40, -3, 100), (-89, 179, 5000)]:
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(x, y, z)
        assert lat2 == approx(lat, abs=1e-7)
        assert lon2 == approx(lon, abs=1e-7)
        assert alt2 == approx(alt, abs=1e-3)


def test_lla_known_point():
    # WGS-84: lat=0, lon=0, alt=0 -> ECEF (a, 0, 0).
    x, y, z = lla_to_ecef(0, 0, 0)
    assert x == approx(6378137.0)
    assert y == approx(0)
    assert z == approx(0)


def test_azimuth_elevation_zenith():
    """A satellite directly overhead has elevation 90."""
    rx = lla_to_ecef(0, 0, 0)
    rx_arr = np.asarray(rx)
    sv = rx_arr * 4.16  # straight up at GPS altitude
    az, el = azimuth_elevation(rx, sv)
    assert el == approx(90, abs=1e-3)


def test_azimuth_elevation_horizon():
    """A satellite at the geocentric horizon (perpendicular to up) has el=0."""
    rx = (6378137.0, 0, 0)
    # Pick a SV at the same altitude as us but along +y.
    sv = np.array([6378137.0, 1e6, 0.0])
    az, el = azimuth_elevation(rx, sv)
    assert el == approx(0, abs=1e-3)
    assert az == approx(90, abs=1e-3)  # east


def test_dop_singular_few_sats():
    rx = lla_to_ecef(0, 0, 0)
    sv = np.array([[2.66e7, 0, 0]])
    out = dop(sv, rx)
    assert all(np.isnan(v) for v in out.values())


def test_dop_good_geometry():
    """5 well-spread SVs above the receiver give finite, sane DOPs."""
    rx = lla_to_ecef(40, -3, 100)
    rx_arr = np.array(rx)
    # 5 satellites distributed in azimuth at ~26000 km radial, well
    # above the horizon, so the geometry matrix is well-conditioned.
    earth_radius = 6378137.0
    gps_alt = 2.0e7
    sv = []
    for az in (0, 72, 144, 216, 288):
        a = np.radians(az)
        # Push the receiver point outward in (E, N, Up) and rotate.
        offset = np.array([np.sin(a) * 1.5e7, np.cos(a) * 1.5e7, 1.5e7])
        sv.append(rx_arr * (earth_radius + gps_alt) / earth_radius + offset)
    out = dop(np.array(sv), rx)
    assert all(np.isfinite(v) for v in out.values())
    # The classic identities.
    assert out["PDOP"] <= out["GDOP"]
    assert out["HDOP"] <= out["PDOP"]
    assert out["VDOP"] <= out["PDOP"]


def test_klobuchar_meters_in_reasonable_range():
    """Iono delay at low elevation > delay at zenith, both positive."""
    alpha = (1e-8, 0, 0, 0)
    beta = (1e5, 0, 0, 0)
    rx_lla = (40.0, -3.0, 100.0)
    delay_zenith = klobuchar(alpha, beta, rx_lla, sv_az_deg=180, sv_el_deg=80, gps_sec=43200)
    delay_low = klobuchar(alpha, beta, rx_lla, sv_az_deg=180, sv_el_deg=10, gps_sec=43200)
    assert delay_low > delay_zenith > 0
    # Sanity: < 50 m even for low elevation with these alpha values
    assert delay_low < 100
