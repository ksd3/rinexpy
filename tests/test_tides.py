"""Tests for the solid-earth tide displacement and the approximate
sun / moon ECEF helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.tides import (
    GM_EARTH,
    GM_MOON,
    GM_SUN,
    R_EARTH,
    moon_position_ecef,
    solid_earth_tide_displacement,
    sun_position_ecef,
)

_AU = 1.495978707e11


def test_sun_distance_close_to_one_au():
    """Approximate sun position should land near 1 AU from Earth."""
    epoch = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    pos = sun_position_ecef(epoch)
    r = np.linalg.norm(pos)
    # Earth-Sun distance varies between 0.983 AU (perihelion) and 1.017 AU.
    assert 0.97 * _AU < r < 1.04 * _AU, f"sun distance {r/_AU:.3f} AU"


def test_moon_distance_in_expected_band():
    """Approximate moon position should land at 350,000 - 410,000 km."""
    epoch = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    pos = moon_position_ecef(epoch)
    r = np.linalg.norm(pos)
    assert 3.4e8 < r < 4.2e8, f"moon distance {r/1e3:.0f} km"


def test_solid_tide_peak_radial_displacement_under_60cm():
    """Peak radial tide is < ~60 cm; smoke check that our model agrees."""
    # Place station at the equator under the moon. Use a synthetic moon
    # directly overhead at typical distance, no sun.
    station = np.array([R_EARTH, 0.0, 0.0])
    sun_far = np.array([1e30, 0.0, 0.0])  # so far the sun contribution is ~0
    moon_overhead = np.array([3.84e8, 0.0, 0.0])
    dr = solid_earth_tide_displacement(
        station, sun_ecef=sun_far, moon_ecef=moon_overhead
    )
    # Most of the displacement is radial (h2 * full Legendre).
    radial = float(np.dot(dr, station) / np.linalg.norm(station))
    assert 0.1 < radial < 0.5, f"moon-overhead radial tide {radial:.3f} m"


def test_solid_tide_zero_when_body_perpendicular():
    """When the body is at the station's local horizon, the radial
    contribution vanishes (cos_psi = 0 makes (3/2 cos^2 - 1/2) = -0.5,
    so it's actually negative). The horizontal piece is zero. Check
    that the displacement magnitude drops well below the peak."""
    station = np.array([R_EARTH, 0.0, 0.0])
    # Body in the +y direction: perpendicular to station r_hat (which is +x).
    moon = np.array([0.0, 3.84e8, 0.0])
    sun = np.array([0.0, _AU, 0.0])
    dr = solid_earth_tide_displacement(station, sun_ecef=sun, moon_ecef=moon)
    # Radial component (the dominant part) at cos_psi=0 is h2 * (-0.5) * factor
    # which is negative and smaller magnitude than the cos_psi=1 maximum.
    assert np.linalg.norm(dr) < 0.3


def test_solid_tide_uses_approximate_bodies_when_epoch_given():
    """If sun_ecef / moon_ecef aren't supplied, the function falls back
    to the approximate sun_position_ecef / moon_position_ecef helpers."""
    station = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    epoch = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    dr = solid_earth_tide_displacement(station, epoch=epoch)
    # Magnitude should be sub-meter on real geometries.
    assert 0.0 < np.linalg.norm(dr) < 1.0


def test_solid_tide_rejects_missing_inputs():
    station = np.array([R_EARTH, 0.0, 0.0])
    with pytest.raises(ValueError, match="Either"):
        solid_earth_tide_displacement(station)
    with pytest.raises(ValueError, match="Either"):
        solid_earth_tide_displacement(
            station, sun_ecef=np.array([1e11, 0, 0])
        )


def test_solid_tide_scales_with_mass_ratio():
    """Doubling GM_body in the formula via a body twice as massive (at the
    same distance) should double the tide contribution."""
    station = np.array([R_EARTH, 0.0, 0.0])
    moon_overhead = np.array([3.84e8, 0.0, 0.0])
    # Single-body solid earth tide: use a very far sun so its
    # contribution is negligible, vary the moon's effective GM by changing
    # distance only (factor depends on 1/R^3). Pulling the moon closer by
    # 2^(1/3) doubles the factor.
    sun_far = np.array([1e30, 0.0, 0.0])
    dr1 = solid_earth_tide_displacement(
        station, sun_ecef=sun_far, moon_ecef=moon_overhead
    )
    moon_closer = moon_overhead / 2.0 ** (1.0 / 3.0)
    dr2 = solid_earth_tide_displacement(
        station, sun_ecef=sun_far, moon_ecef=moon_closer
    )
    # Allow 5% tolerance because the position vector also changes the
    # geometry slightly when we scale the moon distance.
    radial1 = float(np.dot(dr1, station) / np.linalg.norm(station))
    radial2 = float(np.dot(dr2, station) / np.linalg.norm(station))
    assert radial2 == approx(2.0 * radial1, rel=0.05)


def test_constants_match_iers_2010():
    """Spot-check the IERS Conventions 2010 numerical values."""
    assert GM_EARTH == approx(3.986004418e14)
    assert GM_SUN == approx(1.32712442099e20)
    assert GM_MOON == approx(4.9048695e12)
    assert R_EARTH == 6378137.0
