"""Tests for ECEF <-> ENU conversion helpers."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import ecef_to_enu, enu_to_ecef, lla_to_ecef


def test_reference_point_maps_to_origin():
    """The reference point in ECEF maps to (0, 0, 0) ENU."""
    ref = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    enu = ecef_to_enu(ref, ref)
    np.testing.assert_allclose(enu, [0.0, 0.0, 0.0], atol=1e-6)


def test_point_straight_up_maps_to_pure_up():
    """A point 100 m above the reference along the local vertical maps to ENU (0, 0, 100)."""
    lat, lon, alt = 40.0, -3.0, 100.0
    ref = np.array(lla_to_ecef(lat, lon, alt))
    above = np.array(lla_to_ecef(lat, lon, alt + 100.0))
    enu = ecef_to_enu(above, ref)
    assert enu[0] == approx(0.0, abs=1e-3)
    assert enu[1] == approx(0.0, abs=1e-3)
    assert enu[2] == approx(100.0, abs=1e-3)


def test_point_to_the_east_lands_on_east_axis():
    """Moving north of the reference along a great-circle leaves only a
    small east bias from spherical-earth approximation."""
    lat, lon = 40.0, -3.0
    ref = np.array(lla_to_ecef(lat, lon, 100.0))
    # 0.001 degree of longitude east at lat=40 is roughly 85 m east.
    east_point = np.array(lla_to_ecef(lat, lon + 0.001, 100.0))
    enu = ecef_to_enu(east_point, ref)
    assert enu[0] > 0.0
    assert abs(enu[1]) < 0.1  # very small north component
    assert abs(enu[2]) < 0.1  # very small up component
    # 0.001 deg longitude at lat=40 = ~85 m horizontally.
    assert 80.0 < enu[0] < 90.0


def test_enu_round_trip_is_identity():
    """ENU -> ECEF -> ENU should return the original point."""
    ref = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    target = ref + np.array([12.5, -7.3, 4.1])
    enu = ecef_to_enu(target, ref)
    back = enu_to_ecef(enu, ref)
    np.testing.assert_allclose(back, target, atol=1e-6)


def test_batch_input_returns_batch_output():
    """A (n, 3) input should give a (n, 3) ENU array."""
    ref = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    pts = ref + np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [10, 20, 30]],
                          dtype=float)
    enu = ecef_to_enu(pts, ref)
    assert enu.shape == (4, 3)
    # Round-trip per row.
    back = enu_to_ecef(enu, ref)
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_ecef_to_enu_rejects_bad_shapes():
    ref = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        ecef_to_enu(np.array([1.0, 2.0]), ref)
    with pytest.raises(ValueError, match="shape"):
        ecef_to_enu(np.zeros((3, 2)), ref)
    with pytest.raises(ValueError, match="ref_ecef"):
        ecef_to_enu(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0]))


def test_enu_to_ecef_rejects_bad_shapes():
    ref = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        enu_to_ecef(np.array([1.0, 2.0]), ref)


def test_enu_axes_orthonormal_and_right_handed():
    """The implicit ENU basis vectors at any latitude/longitude form an
    orthonormal right-handed frame."""
    ref = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    e_axis = enu_to_ecef(np.array([1.0, 0.0, 0.0]), ref) - ref
    n_axis = enu_to_ecef(np.array([0.0, 1.0, 0.0]), ref) - ref
    u_axis = enu_to_ecef(np.array([0.0, 0.0, 1.0]), ref) - ref
    # Each axis is unit length.
    for v in (e_axis, n_axis, u_axis):
        assert np.linalg.norm(v) == approx(1.0, abs=1e-9)
    # Pairwise orthogonal.
    assert e_axis @ n_axis == approx(0.0, abs=1e-9)
    assert e_axis @ u_axis == approx(0.0, abs=1e-9)
    assert n_axis @ u_axis == approx(0.0, abs=1e-9)
    # Right-handed: E x N = U.
    cross = np.cross(e_axis, n_axis)
    np.testing.assert_allclose(cross, u_axis, atol=1e-9)
