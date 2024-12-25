"""Tests for the ECEF <-> ECI low-precision transforms."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import ecef_to_eci, eci_to_ecef


def test_round_trip_no_eop():
    """ECEF -> ECI -> ECEF returns the original point to numerical precision."""
    pos = np.array([4.0e6, 1.0e6, 3.0e6])
    epoch = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    eci = ecef_to_eci(pos, epoch)
    back = eci_to_ecef(eci, epoch)
    np.testing.assert_allclose(back, pos, atol=1e-6)


def test_round_trip_batched():
    """Batch (n, 3) input round-trips equally."""
    pts = np.array([
        [4.0e6, 1.0e6, 3.0e6],
        [1.0e6, 2.0e6, -5.0e6],
        [-3.0e6, 4.0e6, 0.0],
    ])
    epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    eci = ecef_to_eci(pts, epoch)
    assert eci.shape == (3, 3)
    back = eci_to_ecef(eci, epoch)
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_z_axis_invariant():
    """The Earth's polar axis (z) is shared between ECEF and ECI."""
    z_only = np.array([0.0, 0.0, 6371e3])
    epoch = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    eci = ecef_to_eci(z_only, epoch)
    assert eci[0] == approx(0.0, abs=1e-6)
    assert eci[1] == approx(0.0, abs=1e-6)
    assert eci[2] == approx(z_only[2], abs=1e-6)


def test_rotation_at_unique_epoch():
    """The ECEF x-axis rotates relative to ECI by GMST at the chosen epoch.

    At a known epoch the longitude of the Greenwich meridian in ECI is GMST.
    Using J2000.0 epoch (2000-01-01 12:00 UT1), GMST is well-defined
    (~18h 41m 50.5s = ~280 degrees), so the ECEF +X direction projected
    to ECI lands close to longitude 280 degrees, NOT along the ECI +X.
    """
    import math
    x_hat = np.array([1.0, 0.0, 0.0])
    epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    eci = ecef_to_eci(x_hat, epoch)
    angle_deg = math.degrees(math.atan2(eci[1], eci[0]))
    # GMST at J2000.0 noon UT1 is roughly 18h 41m 50.5s, which is
    # 280.4606 degrees.
    assert 270.0 < (angle_deg % 360.0) < 290.0


def test_polar_motion_correction_is_small():
    """Applying polar motion via EOP changes the result by at most a meter or
    so on Earth's surface (polar motion is sub-arcsec)."""
    import math
    import xarray as xr
    from rinexpy.eop import load_eop
    # Build a tiny in-memory EOP-like dataset.
    times = np.array(
        ["2024-06-20T00:00:00", "2024-06-21T00:00:00", "2024-06-22T00:00:00"],
        dtype="datetime64[ns]",
    )
    eop = xr.Dataset(
        {
            "x": (("time",), [0.1, 0.1, 0.1]),    # arcseconds
            "y": (("time",), [0.2, 0.2, 0.2]),
            "ut1_utc": (("time",), [-0.05, -0.05, -0.05]),
            "lod": (("time",), [0.0, 0.0, 0.0]),
            "dx": (("time",), [0.0, 0.0, 0.0]),
            "dy": (("time",), [0.0, 0.0, 0.0]),
            "x_err": (("time",), [0.0, 0.0, 0.0]),
            "y_err": (("time",), [0.0, 0.0, 0.0]),
            "ut1_utc_err": (("time",), [0.0, 0.0, 0.0]),
            "lod_err": (("time",), [0.0, 0.0, 0.0]),
            "dx_err": (("time",), [0.0, 0.0, 0.0]),
            "dy_err": (("time",), [0.0, 0.0, 0.0]),
        },
        coords={"time": times},
    )
    pos = np.array([6.378e6, 0.0, 0.0])
    epoch = datetime(2024, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
    no_eop = ecef_to_eci(pos, epoch)
    with_eop = ecef_to_eci(pos, epoch, eop=eop)
    diff = np.linalg.norm(with_eop - no_eop)
    # x_p=0.1", y_p=0.2", at radius 6378 km, expect a few meters of
    # difference. UT1-UTC also contributes via GMST.
    assert 0.5 < diff < 100.0, f"polar-motion + UT1 diff {diff:.3f} m"


def test_ecef_to_eci_rejects_bad_shape():
    epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match=r"must be \(3,\)"):
        ecef_to_eci(np.array([1.0, 2.0]), epoch)


def test_eci_to_ecef_rejects_bad_shape():
    epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match=r"must be \(3,\)"):
        eci_to_ecef(np.zeros((3, 2)), epoch)
