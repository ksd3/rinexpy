"""Tests for the Keplerian -> ECEF conversion."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import rinexpy as rp
from rinexpy.keplerian import keplerian2ecef

from .conftest import fixture


def test_keplerian_gps_sanity():
    """ECEF position magnitudes should be on the order of Earth radius+GPS altitude."""
    nav = rp.load(fixture("demo.10n"))
    sv = nav.sel(sv="G13").dropna(dim="time", how="all")
    if sv.time.size == 0:
        pytest.skip("no usable data in fixture for G13")
    X, Y, Z = keplerian2ecef(sv)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    # GPS orbit altitude is ~20200 km; total radius ~26600 km = 2.66e7 m.
    assert np.all(np.abs(r - 2.66e7) < 1e6)


def test_keplerian_unsupported_system():
    nav = rp.load(fixture("demo.10n"))
    sv = nav.copy()
    sv.attrs["svtype"] = ["Z"]
    with pytest.raises(ValueError, match="unsupported"):
        keplerian2ecef(sv)


def test_keplerian_constants():
    """The module-level GM and OMEGA_E constants match the WGS-84 / GPS-ICD."""
    from rinexpy.keplerian import _GM, _OMEGA_E

    assert approx(3.986004418e14) == _GM
    assert approx(7.2921151467e-5) == _OMEGA_E
