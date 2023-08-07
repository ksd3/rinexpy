"""Tests for the RINEX-2 NAV reader."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from pytest import approx

import rinexpy as rp

from .conftest import fixture


def test_navtime2():
    times = rp.gettime(fixture("ab422100.18n"))
    assert times[0] == datetime(2018, 7, 29, 1, 59, 44)
    assert times[-1] == datetime(2018, 7, 30)


def test_nav2_data():
    nav = rp.load(fixture("ab422100.18n"))
    nav0 = nav.sel(time="2018-07-29T03:59:44").dropna(dim="sv", how="all")
    assert nav0.sv.values.tolist() == ["G18", "G20", "G24", "G27"]

    g20 = nav0.sel(sv="G20")
    expected = [
        5.1321554929e-4,
        6.821210263e-13,
        0.0,
        11,
        -74.125,
        4.944134514e-09,
        0.736990015,
        -3.810971975327e-06,
        4.055858473293e-03,
        1.130439341068e-5,
        5.153679727554e3,
        14384,
        -2.980232238770e-8,
        -2.942741,
        -5.587935447693e-8,
        9.291603197140e-01,
        144.8125,
        2.063514928857,
        -8.198555788471e-09,
        2.935836575092e-10,
        1,
        2012,
        0.0,
        2.0,
        0.0,
        -8.381903171539e-09,
        11,
        9456,
        4,
    ]
    assert g20.to_array().values == approx(expected)


def test_nav2_galileo():
    nav = rp.load(fixture("ceda2100.18e"))
    e18 = nav.sel(sv="E18").dropna(dim="time", how="all")
    assert rp.to_datetime(e18.time) == datetime(2018, 7, 29, 12, 40)


def test_nav2_mangled():
    nav = rp.load(fixture("14601736.18n"))
    times = rp.to_datetime(nav.time)
    assert times == datetime(2018, 6, 22, 8)


def test_nav2_glonass_units():
    """GLONASS positions/velocities should be reported in meters, not km."""
    nav = rp.load(fixture("p1462100.18g"))
    # All position values should have magnitudes typical of meters (1e6+).
    nonzero = nav.X.values[~np.isnan(nav.X.values)]
    if nonzero.size:
        assert np.median(np.abs(nonzero)) > 1e6


def test_nav2_tlim_past_eof():
    nav = rp.load(
        fixture("p1462100.18g"),
        tlim=("2018-07-29T23:45", "2018-07-30"),
    )
    times = rp.to_datetime(nav.time)
    assert times == datetime(2018, 7, 29, 23, 45)


def test_nav2_ionospheric_correction():
    nav = rp.load(fixture("14601736.18n"))
    assert nav.attrs["ionospheric_corr_GPS"] == approx(
        [
            0.4657e-08,
            0.1490e-07,
            -0.5960e-07,
            -0.1192e-06,
            0.8192e05,
            0.9830e05,
            -0.6554e05,
            -0.5243e06,
        ]
    )


@pytest.mark.needs_netcdf
def test_nav2_vs_reference_nc():
    pytest.importorskip("netCDF4")
    truth = xr.open_dataset(fixture("r2all.nc"), group="NAV")
    nav = rp.load(fixture("demo.10n"))
    assert nav.equals(truth)
