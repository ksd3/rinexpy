"""Tests for the RINEX-2 OBS reader."""

from __future__ import annotations

from datetime import datetime

import pytest
import xarray as xr
from pytest import approx

import rinexpy as rp

from .conftest import fixture


def test_obs2_fast_slow_equal():
    fn = fixture("minimal2.10o")
    fast = rp.load(fn, fast=True)
    slow = rp.load(fn, fast=False)
    assert fast.equals(slow)
    assert fast.fast_processing
    assert not slow.fast_processing


def test_obs2_meas_continuation():
    """OBS2 file with > 10 obs types."""
    obs = rp.load(fixture("ab430140.18o.zip"), verbose=True)
    expected = {
        "L1", "L2", "C1", "P2", "P1", "S1", "S2", "C2",
        "L5", "C5", "S5", "L6", "C6", "S6",
        "L7", "C7", "S7", "L8", "C8", "S8",
    }
    assert expected.issubset(set(obs.data_vars))
    assert rp.to_datetime(obs.time).size == 9


def test_obs2_meas_one():
    obs = rp.load(fixture("demo.10o"), meas="C1")
    assert "L1" not in obs
    assert obs.C1.shape == (2, 14)
    assert obs.C1.sel(sv="G07").values == approx([22227666.76, 25342359.37])


def test_obs2_meas_two_nonsequential():
    obs = rp.load(fixture("demo.10o"), meas=["L1", "S1"])
    assert "L2" not in obs
    L1 = obs["L1"]
    assert L1.shape == (2, 14)
    assert L1.sel(sv="G07").values == approx(
        [118767195.32608, 133174968.81808]
    )
    S1 = obs["S1"]
    assert (S1.sel(sv="R23") == approx([39.0, 79.0])).all()


def test_obs2_meas_some_missing_systems():
    obs = rp.load(fixture("demo.10o"), meas=["S2"])
    S2 = obs["S2"]
    assert S2.shape == (2, 10)
    assert (S2.sel(sv="G13") == approx([40.0, 80.0])).all()
    with pytest.raises(KeyError):
        S2.sel(sv="R23")


def test_obs2_meas_all_missing():
    obs = rp.load(fixture("demo.10o"), meas="nonsense")
    assert len(obs) == 0


def test_obs2_meas_wildcard():
    obs = rp.load(fixture("demo.10o"), meas="P")
    assert "L1" not in obs
    assert "P1" in obs and "P2" in obs
    assert len(obs.data_vars) == 2


def test_obs2_mangled_data():
    obs = rp.load(fixture("14601736.18o"))
    times = rp.to_datetime(obs.time)
    assert (
        times
        == (
            datetime(2018, 6, 22, 6, 17, 30),
            datetime(2018, 6, 22, 6, 17, 45),
            datetime(2018, 6, 22, 6, 18),
        )
    ).all()


def test_obs2_one_sv():
    obs = rp.load(fixture("rinex2onesat.10o"))
    assert len(obs.sv) == 1
    assert obs.sv.item() == "G13"


@pytest.mark.parametrize("use", (None, {"G", "R", "S"}))
def test_obs2_all_systems(use):
    pytest.importorskip("netCDF4")
    truth = xr.open_dataset(fixture("r2all.nc"), group="OBS")
    obs = rp.load(fixture("demo.10o"), use=use)
    assert obs.equals(truth)
    assert obs.position == approx([4789028.4701, 176610.0133, 4195017.031])


@pytest.mark.parametrize("use", ("G", ["G"]))
def test_obs2_one_system(use):
    pytest.importorskip("netCDF4")
    truth = xr.open_dataset(fixture("r2G.nc"), group="OBS")
    obs = rp.load(fixture("demo.10o"), use=use)
    assert obs.equals(truth)


def test_obs2_multi_system():
    pytest.importorskip("netCDF4")
    truth = xr.open_dataset(fixture("r2GR.nc"), group="OBS")
    obs = rp.load(fixture("demo.10o"), use=("G", "R"))
    assert obs.equals(truth)


def test_obs2_indicators():
    pytest.importorskip("netCDF4")
    obs = rp.load(fixture("demo.10o"), useindicators=True)
    truth = rp.rinexobs(fixture("r2all_indicators.nc"), group="OBS")
    assert obs.equals(truth)


def test_obs2_meas_indicators():
    pytest.importorskip("netCDF4")
    obs = rp.load(fixture("demo.10o"), meas="C1", useindicators=True)
    truth = rp.rinexobs(fixture("r2_C1_indicators.nc"), group="OBS")
    assert obs.equals(truth)


@pytest.mark.parametrize(
    "fn,tname",
    [("demo.10o", "GPS"), ("default_time_system2.10o", "GLO")],
)
def test_obs2_time_system(fn, tname):
    obs = rp.load(fixture(fn))
    assert obs.attrs["time_system"] == tname


def test_obs2_wrong_header_count():
    obs = rp.load(fixture("wrong_obs2_count.10o"))
    s2 = obs["S2"].dropna(dim="sv", how="all")
    assert s2.sel(sv="G31").item() == approx(63.0)
