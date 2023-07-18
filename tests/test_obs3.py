"""Tests for the RINEX-3 OBS reader."""

from __future__ import annotations

from datetime import datetime

import pytest
import xarray as xr
from pytest import approx

import rinexpy as rp

from .conftest import fixture


def test_obs3_contents():
    obs = rp.load(fixture("obs3.01gage.10o"))
    expected = {"L1C", "L2P", "C1P", "C2P", "C1C", "S1C", "S1P", "S2P"}
    assert expected.issubset(set(obs.data_vars))


def test_obs3_meas_one():
    obs = rp.load(fixture("obs3.01gage.10o"), meas="C1C")
    assert "L1C" not in obs
    C1C = obs["C1C"]
    assert C1C.shape == (2, 14)
    assert (C1C.sel(sv="G07") == approx([22227666.76, 25342359.37])).all()


def test_obs3_meas_two_nonsequential():
    obs = rp.load(fixture("obs3.01gage.10o"), meas=["L1C", "S1C"])
    assert "L2P" not in obs
    L1C = obs["L1C"].dropna(dim="sv", how="all")
    assert L1C.shape == (2, 14)
    assert (L1C.sel(sv="G07") == approx([118767195.32608, 133174968.81808])).all()
    S1C = obs["S1C"].dropna(dim="sv", how="all")
    assert S1C.shape == (2, 4)
    assert (S1C.sel(sv="R23") == approx([39.0, 79.0])).all()


def test_obs3_meas_some_missing():
    obs = rp.load(fixture("obs3.01gage.10o"), meas=["S2P"])
    S2P = obs["S2P"].dropna(dim="sv", how="all")
    assert S2P.shape == (2, 10)
    assert (S2P.sel(sv="G13") == approx([40.0, 80.0])).all()
    assert "R23" not in S2P.sv


def test_obs3_meas_all_missing():
    obs = rp.load(fixture("obs3.01gage.10o"), meas="nonsense")
    assert len(obs.data_vars) == 0


def test_obs3_meas_wildcard():
    obs = rp.load(fixture("obs3.01gage.10o"), meas="C")
    assert "L1C" not in obs
    assert "C1P" in obs and "C2P" in obs and "C1C" in obs
    assert len(obs.data_vars) == 3


def test_obs3_junk_time():
    times = rp.gettime(fixture("junk_time_obs3.10o"))
    assert times.tolist() == [
        datetime(2010, 3, 5, 0, 0, 30),
        datetime(2010, 3, 5, 0, 1, 30),
    ]


def test_obs3_zip():
    fn = fixture("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
    obs = rp.load(fn)
    expected = [
        "E04", "E09", "E12", "E24",
        "G02", "G05", "G06", "G07", "G09", "G12", "G13", "G17", "G19", "G25", "G30",
        "R01", "R02", "R08", "R22", "R23", "R24",
        "S20", "S31", "S35", "S38",
    ]
    assert obs.sv.values.tolist() == expected
    times = rp.gettime(fn)
    assert times.tolist() == [
        datetime(2018, 5, 13, 1, 30),
        datetime(2018, 5, 13, 1, 30, 30),
        datetime(2018, 5, 13, 1, 31),
    ]


def test_obs3_bad_system():
    with pytest.raises(KeyError):
        rp.load(fixture("obs3.01gage.10o"), use="Z")


@pytest.mark.parametrize("use", ("G", ["G"]))
def test_obs3_one_system(use):
    pytest.importorskip("netCDF4")
    truth = xr.open_dataset(fixture("r3G.nc"), group="OBS")
    obs = rp.load(fixture("obs3.01gage.10o"), use=use)
    assert obs.equals(truth)


def test_obs3_multi_system():
    pytest.importorskip("netCDF4")
    truth = xr.open_dataset(fixture("r3GR.nc"), group="OBS")
    obs = rp.load(fixture("obs3.01gage.10o"), use=("G", "R"))
    assert obs.equals(truth)


def test_obs3_all_systems():
    pytest.importorskip("netCDF4")
    truth = rp.rinexobs(fixture("r3all.nc"), group="OBS")
    obs = rp.load(fixture("obs3.01gage.10o"))
    assert obs.equals(truth)


def test_obs3_all_indicators():
    pytest.importorskip("netCDF4")
    truth = rp.rinexobs(fixture("r3all_indicators.nc"), group="OBS")
    obs = rp.load(fixture("obs3.01gage.10o"), useindicators=True)
    assert obs.equals(truth)


@pytest.mark.parametrize(
    "fn,tname",
    [("obs3.01gage.10o", "GPS"), ("default_time_system3.10o", "GAL")],
)
def test_obs3_time_system(fn, tname):
    obs = rp.load(fixture(fn))
    assert obs.attrs["time_system"] == tname
