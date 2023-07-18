"""Tests for header parsers and rinexinfo()."""

from __future__ import annotations

import pytest
from pytest import approx

from rinexpy.headers import (
    navheader2,
    navheader3,
    obsheader2,
    obsheader3,
    rinexheader,
    rinexinfo,
)

from .conftest import fixture


def test_rinexinfo_obs2():
    info = rinexinfo(fixture("demo.10o"))
    assert info["version"] == 2.11
    assert info["rinextype"] == "obs"
    assert info["systems"] == "M"


def test_rinexinfo_nav2():
    info = rinexinfo(fixture("demo.10n"))
    assert info["version"] == 2.11
    assert info["rinextype"] == "nav"


def test_rinexinfo_obs3():
    info = rinexinfo(fixture("obs3.01gage.10o"))
    assert info["version"] == 3.01
    assert info["rinextype"] == "obs"


def test_rinexinfo_sp3():
    info = rinexinfo(fixture("igs19362.sp3c"))
    assert info["rinextype"] == "sp3"
    assert info["version"] == "c"


def test_rinexinfo_nc(tmp_path):
    pytest.importorskip("netCDF4")
    info = rinexinfo(fixture("demo_nav3.17n.nc"))
    assert "nav" in info["rinextype"]


def test_obsheader2_position():
    hdr = obsheader2(fixture("demo.10o"))
    assert hdr["position"] == approx([4789028.4701, 176610.0133, 4195017.031])
    assert "fields" in hdr
    assert hdr["Nobs"] == len(hdr["fields"])


def test_obsheader2_meas_filter():
    hdr = obsheader2(fixture("demo.10o"), meas=["C1"])
    assert hdr["fields"] == ["C1"]


def test_obsheader3_per_system_fields():
    hdr = obsheader3(fixture("obs3.01gage.10o"))
    assert "G" in hdr["fields"]
    assert "L1C" in hdr["fields"]["G"]


def test_obsheader3_use_filter():
    hdr = obsheader3(fixture("obs3.01gage.10o"), use={"G"})
    assert set(hdr["fields"]) == {"G"}


def test_obsheader3_use_unknown():
    with pytest.raises(KeyError):
        obsheader3(fixture("obs3.01gage.10o"), use={"Z"})


def test_navheader2():
    hdr = navheader2(fixture("demo.10n"))
    assert hdr["filetype"] == "N"


def test_navheader3_iono_corr():
    hdr = navheader3(fixture("demo_nav3.17n"))
    assert hdr["IONOSPHERIC CORR"]["GPSA"] == approx(
        [1.1176e-08, -1.4901e-08, -5.9605e-08, 1.1921e-07]
    )
    assert hdr["TIME SYSTEM CORR"]["GPUT"] == approx(
        [-3.7252902985e-09, -1.065814104e-14, 61440, 1976]
    )


def test_rinexheader_dispatches_obs2():
    hdr = rinexheader(fixture("demo.10o"))
    assert "fields" in hdr  # OBS-specific


def test_rinexheader_dispatches_nav3():
    hdr = rinexheader(fixture("demo_nav3.17n"))
    assert "IONOSPHERIC CORR" in hdr  # NAV3-specific
