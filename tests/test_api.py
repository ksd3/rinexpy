"""Tests for the high-level public API in :mod:`rinexpy.api`."""

from __future__ import annotations

import importlib.resources as ir
from datetime import datetime

import numpy as np
import pytest

import rinexpy as rp

from .conftest import fixture


def test_load_obs2():
    obs = rp.load(fixture("demo.10o"))
    assert obs.attrs["rinextype"] == "obs"
    assert "C1" in obs.data_vars


def test_load_nav2():
    nav = rp.load(fixture("demo.10n"))
    assert nav.attrs["rinextype"] == "nav"


def test_load_nav3():
    nav = rp.load(fixture("demo_nav3.17n"))
    assert nav.attrs["rinextype"] == "nav"


def test_load_sp3_via_dispatch():
    ds = rp.load(fixture("example1.sp3a"))
    assert "position" in ds.data_vars


def test_load_nc_obs(tmp_path):
    pytest.importorskip("netCDF4")
    out = tmp_path / "out.nc"
    rp.load(fixture("demo.10o"), out=out)
    re = rp.load(out)
    assert "C1" in re.data_vars


def test_load_nc_nav(tmp_path):
    pytest.importorskip("netCDF4")
    out = tmp_path / "out.nc"
    rp.load(fixture("demo.10n"), out=out)
    re = rp.load(out)
    assert "SVclockBias" in re.data_vars


def test_load_nc_both(tmp_path):
    pytest.importorskip("netCDF4")
    out = tmp_path / "out.nc"
    rp.load(fixture("demo.10o"), out=out)
    rp.load(fixture("demo.10n"), out=out)
    both = rp.load(out)
    assert isinstance(both, dict)
    assert "obs" in both and "nav" in both


def test_load_tlim_strings():
    nav = rp.load(
        fixture("CEDA00USA_R_20182100000_01D_MN.rnx.gz"),
        tlim=("2018-07-29T23", "2018-07-29T23:30"),
    )
    times = rp.to_datetime(nav.time)
    assert times == datetime(2018, 7, 29, 23)


def test_load_invalid_tlim_order():
    with pytest.raises(ValueError):
        rp.load(fixture("demo.10o"), tlim=("2018-01-02", "2018-01-01"))


def test_batch_convert(tmp_path):
    pytest.importorskip("netCDF4")
    src = fixture("demo.10o")
    out = tmp_path
    written = rp.batch_convert(src.parent, src.name, out)
    assert len(written) == 1


def test_gettime_obs2():
    times = rp.gettime(fixture("demo.10o"))
    assert times.size == 2


def test_gettime_nav3():
    times = rp.gettime(fixture("demo_nav3.17n"))
    assert times.size >= 1


def test_rinexheader_dispatch():
    hdr = rp.rinexheader(fixture("demo.10o"))
    assert hdr["rinextype"] == "obs"


def test_to_datetime():
    obs = rp.load(fixture("demo.10o"))
    t = rp.to_datetime(obs.time)
    assert t.size == 2


def test_importlib_resources_compat():
    """Mimic georinex test pattern using importlib.resources."""
    fn = ir.files(f"{__package__}.data") / "demo.10o"
    obs = rp.load(fn)
    assert obs.C1.size > 0
