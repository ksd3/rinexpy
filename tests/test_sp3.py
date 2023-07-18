"""Tests for the SP3 ephemeris reader."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from rinexpy.sp3 import load_sp3

from .conftest import fixture


def test_sp3a():
    ds = load_sp3(fixture("example1.sp3a"))
    assert "position" in ds.data_vars
    assert "clock" in ds.data_vars
    assert "velocity" in ds.data_vars
    assert ds.sv.size > 0


def test_sp3a_gz():
    ds = load_sp3(fixture("example1.sp3a.gz"))
    plain = load_sp3(fixture("example1.sp3a"))
    np.testing.assert_array_equal(ds.position.values, plain.position.values)
    np.testing.assert_array_equal(ds.sv.values, plain.sv.values)


def test_sp3c():
    ds = load_sp3(fixture("igs19362.sp3c"))
    assert ds.attrs["coord_sys"] == "IGS14"
    assert ds.sv.size == 32
    # All times should be on the same day.
    assert ds.time.values[0].astype("datetime64[D]") == np.datetime64("2017-02-14")


def test_sp3d():
    ds = load_sp3(fixture("minimal.sp3d"))
    assert ds.sv.size > 0


def test_sp3_missing_first_line():
    with pytest.raises(ValueError):
        load_sp3(fixture("blank.sp3"))


def test_sp3_truncated():
    # Truncated file should still parse what it can without crashing.
    ds = load_sp3(fixture("truncated.sp3"))
    assert ds.sv.size > 0


def test_sp3_write(tmp_path):
    pytest.importorskip("netCDF4")
    out = tmp_path / "out.nc"
    load_sp3(fixture("example1.sp3a"), outfn=out)
    assert out.is_file()
    assert out.stat().st_size > 1000
