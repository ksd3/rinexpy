"""Tests for asyncio, zarr, lazy, mmap, and error-message helpers."""

from __future__ import annotations

import asyncio

import pytest

import rinexpy as rp
from rinexpy._errors import LineCountingStream, format_parse_error
from rinexpy.asyncio import aload, aload_many

from .conftest import fixture


def test_aload():
    obs = asyncio.run(aload(fixture("demo.10o")))
    assert obs.attrs["rinextype"] == "obs"


def test_aload_many():
    files = [fixture("demo.10o"), fixture("demo.10n")]
    results = asyncio.run(aload_many(files))
    assert len(results) == 2
    assert results[0].attrs["rinextype"] == "obs"
    assert results[1].attrs["rinextype"] == "nav"


def test_aload_many_with_error_in_one_file(tmp_path):
    """A failing file produces an Exception in the result list, not a crash."""
    bad = tmp_path / "garbage.rnx"
    bad.write_text("this is not a rinex file\n")
    files = [fixture("demo.10o"), bad]
    results = asyncio.run(aload_many(files))
    assert results[0].attrs["rinextype"] == "obs"
    assert isinstance(results[1], Exception)


def test_to_zarr_roundtrip(tmp_path):
    pytest.importorskip("zarr")
    import xarray as xr

    from rinexpy.zarr_io import to_zarr

    obs = rp.load(fixture("demo.10o"))
    out = tmp_path / "store.zarr"
    to_zarr(obs, out)
    re = xr.open_zarr(out)
    assert re.equals(obs)


def test_load_lazy_returns_xarray():
    from rinexpy.lazy import load_lazy

    ds = load_lazy([fixture("demo.10o")])
    assert ds.time.size > 0


def test_line_counting_stream_increments():
    import io

    src = io.StringIO("a\nb\nc\n")
    s = LineCountingStream(src, name="x.txt")
    assert s.line_no == 0
    s.readline()
    assert s.line_no == 1
    list(s)
    assert s.line_no == 3


def test_format_parse_error_includes_line():
    msg = format_parse_error("foo.18o", 42, "BAD LINE HERE", "boom")
    assert "foo.18o:42" in msg
    assert "boom" in msg
    assert "BAD LINE HERE" in msg
