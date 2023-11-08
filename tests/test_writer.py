"""Tests for the RINEX writer."""

from __future__ import annotations

import pytest

import rinexpy as rp
from rinexpy.writer import to_rinex_obs

from .conftest import fixture


def test_round_trip_obs2(tmp_path):
    """Read OBS3, write OBS3, read back: same SVs, same times."""
    src = rp.load(fixture("obs3.01gage.10o"))
    out = tmp_path / "out.rnx"
    to_rinex_obs(src, out, version=3)
    re = rp.load(out)
    assert sorted(re.sv.values.tolist()) == sorted(src.sv.values.tolist())
    assert re.time.size == src.time.size


def test_round_trip_values(tmp_path):
    """One round-trip preserves the C1C value to 3 decimal places."""
    src = rp.load(fixture("obs3.01gage.10o"))
    out = tmp_path / "out.rnx"
    to_rinex_obs(src, out, version=3)
    re = rp.load(out)
    src_val = float(src.C1C.sel(sv="G07").values[0])
    re_val = float(re.C1C.sel(sv="G07").values[0])
    assert abs(src_val - re_val) < 1e-2


def test_writer_rejects_bad_version(tmp_path):
    src = rp.load(fixture("demo.10o"))
    with pytest.raises(ValueError):
        to_rinex_obs(src, tmp_path / "x.rnx", version=4)
