"""Tests for stitch_sp3 (multi-day SP3 concatenation with dedup)."""

from __future__ import annotations

import numpy as np
import pytest

import rinexpy as rp
from rinexpy.sp3 import stitch_sp3


def test_stitch_single_file_returns_unchanged():
    """One file in -> identical-shape dataset out."""
    one = stitch_sp3("tests/data/igs19362.sp3c")
    direct = rp.load_sp3("tests/data/igs19362.sp3c")
    assert one.time.size == direct.time.size
    assert one.sv.size == direct.sv.size
    np.testing.assert_array_equal(one.time.values, direct.time.values)


def test_stitch_same_file_twice_dedups():
    """Passing the same SP3 file twice should still produce one timeline."""
    a = rp.load_sp3("tests/data/igs19362.sp3c")
    stitched = stitch_sp3("tests/data/igs19362.sp3c", "tests/data/igs19362.sp3c")
    # Duplicate epochs collapse to one each.
    assert stitched.time.size == a.time.size
    # No duplicate timestamps remain.
    times = stitched.time.values
    assert len(np.unique(times)) == len(times)


def test_stitch_orders_time_axis_chronologically():
    """The output time axis is sorted regardless of input order."""
    stitched = stitch_sp3(
        "tests/data/igs19362.sp3c", "tests/data/igs19362.sp3c"
    )
    times = stitched.time.values
    assert np.all(times[1:] >= times[:-1])


def test_stitch_preserves_position_values():
    """Values at common epochs survive the concat + dedup."""
    one = rp.load_sp3("tests/data/igs19362.sp3c")
    stitched = stitch_sp3("tests/data/igs19362.sp3c")
    t = one.time.values[5]
    sv = "G07"
    a = one.position.sel(time=t, sv=sv).values
    b = stitched.position.sel(time=t, sv=sv).values
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_stitch_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        stitch_sp3()
