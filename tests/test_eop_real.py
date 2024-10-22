"""Real-file integration tests for the IERS EOP C04 reader."""

from __future__ import annotations

import shutil
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from rinexpy.eop import interp_eop, load_eop

_CACHE = Path("/tmp/igs_real_cache")
_URL = "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04_IAU2000.62-now"


def _fetch_eop() -> Path:
    path = _CACHE / "eopc04.txt"
    if path.exists() and path.stat().st_size > 100_000:
        return path
    _CACHE.mkdir(exist_ok=True)
    try:
        with urllib.request.urlopen(_URL, timeout=60) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception:
        if path.exists():
            path.unlink()
        pytest.skip(f"Cannot reach {_URL}; skip real EOP test")
    if path.stat().st_size < 100_000:
        pytest.skip("Downloaded EOP file is suspiciously small")
    return path


@pytest.fixture(scope="module")
def eop_real():
    return load_eop(str(_fetch_eop()))


def test_eop_spans_decades(eop_real):
    """The C04 series starts in 1962; we should get tens of thousands of rows."""
    assert isinstance(eop_real, xr.Dataset)
    assert eop_real.time.size > 20_000
    first = eop_real.time.values[0]
    assert str(first).startswith("1962")


def test_eop_values_in_physical_band(eop_real):
    """Polar motion ~ 1 arcsec; UT1-UTC < 1 s; LOD a few ms."""
    assert np.all(np.abs(eop_real["x"].values) < 1.0)
    assert np.all(np.abs(eop_real["y"].values) < 1.0)
    assert np.all(np.abs(eop_real["ut1_utc"].values) < 1.5)
    # LOD is typically -1 to +5 ms (we report in seconds).
    assert np.all(np.abs(eop_real["lod"].values) < 0.01)


def test_eop_interp_matches_known_date(eop_real):
    """Interpolating at a date present in the series returns that row exactly."""
    # Pick a mid-table date.
    idx = eop_real.time.size // 2
    t = eop_real.time.values[idx]
    out = interp_eop(eop_real, t)
    assert out["x"] == pytest.approx(float(eop_real["x"].isel(time=idx).values))
    assert out["y"] == pytest.approx(float(eop_real["y"].isel(time=idx).values))
    assert out["ut1_utc"] == pytest.approx(
        float(eop_real["ut1_utc"].isel(time=idx).values)
    )


def test_eop_interp_between_two_days_is_a_mean(eop_real):
    """The C04 series is daily; interpolating at 12:00 UTC should give the
    arithmetic mean of the two bracketing days (within machine epsilon)."""
    t0 = eop_real.time.values[100]
    t1 = eop_real.time.values[101]
    mid = t0 + (t1 - t0) // 2
    mid_out = interp_eop(eop_real, mid)
    avg_x = 0.5 * (
        float(eop_real["x"].isel(time=100).values)
        + float(eop_real["x"].isel(time=101).values)
    )
    assert mid_out["x"] == pytest.approx(avg_x, rel=1e-6)


def test_eop_rejects_out_of_range_query(eop_real):
    """A query before 1962 (or far in the future) raises."""
    with pytest.raises(ValueError, match="outside"):
        interp_eop(eop_real, datetime(1900, 1, 1))
    with pytest.raises(ValueError, match="outside"):
        interp_eop(eop_real, datetime(2999, 1, 1))


def test_eop_known_leap_second_jump():
    """UT1-UTC should jump by ~1 second at the 2017-01-01 leap second.

    This isn't a smooth function: UT1 follows Earth rotation while UTC
    adds a leap second roughly every 1-2 years.
    """
    eop = load_eop(str(_fetch_eop()))
    # Find the rows just before and just after 2017-01-01.
    times = eop.time.values
    pre = np.searchsorted(times, np.datetime64("2016-12-31", "ns"))
    post = np.searchsorted(times, np.datetime64("2017-01-01", "ns"))
    if pre >= eop.time.size or post >= eop.time.size:
        pytest.skip("EOP file doesn't cover 2017-01-01")
    diff = float(eop["ut1_utc"].isel(time=post).values) - float(
        eop["ut1_utc"].isel(time=pre).values
    )
    # At a positive leap second, UTC is retarded by 1 s (the 23:59:60 step),
    # which means for a given UT1 the (UT1 - UTC) difference INCREASES by
    # ~1 s. Pre-leap UT1-UTC is typically -0.4 s; post-leap is ~+0.6 s.
    assert diff > 0.9, f"UT1-UTC jump across 2017-01-01 leap was {diff:+.3f} s"
