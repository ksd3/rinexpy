"""Tests for the broadcast group-delay (TGD/BGD) correction helpers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from pytest import approx

from rinexpy.positioning import apply_tgd_correction, tgd_from_nav

_C = 299_792_458.0


def _make_nav(tgd_by_sv_and_time: dict, field: str = "TGD") -> xr.Dataset:
    """Build a minimal NAV dataset: (time, sv) with one named field."""
    svs = sorted(tgd_by_sv_and_time)
    all_times = sorted({t for vals in tgd_by_sv_and_time.values() for t in vals})
    grid = np.full((len(all_times), len(svs)), np.nan)
    for j, sv in enumerate(svs):
        for i, t in enumerate(all_times):
            grid[i, j] = tgd_by_sv_and_time[sv].get(t, np.nan)
    return xr.Dataset(
        {field: (("time", "sv"), grid)},
        coords={
            "time": np.array(all_times, dtype="datetime64[ns]"),
            "sv": svs,
        },
    )


def test_apply_tgd_subtracts_c_times_tgd():
    """Single SV, gamma=1: result is PR - c * TGD."""
    pr = np.array([2.5e7])
    tgd = {"G01": 1e-8}
    out = apply_tgd_correction(pr, ["G01"], tgd)
    assert out[0] == approx(2.5e7 - _C * 1e-8, abs=1e-6)


def test_apply_tgd_gamma_zero_is_passthrough():
    """gamma=0 (ionosphere-free combination): pseudoranges unchanged."""
    pr = np.array([2.5e7, 2.6e7])
    tgd = {"G01": 5e-9, "G02": 1e-8}
    out = apply_tgd_correction(pr, ["G01", "G02"], tgd, gamma=0.0)
    assert out[0] == pr[0]
    assert out[1] == pr[1]


def test_apply_tgd_gamma_l2_uses_squared_freq_ratio():
    """gamma=(f1/f2)**2 (L2 frequency): correction scales accordingly."""
    pr = np.array([2.5e7])
    tgd = {"G01": 1e-8}
    gamma_l2 = (1575.42e6 / 1227.60e6) ** 2
    out = apply_tgd_correction(pr, ["G01"], tgd, gamma=gamma_l2)
    assert out[0] == approx(2.5e7 - _C * gamma_l2 * 1e-8, abs=1e-6)


def test_apply_tgd_skips_missing_svs():
    """SVs not in the TGD map pass through unchanged."""
    pr = np.array([2.5e7, 2.6e7, 2.7e7])
    tgd = {"G01": 1e-8}  # only G01 has a value
    out = apply_tgd_correction(pr, ["G01", "G99", "R03"], tgd)
    assert out[0] == approx(2.5e7 - _C * 1e-8, abs=1e-6)
    assert out[1] == pr[1]
    assert out[2] == pr[2]


def test_apply_tgd_skips_nan_tgd():
    """A NaN TGD in the map is treated as missing, no correction."""
    pr = np.array([2.5e7])
    out = apply_tgd_correction(pr, ["G01"], {"G01": np.nan})
    assert out[0] == pr[0]


def test_apply_tgd_returns_copy():
    """Input is not mutated."""
    pr = np.array([2.5e7])
    pr_before = pr.copy()
    apply_tgd_correction(pr, ["G01"], {"G01": 1e-8})
    assert np.array_equal(pr, pr_before)


def test_tgd_from_nav_picks_latest_record():
    """Two records before the epoch: pick the most recent one."""
    nav = _make_nav(
        {
            "G01": {
                "2024-03-14T00:00:00": 5e-9,
                "2024-03-14T02:00:00": 6e-9,  # latest before query
                "2024-03-14T10:00:00": 7e-9,  # future
            }
        }
    )
    tgd = tgd_from_nav(nav, datetime(2024, 3, 14, 5, 0))
    assert tgd == {"G01": approx(6e-9)}


def test_tgd_from_nav_skips_sv_without_record_at_or_before_epoch():
    """If the only record is in the future, skip that SV."""
    nav = _make_nav({"G01": {"2024-03-14T10:00:00": 5e-9}})
    tgd = tgd_from_nav(nav, datetime(2024, 3, 14, 5, 0))
    assert "G01" not in tgd


def test_tgd_from_nav_drops_nan_values():
    """A NaN record at the query time is treated as missing."""
    nav = _make_nav({"G01": {"2024-03-14T00:00:00": float("nan")}})
    tgd = tgd_from_nav(nav, datetime(2024, 3, 14, 5, 0))
    assert "G01" not in tgd


def test_tgd_from_nav_missing_field_returns_empty():
    """If the NAV dataset has no TGD variable at all, return {}."""
    ds = xr.Dataset(
        {"OTHER": (("time", "sv"), np.zeros((1, 1)))},
        coords={"time": np.array(["2024-03-14T00:00:00"], dtype="datetime64[ns]"),
                "sv": ["G01"]},
    )
    assert tgd_from_nav(ds, datetime(2024, 3, 14, 5, 0)) == {}


def test_tgd_from_nav_galileo_bgd_field():
    """The same machinery works for Galileo BGD fields by passing field=."""
    nav = _make_nav(
        {"E01": {"2024-03-14T00:00:00": 2e-9}}, field="BGDe5a"
    )
    tgd = tgd_from_nav(nav, datetime(2024, 3, 14, 5, 0), field="BGDe5a")
    assert tgd == {"E01": approx(2e-9)}


def test_apply_tgd_with_nav_extracted_values_roundtrip():
    """End-to-end: pull TGD from a NAV, apply to pseudoranges."""
    nav = _make_nav({"G01": {"2024-03-14T00:00:00": 1e-8},
                     "G02": {"2024-03-14T00:00:00": 5e-9}})
    epoch = datetime(2024, 3, 14, 5, 0)
    pr = np.array([2.5e7, 2.6e7])
    sv = ["G01", "G02"]
    tgd = tgd_from_nav(nav, epoch)
    corrected = apply_tgd_correction(pr, sv, tgd)
    assert corrected[0] == approx(2.5e7 - _C * 1e-8, abs=1e-6)
    assert corrected[1] == approx(2.6e7 - _C * 5e-9, abs=1e-6)
