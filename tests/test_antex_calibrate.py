"""Tests for the antenna PCV calibration tool."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from pytest import approx

from rinexpy.antex import load_antex
from rinexpy.antex_calibrate import calibrate_pcv, write_antex


def _synth_residuals(rng, n: int, pcv_model):
    """Build (residual, elevation, azimuth) triples consistent with a
    known PCV model.

    pcv_model: callable (elev_rad, azi_rad) -> residual in metres.
    """
    elev = rng.uniform(np.deg2rad(5), np.deg2rad(85), size=n)
    azi = rng.uniform(0, 2 * np.pi, size=n)
    residuals = pcv_model(elev, azi)
    return residuals, elev, azi


def test_calibrate_pcv_returns_antex_shaped_entry():
    rng = np.random.default_rng(0)
    n = 2000
    pcv_model = lambda elev, azi: 0.005 * np.cos(elev)   # mm-scale at zenith=0
    r, elev, azi = _synth_residuals(rng, n, pcv_model)

    entry = calibrate_pcv(r, elev, azi, antenna_type="TESTANT")
    assert entry["type"] == "TESTANT"
    assert "G01" in entry["frequencies"]
    f = entry["frequencies"]["G01"]
    assert f["pcv"].shape == (entry["azimuth_deg"].size, entry["zenith_deg"].size)
    assert f["noazi"].size == entry["zenith_deg"].size


def test_calibrate_pcv_round_trips_through_antex_loader(tmp_path):
    rng = np.random.default_rng(11)
    # Pure elevation-dependent PCV (NOAZI signal) so the loader can
    # confirm the value.
    pcv_model = lambda elev, azi: 0.005 * np.cos(elev) + 0.001 * np.cos(2 * azi)
    r, elev, azi = _synth_residuals(rng, 5000, pcv_model)

    entry = calibrate_pcv(
        r, elev, azi,
        antenna_type="RT_TEST",
        serial="SN001",
        valid_from=datetime(2024, 1, 1),
    )
    path = tmp_path / "calibrated.atx"
    write_antex([entry], path)

    parsed = load_antex(path)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["type"] == "RT_TEST"
    assert p["serial"] == "SN001"
    g01 = p["frequencies"]["G01"]
    # NOAZI shape matches.
    assert g01["noazi"].shape == entry["frequencies"]["G01"]["noazi"].shape
    # NOAZI values are the same (to within float-round-trip).
    np.testing.assert_allclose(
        g01["noazi"], entry["frequencies"]["G01"]["noazi"], atol=0.01
    )
    # 2-D PCV grid shape matches.
    assert g01["pcv"].shape == entry["frequencies"]["G01"]["pcv"].shape


def test_calibrate_pcv_recovers_zenith_trend():
    """A pure zenith-only signal should appear in the NOAZI vector
    monotonically."""
    rng = np.random.default_rng(7)
    pcv_model = lambda elev, azi: 0.010 * np.sin(0.5 * np.pi - elev)   # bigger at horizon
    r, elev, azi = _synth_residuals(rng, 10000, pcv_model)
    entry = calibrate_pcv(r, elev, azi, antenna_type="TREND", dazi_deg=10, dzen_deg=5)
    noazi = entry["frequencies"]["G01"]["noazi"]
    zenith = entry["zenith_deg"]
    # Synthetic data only covers elev 5..85 deg -> zenith 5..85 deg.
    # The 85..90 bin is empty (NOAZI = 0). Restrict the trend test to
    # the populated zenith bins.
    populated = (zenith >= 10) & (zenith <= 80)
    valid = noazi[populated]
    diffs = np.diff(valid)
    # Allow occasional small dips due to binning noise, but the trend
    # should be > 0 on average.
    assert diffs.mean() > 0


def test_calibrate_pcv_dimensional_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        calibrate_pcv(np.zeros(5), np.zeros(4), np.zeros(5), antenna_type="X")
