"""Tests for the MP1/MP2 multipath metrics."""

from __future__ import annotations

import numpy as np
from pytest import approx

from rinexpy.qc import mp1, mp2, multipath_rms

_C = 299_792_458.0
_F_L1 = 1575.42e6
_F_L2 = 1227.60e6
_LAMBDA_L1 = _C / _F_L1
_LAMBDA_L2 = _C / _F_L2


def _make_synthetic(n=200, *, code_rms=0.5, phase_rms=0.005, seed=0):
    """Synthetic dual-freq obs with shared geometry, constant ambiguity, no iono."""
    rng = np.random.default_rng(seed)
    rho = 2.5e7 + np.arange(n) * 100.0
    n1_m = 1000.0 * _LAMBDA_L1
    n2_m = 1500.0 * _LAMBDA_L2
    p1 = rho + rng.normal(0, code_rms, n)
    p2 = rho + rng.normal(0, code_rms, n)
    l1 = rho + n1_m + rng.normal(0, phase_rms, n)
    l2 = rho + n2_m + rng.normal(0, phase_rms, n)
    return p1, p2, l1, l2


def test_mp1_cancels_geometry():
    """Noiseless inputs: MP1 series is a constant (the carrier ambiguity combo)."""
    p1, _, l1, l2 = _make_synthetic(code_rms=0, phase_rms=0)
    series = mp1(p1, l1, l2)
    assert series.std() < 1e-6


def test_mp2_cancels_geometry():
    """Same for MP2."""
    _, p2, l1, l2 = _make_synthetic(code_rms=0, phase_rms=0)
    series = mp2(p2, l1, l2)
    assert series.std() < 1e-6


def test_mp1_picks_up_code_noise():
    """0.5 m code noise => MP1 std picks up roughly the same scale."""
    p1, _, l1, l2 = _make_synthetic(code_rms=0.5, phase_rms=0.005, seed=42)
    s = np.std(mp1(p1, l1, l2))
    assert 0.3 < s < 1.0


def test_multipath_rms_no_slips():
    """No slip mask: RMS equals std around the mean."""
    p1, _, l1, l2 = _make_synthetic(code_rms=0.5, seed=42)
    series = mp1(p1, l1, l2)
    rms = multipath_rms(series)
    expected = float(np.std(series - series.mean()))
    assert rms == approx(expected, rel=1e-6)


def test_multipath_rms_splits_on_slips():
    """With slips, RMS is averaged over per-arc detrended series."""
    n = 300
    p1, _, l1, l2 = _make_synthetic(n=n, code_rms=0.5, seed=7)
    series = mp1(p1, l1, l2)
    slips = np.zeros(n, dtype=bool)
    slips[100] = True
    slips[200] = True
    rms = multipath_rms(series, slips=slips)
    # The per-arc RMS should be close to the full-series RMS for a
    # stationary noise series (no per-arc bias jumps).
    assert 0.3 < rms < 1.0


def test_multipath_rms_handles_nan():
    """NaN epochs are dropped, not propagated."""
    s = np.array([1.0, np.nan, 2.0, 3.0, np.nan])
    rms = multipath_rms(s)
    expected = float(np.std(np.array([1.0, 2.0, 3.0]) - 2.0))
    assert rms == approx(expected, abs=1e-9)


def test_multipath_rms_returns_nan_on_empty():
    """All-NaN input gives NaN."""
    s = np.array([np.nan, np.nan])
    assert np.isnan(multipath_rms(s))


def test_mp_formulae_match_expected_constants():
    """Spot-check the GPS L1/L2 weight constants vs the textbook values."""
    p1, p2, l1, l2 = _make_synthetic(n=1, code_rms=0, phase_rms=0)
    # Pick concrete inputs and compute mp1 by hand.
    p1[0] = 100.0
    l1[0] = 50.0
    l2[0] = 40.0
    alpha = (_F_L1 / _F_L2) ** 2
    k = 2.0 / (alpha - 1.0)
    expected_mp1 = p1[0] - (1 + k) * l1[0] + k * l2[0]
    assert mp1(p1, l1, l2)[0] == approx(expected_mp1, abs=1e-9)
    k2 = 2.0 * alpha / (alpha - 1.0)
    expected_mp2 = p2[0] - k2 * l1[0] + (k2 - 1) * l2[0]
    assert mp2(p2, l1, l2)[0] == approx(expected_mp2, abs=1e-9)
