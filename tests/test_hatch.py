"""Tests for the Hatch filter (carrier-smoothed code pseudorange)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.qc import hatch_filter

_C = 299_792_458.0
_F_L1 = 1575.42e6
_LAMBDA_L1 = _C / _F_L1


def test_first_epoch_passes_pr_through():
    """At the very first sample, the filter has no history and returns the raw PR."""
    pr = np.array([2.5e7])
    phi = np.array([1.0e8 * _LAMBDA_L1])
    out = hatch_filter(pr, phi)
    assert out[0] == approx(2.5e7)


def test_noiseless_inputs_return_unchanged():
    """Clean range + phase: smoothed PR equals the input."""
    n = 50
    rho = 2.5e7 + np.arange(n) * 100.0
    pr = rho.copy()
    phi = rho.copy()  # phase already in meters
    out = hatch_filter(pr, phi)
    np.testing.assert_allclose(out, rho)


def test_smoothing_reduces_code_noise():
    """Inject zero-mean code noise; smoothed RMS goes down with the window."""
    rng = np.random.default_rng(7)
    n = 500
    rho = 2.5e7 + np.arange(n) * 100.0
    pr = rho + rng.normal(0, 0.5, n)        # 0.5 m code noise
    phi = rho + rng.normal(0, 0.005, n)     # mm-level carrier noise
    smoothed = hatch_filter(pr, phi, window=100)
    # Look at the steady-state region (after window ramp-up).
    err_raw = np.std(pr[200:] - rho[200:])
    err_smoothed = np.std(smoothed[200:] - rho[200:])
    assert err_smoothed < err_raw / 3       # at least 3x noise reduction


def test_slip_mask_resets_filter():
    """A slip at epoch K snaps the smoothed PR back toward the raw value at K."""
    rng = np.random.default_rng(11)
    n = 200
    rho = 2.5e7 + np.arange(n) * 100.0
    pr = rho + rng.normal(0, 0.5, n)
    phi = rho + rng.normal(0, 0.005, n)
    # Inject a 10 m PHASE slip starting at epoch 100. Without a reset, the
    # filter will track the slip forever; with a reset, it re-initialises.
    phi[100:] += 10.0
    slips = np.zeros(n, dtype=bool)
    slips[100] = True
    out = hatch_filter(pr, phi, slips=slips)
    # At the reset epoch, smoothed == raw.
    assert out[100] == approx(pr[100])
    # After a few epochs of smoothing, the filter has tracked phase again
    # and stays bounded.
    assert np.abs(out[150] - rho[150]) < 1.0


def test_nan_epoch_resets_filter():
    """A NaN gap forces a fresh re-initialisation on the next valid epoch."""
    pr = np.array([100.0, 101.0, np.nan, 102.0, 103.0])
    phi = np.array([100.0, 101.0, np.nan, 102.0, 103.0])
    out = hatch_filter(pr, phi, window=10)
    assert np.isnan(out[2])
    # Epoch 3 is the first valid one after the gap; smoother starts over.
    assert out[3] == approx(pr[3])


def test_window_of_one_returns_raw():
    """window=1 means no smoothing; output is the raw pseudorange."""
    pr = np.array([1.0, 2.0, 3.0, 4.0])
    phi = np.array([0.0, 1.0, 2.0, 3.0])
    out = hatch_filter(pr, phi, window=1)
    np.testing.assert_allclose(out, pr)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="match shape"):
        hatch_filter(np.zeros(5), np.zeros(4))


def test_window_zero_raises():
    with pytest.raises(ValueError, match="window must be"):
        hatch_filter(np.zeros(5), np.zeros(5), window=0)
