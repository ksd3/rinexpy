"""Tests for the GNSS-R reflectometry altimetry retrieval."""

from __future__ import annotations

import numpy as np
from pytest import approx

from rinexpy.gnssr import detrend_snr, snr_to_sea_height

_LAMBDA_L1 = 0.190293672798365


def _synth_snr(elev_rad: np.ndarray, height_m: float, *, amplitude: float = 2.0,
               noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Build a Larson-style synthetic SNR arc:

        SNR = trend(sin elev) + amplitude * cos(2 pi f sin elev)
    """
    rng = np.random.default_rng(seed)
    s = np.sin(elev_rad)
    trend = 45.0 - 30.0 * s + 10.0 * s ** 2  # smooth direct-signal envelope
    f = 2.0 * height_m / _LAMBDA_L1
    osc = amplitude * np.cos(2 * np.pi * f * s)
    eps = rng.normal(scale=noise, size=elev_rad.shape) if noise > 0 else 0.0
    return trend + osc + eps


def test_detrend_removes_polynomial_envelope():
    elev = np.deg2rad(np.linspace(5, 30, 200))
    snr = _synth_snr(elev, height_m=3.0, amplitude=2.0)
    residual = detrend_snr(snr, elev, order=4)
    # The residual is the oscillation; mean ~ 0 and std close to amp/sqrt(2).
    assert abs(residual.mean()) < 1e-3
    assert 0.7 < residual.std() < 2.5


def test_snr_to_sea_height_recovers_known_height():
    elev = np.deg2rad(np.linspace(5, 30, 400))
    true_h = 4.5
    snr = _synth_snr(elev, height_m=true_h, amplitude=3.0, noise=0.1)
    out = snr_to_sea_height(snr, elev, wavelength_m=_LAMBDA_L1, n_freqs=2048)
    assert out["height_m"] == approx(true_h, abs=0.1)


def test_snr_to_sea_height_short_arc_raises():
    import pytest
    elev = np.deg2rad(np.linspace(5, 10, 4))
    snr = _synth_snr(elev, height_m=2.0)
    with pytest.raises(ValueError):
        snr_to_sea_height(snr, elev, wavelength_m=_LAMBDA_L1)


def test_snr_to_sea_height_dimensional_check():
    import pytest
    elev = np.deg2rad(np.linspace(5, 30, 100))
    snr = _synth_snr(elev, height_m=2.0)
    with pytest.raises(ValueError):
        snr_to_sea_height(snr[:50], elev, wavelength_m=_LAMBDA_L1)
