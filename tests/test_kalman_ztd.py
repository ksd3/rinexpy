"""Tests for the ZWD-augmented static PPP filter."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.kalman_ztd import StaticPPPFilterZTD

C = 299_792_458.0
F_L1 = 1575.42e6
F_L2 = 1227.60e6
LAMBDA_L1 = C / F_L1


def _sky() -> np.ndarray:
    return np.array([
        [ 1.5e7,  0.0,    2.0e7],
        [-1.5e7,  0.5e7,  2.0e7],
        [ 0.5e7,  1.5e7,  1.8e7],
        [-0.5e7, -1.5e7,  1.5e7],
        [ 2.0e7, -0.5e7,  1.0e7],
        [-2.0e7,  1.0e7,  0.8e7],
        [ 1.0e7, -1.7e7,  1.2e7],
        [-1.0e7,  0.0,    2.2e7],
    ])


def _wet_mapping(sv, rx_ecef):
    """1/sin(el) wet mapping factor per SV (simple but sufficient)."""
    diff = sv - rx_ecef
    rho = np.linalg.norm(diff, axis=1)
    cos_zen = -np.dot(diff, rx_ecef / np.linalg.norm(rx_ecef)) / rho
    # Cap so the test doesn't blow up at horizons.
    return np.clip(1.0 / np.sqrt(np.maximum(1.0 - cos_zen ** 2, 1e-6)), 1.0, 10.0)


def test_initial_state():
    f = StaticPPPFilterZTD(n_sv=8, initial_position=(0.0, 0.0, 0.0),
                            initial_zwd_m=0.15)
    assert f.position == (0.0, 0.0, 0.0)
    assert f.zwd_m == pytest.approx(0.15)
    assert f.zwd_sigma_m > 0
    assert f.ambiguities_m.shape == (8,)


def test_predict_grows_zwd_variance():
    f = StaticPPPFilterZTD(n_sv=4, initial_position=(0.0, 0.0, 0.0),
                            sigma_zwd_rate_m_per_sqrt_hr=0.01)
    v0 = f.P[4, 4]
    f.predict(dt=3600.0)
    v1 = f.P[4, 4]
    # Variance grows by sigma^2 * dt = 0.01^2 * (1/3600) * 3600 = 1e-4.
    assert v1 > v0
    assert v1 - v0 == pytest.approx(0.01 ** 2, rel=1e-9)


def test_zwd_recovers_from_observations():
    """Inject a known ZWD truth into synthetic observations; the filter
    should converge to it within a few epochs.

    Setup: receiver at the equator with a known position. True ZWD =
    0.12 m. Observations have no ambiguity (pure code, no phase) so
    every update is unambiguous about geometry vs ZWD.
    """
    truth_pos = np.array([6_378_137.0, 0.0, 0.0])
    truth_zwd = 0.12
    sv = _sky()
    mw = _wet_mapping(sv, truth_pos)
    rho_truth = np.linalg.norm(sv - truth_pos, axis=1)
    pr_truth = rho_truth + mw * truth_zwd   # clock = 0, no ambiguity for code

    f = StaticPPPFilterZTD(
        n_sv=sv.shape[0], initial_position=tuple(truth_pos),
        initial_zwd_m=0.0,
        sigma_position_init=1e-3,   # pin position tight (already known)
        sigma_zwd_init=1.0,         # let ZWD float
        sigma_clock_init=1e-3,      # pin clock tight (it's zero)
        sigma_code=0.1,
    )
    for _ in range(20):
        # NaN phase => no phase update => no ambiguity coupling.
        ph = np.full(sv.shape[0], np.nan)
        f.update(sv, np.zeros(sv.shape[0]), pr_truth, ph, wet_mapping=mw)
    # ZWD should converge to ~truth.
    assert abs(f.zwd_m - truth_zwd) < 0.01


def test_position_holds_with_phase_and_zwd_updates():
    """Full code+phase observations with known truth ZWD; position
    stays at truth and ZWD converges."""
    truth_pos = np.array([6_378_137.0, 0.0, 0.0])
    truth_zwd = 0.15
    sv = _sky()
    mw = _wet_mapping(sv, truth_pos)
    rho_truth = np.linalg.norm(sv - truth_pos, axis=1)
    true_amb = np.array([100., -50., 200., 75., -25., 0., 150., -100.])
    pr = rho_truth + mw * truth_zwd
    ph = rho_truth + mw * truth_zwd + true_amb

    f = StaticPPPFilterZTD(
        n_sv=sv.shape[0], initial_position=tuple(truth_pos),
        initial_zwd_m=0.0,
        sigma_position_init=0.1,
        sigma_zwd_init=1.0,
        sigma_clock_init=1.0,
    )
    for _ in range(30):
        f.update(sv, np.zeros(sv.shape[0]), pr, ph, wet_mapping=mw)
    assert abs(f.zwd_m - truth_zwd) < 0.02
    err = np.linalg.norm(np.array(f.position) - truth_pos)
    assert err < 0.1


def test_shape_validation_raises():
    f = StaticPPPFilterZTD(n_sv=4, initial_position=(0.0, 0.0, 0.0))
    sv = _sky()[:4]
    with pytest.raises(ValueError, match="shape mismatch"):
        f.update(sv, np.zeros(4), np.zeros(4), np.zeros(4),
                 wet_mapping=np.zeros(3))


def test_predict_negative_dt_rejected():
    f = StaticPPPFilterZTD(n_sv=4, initial_position=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=">= 0"):
        f.predict(dt=-1.0)
