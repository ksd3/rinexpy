"""Tests for the 16-state tightly-coupled GNSS/INS EKF."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.imu_tight import TightINS16


def _sv_set() -> np.ndarray:
    """5 GPS satellites in roughly GPS-orbit geometry."""
    return np.array([
        [ 1.5e7,  0.0e7,  2.0e7],
        [-1.5e7,  0.5e7,  2.0e7],
        [ 0.5e7,  1.5e7,  1.8e7],
        [-0.5e7, -1.5e7,  1.5e7],
        [ 2.0e7, -0.5e7,  1.0e7],
    ])


def test_predict_does_not_modify_clock_when_no_drift():
    f = TightINS16(sigma_clock_drift_m_per_s=0.0)
    f.clock_bias_m = 42.0
    for _ in range(100):
        f.predict(np.zeros(3), np.zeros(3), dt=0.01)
    # No drift -> bias unchanged.
    assert f.clock_bias_m == pytest.approx(42.0)


def test_predict_inflates_clock_variance_with_drift():
    f = TightINS16(sigma_clock_drift_m_per_s=1.0)
    p0 = f.P[15, 15]
    for _ in range(100):
        f.predict(np.zeros(3), np.zeros(3), dt=0.01)
    p1 = f.P[15, 15]
    assert p1 > p0


def test_pseudorange_update_pulls_position():
    """One-shot batch update with 5 SVs from a known truth recovers
    position to the noise floor."""
    truth_pos = np.array([6_378_137.0, 0.0, 0.0])
    truth_clock = 12.34   # meters
    sv = _sv_set()
    rho_truth = np.linalg.norm(sv - truth_pos, axis=1) + truth_clock

    f = TightINS16()
    f.position = truth_pos + np.array([100.0, 50.0, -30.0])   # ~120 m off
    f.P = np.eye(16) * 100.0
    f.P[6:9, 6:9] = np.eye(3) * 1e-6   # pin attitude tight (irrelevant here)
    # Do a few iterated updates (linearization improves with each step).
    for _ in range(8):
        f.update_pseudoranges(sv, rho_truth, sigma_pr=0.1)
    np.testing.assert_allclose(f.position, truth_pos, atol=0.05)
    assert f.clock_bias_m == pytest.approx(truth_clock, abs=0.05)


def test_pseudorange_update_returns_residuals_shape():
    f = TightINS16()
    sv = _sv_set()
    pr = np.linalg.norm(sv - f.position, axis=1) + f.clock_bias_m
    out = f.update_pseudoranges(sv, pr, sigma_pr=1.0)
    assert out["n_obs"] == sv.shape[0]
    assert out["residuals"].shape == (sv.shape[0],)


def test_pseudorange_update_rejects_shape_mismatch():
    f = TightINS16()
    sv = _sv_set()
    with pytest.raises(ValueError, match="match sv count"):
        f.update_pseudoranges(sv, np.zeros(3))


def test_pseudorange_update_accepts_per_sv_sigma():
    f = TightINS16()
    f.P = np.eye(16) * 10.0
    sv = _sv_set()
    truth_pos = np.array([6_378_137.0, 0.0, 0.0])
    f.position = truth_pos.copy()
    rho = np.linalg.norm(sv - f.position, axis=1) + f.clock_bias_m
    rho[0] += 50.0   # outlier on the first SV
    sigma = np.array([100.0, 1.0, 1.0, 1.0, 1.0])   # downweight the outlier
    f.update_pseudoranges(sv, rho, sigma_pr=sigma)
    # Position should stay close to truth because the outlier is downweighted.
    err = np.linalg.norm(f.position - truth_pos)
    assert err < 5.0


def test_tight_filter_runs_long_kinematic_sequence():
    """Static body. IMU reports specific force balancing gravity, plus a
    constant acceleration bias. GNSS pseudoranges at 1 Hz drive the
    filter; the position should stay tight on the truth."""
    truth_pos = np.array([6_378_137.0, 0.0, 0.0])
    sv = _sv_set()
    truth_clock = 5.0

    f = TightINS16(sigma_accel_bias_rw=1e-3)
    f.position = truth_pos.copy()
    f.clock_bias_m = truth_clock
    f.P = np.eye(16) * 1e-3
    f.P[9:12, 9:12] = np.eye(3) * 0.01   # learn the accel bias

    # IMU specific force: -gravity (so the body stays still) plus a bias.
    bias = np.array([0.05, 0.0, 0.0])
    reported_accel = -f.gravity + bias

    dt_imu = 0.01
    rng = np.random.default_rng(0)
    last_gnss = 0.0
    t = 0.0
    while t < 20.0:
        f.predict(reported_accel, np.zeros(3), dt=dt_imu)
        t += dt_imu
        if t - last_gnss >= 1.0:
            rho = (
                np.linalg.norm(sv - truth_pos, axis=1)
                + truth_clock
                + rng.normal(0.0, 0.3, sv.shape[0])
            )
            f.update_pseudoranges(sv, rho, sigma_pr=0.3)
            last_gnss = t

    err = np.linalg.norm(f.position - truth_pos)
    # 20 s with 0.05 m/s^2 unmodeled bias would open-loop drift to 10 m;
    # tight coupling should keep us at sub-meter.
    assert err < 1.0
