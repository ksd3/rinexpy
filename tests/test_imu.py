"""Tests for the 15-state loosely-coupled GNSS/INS EKF."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.imu import (
    GRAVITY_M_PER_S2,
    LooseINS15,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
    quat_to_matrix,
)


def test_quat_identity_round_trip():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = quat_to_matrix(q)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_quat_rotation_90deg_about_z():
    """90° rotation about z maps +x to +y."""
    q = quat_from_axis_angle(np.array([0.0, 0.0, np.pi / 2]))
    R = quat_to_matrix(q)
    v_in = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(R @ v_in, [0.0, 1.0, 0.0], atol=1e-12)


def test_quat_mul_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    p = quat_normalize(p)
    np.testing.assert_allclose(quat_mul(q, p), p)
    np.testing.assert_allclose(quat_mul(p, q), p)


def test_predict_no_motion_no_gravity_stays_still():
    """Zero IMU samples + zero gravity -> filter stays at initial state."""
    f = LooseINS15(gravity=np.zeros(3))
    f.position = np.array([1.0, 2.0, 3.0])
    f.velocity = np.array([0.0, 0.0, 0.0])
    for _ in range(100):
        f.predict(np.zeros(3), np.zeros(3), dt=0.01)
    np.testing.assert_allclose(f.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(f.velocity, [0.0, 0.0, 0.0])


def test_predict_freefall_drops_under_gravity():
    """Zero specific-force IMU (the body's accelerating with the freely
    falling frame) leads to velocity accruing under gravity."""
    f = LooseINS15()   # default gravity = [0, 0, -g]
    f.position = np.array([0.0, 0.0, 1000.0])
    for _ in range(100):
        f.predict(np.zeros(3), np.zeros(3), dt=0.01)
    t = 1.0
    expected_v = -GRAVITY_M_PER_S2 * t
    expected_z = 1000.0 + 0.5 * (-GRAVITY_M_PER_S2) * t * t
    assert f.velocity[2] == pytest.approx(expected_v, abs=1e-6)
    assert f.position[2] == pytest.approx(expected_z, abs=1e-3)


def test_predict_balanced_accel_holds_still():
    """An IMU sample equal to -g in body frame (specific force pointing
    'up') exactly balances gravity; the body stays still."""
    f = LooseINS15()
    # Body frame is the navigation frame here -> specific force == -g.
    accel = -f.gravity   # = [0, 0, +g]
    for _ in range(100):
        f.predict(accel, np.zeros(3), dt=0.01)
    np.testing.assert_allclose(f.position, [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(f.velocity, [0.0, 0.0, 0.0], atol=1e-9)


def test_gnss_update_pulls_position_toward_measurement():
    """A GNSS fix offset from the predicted position pulls the state."""
    f = LooseINS15()
    f.position = np.array([10.0, 20.0, 30.0])
    f.P = np.eye(15) * 1.0   # large prior uncertainty
    z = np.array([15.0, 22.0, 28.0])
    f.update_gnss(z, R_gnss=1e-4)
    # After update, position should be very close to z because R << P
    # (Kalman gain ~ P/(P+R) = 0.9999, so the update absorbs ~99.99 %
    # of the residual).
    np.testing.assert_allclose(f.position, z, atol=1e-3)


def test_gnss_update_reduces_covariance():
    f = LooseINS15()
    f.P = np.eye(15) * 1.0
    trace_before = np.trace(f.P[0:3, 0:3])
    f.update_gnss(np.zeros(3), R_gnss=0.1)
    trace_after = np.trace(f.P[0:3, 0:3])
    assert trace_after < trace_before


def test_bias_estimation_converges():
    """Constant accelerometer-bias offset; static GNSS fixes should drive
    the estimated bias toward the truth.

    Setup: body is truly stationary at the origin. The IMU reports a
    constant non-zero specific-force bias of (0.1, 0, +g). After many
    predict / update cycles with the GNSS clamped at the origin and
    a generous prior on the bias, the filter's bias estimate should
    drift toward (0.1, 0, 0).
    """
    f = LooseINS15(
        sigma_accel=0.001,            # tight on the IMU noise
        sigma_accel_bias_rw=1e-3,     # let the bias move
    )
    # Inflate the bias-state covariance so the filter is willing to learn it.
    f.P[9:12, 9:12] = np.eye(3) * 1.0
    f.P[0:3, 0:3] = np.eye(3) * 100.0
    truth_bias_body = np.array([0.1, 0.0, 0.0])
    # Reported specific force: truth (-gravity to stay still) + bias.
    reported_accel = -f.gravity + truth_bias_body
    for _ in range(2000):
        f.predict(reported_accel, np.zeros(3), dt=0.01)
        # GNSS fix at the (true) origin.
        f.update_gnss(np.zeros(3), R_gnss=0.0001)
    # Bias should be within a few cm/s^2 of truth in the dominant axis.
    assert abs(f.accel_bias[0] - 0.1) < 0.02


def test_static_drift_corrected_by_gnss():
    """Without GNSS, an IMU with a 0.05 m/s² bias drifts ~25 m in 30 s.
    With GNSS updates at 1 Hz, the position stays close to truth."""
    truth = np.array([100.0, 200.0, 50.0])
    f = LooseINS15()
    f.position = truth.copy()
    f.P = np.eye(15) * 1e-3
    f.P[9:12, 9:12] = np.eye(3) * 0.01   # allow bias to learn
    reported_accel = -f.gravity + np.array([0.05, 0.0, 0.0])  # 0.05 m/s² bias
    dt_imu = 0.01      # 100 Hz IMU
    dt_gnss = 1.0      # 1 Hz GNSS
    t = 0.0
    while t < 30.0:
        f.predict(reported_accel, np.zeros(3), dt=dt_imu)
        t += dt_imu
        if abs(round(t) - t) < 1e-9 and round(t) >= 1:
            f.update_gnss(truth, R_gnss=0.01)
    err = np.linalg.norm(f.position - truth)
    assert err < 5.0   # well under the open-loop 25 m drift
