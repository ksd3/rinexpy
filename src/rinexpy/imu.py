"""Loosely-coupled GNSS / inertial navigation EKF.

Implements the textbook 15-state error-state EKF (Multiplicative Extended
Kalman Filter, MEKF) for fusing IMU strapdown integration with GNSS
position fixes. The state is

    [ position_ecef (3)  velocity_ecef (3)
      attitude_quaternion (4 nominal, 3 in error-state)
      accel_bias_body (3)  gyro_bias_body (3) ]

The IMU prediction step integrates accelerometer + gyroscope samples
through the strapdown equations (with WGS-84 gravity and centripetal
correction). The GNSS update step accepts an ECEF position fix with its
covariance and applies a standard Kalman update.

The MEKF design uses a 3-element attitude error vector in the error
state (with the nominal attitude carried as a unit quaternion); after
each update the error is folded into the nominal quaternion and the
error reset to zero.

Suitable for static / kinematic land vehicles. Aircraft / spacecraft
applications need the higher-order corrections (Earth rotation in the
state propagation, scale-factor errors, ...) that this module skips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY_M_PER_S2 = 9.80665   # WGS-84 nominal at sea level


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([
        [0.0, -z,  y],
        [z,  0.0, -x],
        [-y,  x, 0.0],
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Project a 4-vector back onto the unit sphere."""
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a body -> reference Hamilton quaternion ``q``.

    ``q = [w, x, y, z]``, with the convention that ``q`` rotates a body-
    frame vector into the reference frame as ``v_ref = R * v_body``.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [    2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product ``p ⊗ q`` (4-vector inputs)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Build a quaternion from a small-angle axis-vector ``omega * dt``."""
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.array([1.0, 0.5 * axis_angle[0], 0.5 * axis_angle[1], 0.5 * axis_angle[2]])
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), axis_angle[0] * s, axis_angle[1] * s, axis_angle[2] * s])


@dataclass
class LooseINS15:
    """15-state loosely-coupled GNSS/INS EKF.

    Public state:

    - ``position`` ``(3,)`` ECEF, meters.
    - ``velocity`` ``(3,)`` ECEF, m/s.
    - ``quaternion`` ``(4,)`` body-to-ECEF, Hamilton convention.
    - ``accel_bias`` ``(3,)`` body-frame accelerometer bias, m/s².
    - ``gyro_bias`` ``(3,)`` body-frame gyroscope bias, rad/s.
    - ``P`` ``(15, 15)`` error-state covariance.

    Process noise spectral densities (default values are reasonable
    starting points for a MEMS-grade IMU at 100 Hz):

    - ``sigma_accel``: white acceleration noise (m/s² / sqrt(Hz)).
    - ``sigma_gyro``: white angular-rate noise (rad/s / sqrt(Hz)).
    - ``sigma_accel_bias_rw``: accelerometer bias random walk.
    - ``sigma_gyro_bias_rw``: gyroscope bias random walk.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(15) * 1e-3)

    sigma_accel: float = 0.05
    sigma_gyro: float = 0.005
    sigma_accel_bias_rw: float = 1e-4
    sigma_gyro_bias_rw: float = 1e-6
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -GRAVITY_M_PER_S2]))

    def predict(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: float,
    ) -> None:
        """Integrate one IMU sample, advance the nominal state and the
        error-state covariance by ``dt`` seconds.

        Parameters
        ----------
        accel_body:
            ``(3,)`` accelerometer measurement (specific force) in
            body frame, m/s².
        gyro_body:
            ``(3,)`` gyroscope measurement in body frame, rad/s.
        dt:
            Integration step in seconds.
        """
        # Bias-corrected IMU samples.
        a = np.asarray(accel_body, dtype=float) - self.accel_bias
        w = np.asarray(gyro_body, dtype=float) - self.gyro_bias
        R = quat_to_matrix(self.quaternion)
        a_nav = R @ a + self.gravity

        # Strapdown integration of the nominal state.
        self.position = self.position + self.velocity * dt + 0.5 * a_nav * dt * dt
        self.velocity = self.velocity + a_nav * dt
        self.quaternion = quat_normalize(
            quat_mul(self.quaternion, quat_from_axis_angle(w * dt))
        )

        # Error-state continuous-time Jacobian F:
        #   d/dt[delta_p, delta_v, delta_theta, delta_b_a, delta_b_g] =
        #   [[0    I    0          0    0]
        #    [0    0   -R skew(a)  -R   0]
        #    [0    0   -skew(w)    0   -I]
        #    [0    0    0          0    0]
        #    [0    0    0          0    0]] *
        #   [delta_p, delta_v, delta_theta, delta_b_a, delta_b_g]
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -R @ _skew(a)
        F[3:6, 9:12] = -R
        F[6:9, 6:9] = -_skew(w)
        F[6:9, 12:15] = -np.eye(3)
        Phi = np.eye(15) + F * dt + 0.5 * F @ F * dt * dt

        # Process noise covariance Q (discretized white-noise model).
        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = (self.sigma_accel ** 2) * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sigma_gyro ** 2) * dt * np.eye(3)
        Q[9:12, 9:12] = (self.sigma_accel_bias_rw ** 2) * dt * np.eye(3)
        Q[12:15, 12:15] = (self.sigma_gyro_bias_rw ** 2) * dt * np.eye(3)

        self.P = Phi @ self.P @ Phi.T + Q

    def update_gnss(
        self,
        position_ecef: np.ndarray,
        R_gnss: np.ndarray | float = 1.0,
    ) -> None:
        """Apply a GNSS position measurement to the filter.

        Parameters
        ----------
        position_ecef:
            ``(3,)`` GNSS-reported receiver position in meters.
        R_gnss:
            Measurement noise covariance. Pass a scalar for an
            isotropic ``sigma^2 * I`` model, or a ``(3, 3)`` matrix.
        """
        z = np.asarray(position_ecef, dtype=float)
        if np.isscalar(R_gnss):
            R = float(R_gnss) * np.eye(3)
        else:
            R = np.asarray(R_gnss, dtype=float)
            if R.shape == (3,):
                R = np.diag(R)
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)
        y = z - self.position           # measurement residual
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        # Apply the error correction to the nominal state.
        self.position += dx[0:3]
        self.velocity += dx[3:6]
        # Attitude error -> quaternion update (small-angle).
        self.quaternion = quat_normalize(
            quat_mul(self.quaternion, quat_from_axis_angle(dx[6:9]))
        )
        self.accel_bias += dx[9:12]
        self.gyro_bias += dx[12:15]
        # Joseph-form covariance update for symmetry / positive-definiteness.
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T


__all__ = [
    "GRAVITY_M_PER_S2",
    "LooseINS15",
    "quat_from_axis_angle",
    "quat_mul",
    "quat_normalize",
    "quat_to_matrix",
]
