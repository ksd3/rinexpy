"""Tightly-coupled GNSS / inertial navigation EKF.

Whereas :class:`rinexpy.imu.LooseINS15` consumes a pre-computed GNSS
position fix as its measurement, the tightly-coupled filter accepts
raw pseudoranges directly. This is operationally more robust: a single
visible satellite still contributes information, the integer-ambiguity
candidate space stays consistent across outages, and the EKF's own
sigma estimate weights every observation.

State vector (16 nominal scalars, 16 error-state dimensions):

    [ position_ecef (3)
      velocity_ecef (3)
      attitude_quaternion (4 nominal, 3 in error-state)
      accel_bias_body (3)
      gyro_bias_body (3)
      receiver_clock_bias_m (1) ]

The clock bias is carried in meters (i.e. ``c * dt_rx``) so the
pseudorange measurement model is

    rho_i = ||sv_i - p|| + b_rx + noise

with no need for the speed of light in the H matrix. Receiver clock
drift is modeled as a random walk; pass ``sigma_clock_drift_m_per_s``
to tune the process noise on the clock bias state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imu import (
    GRAVITY_M_PER_S2,
    _skew,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
    quat_to_matrix,
)


@dataclass
class TightINS16:
    """16-state tightly-coupled GNSS/INS EKF.

    Public state:

    - ``position`` ``(3,)`` ECEF, meters.
    - ``velocity`` ``(3,)`` ECEF, m/s.
    - ``quaternion`` ``(4,)`` body-to-ECEF, Hamilton.
    - ``accel_bias`` ``(3,)`` body-frame, m/s².
    - ``gyro_bias`` ``(3,)`` body-frame, rad/s.
    - ``clock_bias_m`` (scalar) receiver clock bias in meters
      (``= c * dt_rx``).
    - ``P`` ``(16, 16)`` error-state covariance.

    Process noise spectral densities are the same set as
    :class:`rinexpy.imu.LooseINS15` plus
    ``sigma_clock_drift_m_per_s`` for the random-walk clock model.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    clock_bias_m: float = 0.0
    P: np.ndarray = field(default_factory=lambda: np.eye(16) * 1e-3)

    sigma_accel: float = 0.05
    sigma_gyro: float = 0.005
    sigma_accel_bias_rw: float = 1e-4
    sigma_gyro_bias_rw: float = 1e-6
    sigma_clock_drift_m_per_s: float = 1.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -GRAVITY_M_PER_S2]))

    def predict(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: float,
    ) -> None:
        """Integrate one IMU sample and advance the error-state covariance."""
        a = np.asarray(accel_body, dtype=float) - self.accel_bias
        w = np.asarray(gyro_body, dtype=float) - self.gyro_bias
        R = quat_to_matrix(self.quaternion)
        a_nav = R @ a + self.gravity

        self.position = self.position + self.velocity * dt + 0.5 * a_nav * dt * dt
        self.velocity = self.velocity + a_nav * dt
        self.quaternion = quat_normalize(
            quat_mul(self.quaternion, quat_from_axis_angle(w * dt))
        )
        # clock_bias_m: random walk (constant drift), no deterministic update.

        F = np.zeros((16, 16))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -R @ _skew(a)
        F[3:6, 9:12] = -R
        F[6:9, 6:9] = -_skew(w)
        F[6:9, 12:15] = -np.eye(3)
        # Clock bias has zero deterministic propagation; pure random walk.
        Phi = np.eye(16) + F * dt + 0.5 * F @ F * dt * dt

        Q = np.zeros((16, 16))
        Q[3:6, 3:6] = (self.sigma_accel ** 2) * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sigma_gyro ** 2) * dt * np.eye(3)
        Q[9:12, 9:12] = (self.sigma_accel_bias_rw ** 2) * dt * np.eye(3)
        Q[12:15, 12:15] = (self.sigma_gyro_bias_rw ** 2) * dt * np.eye(3)
        Q[15, 15] = (self.sigma_clock_drift_m_per_s ** 2) * dt

        self.P = Phi @ self.P @ Phi.T + Q

    def update_pseudoranges(
        self,
        sv_ecef: np.ndarray,
        pseudoranges: np.ndarray,
        sigma_pr: float | np.ndarray = 5.0,
    ) -> dict:
        """Apply a batch of pseudorange measurements at one epoch.

        For each satellite the measurement model is

            rho_i = ||sv_i - p|| + b_rx + epsilon_i

        with ``b_rx`` the receiver clock bias in meters. The
        Jacobian row for SV ``i`` is

            H_i = [ -u_i,  0_3,  0_3,  0_3,  0_3,  1 ]

        where ``u_i = (sv_i - p) / ||sv_i - p||`` is the unit
        line-of-sight from the receiver to the satellite. Joseph-form
        covariance update keeps ``P`` symmetric and positive-definite
        across long sequences.

        Parameters
        ----------
        sv_ecef:
            ``(n_sv, 3)`` satellite ECEF positions at signal emission
            time (caller is expected to have already applied the standard
            light-time / Earth-rotation correction).
        pseudoranges:
            ``(n_sv,)`` observed pseudoranges in meters.
        sigma_pr:
            Pseudorange 1-sigma noise (m). Scalar for an isotropic
            ``sigma^2 * I_n`` model, or a length-``n_sv`` array for
            elevation-weighted noise.

        Returns
        -------
        dict
            ``{"residuals": ndarray, "n_obs": int}``.

        Raises
        ------
        ValueError
            If fewer than 1 satellite is supplied or the shapes don't match.
        """
        sv = np.asarray(sv_ecef, dtype=float)
        pr = np.asarray(pseudoranges, dtype=float)
        if sv.ndim != 2 or sv.shape[1] != 3:
            raise ValueError(f"sv_ecef must be (n_sv, 3); got {sv.shape}")
        if pr.shape != (sv.shape[0],):
            raise ValueError(f"pseudoranges must match sv count; got {pr.shape}")
        n = sv.shape[0]
        if n < 1:
            raise ValueError("update_pseudoranges needs >= 1 satellite")

        diff = sv - self.position
        rho = np.linalg.norm(diff, axis=1)
        u = diff / rho[:, None]   # unit line-of-sight
        predicted = rho + self.clock_bias_m
        residuals = pr - predicted

        H = np.zeros((n, 16))
        H[:, 0:3] = -u
        H[:, 15] = 1.0

        if np.isscalar(sigma_pr):
            R = float(sigma_pr) ** 2 * np.eye(n)
        else:
            sigma = np.asarray(sigma_pr, dtype=float)
            if sigma.shape != (n,):
                raise ValueError(f"sigma_pr array must have shape ({n},)")
            R = np.diag(sigma ** 2)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residuals

        self.position += dx[0:3]
        self.velocity += dx[3:6]
        self.quaternion = quat_normalize(
            quat_mul(self.quaternion, quat_from_axis_angle(dx[6:9]))
        )
        self.accel_bias += dx[9:12]
        self.gyro_bias += dx[12:15]
        self.clock_bias_m += float(dx[15])

        IKH = np.eye(16) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return {"residuals": residuals, "n_obs": n}


__all__ = ["TightINS16"]
