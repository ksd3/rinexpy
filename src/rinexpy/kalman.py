"""Static-receiver Precise Point Positioning extended Kalman filter.

Per-epoch sequential filter for cm-level static PPP. The state is

    x = [px, py, pz, c * dt_rx, N_1, ..., N_n_sv]

with units meters / meters / meters / meters / meters. Receiver-clock
bias is parameterised as ``c * dt_rx`` so the observation matrix has
``1`` in the clock slot (not the speed of light); ambiguities are
iono-free phase ambiguities in meters as defined by

    B_IF = (alpha * N1 * lambda1 - N2 * lambda2) / (alpha - 1)

with ``alpha = (f1/f2)^2``. The dynamics are:

- Position: identity (static receiver).
- Receiver clock: random walk with rate ``sigma_clock_rate_m``
  (driven by the receiver oscillator stability; ~10 m/sqrt(s) for a
  typical TCXO, much smaller for an OCXO).
- Ambiguities: identity (constant between cycle slips).

Measurement updates are processed one observation at a time, doing a
scalar EKF update per (SV, code/phase). The first phase observation
for any SV initialises that SV's ambiguity from the code-phase
difference at large variance; subsequent phase observations refine it.

For cycle slips: call :meth:`StaticPPPFilter.reset_ambiguity` (or
:meth:`reset_ambiguities` for a batch) to wipe the per-SV ambiguity
estimate and start the next observation as a fresh initialisation.
"""

from __future__ import annotations

import numpy as np

from . import _native

_C = 299_792_458.0


class StaticPPPFilter:
    """Sequential PPP filter for a static GNSS receiver.

    Parameters
    ----------
    n_sv:
        Number of SVs the state tracks. The same index is used across
        epochs; pre-allocate enough slots for every SV you might see.
    initial_position:
        Initial guess of the receiver ECEF position, in meters. Pull
        from a code-only SPP first pass or the RINEX header
        ``APPROX POSITION XYZ``.
    sigma_code, sigma_phase:
        1-sigma measurement noise on iono-free code and phase
        observations. Defaults 1.0 m and 5 mm.
    sigma_position_init:
        1-sigma initial position uncertainty. Default 10 m. Smaller
        values speed convergence if your initial guess is tight.
    sigma_clock_init:
        1-sigma initial receiver-clock uncertainty in meters. Default
        300 m (= 1 us light-time; receivers can be that far off until
        the filter pins them).
    sigma_clock_rate_m:
        Receiver-clock process-noise rate in meters per sqrt(s).
        Default 10. Increase for noisier oscillators, decrease for
        atomic-disciplined receivers.
    sigma_position_rate_m:
        Position process-noise rate in meters per sqrt(s). Default 0
        for a truly static receiver. Set to a positive value to enable
        kinematic tracking: a typical pedestrian-speed PPP setup uses
        ~0.3, a vehicle ~3, a low-Earth-orbit platform ~30. Position
        variance grows by ``sigma_position_rate_m^2 * dt`` per
        :meth:`predict`.
    sigma_ambig_init_m:
        1-sigma initial uncertainty of an ambiguity slot. Default
        1000 m, big enough to let any reasonable starting value
        decay quickly.
    """

    def __init__(
        self,
        n_sv: int,
        initial_position: tuple[float, float, float],
        *,
        sigma_code: float = 1.0,
        sigma_phase: float = 0.005,
        sigma_position_init: float = 10.0,
        sigma_clock_init: float = 300.0,
        sigma_clock_rate_m: float = 10.0,
        sigma_position_rate_m: float = 0.0,
        sigma_ambig_init_m: float = 1000.0,
    ) -> None:
        if sigma_position_rate_m < 0:
            raise ValueError(
                f"sigma_position_rate_m must be >= 0, got "
                f"{sigma_position_rate_m}"
            )
        self.n_sv = n_sv
        self._n_state = 4 + n_sv
        self.x = np.zeros(self._n_state)
        self.x[:3] = np.asarray(initial_position, dtype=float)
        var = np.empty(self._n_state)
        var[:3] = sigma_position_init ** 2
        var[3] = sigma_clock_init ** 2
        var[4:] = sigma_ambig_init_m ** 2
        self.P = np.diag(var)
        self.sigma_code = sigma_code
        self.sigma_phase = sigma_phase
        self.sigma_clock_rate_m = sigma_clock_rate_m
        self.sigma_position_rate_m = sigma_position_rate_m
        self.sigma_ambig_init_m = sigma_ambig_init_m
        self._ambig_initialised = np.zeros(n_sv, dtype=bool)
        # Per-SV previous Melbourne-Wuebbena value in wide-lane cycles,
        # used by update_with_slip_check.
        self._prev_mw_wl_cycles = np.full(n_sv, np.nan)
        # Pre-allocated sparse-H buffers for the native dispatch path.
        # We keep two buffers (code: 4-wide; phase: 5-wide) and mutate
        # them per call instead of re-allocating tiny ndarrays.
        self._h_idx_code = np.array([0, 1, 2, 3], dtype=np.int32)
        self._h_val_code = np.zeros(4, dtype=np.float64)
        self._h_val_code[3] = 1.0
        self._h_idx_phase = np.array([0, 1, 2, 3, 0], dtype=np.int32)
        self._h_val_phase = np.zeros(5, dtype=np.float64)
        self._h_val_phase[3] = 1.0
        self._h_val_phase[4] = 1.0

    @property
    def position(self) -> tuple[float, float, float]:
        """ECEF receiver position from the current state (meters)."""
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def clock_bias_s(self) -> float:
        """Receiver clock bias in seconds (== state[3] / c)."""
        return float(self.x[3] / _C)

    @property
    def ambiguities_m(self) -> np.ndarray:
        """Copy of the iono-free phase ambiguities in meters, per SV slot."""
        return self.x[4 : 4 + self.n_sv].copy()

    @property
    def position_sigma(self) -> tuple[float, float, float]:
        """1-sigma position uncertainty per axis from the covariance diagonal."""
        d = np.diag(self.P)
        return (float(np.sqrt(d[0])), float(np.sqrt(d[1])), float(np.sqrt(d[2])))

    def predict(self, dt: float) -> None:
        """Time update.

        Receiver clock random-walks with variance
        ``sigma_clock_rate_m^2 * dt``. Position random-walks with
        ``sigma_position_rate_m^2 * dt`` per axis when that parameter
        is set (the default 0 keeps the filter strictly static).
        Ambiguities are constant.
        """
        if dt < 0:
            raise ValueError(f"dt must be >= 0, got {dt}")
        self.P[3, 3] += self.sigma_clock_rate_m ** 2 * dt
        if self.sigma_position_rate_m > 0.0:
            growth = self.sigma_position_rate_m ** 2 * dt
            for i in range(3):
                self.P[i, i] += growth

    def reset_ambiguity(self, sv_index: int) -> None:
        """Wipe one SV's ambiguity (e.g. after a cycle slip).

        The slot is set to 0, all covariance row / column entries are
        zeroed, and the diagonal restored to ``sigma_ambig_init_m^2``,
        so the next phase observation re-initialises it.
        """
        i = 4 + sv_index
        self.x[i] = 0.0
        self.P[i, :] = 0.0
        self.P[:, i] = 0.0
        self.P[i, i] = self.sigma_ambig_init_m ** 2
        self._ambig_initialised[sv_index] = False
        # Force the MW history to re-baseline on the next slip-aware call.
        self._prev_mw_wl_cycles[sv_index] = np.nan

    def reset_ambiguities(self, sv_indices) -> None:
        """Batch wipe ambiguity slots; see :meth:`reset_ambiguity`."""
        for i in sv_indices:
            self.reset_ambiguity(int(i))

    def update(
        self,
        sv_ecef: np.ndarray,
        sat_clock_s: np.ndarray,
        pr_if: np.ndarray,
        phase_if: np.ndarray,
        *,
        tropo_m: np.ndarray | None = None,
    ) -> None:
        """Measurement update for one epoch.

        Processes each finite code / phase observation as a scalar EKF
        update. The first phase observation for a given SV initialises
        that SV's ambiguity from the code-phase difference (so the
        filter doesn't reject good phase data because the ambiguity
        slot was at zero).

        Parameters
        ----------
        sv_ecef:
            ``(n_sv, 3)`` satellite ECEF positions at signal-emission
            time (run them through
            :func:`rinexpy.positioning.apply_light_time_and_earth_rotation`
            first).
        sat_clock_s:
            ``(n_sv,)`` precise satellite clock offsets in seconds.
        pr_if:
            ``(n_sv,)`` iono-free code pseudoranges, NaN for missing
            observations.
        phase_if:
            ``(n_sv,)`` iono-free phase observations, NaN for missing.
        tropo_m:
            Optional ``(n_sv,)`` slant tropospheric delay in meters.
        """
        sv = np.asarray(sv_ecef, dtype=float)
        dt_sv = np.asarray(sat_clock_s, dtype=float)
        pr = np.asarray(pr_if, dtype=float)
        ph = np.asarray(phase_if, dtype=float)
        n_sv = sv.shape[0]
        if (
            n_sv != self.n_sv
            or dt_sv.shape != (n_sv,)
            or pr.shape != (n_sv,)
            or ph.shape != (n_sv,)
        ):
            raise ValueError(
                "shape mismatch: expected n_sv = "
                f"{self.n_sv}, got sv_ecef.shape={sv.shape}, "
                f"sat_clock_s.shape={dt_sv.shape}, "
                f"pr_if.shape={pr.shape}, phase_if.shape={ph.shape}"
            )
        if tropo_m is None:
            tropo = np.zeros(n_sv)
        else:
            tropo = np.asarray(tropo_m, dtype=float)
            if tropo.shape != (n_sv,):
                raise ValueError("tropo_m shape mismatch")

        # Apply precise sat clock and tropo corrections to the observations.
        pr_corr = pr + _C * dt_sv - tropo
        ph_corr = ph + _C * dt_sv - tropo

        # Two passes per epoch: process all code observations first to
        # nail down position and clock, then process all phase
        # observations. Ambiguities are initialised in a third sweep
        # using the converged geometry so every SV's ambig slot
        # represents the same position / clock snapshot. This avoids
        # the "ambig j absorbs the position error that was still
        # there when SV j was first seen but isn't there anymore"
        # failure mode that single-pass per-SV updates suffer.
        for j in range(n_sv):
            if not np.isfinite(pr_corr[j]):
                continue
            diff = sv[j] - self.x[:3]
            rho = float(np.linalg.norm(diff))
            if rho == 0.0:
                continue
            u = -diff / rho
            self._scalar_update(u, code=True, sv_index=j,
                                obs=pr_corr[j], rho=rho)

        # After all code updates: initialise any new-SV ambiguities
        # against the now-better-known position and clock.
        for j in range(n_sv):
            if not np.isfinite(ph_corr[j]):
                continue
            if self._ambig_initialised[j]:
                continue
            diff = sv[j] - self.x[:3]
            rho = float(np.linalg.norm(diff))
            if rho == 0.0:
                continue
            self.x[4 + j] = ph_corr[j] - rho - self.x[3]
            self._ambig_initialised[j] = True

        for j in range(n_sv):
            if not np.isfinite(ph_corr[j]):
                continue
            diff = sv[j] - self.x[:3]
            rho = float(np.linalg.norm(diff))
            if rho == 0.0:
                continue
            u = -diff / rho
            self._scalar_update(u, code=False, sv_index=j,
                                obs=ph_corr[j], rho=rho)

    def update_with_slip_check(
        self,
        sv_ecef: np.ndarray,
        sat_clock_s: np.ndarray,
        p1_m: np.ndarray,
        p2_m: np.ndarray,
        phi1_cycles: np.ndarray,
        phi2_cycles: np.ndarray,
        *,
        tropo_m: np.ndarray | None = None,
        slip_threshold_cycles: float = 2.0,
        f1: float = 1575.42e6,
        f2: float = 1227.60e6,
    ) -> np.ndarray:
        """Slip-aware measurement update.

        Computes the Melbourne-Wuebbena combination per SV from the raw
        L1 / L2 code + phase observations and compares it to the
        previous epoch's MW value. Any SV whose first-difference exceeds
        ``slip_threshold_cycles`` (in wide-lane cycles) is flagged as
        having slipped, and its ambiguity slot is wiped via
        :meth:`reset_ambiguity` before the regular measurement update
        runs.

        The iono-free code and phase observations are formed internally
        from the raw L1 / L2 inputs.

        Parameters
        ----------
        sv_ecef:
            ``(n_sv, 3)`` satellite ECEF positions at signal emission.
        sat_clock_s:
            ``(n_sv,)`` precise satellite clocks in seconds.
        p1_m, p2_m:
            ``(n_sv,)`` raw L1 / L2 code pseudoranges in meters.
        phi1_cycles, phi2_cycles:
            ``(n_sv,)`` raw L1 / L2 carrier phase in cycles.
        tropo_m:
            Optional ``(n_sv,)`` slant tropospheric delay in meters.
        slip_threshold_cycles:
            MW first-difference threshold in wide-lane cycles. Default 2.0,
            which is loose enough to be stable on realistic code-noise
            data without missing real one-cycle slips. Lower for cleaner
            signals.
        f1, f2:
            Carrier frequencies. Defaults are GPS L1, L2.

        Returns
        -------
        ndarray
            Boolean ``(n_sv,)`` mask: ``True`` where a slip was flagged
            and the ambiguity was reset before this epoch's update.
        """
        from .positioning import iono_free_phase, iono_free_pseudorange

        p1 = np.asarray(p1_m, dtype=float)
        p2 = np.asarray(p2_m, dtype=float)
        phi1 = np.asarray(phi1_cycles, dtype=float)
        phi2 = np.asarray(phi2_cycles, dtype=float)
        if (
            p1.shape != (self.n_sv,) or p2.shape != (self.n_sv,)
            or phi1.shape != (self.n_sv,) or phi2.shape != (self.n_sv,)
        ):
            raise ValueError(
                f"L1/L2 observation arrays must have shape ({self.n_sv},)"
            )

        # MW in wide-lane cycles per SV:
        #   MW_cycles = (phi1 - phi2) - (f1 - f2)/(f1 + f2) * (p1/lam1 + p2/lam2)
        lam1 = _C / f1
        lam2 = _C / f2
        mw_cycles = (phi1 - phi2) - (
            (f1 - f2) / (f1 + f2) * (p1 / lam1 + p2 / lam2)
        )

        slip_mask = np.zeros(self.n_sv, dtype=bool)
        for j in range(self.n_sv):
            if not np.isfinite(mw_cycles[j]):
                continue
            prev = self._prev_mw_wl_cycles[j]
            if np.isfinite(prev):
                if abs(mw_cycles[j] - prev) > slip_threshold_cycles:
                    self.reset_ambiguity(j)
                    slip_mask[j] = True
            self._prev_mw_wl_cycles[j] = mw_cycles[j]

        # Form iono-free observations and run the standard update.
        l1_m = phi1 * lam1
        l2_m = phi2 * lam2
        pr_if = iono_free_pseudorange(p1, p2, f1=f1, f2=f2)
        phase_if = iono_free_phase(l1_m, l2_m, f1=f1, f2=f2)
        self.update(sv_ecef, sat_clock_s, pr_if, phase_if, tropo_m=tropo_m)
        return slip_mask

    def _scalar_update(
        self,
        u: np.ndarray,
        *,
        code: bool,
        sv_index: int,
        obs: float,
        rho: float,
    ) -> None:
        """Process one scalar code or phase observation.

        Dispatches to the C++ Joseph-form kernel (O(n^2) via sparse-H
        exploitation, ~10-30x faster) when ``rinexpy_native`` is
        importable; falls back to the dense numpy version below.
        """
        r = self.sigma_code ** 2 if code else self.sigma_phase ** 2
        if _native.have_kalman_scalar_update():
            # Mutate pre-allocated sparse-H buffers instead of allocating
            # new ndarrays per call.
            if code:
                pred = rho + self.x[3]
                self._h_val_code[0] = u[0]
                self._h_val_code[1] = u[1]
                self._h_val_code[2] = u[2]
                idx = self._h_idx_code
                val = self._h_val_code
            else:
                pred = rho + self.x[3] + self.x[4 + sv_index]
                self._h_val_phase[0] = u[0]
                self._h_val_phase[1] = u[1]
                self._h_val_phase[2] = u[2]
                self._h_idx_phase[4] = 4 + sv_index
                idx = self._h_idx_phase
                val = self._h_val_phase
            innovation = float(obs - pred)
            if not (self.x.flags.c_contiguous and self.x.dtype == np.float64):
                self.x = np.ascontiguousarray(self.x, dtype=np.float64)
            if not (self.P.flags.c_contiguous and self.P.dtype == np.float64):
                self.P = np.ascontiguousarray(self.P, dtype=np.float64)
            _native.kalman_scalar_update_sparse(
                self.x, self.P, idx, val, innovation, r,
            )
            return

        n = self._n_state
        h = np.zeros(n)
        h[:3] = u
        h[3] = 1.0
        if code:
            pred = rho + self.x[3]
        else:
            h[4 + sv_index] = 1.0
            pred = rho + self.x[3] + self.x[4 + sv_index]
        innovation = obs - pred

        # K = P @ h^T / (h @ P @ h^T + r)
        ph = self.P @ h
        s = float(h @ ph + r)
        if s <= 0.0:
            return
        k = ph / s
        self.x = self.x + k * innovation
        # Joseph form for numerical stability of P:
        #   P_new = (I - K H) P (I - K H)^T + K R K^T
        # For scalar H and R this collapses to:
        I_KH = np.eye(n) - np.outer(k, h)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(k, k) * r


#: The named EKF entry point per the roadmap acceptance API.
#: ``GNSSFilter`` is an alias for :class:`StaticPPPFilter` - the same
#: state ``[px, py, pz, c*dt, N_1, ..., N_n_sv]`` handles both the
#: static (``sigma_position_rate_m = 0``, the default) and kinematic
#: (``sigma_position_rate_m > 0``) cases.
GNSSFilter = StaticPPPFilter


__all__ = ["GNSSFilter", "StaticPPPFilter"]
