"""Static PPP EKF with tropospheric ZWD as an additional filter state.

The base :class:`rinexpy.kalman.StaticPPPFilter` parameterizes the state
as ``[pos, clock, ambiguities]``. For sub-cm static PPP the residual
zenith wet tropospheric delay (ZWD) is the next biggest unmodelled
contributor (3-30 cm of slant delay at moderate elevation). This module
adds it as a random-walk state. The observation equation becomes

    pr_if  = ||sv - p|| + c*dt_rx + tropo_apriori + m_w(el) * ZWD + e_p
    phi_if = ||sv - p|| + c*dt_rx + tropo_apriori + m_w(el) * ZWD
             + N_iono_free + e_phi

with the per-SV wet mapping function ``m_w`` supplied by the caller
(typically from :func:`rinexpy.geodesy.niell_mapping` or
:func:`rinexpy.geodesy.vmf1`). The hydrostatic component is folded
into ``tropo_apriori`` (Saastamoinen ZHD is accurate to ~mm).

State vector

    x = [px, py, pz, c*dt_rx, ZWD, N_1, ..., N_n_sv]
"""

from __future__ import annotations

import numpy as np

_C = 299_792_458.0


class StaticPPPFilterZTD:
    """PPP EKF with a ZWD state in addition to position / clock /
    ambiguities.

    Parameters
    ----------
    n_sv:
        Maximum number of tracked satellites.
    initial_position:
        ECEF starting guess.
    initial_zwd_m:
        ZWD initial value in meters; default 0.1 (roughly the global
        mean wet delay).
    sigma_code, sigma_phase:
        Observation noise (1-sigma, meters).
    sigma_position_init, sigma_clock_init, sigma_zwd_init, sigma_ambig_init_m:
        Prior 1-sigma uncertainties (m).
    sigma_clock_rate_m, sigma_zwd_rate_m, sigma_position_rate_m:
        Random-walk rate spectral densities. Defaults: 10 m/sqrt(s) on
        the clock, 0.01 m/sqrt(hr) on the ZWD (Boehm & Schuh 2007),
        0 on position (static).
    """

    def __init__(
        self,
        n_sv: int,
        initial_position: tuple[float, float, float],
        *,
        initial_zwd_m: float = 0.1,
        sigma_code: float = 1.0,
        sigma_phase: float = 0.005,
        sigma_position_init: float = 10.0,
        sigma_clock_init: float = 300.0,
        sigma_zwd_init: float = 0.5,
        sigma_clock_rate_m: float = 10.0,
        sigma_zwd_rate_m_per_sqrt_hr: float = 0.01,
        sigma_position_rate_m: float = 0.0,
        sigma_ambig_init_m: float = 1000.0,
    ) -> None:
        if sigma_position_rate_m < 0:
            raise ValueError(
                f"sigma_position_rate_m must be >= 0, got {sigma_position_rate_m}"
            )
        self.n_sv = n_sv
        self._n_state = 5 + n_sv   # +1 for ZWD
        self.x = np.zeros(self._n_state)
        self.x[:3] = np.asarray(initial_position, dtype=float)
        self.x[4] = initial_zwd_m
        var = np.empty(self._n_state)
        var[:3] = sigma_position_init ** 2
        var[3] = sigma_clock_init ** 2
        var[4] = sigma_zwd_init ** 2
        var[5:] = sigma_ambig_init_m ** 2
        self.P = np.diag(var)
        self.sigma_code = sigma_code
        self.sigma_phase = sigma_phase
        self.sigma_clock_rate_m = sigma_clock_rate_m
        # Convert m / sqrt(hr) -> m / sqrt(s) once at construction.
        self.sigma_zwd_rate_m_per_sqrt_s = sigma_zwd_rate_m_per_sqrt_hr / np.sqrt(3600.0)
        self.sigma_position_rate_m = sigma_position_rate_m
        self.sigma_ambig_init_m = sigma_ambig_init_m
        self._ambig_initialised = np.zeros(n_sv, dtype=bool)

    @property
    def position(self) -> tuple[float, float, float]:
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def clock_bias_s(self) -> float:
        return float(self.x[3] / _C)

    @property
    def zwd_m(self) -> float:
        """Estimated zenith wet delay in meters."""
        return float(self.x[4])

    @property
    def ambiguities_m(self) -> np.ndarray:
        return self.x[5 : 5 + self.n_sv].copy()

    @property
    def position_sigma(self) -> tuple[float, float, float]:
        d = np.diag(self.P)
        return (float(np.sqrt(d[0])), float(np.sqrt(d[1])), float(np.sqrt(d[2])))

    @property
    def zwd_sigma_m(self) -> float:
        return float(np.sqrt(self.P[4, 4]))

    def predict(self, dt: float) -> None:
        """Random-walk clock (rate ``sigma_clock_rate_m``), ZWD
        (rate ``sigma_zwd_rate_m_per_sqrt_hr``), and position
        (rate ``sigma_position_rate_m``).
        """
        if dt < 0:
            raise ValueError(f"dt must be >= 0, got {dt}")
        self.P[3, 3] += self.sigma_clock_rate_m ** 2 * dt
        self.P[4, 4] += self.sigma_zwd_rate_m_per_sqrt_s ** 2 * dt
        if self.sigma_position_rate_m > 0.0:
            growth = self.sigma_position_rate_m ** 2 * dt
            for i in range(3):
                self.P[i, i] += growth

    def reset_ambiguity(self, sv_index: int) -> None:
        i = 5 + sv_index
        self.x[i] = 0.0
        self.P[i, :] = 0.0
        self.P[:, i] = 0.0
        self.P[i, i] = self.sigma_ambig_init_m ** 2
        self._ambig_initialised[sv_index] = False

    def update(
        self,
        sv_ecef: np.ndarray,
        sat_clock_s: np.ndarray,
        pr_if: np.ndarray,
        phase_if: np.ndarray,
        wet_mapping: np.ndarray,
        *,
        tropo_apriori_m: np.ndarray | None = None,
    ) -> None:
        """Measurement update with a wet-mapping ZWD term.

        Parameters
        ----------
        sv_ecef:
            ``(n_sv, 3)`` satellite ECEF positions at signal emission.
        sat_clock_s:
            ``(n_sv,)`` precise satellite clocks in seconds.
        pr_if, phase_if:
            ``(n_sv,)`` iono-free code and phase observations in meters.
        wet_mapping:
            ``(n_sv,)`` wet mapping function values for each SV at the
            current epoch's elevation angle. Multiply ZWD by these to
            get the slant wet delay.
        tropo_apriori_m:
            Optional ``(n_sv,)`` slant a-priori delay (hydrostatic +
            modelled wet) already applied to remove the deterministic
            component. The filter then estimates the residual ZWD on
            top of zero a-priori.
        """
        sv = np.asarray(sv_ecef, dtype=float)
        dt_sv = np.asarray(sat_clock_s, dtype=float)
        pr = np.asarray(pr_if, dtype=float)
        ph = np.asarray(phase_if, dtype=float)
        mw = np.asarray(wet_mapping, dtype=float)
        n_sv = sv.shape[0]
        if (
            n_sv != self.n_sv
            or dt_sv.shape != (n_sv,)
            or pr.shape != (n_sv,)
            or ph.shape != (n_sv,)
            or mw.shape != (n_sv,)
        ):
            raise ValueError(
                f"shape mismatch (n_sv={self.n_sv} expected); "
                f"got sv={sv.shape}, clock={dt_sv.shape}, "
                f"pr={pr.shape}, ph={ph.shape}, mw={mw.shape}"
            )
        if tropo_apriori_m is None:
            tropo = np.zeros(n_sv)
        else:
            tropo = np.asarray(tropo_apriori_m, dtype=float)

        pr_corr = pr + _C * dt_sv - tropo
        ph_corr = ph + _C * dt_sv - tropo

        # Pass 1: code observations (no ambiguity dependence).
        for j in range(n_sv):
            if not np.isfinite(pr_corr[j]):
                continue
            diff = sv[j] - self.x[:3]
            rho = float(np.linalg.norm(diff))
            if rho == 0.0:
                continue
            u = -diff / rho
            self._scalar_update(u, code=True, sv_index=j,
                                obs=pr_corr[j], rho=rho, m_wet=mw[j])

        # Pass 2: initialise any not-yet-initialised ambiguities.
        for j in range(n_sv):
            if not np.isfinite(ph_corr[j]):
                continue
            if self._ambig_initialised[j]:
                continue
            diff = sv[j] - self.x[:3]
            rho = float(np.linalg.norm(diff))
            if rho == 0.0:
                continue
            # phi = rho + clock + m_wet * ZWD + amb
            self.x[5 + j] = (
                ph_corr[j] - rho - self.x[3] - mw[j] * self.x[4]
            )
            self._ambig_initialised[j] = True

        # Pass 3: phase observations.
        for j in range(n_sv):
            if not np.isfinite(ph_corr[j]):
                continue
            diff = sv[j] - self.x[:3]
            rho = float(np.linalg.norm(diff))
            if rho == 0.0:
                continue
            u = -diff / rho
            self._scalar_update(u, code=False, sv_index=j,
                                obs=ph_corr[j], rho=rho, m_wet=mw[j])

    def _scalar_update(
        self,
        u: np.ndarray,
        *,
        code: bool,
        sv_index: int,
        obs: float,
        rho: float,
        m_wet: float,
    ) -> None:
        n = self._n_state
        h = np.zeros(n)
        h[:3] = u
        h[3] = 1.0
        h[4] = m_wet     # ZWD column
        if code:
            pred = rho + self.x[3] + m_wet * self.x[4]
            r = self.sigma_code ** 2
        else:
            h[5 + sv_index] = 1.0
            pred = rho + self.x[3] + m_wet * self.x[4] + self.x[5 + sv_index]
            r = self.sigma_phase ** 2
        innovation = obs - pred

        ph = self.P @ h
        s = float(h @ ph + r)
        if s <= 0.0:
            return
        k = ph / s
        self.x = self.x + k * innovation
        I_KH = np.eye(n) - np.outer(k, h)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(k, k) * r


__all__ = ["StaticPPPFilterZTD"]
