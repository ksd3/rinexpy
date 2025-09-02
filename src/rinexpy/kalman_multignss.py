"""Multi-constellation PPP EKF with per-system inter-system biases.

The GPS clock state ``c * dt_rx`` parameterizes the receiver's clock
against a GPS reference. When the same receiver tracks Galileo / BeiDou
/ GLONASS, each non-GPS constellation has its own time scale and the
receiver hardware has a constellation-specific code-phase bias inside
its correlator. Both effects are absorbed into an inter-system bias
(ISB) state in meters:

    pr_obs[j]  = ||sv_j - p|| + c*dt_rx + ISB[const(j)]
                 + m_w(el_j) * ZWD + tropo_apriori[j] + e_p
    phi_obs[j] = ||sv_j - p|| + c*dt_rx + ISB[const(j)]
                 + m_w(el_j) * ZWD + tropo_apriori[j]
                 + N_iono_free[j] + e_phi

with ``const(j)`` mapping each SV slot to its constellation. The GPS
constellation has its ISB pinned to zero (it's defined as the reference
into which c*dt_rx is referenced). Other constellations get one ISB
state each, with random-walk dynamics (sigma_isb_rate_m_per_sqrt_hr).

State vector (n_isb = number of non-GPS constellations actually present):

    x = [px, py, pz, c*dt_rx, ZWD,
         ISB_const1, ISB_const2, ...,
         N_1, ..., N_n_sv]
"""

from __future__ import annotations

import numpy as np

from . import _native

_C = 299_792_458.0


class StaticPPPFilterMultiGNSS:
    """PPP EKF with per-constellation ISBs and a ZWD state.

    Parameters
    ----------
    n_sv:
        Maximum number of tracked satellites across all constellations.
    constellations:
        ``(n_sv,)`` array-like of constellation labels for each SV slot.
        Conventional letters: ``'G'`` GPS (reference, no ISB),
        ``'E'`` Galileo, ``'C'`` BeiDou, ``'R'`` GLONASS,
        ``'J'`` QZSS, ``'I'`` NavIC. Any label other than 'G' is
        allocated an ISB state.
    initial_position:
        ECEF starting guess.
    Other parameters:
        Same role as :class:`rinexpy.kalman_ztd.StaticPPPFilterZTD`.
    """

    def __init__(
        self,
        n_sv: int,
        constellations,
        initial_position: tuple[float, float, float],
        *,
        initial_zwd_m: float = 0.1,
        sigma_code: float = 1.0,
        sigma_phase: float = 0.005,
        sigma_position_init: float = 10.0,
        sigma_clock_init: float = 300.0,
        sigma_zwd_init: float = 0.5,
        sigma_isb_init: float = 100.0,
        sigma_clock_rate_m: float = 10.0,
        sigma_zwd_rate_m_per_sqrt_hr: float = 0.01,
        sigma_isb_rate_m_per_sqrt_hr: float = 0.0,
        sigma_position_rate_m: float = 0.0,
        sigma_ambig_init_m: float = 1000.0,
    ) -> None:
        if sigma_position_rate_m < 0:
            raise ValueError(
                f"sigma_position_rate_m must be >= 0, got {sigma_position_rate_m}"
            )
        consts = list(constellations)
        if len(consts) != n_sv:
            raise ValueError(
                f"constellations length {len(consts)} != n_sv {n_sv}"
            )
        self.n_sv = n_sv
        self.constellations = consts
        # Build the ISB index map: every non-'G' label that appears
        # gets one state, sorted by first appearance for stability.
        self._isb_map: dict[str, int] = {}
        for c in consts:
            if c != "G" and c not in self._isb_map:
                self._isb_map[c] = len(self._isb_map)
        self.n_isb = len(self._isb_map)
        # State layout: position (3), clock (1), ZWD (1), ISBs (n_isb), ambs.
        self._idx_clock = 3
        self._idx_zwd = 4
        self._idx_isb_start = 5
        self._idx_amb_start = 5 + self.n_isb
        self._n_state = self._idx_amb_start + n_sv

        self.x = np.zeros(self._n_state)
        self.x[:3] = np.asarray(initial_position, dtype=float)
        self.x[self._idx_zwd] = initial_zwd_m
        var = np.empty(self._n_state)
        var[:3] = sigma_position_init ** 2
        var[self._idx_clock] = sigma_clock_init ** 2
        var[self._idx_zwd] = sigma_zwd_init ** 2
        if self.n_isb:
            var[self._idx_isb_start : self._idx_amb_start] = sigma_isb_init ** 2
        var[self._idx_amb_start :] = sigma_ambig_init_m ** 2
        self.P = np.diag(var)

        self.sigma_code = sigma_code
        self.sigma_phase = sigma_phase
        self.sigma_clock_rate_m = sigma_clock_rate_m
        self.sigma_zwd_rate_m_per_sqrt_s = sigma_zwd_rate_m_per_sqrt_hr / np.sqrt(3600.0)
        self.sigma_isb_rate_m_per_sqrt_s = sigma_isb_rate_m_per_sqrt_hr / np.sqrt(3600.0)
        self.sigma_position_rate_m = sigma_position_rate_m
        self.sigma_ambig_init_m = sigma_ambig_init_m
        self._ambig_initialised = np.zeros(n_sv, dtype=bool)

    # ---- accessors ----

    @property
    def position(self) -> tuple[float, float, float]:
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def clock_bias_s(self) -> float:
        return float(self.x[self._idx_clock] / _C)

    @property
    def zwd_m(self) -> float:
        return float(self.x[self._idx_zwd])

    def isb_m(self, constellation: str) -> float:
        """Inter-system bias in meters for one constellation label.

        Returns 0.0 for GPS (the reference) and raises if the label
        isn't present in this filter's mapping.
        """
        if constellation == "G":
            return 0.0
        if constellation not in self._isb_map:
            raise KeyError(
                f"constellation {constellation!r} not in this filter; "
                f"available: {list(self._isb_map)}"
            )
        return float(self.x[self._idx_isb_start + self._isb_map[constellation]])

    @property
    def ambiguities_m(self) -> np.ndarray:
        return self.x[self._idx_amb_start :].copy()

    @property
    def position_sigma(self) -> tuple[float, float, float]:
        d = np.diag(self.P)
        return (float(np.sqrt(d[0])), float(np.sqrt(d[1])), float(np.sqrt(d[2])))

    # ---- dynamics ----

    def predict(self, dt: float) -> None:
        if dt < 0:
            raise ValueError(f"dt must be >= 0, got {dt}")
        self.P[self._idx_clock, self._idx_clock] += (
            self.sigma_clock_rate_m ** 2 * dt
        )
        self.P[self._idx_zwd, self._idx_zwd] += (
            self.sigma_zwd_rate_m_per_sqrt_s ** 2 * dt
        )
        for k in range(self.n_isb):
            i = self._idx_isb_start + k
            self.P[i, i] += self.sigma_isb_rate_m_per_sqrt_s ** 2 * dt
        if self.sigma_position_rate_m > 0.0:
            growth = self.sigma_position_rate_m ** 2 * dt
            for i in range(3):
                self.P[i, i] += growth

    def reset_ambiguity(self, sv_index: int) -> None:
        i = self._idx_amb_start + sv_index
        self.x[i] = 0.0
        self.P[i, :] = 0.0
        self.P[:, i] = 0.0
        self.P[i, i] = self.sigma_ambig_init_m ** 2
        self._ambig_initialised[sv_index] = False

    # ---- measurement update ----

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
            raise ValueError(f"shape mismatch (n_sv={self.n_sv} expected)")
        if tropo_apriori_m is None:
            tropo = np.zeros(n_sv)
        else:
            tropo = np.asarray(tropo_apriori_m, dtype=float)

        pr_corr = pr + _C * dt_sv - tropo
        ph_corr = ph + _C * dt_sv - tropo

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

        for j in range(n_sv):
            if not np.isfinite(ph_corr[j]):
                continue
            if self._ambig_initialised[j]:
                continue
            diff = sv[j] - self.x[:3]
            rho = float(np.linalg.norm(diff))
            if rho == 0.0:
                continue
            isb_val = self._isb_value_for_sv(j)
            self.x[self._idx_amb_start + j] = (
                ph_corr[j] - rho - self.x[self._idx_clock]
                - mw[j] * self.x[self._idx_zwd] - isb_val
            )
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
                                obs=ph_corr[j], rho=rho, m_wet=mw[j])

    # ---- internals ----

    def _isb_index_for_sv(self, sv_index: int) -> int | None:
        c = self.constellations[sv_index]
        if c == "G":
            return None
        return self._idx_isb_start + self._isb_map[c]

    def _isb_value_for_sv(self, sv_index: int) -> float:
        idx = self._isb_index_for_sv(sv_index)
        return 0.0 if idx is None else float(self.x[idx])

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
        isb_idx = self._isb_index_for_sv(sv_index)
        isb_val = self._isb_value_for_sv(sv_index)
        if code:
            r = self.sigma_code ** 2
            pred = rho + self.x[self._idx_clock] + m_wet * self.x[self._idx_zwd] + isb_val
        else:
            amb_idx = self._idx_amb_start + sv_index
            r = self.sigma_phase ** 2
            pred = (
                rho + self.x[self._idx_clock] + m_wet * self.x[self._idx_zwd]
                + isb_val + self.x[amb_idx]
            )
        innovation = obs - pred

        if _native.have_kalman_scalar_update():
            # Build the sparse H spec: [0,1,2] = u (LoS), clock idx = 1,
            # zwd idx = m_wet, optional ISB idx = 1, optional amb idx = 1.
            idx_list = [0, 1, 2, self._idx_clock, self._idx_zwd]
            val_list = [float(u[0]), float(u[1]), float(u[2]), 1.0, float(m_wet)]
            if isb_idx is not None:
                idx_list.append(isb_idx)
                val_list.append(1.0)
            if not code:
                idx_list.append(self._idx_amb_start + sv_index)
                val_list.append(1.0)
            idx = np.asarray(idx_list, dtype=np.int32)
            val = np.asarray(val_list, dtype=np.float64)
            if not (self.x.flags.c_contiguous and self.x.dtype == np.float64):
                self.x = np.ascontiguousarray(self.x, dtype=np.float64)
            if not (self.P.flags.c_contiguous and self.P.dtype == np.float64):
                self.P = np.ascontiguousarray(self.P, dtype=np.float64)
            _native.kalman_scalar_update_sparse(
                self.x, self.P, idx, val, float(innovation), r,
            )
            return

        n = self._n_state
        h = np.zeros(n)
        h[:3] = u
        h[self._idx_clock] = 1.0
        h[self._idx_zwd] = m_wet
        if isb_idx is not None:
            h[isb_idx] = 1.0
        if not code:
            h[self._idx_amb_start + sv_index] = 1.0

        ph = self.P @ h
        s = float(h @ ph + r)
        if s <= 0.0:
            return
        k = ph / s
        self.x = self.x + k * innovation
        I_KH = np.eye(n) - np.outer(k, h)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(k, k) * r


__all__ = ["StaticPPPFilterMultiGNSS"]
