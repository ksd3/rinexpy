"""PPP-RTK fusion: parallel PPP filter + single-baseline RTK with
baseline-length-weighted blending of their position estimates.

Short-baseline RTK (rover within a few tens of km of a base station)
converges to cm precision in seconds thanks to the double-differences
cancelling the receiver / satellite clocks and most of the atmosphere.
Long-baseline PPP converges slower (tens of minutes) but doesn't need
a base. PPP-RTK fusion runs both simultaneously and blends:

- When the baseline is short, RTK dominates -- its sigma is essentially
  the carrier noise.
- When the baseline grows past ~50 km, the differential atmosphere
  bleeds back in, RTK sigma grows ~1 mm / km, and PPP starts to win.
- The natural inverse-variance weighting handles the crossover without
  any explicit switch.

This class is an orchestration layer over :class:`StaticPPPFilterZTD`
and :func:`rinexpy.rtk.double_difference_solve`. It does not introduce
a new estimator; the two underlying solvers run on their own state and
the fusion is at the position-output level only.
"""

from __future__ import annotations

import numpy as np

from .kalman_ztd import StaticPPPFilterZTD
from .multifreq import LAMBDA_L1
from .rtk import double_difference_solve


class PPPRTKFusion:
    """Parallel PPP + single-baseline RTK with inverse-variance blending.

    Parameters
    ----------
    n_sv:
        Maximum number of tracked satellites.
    initial_position:
        ECEF starting guess.
    base_position:
        Optional base station ECEF. Required to call
        :meth:`update_rtk`; can be set later via the attribute of the
        same name.
    rtk_sigma_floor_m:
        Floor on the RTK position sigma (default 1 cm). The model
        ``sigma = floor + ppm_per_km * baseline_km / 1000`` grows
        linearly past the floor.
    rtk_sigma_ppm_per_km:
        ppm scale for the RTK sigma growth model. Default 1 ppm =
        1 mm / km, which is a typical short-baseline standard.
    **ppp_kwargs:
        Forwarded to :class:`StaticPPPFilterZTD`.
    """

    def __init__(
        self,
        n_sv: int,
        initial_position: tuple[float, float, float],
        *,
        base_position: tuple[float, float, float] | None = None,
        rtk_sigma_floor_m: float = 0.01,
        rtk_sigma_ppm_per_km: float = 1.0,
        **ppp_kwargs,
    ) -> None:
        self.ppp = StaticPPPFilterZTD(n_sv, initial_position, **ppp_kwargs)
        self.base_position = (
            np.asarray(base_position, dtype=float) if base_position is not None else None
        )
        self.rtk_position: np.ndarray | None = None
        self.rtk_sigma_m: float = float("inf")
        self.rtk_sigma_floor_m = float(rtk_sigma_floor_m)
        self.rtk_sigma_ppm_per_km = float(rtk_sigma_ppm_per_km)

    @property
    def baseline_km(self) -> float | None:
        """Rover-base distance in km from the latest RTK solution, or
        ``None`` if no RTK fix has been taken yet."""
        if self.base_position is None or self.rtk_position is None:
            return None
        d = self.rtk_position - self.base_position
        return float(np.linalg.norm(d) / 1000.0)

    def update_ppp(
        self,
        sv_ecef: np.ndarray,
        sat_clock_s: np.ndarray,
        pr_if: np.ndarray,
        phase_if: np.ndarray,
        wet_mapping: np.ndarray,
        *,
        tropo_apriori_m: np.ndarray | None = None,
    ) -> None:
        """Run one PPP measurement update.

        Same arguments as :meth:`StaticPPPFilterZTD.update`; this is a
        thin pass-through.
        """
        self.ppp.update(
            sv_ecef, sat_clock_s, pr_if, phase_if,
            wet_mapping=wet_mapping,
            tropo_apriori_m=tropo_apriori_m,
        )

    def update_rtk(
        self,
        rover_pr: np.ndarray,
        base_pr: np.ndarray,
        rover_phase: np.ndarray,
        base_phase: np.ndarray,
        sv_positions_ecef: np.ndarray,
        *,
        wavelength: float = LAMBDA_L1,
    ) -> dict:
        """Run one single-baseline RTK fix against the configured base.

        Returns the same dict as :func:`rinexpy.rtk.double_difference_solve`;
        the rover position is cached as the live RTK estimate and the
        RTK sigma model is evaluated against the resulting baseline.
        """
        if self.base_position is None:
            raise ValueError("base_position must be set before update_rtk()")
        result = double_difference_solve(
            rover_pr, base_pr, rover_phase, base_phase,
            sv_positions_ecef=sv_positions_ecef,
            base_position_ecef=tuple(float(x) for x in self.base_position),
            wavelength=wavelength,
        )
        self.rtk_position = np.array(result["rover_position"])
        b_km = self.baseline_km or 0.0
        # sigma = floor + ppm * baseline_km
        # ppm = mm/km, so 1 ppm = 1 mm/km = 1e-3 m/km
        self.rtk_sigma_m = self.rtk_sigma_floor_m + (
            self.rtk_sigma_ppm_per_km * 1e-3 * b_km
        )
        return result

    @property
    def ppp_position(self) -> tuple[float, float, float]:
        return self.ppp.position

    @property
    def ppp_sigma(self) -> tuple[float, float, float]:
        return self.ppp.position_sigma

    @property
    def fused_position(self) -> tuple[float, float, float]:
        """Inverse-variance-weighted blend of the PPP and RTK positions.

        When no RTK fix has been taken yet, returns the PPP estimate as-is.
        """
        pos_ppp = np.array(self.ppp.position)
        if self.rtk_position is None:
            return tuple(float(x) for x in pos_ppp)
        sigma_ppp = np.array(self.ppp.position_sigma)
        # Per-axis inverse-variance weighting.
        w_ppp = 1.0 / (sigma_ppp ** 2)
        w_rtk = 1.0 / (self.rtk_sigma_m ** 2)
        fused = (w_ppp * pos_ppp + w_rtk * self.rtk_position) / (w_ppp + w_rtk)
        return tuple(float(x) for x in fused)

    @property
    def fused_sigma(self) -> tuple[float, float, float]:
        """Per-axis 1-sigma of the fused position."""
        sigma_ppp = np.array(self.ppp.position_sigma)
        if self.rtk_position is None:
            return tuple(float(x) for x in sigma_ppp)
        w_ppp = 1.0 / (sigma_ppp ** 2)
        w_rtk = 1.0 / (self.rtk_sigma_m ** 2)
        fused_var = 1.0 / (w_ppp + w_rtk)
        return tuple(float(np.sqrt(v)) for v in fused_var)

    @property
    def rtk_weight(self) -> float:
        """Fraction of total weight assigned to the RTK estimate (axis-
        averaged), in [0, 1]. 1.0 means RTK fully dominates the fused
        position; 0.0 means PPP fully dominates."""
        if self.rtk_position is None:
            return 0.0
        sigma_ppp = np.array(self.ppp.position_sigma)
        avg_var_ppp = float(np.mean(sigma_ppp ** 2))
        w_ppp = 1.0 / avg_var_ppp
        w_rtk = 1.0 / (self.rtk_sigma_m ** 2)
        return float(w_rtk / (w_ppp + w_rtk))


__all__ = ["PPPRTKFusion"]
