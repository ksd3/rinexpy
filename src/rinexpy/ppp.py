"""Single-receiver Precise Point Positioning (PPP) driver.

This module is the roadmap-named entry point for cm-level static PPP:

    rinexpy.ppp.ppp_solve(obs, sp3, clk, ...)

It composes the existing building blocks:

- :func:`rinexpy.interp.interpolate_sp3` for precise satellite positions.
- :func:`rinexpy.clk.interpolate_clk` for precise satellite clocks.
- :func:`rinexpy.geodesy.saastamoinen` for the wet+dry tropospheric
  slant delay.
- :class:`rinexpy.kalman.StaticPPPFilter` (a.k.a. ``GNSSFilter``) as
  the per-epoch sequential filter.

For each obs epoch, the driver:

1. Interpolates SP3 positions and CLK clocks to the epoch's timestamp.
2. Forms the dual-frequency iono-free code and phase combinations.
3. Computes per-SV slant tropospheric delay (Saastamoinen).
4. Feeds the iono-free observations to :meth:`StaticPPPFilter.update`.

The output is the final converged static position estimate plus a
per-epoch trace of position / clock bias values.

The function is deliberately liberal about which observation codes it
pulls from the obs Dataset - it walks a priority list of (L1-band,
L2-band) code/phase pairs (``C1C/C2W``, ``C1W/C2W``, ``C1C/C2L``,
``C1P/C2P``...) and uses the first complete pair it finds. Pass
``obs_codes={"code1", "code2", "phase1", "phase2"}`` to force a
specific combination.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

from .clk import interpolate_clk
from .geodesy import ecef_to_lla, saastamoinen
from .interp import interpolate_sp3
from .kalman import StaticPPPFilter
from .multifreq import F1, F2, LAMBDA_L1, LAMBDA_L2

log = logging.getLogger(__name__)

_C = 299_792_458.0
_ALPHA_IF = F1 ** 2 / (F1 ** 2 - F2 ** 2)
_BETA_IF = F2 ** 2 / (F1 ** 2 - F2 ** 2)

#: Priority list of (code-L1, code-L2, phase-L1, phase-L2) GPS RINEX 3
#: observation codes. Tried in order; first all-present quadruple wins.
_DEFAULT_CODE_PRIORITY = (
    ("C1C", "C2W", "L1C", "L2W"),
    ("C1W", "C2W", "L1W", "L2W"),
    ("C1C", "C2L", "L1C", "L2L"),
    ("C1C", "C2X", "L1C", "L2X"),
    ("C1P", "C2P", "L1P", "L2P"),
)


def _pick_obs_codes(obs: xr.Dataset) -> tuple[str, str, str, str] | None:
    """Return the first observation-code quadruple where every variable
    exists in the obs Dataset, or None."""
    have = set(obs.data_vars)
    for codes in _DEFAULT_CODE_PRIORITY:
        if all(c in have for c in codes):
            return codes
    return None


def _iono_free_code(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return _ALPHA_IF * p1 - _BETA_IF * p2


def _iono_free_phase(phi1_m: np.ndarray, phi2_m: np.ndarray) -> np.ndarray:
    return _ALPHA_IF * phi1_m - _BETA_IF * phi2_m


def ppp_solve(
    obs: xr.Dataset,
    sp3: xr.Dataset,
    clk: xr.Dataset,
    *,
    initial_position_ecef: tuple[float, float, float] | None = None,
    obs_codes: tuple[str, str, str, str] | None = None,
    sigma_code: float = 1.0,
    sigma_phase: float = 0.005,
    elevation_mask_deg: float = 7.0,
    max_epochs: int | None = None,
    apply_tropo: bool = True,
) -> dict[str, Any]:
    """Static-receiver PPP driver.

    Parameters
    ----------
    obs:
        RINEX-3 OBS xarray Dataset (as returned by
        :func:`rinexpy.rinexobs3`). Must contain ``time``, ``sv``, and
        at least one ``(code-L1, code-L2, phase-L1, phase-L2)``
        quadruple of GPS observation variables.
    sp3:
        SP3 xarray Dataset spanning the obs epochs.
    clk:
        CLK xarray Dataset spanning the obs epochs.
    initial_position_ecef:
        Receiver position prior (m). Defaults to the obs header's
        ``position`` attr, falling back to Earth's center.
    obs_codes:
        ``(code1, code2, phase1, phase2)`` to use. If omitted, the
        first all-present quadruple from :data:`_DEFAULT_CODE_PRIORITY`
        is picked.
    sigma_code, sigma_phase:
        Iono-free code and phase 1-sigma measurement noise (m). Defaults
        1.0 m and 5 mm.
    elevation_mask_deg:
        Minimum elevation for an SV to be used. SVs below this are
        masked out per-epoch. Default 7 deg.
    max_epochs:
        Cap on number of obs epochs to process (debugging knob).
        Default None (= every epoch).
    apply_tropo:
        Whether to apply the Saastamoinen tropospheric slant correction.
        Default True.

    Returns
    -------
    dict
        ``{"position": (x, y, z) ECEF in m, "lla": (lat, lon, alt),
        "clock_bias_s": float, "position_sigma_m": (sx, sy, sz),
        "n_epochs": int, "trace": list[dict] | None,
        "obs_codes": tuple, "filter": StaticPPPFilter}``. Each entry
        of ``trace`` is one epoch's ``{epoch, position, clock_bias_s}``.

    Raises
    ------
    ValueError
        If no valid observation-code quadruple is available in ``obs``,
        or the SP3 / CLK time ranges do not overlap the obs window.
    """
    codes = obs_codes or _pick_obs_codes(obs)
    if codes is None:
        raise ValueError(
            "No usable L1/L2 obs-code quadruple in the dataset; supply obs_codes=..."
        )
    code1_name, code2_name, phase1_name, phase2_name = codes

    # Restrict to GPS SVs - the iono-free L1/L2 frequencies above are
    # GPS-specific. Multi-GNSS PPP is a separate module.
    sv_labels = [str(s) for s in obs.sv.values]
    gps_mask = np.array([s.startswith("G") for s in sv_labels], dtype=bool)
    gps_indices = np.where(gps_mask)[0]
    if gps_indices.size == 0:
        raise ValueError("No GPS satellites in obs Dataset")
    gps_svs = [sv_labels[i] for i in gps_indices]
    n_sv = gps_indices.size

    if initial_position_ecef is None:
        approx = obs.attrs.get("position") or obs.attrs.get("approx_position")
        if approx is not None and len(approx) == 3:
            initial_position_ecef = tuple(float(x) for x in approx)
        else:
            initial_position_ecef = (0.0, 0.0, 0.0)

    flt = StaticPPPFilter(
        n_sv=n_sv,
        initial_position=initial_position_ecef,
        sigma_code=sigma_code,
        sigma_phase=sigma_phase,
    )

    c1 = obs[code1_name].values[:, gps_indices]
    c2 = obs[code2_name].values[:, gps_indices]
    l1 = obs[phase1_name].values[:, gps_indices]
    l2 = obs[phase2_name].values[:, gps_indices]

    times = obs.time.values
    if max_epochs is not None:
        times = times[:max_epochs]

    trace: list[dict[str, Any]] = []
    last_epoch: datetime | None = None
    for k, t64 in enumerate(times):
        # Skip epochs outside SP3 / CLK coverage.
        epoch = _as_datetime(t64)
        if (
            epoch < _as_datetime(sp3.time.values[0])
            or epoch > _as_datetime(sp3.time.values[-1])
        ):
            continue

        sp3_at = interpolate_sp3(sp3, epoch)
        sv_pos_full = sp3_at.position.values  # (n_sv_sp3, 3)
        sp3_sv_labels = [str(s) for s in sp3_at.sv.values]
        sp3_index = {s: i for i, s in enumerate(sp3_sv_labels)}

        sv_ecef = np.full((n_sv, 3), np.nan)
        sat_clock_s = np.full(n_sv, np.nan)
        for j, sv in enumerate(gps_svs):
            if sv in sp3_index:
                sv_ecef[j] = sv_pos_full[sp3_index[sv]]
                sat_clock_s[j] = interpolate_clk(clk, sv, epoch)

        # Iono-free code (m) and phase (m).
        pr1 = c1[k]
        pr2 = c2[k]
        ph1_m = l1[k] * LAMBDA_L1
        ph2_m = l2[k] * LAMBDA_L2
        pr_if = _iono_free_code(pr1, pr2)
        ph_if = _iono_free_phase(ph1_m, ph2_m)

        # Elevation mask + finite checks.
        rx_guess = np.array(flt.position)
        diff = sv_ecef - rx_guess
        rho = np.linalg.norm(diff, axis=1)
        rho[rho == 0] = np.nan
        # Elevation from the rover at the current iterate.
        try:
            lat, lon, alt = ecef_to_lla(*flt.position)
        except (ValueError, ZeroDivisionError):
            lat = lon = alt = 0.0
        elev_deg = _elevation_deg(rx_guess, sv_ecef)

        tropo_m = np.zeros(n_sv)
        if apply_tropo:
            for j in range(n_sv):
                if not np.isfinite(elev_deg[j]) or elev_deg[j] <= 0:
                    continue
                tropo_m[j] = saastamoinen(float(elev_deg[j]), alt)

        masked = ~np.isfinite(elev_deg) | (elev_deg < elevation_mask_deg)
        pr_if[masked] = np.nan
        ph_if[masked] = np.nan

        dt = 0.0 if last_epoch is None else max(0.0, (epoch - last_epoch).total_seconds())
        flt.predict(dt=dt)
        # Replace NaN sat positions with origin so the filter's
        # NaN-skip logic kicks in on the pr/ph side.
        sv_ecef_safe = np.where(np.isnan(sv_ecef), 0.0, sv_ecef)
        sat_clock_safe = np.where(np.isnan(sat_clock_s), 0.0, sat_clock_s)
        flt.update(sv_ecef_safe, sat_clock_safe, pr_if, ph_if, tropo_m=tropo_m)

        last_epoch = epoch
        trace.append({
            "epoch": epoch,
            "position": flt.position,
            "clock_bias_s": flt.clock_bias_s,
        })

    pos = flt.position
    sigma = flt.position_sigma
    try:
        lla = ecef_to_lla(*pos)
    except (ValueError, ZeroDivisionError):
        lla = (float("nan"),) * 3
    return {
        "position": tuple(float(x) for x in pos),
        "lla": lla,
        "clock_bias_s": float(flt.clock_bias_s),
        "position_sigma_m": tuple(float(x) for x in sigma),
        "n_epochs": len(trace),
        "trace": trace,
        "obs_codes": codes,
        "filter": flt,
    }


def _as_datetime(t) -> datetime:
    """Coerce a numpy datetime64 / datetime / xarray scalar to datetime."""
    if isinstance(t, datetime):
        return t
    return np.datetime64(t, "us").astype(object)


def _elevation_deg(rx_ecef: np.ndarray, sv_ecef: np.ndarray) -> np.ndarray:
    """Per-SV elevation in degrees from rx_ecef, vectorised over (n_sv, 3)."""
    try:
        lat, lon, _ = ecef_to_lla(*rx_ecef)
    except (ValueError, ZeroDivisionError):
        return np.full(sv_ecef.shape[0], np.nan)
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    sl, cl = np.sin(lat_r), np.cos(lat_r)
    sg, cg = np.sin(lon_r), np.cos(lon_r)
    R = np.array([
        [-sg, cg, 0.0],
        [-sl * cg, -sl * sg, cl],
        [cl * cg, cl * sg, sl],
    ])
    enu = (sv_ecef - rx_ecef) @ R.T
    horiz = np.linalg.norm(enu[:, :2], axis=1)
    elev = np.degrees(np.arctan2(enu[:, 2], horiz))
    elev[np.isnan(elev)] = -1.0
    return elev


__all__ = ["ppp_solve"]
