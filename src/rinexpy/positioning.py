"""Single-point positioning (SPP) — iterative least-squares solver.

Given pseudorange observations and broadcast ephemeris, solve for the
receiver position (x, y, z) and clock bias dt. The implementation is the
classical 4-unknown linearised LSQ:

    rho = ||X_sv - X_rx|| + c * dt + e

iterated until ``|delta| < tol`` (default 1e-3 m). Standard SPP error
sources (ionosphere, troposphere, Earth rotation) are not corrected by
default; pass ``apply_iono=True`` plus a NAV dataset with broadcast
``ION ALPHA`` / ``ION BETA`` to apply Klobuchar.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from .geodesy import ecef_to_lla, klobuchar
from .gpstime import datetime_to_gps

log = logging.getLogger(__name__)

_C = 299_792_458.0  # speed of light, m/s


def spp_solve(
    sv_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    *,
    initial_guess: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_iter: int = 10,
    tol: float = 1e-3,
) -> dict:
    """Single-point positioning least-squares.

    Parameters
    ----------
    sv_ecef:
        ``(n_sv, 3)`` satellite ECEF positions in meters at signal-emission
        time. The caller is expected to have applied the standard
        light-time correction (typically ~70 ms ahead of receive time).
    pseudoranges:
        ``(n_sv,)`` measured pseudoranges in meters.
    initial_guess:
        ECEF receiver guess; default is Earth's center, which converges
        from anywhere on the planet within ~5 iterations.
    max_iter:
        Iteration cap. Raises ``RuntimeError`` if not converged in time.
    tol:
        Convergence tolerance on the position update norm, in meters.

    Returns
    -------
    dict
        ``{"position": (x, y, z) ECEF in m, "clock_bias": dt in s,
        "n_iter": int, "residuals": ndarray of size n_sv,
        "lla": (lat, lon, alt)}``.

    Raises
    ------
    ValueError
        If fewer than 4 satellites are supplied.
    RuntimeError
        If the iteration does not converge within ``max_iter``.
    """
    sv = np.asarray(sv_ecef, dtype=float)
    pr = np.asarray(pseudoranges, dtype=float)
    if sv.shape[0] < 4:
        raise ValueError("SPP needs >= 4 pseudoranges")

    state = np.array([initial_guess[0], initial_guess[1], initial_guess[2], 0.0])
    for it in range(max_iter):
        x, y, z, dt_s = state
        diff = sv - np.array([x, y, z])
        rho = np.linalg.norm(diff, axis=1)
        # Predicted pseudorange = geometric range + c * dt
        pred = rho + _C * dt_s
        residuals = pr - pred
        # Geometry matrix
        unit = -diff / rho[:, None]  # rows are -line-of-sight
        g = np.hstack([unit, np.ones((sv.shape[0], 1)) * _C])
        # Normal equations
        try:
            update, *_ = np.linalg.lstsq(g, residuals, rcond=None)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SPP normal equations singular: {e}") from e
        state += update
        if np.linalg.norm(update[:3]) < tol:
            x, y, z, dt_s = state
            try:
                lat, lon, alt = ecef_to_lla(x, y, z)
            except (ValueError, ZeroDivisionError):
                lat = lon = alt = float("nan")
            return {
                "position": (float(x), float(y), float(z)),
                "clock_bias": float(dt_s),
                "n_iter": it + 1,
                "residuals": residuals,
                "lla": (lat, lon, alt),
            }
    raise RuntimeError(f"SPP did not converge in {max_iter} iterations")


def apply_klobuchar_correction(
    pseudoranges: np.ndarray,
    sv_ecef: np.ndarray,
    rx_ecef: tuple[float, float, float],
    iono_alpha: tuple[float, float, float, float],
    iono_beta: tuple[float, float, float, float],
    epoch: datetime,
) -> np.ndarray:
    """Subtract the Klobuchar L1 ionospheric delay from each pseudorange.

    Parameters
    ----------
    pseudoranges:
        ``(n_sv,)`` measured pseudoranges in meters.
    sv_ecef:
        ``(n_sv, 3)`` satellite ECEF positions.
    rx_ecef:
        Approximate receiver ECEF for the az/el geometry. Re-running SPP
        after applying the correction is the standard pattern.
    iono_alpha, iono_beta:
        Klobuchar 8-coef parameters (4 + 4) from the GPS NAV header.
    epoch:
        Observation time (UTC).

    Returns
    -------
    ndarray
        Corrected pseudoranges (one per SV).
    """
    from .geodesy import azimuth_elevation

    az, el = azimuth_elevation(rx_ecef, sv_ecef)
    lat, lon, alt = ecef_to_lla(*rx_ecef)
    _, sow = datetime_to_gps(epoch)
    out = pseudoranges.copy()
    for i in range(len(pseudoranges)):
        if not np.isfinite(out[i]) or el[i] < 0:
            continue
        out[i] -= klobuchar(iono_alpha, iono_beta, (lat, lon, alt), float(az[i]), float(el[i]), sow)
    return out


__all__ = ["apply_klobuchar_correction", "spp_solve"]
