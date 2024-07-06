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
import math
from datetime import datetime

import numpy as np

from .geodesy import ecef_to_lla, klobuchar
from .gpstime import datetime_to_gps

log = logging.getLogger(__name__)

_C = 299_792_458.0  # speed of light, m/s


# Acklam's coefficients for the inverse standard-normal CDF.
_ACKLAM_A = (
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
)
_ACKLAM_B = (
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
)
_ACKLAM_C = (
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0,
)
_ACKLAM_D = (
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0,
)


def _norm_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Acklam). Accurate to ~1e-9 over (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"_norm_quantile: p must be in (0, 1), got {p}")
    p_low = 0.02425
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
        )
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
        (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
    )


def _chi2_quantile(p: float, df: int) -> float:
    """Wilson-Hilferty chi-squared inverse CDF.

    Accuracy is ~3% at df=5 and tightens to ~1% by df=10. Good enough
    for an integrity threshold; not a scipy.stats.chi2.ppf replacement.
    """
    if df < 1:
        raise ValueError(f"_chi2_quantile: df must be >= 1, got {df}")
    z = _norm_quantile(p)
    return df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3


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


def spp_solve_raim(
    sv_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    *,
    initial_guess: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_iter: int = 10,
    tol: float = 1e-3,
    sigma_pr: float = 5.0,
    p_fa: float = 1e-4,
    max_exclusions: int = 2,
) -> dict:
    """SPP with chi-squared RAIM fault detection and exclusion.

    Wraps :func:`spp_solve`. After each solve, computes the residual
    chi-squared test statistic and compares it against the threshold for
    the given false-alarm probability. If the test fails, drops the SV
    with the largest residual and re-solves, up to ``max_exclusions``
    times.

    Parameters
    ----------
    sv_ecef, pseudoranges, initial_guess, max_iter, tol:
        As for :func:`spp_solve`.
    sigma_pr:
        Assumed 1-sigma pseudorange noise in meters. Default 5.0.
    p_fa:
        Per-epoch false-alarm probability. Default 1e-4 (about one false
        alarm per 10,000 epochs). The chi-squared threshold is
        ``_chi2_quantile(1 - p_fa, df)`` with ``df = n_sv - 4``.
    max_exclusions:
        Cap on how many SVs to drop before giving up.

    Returns
    -------
    dict
        Same keys as :func:`spp_solve`, plus:

        - ``raim_test``: SSE divided by ``sigma_pr ** 2`` (chi-squared
          statistic).
        - ``raim_threshold``: chi-squared threshold for the kept SV set.
        - ``fault_detected``: True if any solve failed the test.
        - ``excluded_svs``: indices (into the input arrays) that were
          dropped.
        - ``raim_failed``: True if RAIM ran out of exclusions without a
          clean solve.

    Raises
    ------
    ValueError
        If fewer than 5 SVs are supplied (RAIM needs at least one degree
        of freedom).
    """
    sv = np.asarray(sv_ecef, dtype=float)
    pr = np.asarray(pseudoranges, dtype=float)
    n = sv.shape[0]
    if n < 5:
        raise ValueError("RAIM needs >= 5 pseudoranges to have any DoF")

    kept = list(range(n))
    excluded: list[int] = []
    fault_detected = False
    sol: dict = {}
    test = float("nan")
    threshold = float("nan")

    for _ in range(max_exclusions + 1):
        sub_sv = sv[kept]
        sub_pr = pr[kept]
        sol = spp_solve(sub_sv, sub_pr, initial_guess=initial_guess,
                        max_iter=max_iter, tol=tol)
        residuals = sol["residuals"]
        sse = float(np.sum(residuals * residuals))
        test = sse / (sigma_pr * sigma_pr)
        df = len(kept) - 4
        if df < 1:
            break
        threshold = _chi2_quantile(1.0 - p_fa, df)
        if test <= threshold:
            sol["raim_test"] = test
            sol["raim_threshold"] = threshold
            sol["fault_detected"] = fault_detected
            sol["excluded_svs"] = list(excluded)
            sol["raim_failed"] = False
            return sol
        fault_detected = True
        worst_in_sub = int(np.argmax(np.abs(residuals)))
        excluded.append(kept[worst_in_sub])
        kept.pop(worst_in_sub)
        if len(kept) < 5:
            break

    sol["raim_test"] = test
    sol["raim_threshold"] = threshold
    sol["fault_detected"] = True
    sol["excluded_svs"] = list(excluded)
    sol["raim_failed"] = True
    return sol


_GPS_F1 = 1575.42e6
_GPS_F2 = 1227.60e6
_GAMMA_L2 = (_GPS_F1 / _GPS_F2) ** 2


def tgd_from_nav(nav, epoch, *, field: str = "TGD") -> dict[str, float]:
    """Extract per-SV group-delay TGD values from a NAV dataset.

    For each SV, picks the latest broadcast record with ``time <= epoch``.
    SVs without a valid record at the query epoch (or with a NaN TGD) are
    omitted from the result.

    Parameters
    ----------
    nav:
        ``xarray.Dataset`` from ``rinexnav`` containing ``TGD`` (GPS),
        ``TGD1``/``TGD2`` (BeiDou), or ``BGDe5a``/``BGDe5b`` (Galileo).
    epoch:
        Query epoch (``datetime`` or ``numpy.datetime64``).
    field:
        Which broadcast field to pull. Default ``"TGD"`` for GPS.

    Returns
    -------
    dict
        ``{sv_label: tgd_seconds}``. Only entries with a valid record.
    """
    epoch_ns = (
        epoch if isinstance(epoch, np.datetime64) else np.datetime64(epoch, "ns")
    )
    out: dict[str, float] = {}
    if field not in nav:
        return out
    arr = nav[field]
    for sv in nav.sv.values:
        try:
            sv_arr = arr.sel(sv=sv).dropna(dim="time")
        except (KeyError, ValueError):
            continue
        if sv_arr.time.size == 0:
            continue
        valid = sv_arr.time.values <= epoch_ns
        if not valid.any():
            continue
        idx = int(np.flatnonzero(valid)[-1])
        v = float(sv_arr.values[idx])
        if np.isfinite(v):
            out[str(sv)] = v
    return out


def apply_tgd_correction(
    pseudoranges: np.ndarray,
    sv_labels: list[str],
    tgd_by_sv: dict[str, float],
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    """Subtract the broadcast group delay from each pseudorange.

    The standard correction is ``PR_corrected = PR - c * gamma * TGD``.
    For GPS, ``gamma=1`` on L1 and ``gamma=(f_L1/f_L2)**2`` on L2. For the
    ionosphere-free L1/L2 combination, TGD cancels and ``gamma=0`` leaves
    the pseudoranges unchanged.

    Parameters
    ----------
    pseudoranges:
        ``(n_sv,)`` pseudoranges in meters.
    sv_labels:
        SV identifiers, parallel to ``pseudoranges``. Entries missing from
        ``tgd_by_sv`` are passed through unchanged.
    tgd_by_sv:
        ``{sv_label: tgd_seconds}`` map, typically from :func:`tgd_from_nav`.
    gamma:
        Frequency scale. 1.0 for the primary frequency, ``(f1/f2)**2`` for
        the secondary, 0 for the ionosphere-free combination.

    Returns
    -------
    ndarray
        Corrected pseudoranges (a copy).
    """
    out = np.asarray(pseudoranges, dtype=float).copy()
    if gamma == 0.0:
        return out
    for i, sv in enumerate(sv_labels):
        tgd = tgd_by_sv.get(sv)
        if tgd is None or not np.isfinite(tgd):
            continue
        out[i] -= _C * gamma * tgd
    return out


__all__ = [
    "apply_klobuchar_correction",
    "apply_tgd_correction",
    "spp_solve",
    "spp_solve_raim",
    "tgd_from_nav",
]
