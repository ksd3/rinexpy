"""Time-transfer helpers: GPS common-view and P3 ionosphere-free
combination for receiver-clock comparison between two stations.

Common-view time transfer compares the receiver clocks at two stations
by tracking the same satellite simultaneously: the satellite clock
cancels in the (rx_A clock - rx_B clock) difference, leaving only the
propagation-delay difference + receiver biases + atmospheric residuals.

P3 combination cancels the first-order ionospheric delay using L1 + L2
pseudoranges with the standard

    P3 = (f1^2 * P1 - f2^2 * P2) / (f1^2 - f2^2)

with f1 = 1575.42 MHz, f2 = 1227.60 MHz for GPS. The output P3 carries
the geometric range + clocks + non-iono atmospheric delays.

Both routines work on epoch-aligned pseudorange observations across the
two stations; the caller is responsible for matching SVs and times
beforehand.
"""

from __future__ import annotations

import numpy as np

_C = 299_792_458.0
_F1_GPS = 1575.42e6
_F2_GPS = 1227.60e6


def p3_combination(
    p1_m: np.ndarray,
    p2_m: np.ndarray,
    *,
    f1: float = _F1_GPS,
    f2: float = _F2_GPS,
) -> np.ndarray:
    """Iono-free P3 pseudorange combination.

    Parameters
    ----------
    p1_m, p2_m:
        L1 and L2 pseudoranges in meters (same shape).
    f1, f2:
        Carrier frequencies. Defaults to GPS L1 / L2.

    Returns
    -------
    ndarray
        Iono-free P3 pseudorange in meters. NaN propagation.
    """
    p1 = np.asarray(p1_m, dtype=float)
    p2 = np.asarray(p2_m, dtype=float)
    if p1.shape != p2.shape:
        raise ValueError(f"P1 and P2 shape mismatch: {p1.shape} vs {p2.shape}")
    alpha = f1 ** 2 / (f1 ** 2 - f2 ** 2)
    beta = f2 ** 2 / (f1 ** 2 - f2 ** 2)
    return alpha * p1 - beta * p2


def common_view_difference(
    pr_station_a_m: np.ndarray,
    pr_station_b_m: np.ndarray,
    sv_ecef: np.ndarray,
    station_a_ecef: tuple[float, float, float],
    station_b_ecef: tuple[float, float, float],
) -> np.ndarray:
    """Common-view receiver-clock-difference estimate per SV.

    For an SV at known ECEF position, the pseudorange equations at
    stations A and B (with sat clock dt_sat and rx clocks dt_A, dt_B)
    are

        PR_A = |X_sv - X_A| + c * (dt_A - dt_sat) + iono_A + tropo_A + ...
        PR_B = |X_sv - X_B| + c * (dt_B - dt_sat) + iono_B + tropo_B + ...

    Subtracting cancels the satellite-clock term and gives

        PR_A - PR_B = (|X_sv - X_A| - |X_sv - X_B|) + c * (dt_A - dt_B)
                    + (atmospheric residuals)

    The geometric range difference is computable from known geometry;
    averaging the residual over many SVs yields ``c * (dt_A - dt_B)``.

    Parameters
    ----------
    pr_station_a_m, pr_station_b_m:
        ``(n_sv,)`` pseudoranges at each station.
    sv_ecef:
        ``(n_sv, 3)`` SV ECEF positions.
    station_a_ecef, station_b_ecef:
        Known ECEF positions of A and B.

    Returns
    -------
    ndarray
        ``(n_sv,)`` per-SV estimates of ``c * (dt_A - dt_B)`` (m).
        Take a median / mean over SVs for the final estimate.
    """
    pa = np.asarray(pr_station_a_m, dtype=float)
    pb = np.asarray(pr_station_b_m, dtype=float)
    sv = np.asarray(sv_ecef, dtype=float)
    xa = np.asarray(station_a_ecef, dtype=float)
    xb = np.asarray(station_b_ecef, dtype=float)
    rho_a = np.linalg.norm(sv - xa, axis=1)
    rho_b = np.linalg.norm(sv - xb, axis=1)
    return (pa - pb) - (rho_a - rho_b)


def estimate_clock_difference_s(
    pr_station_a_m: np.ndarray,
    pr_station_b_m: np.ndarray,
    sv_ecef: np.ndarray,
    station_a_ecef: tuple[float, float, float],
    station_b_ecef: tuple[float, float, float],
    *,
    estimator: str = "median",
) -> float:
    """One-shot estimate of ``dt_A - dt_B`` (seconds) from common-view.

    Wraps :func:`common_view_difference` and aggregates over SVs.
    """
    per_sv = common_view_difference(
        pr_station_a_m, pr_station_b_m, sv_ecef,
        station_a_ecef, station_b_ecef,
    )
    finite = per_sv[np.isfinite(per_sv)]
    if finite.size == 0:
        return float("nan")
    if estimator == "median":
        m = float(np.median(finite))
    elif estimator == "mean":
        m = float(np.mean(finite))
    else:
        raise ValueError(f"unknown estimator: {estimator!r}")
    return m / _C


__all__ = [
    "common_view_difference",
    "estimate_clock_difference_s",
    "p3_combination",
]
