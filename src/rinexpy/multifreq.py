"""Dual-frequency carrier-phase combinations and ambiguity resolution.

For L1+L2 GPS the standard linear combinations of carrier-phase
observables exploit the wide separation in wavelength to break the
single-frequency ambiguity-fix problem in two:

- **Wide-Lane (WL)** — long wavelength (~86 cm), the integer ambiguity
  ``N_WL = N1 - N2`` is easy to fix even from noisy phase alone, and
  the **Melbourne-Wübbena (MW)** combination of phase and code makes
  the float estimate iono-free and geometry-free, so a single epoch is
  often enough.
- **Narrow-Lane (NL)** — short wavelength (~10.7 cm), the integer
  ambiguity ``N_NL = N1 + N2`` is harder. Once ``N_WL`` is fixed,
  ``N1 = (N_WL + N_NL)/2`` decouples nicely.

Frequencies (GPS L-band):

- f1 = 1575.42 MHz, λ1 = c/f1 ≈ 0.190294 m
- f2 = 1227.60 MHz, λ2 = c/f2 ≈ 0.244210 m
- f_WL = f1 - f2 = 347.82 MHz, λ_WL ≈ 0.861918 m
- f_NL = f1 + f2 = 2803.02 MHz, λ_NL ≈ 0.106953 m
"""

from __future__ import annotations

import numpy as np

#: Speed of light, m/s.
_C = 299_792_458.0
#: GPS L1 / L2 frequencies (Hz).
F1 = 1.57542e9
F2 = 1.22760e9

#: Carrier wavelengths (m).
LAMBDA_L1 = _C / F1
LAMBDA_L2 = _C / F2
LAMBDA_WL = _C / (F1 - F2)
LAMBDA_NL = _C / (F1 + F2)


def wide_lane_phase(phi1_cycles: np.ndarray, phi2_cycles: np.ndarray) -> np.ndarray:
    """Wide-Lane carrier-phase combination in cycles of lambda_WL.

    Parameters
    ----------
    phi1_cycles, phi2_cycles:
        L1 and L2 carrier-phase observations in cycles.

    Returns
    -------
    ndarray
        ``phi1 - phi2`` in cycles. For zero geometric range (or once
        geometry is differenced out, e.g. by a baseline solver) the
        result equals ``N1 - N2`` exactly.
    """
    return np.asarray(phi1_cycles) - np.asarray(phi2_cycles)


def narrow_lane_phase(phi1_cycles: np.ndarray, phi2_cycles: np.ndarray) -> np.ndarray:
    """Narrow-Lane carrier-phase combination in cycles of lambda_NL.

    Returns ``phi1 + phi2`` (cycles). Pairs with the WL combination to
    decouple the L1/L2 ambiguities (``N1 = (N_WL + N_NL) / 2``).
    """
    return np.asarray(phi1_cycles) + np.asarray(phi2_cycles)


def melbourne_wubbena(
    phi1_cycles: np.ndarray,
    phi2_cycles: np.ndarray,
    p1_m: np.ndarray,
    p2_m: np.ndarray,
) -> np.ndarray:
    """Melbourne-Wuebbena (MW) combination, in meters.

    Parameters
    ----------
    phi1_cycles, phi2_cycles:
        L1 / L2 carrier-phase observations in cycles.
    p1_m, p2_m:
        L1 / L2 pseudorange observations in meters.

    Returns
    -------
    ndarray
        ``lambda_WL * N_WL + noise`` in meters. The combination removes
        both the geometric range and the ionospheric delay, so a
        single-epoch round-to-nearest-integer of ``MW / lambda_WL`` is
        usually a reliable Wide-Lane ambiguity estimate.

    Notes
    -----
    Per Melbourne (1985) and Wuebbena (1985):

        MW = lambda_WL * (Phi1 - Phi2) - (f1*P1 + f2*P2) / (f1 + f2)

    where Phi_i are phase observations in cycles. The phase term equals
    the WL phase in meters; the code term is the iono-free Narrow-Lane
    pseudorange in meters; their difference cancels geometry and
    ionosphere, leaving only the WL ambiguity.
    """
    phi_diff_cycles = np.asarray(phi1_cycles) - np.asarray(phi2_cycles)
    phase_combo_m = LAMBDA_WL * phi_diff_cycles
    code_combo_m = (F1 * np.asarray(p1_m) + F2 * np.asarray(p2_m)) / (F1 + F2)
    return phase_combo_m - code_combo_m


def resolve_wide_lane(
    phi1_cycles: np.ndarray,
    phi2_cycles: np.ndarray,
    p1_m: np.ndarray,
    p2_m: np.ndarray,
    *,
    sigma_threshold: float = 0.25,
) -> dict:
    """Round-to-nearest WL ambiguity from the Melbourne-Wübbena combination.

    Parameters
    ----------
    phi1_cycles, phi2_cycles, p1_m, p2_m:
        Per-satellite L1/L2 phase (cycles) and pseudorange (m).
    sigma_threshold:
        Maximum acceptable per-SV deviation from the rounded integer,
        in cycles of λ_WL. Anything larger is flagged as unfixed.
        Default 0.25 cycles (~22 cm).

    Returns
    -------
    dict
        ``{"N_WL": ndarray[int], "float_WL": ndarray, "fixed_mask": ndarray[bool],
        "fraction_fixed": float}``. ``N_WL`` is the rounded integer
        ambiguity per SV (zero where the round-to-int is rejected by
        ``sigma_threshold``); ``fixed_mask`` is True where the fix is
        confident.
    """
    mw = melbourne_wubbena(phi1_cycles, phi2_cycles, p1_m, p2_m)
    float_n = mw / LAMBDA_WL
    n_int = np.round(float_n).astype(int)
    residual = float_n - n_int
    fixed_mask = np.abs(residual) < sigma_threshold
    n_int_safe = n_int.copy()
    n_int_safe[~fixed_mask] = 0
    fraction = float(np.mean(fixed_mask)) if fixed_mask.size else 0.0
    return {
        "N_WL": n_int_safe,
        "float_WL": float_n,
        "fixed_mask": fixed_mask,
        "fraction_fixed": fraction,
    }


def split_wl_into_l1_l2(n_wl: np.ndarray, n_nl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(N1, N2)`` from ``(N_WL = N1-N2, N_NL = N1+N2)``.

    Both inputs must already be integer-valued. The result is also
    integer (per definition of the lane combinations).

    Parameters
    ----------
    n_wl, n_nl:
        Integer Wide-Lane and Narrow-Lane ambiguities.

    Returns
    -------
    n1, n2 : ndarray[int]
        Per-frequency integer ambiguities.

    Raises
    ------
    ValueError
        If any element of ``n_wl + n_nl`` is odd (which would mean the
        decomposition is inconsistent and at least one of WL or NL is
        wrong).
    """
    n_wl = np.asarray(n_wl, dtype=int)
    n_nl = np.asarray(n_nl, dtype=int)
    s = n_wl + n_nl
    if np.any(s % 2 != 0):
        raise ValueError(
            "N_WL + N_NL must be even (otherwise the decomposition is inconsistent)"
        )
    n1 = s // 2
    n2 = n1 - n_wl
    return n1, n2


def lambda_dual_freq(
    a_l1_float: np.ndarray,
    a_l2_float: np.ndarray,
    cov_block: np.ndarray | None = None,
    *,
    p1_m: np.ndarray | None = None,
    p2_m: np.ndarray | None = None,
    sigma_threshold: float = 0.25,
) -> dict:
    """Dual-frequency LAMBDA-style integer ambiguity resolution.

    Strategy:

    1. If pseudoranges are supplied, fix Wide-Lane via Melbourne-Wübbena
       (a single-epoch geometry-and-iono-free combination).
    2. Otherwise estimate float ``N_WL = N_L1 - N_L2`` from the float
       ambiguities and round.
    3. Use the WL constraint to recover the L1 ambiguity from the float
       L1 estimate (round to nearest integer; the WL-fixed N2 follows
       directly).

    Parameters
    ----------
    a_l1_float, a_l2_float:
        Float ambiguity vectors for L1 and L2, one entry per satellite.
    cov_block:
        Optional ``(2*n, 2*n)`` joint covariance of the stacked
        ``[a_L1, a_L2]`` vectors. Currently used only for diagnostic
        output; the WL/NL approach is not Cholesky-based.
    p1_m, p2_m:
        Optional L1/L2 pseudorange observations. When supplied, WL is
        resolved via :func:`resolve_wide_lane` (geometry-and-iono-free).
        Otherwise WL is rounded from ``a_l1 - a_l2``.
    sigma_threshold:
        Maximum acceptable rounding residual (cycles) for an SV to
        count as "fixed".

    Returns
    -------
    dict
        ``{"N_L1": ndarray[int], "N_L2": ndarray[int],
        "N_WL": ndarray[int], "fixed_mask": ndarray[bool],
        "fraction_fixed": float}``.
    """
    a_l1 = np.asarray(a_l1_float, dtype=float)
    a_l2 = np.asarray(a_l2_float, dtype=float)
    if a_l1.shape != a_l2.shape:
        raise ValueError("L1 and L2 float ambiguity vectors must have same shape")

    # Step 1: Wide-Lane.
    if p1_m is not None and p2_m is not None:
        wl = resolve_wide_lane(
            a_l1, a_l2, p1_m, p2_m, sigma_threshold=sigma_threshold
        )
        n_wl = wl["N_WL"]
        wl_mask = wl["fixed_mask"]
    else:
        float_wl = a_l1 - a_l2
        n_wl = np.round(float_wl).astype(int)
        wl_residual = float_wl - n_wl
        wl_mask = np.abs(wl_residual) < sigma_threshold

    # Step 2: round L1 to nearest integer; N2 = N1 - N_WL.
    n_l1 = np.round(a_l1).astype(int)
    l1_residual = a_l1 - n_l1
    l1_mask = np.abs(l1_residual) < sigma_threshold

    fixed = wl_mask & l1_mask
    n_l2 = n_l1 - n_wl
    # Where unfixed, zero everything to avoid downstream surprises.
    n_l1_out = np.where(fixed, n_l1, 0)
    n_l2_out = np.where(fixed, n_l2, 0)
    n_wl_out = np.where(fixed, n_wl, 0)
    _ = cov_block  # accepted for API symmetry; unused in this routine

    return {
        "N_L1": n_l1_out,
        "N_L2": n_l2_out,
        "N_WL": n_wl_out,
        "fixed_mask": fixed,
        "fraction_fixed": float(np.mean(fixed)) if fixed.size else 0.0,
    }


__all__ = [
    "F1",
    "F2",
    "LAMBDA_L1",
    "LAMBDA_L2",
    "LAMBDA_NL",
    "LAMBDA_WL",
    "lambda_dual_freq",
    "melbourne_wubbena",
    "narrow_lane_phase",
    "resolve_wide_lane",
    "split_wl_into_l1_l2",
    "wide_lane_phase",
]
