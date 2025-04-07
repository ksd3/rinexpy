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
#: GPS L1 / L2 / L5 frequencies (Hz).
F1 = 1.57542e9
F2 = 1.22760e9
F5 = 1.17645e9

#: Carrier wavelengths (m).
LAMBDA_L1 = _C / F1
LAMBDA_L2 = _C / F2
LAMBDA_L5 = _C / F5
LAMBDA_WL = _C / (F1 - F2)
LAMBDA_NL = _C / (F1 + F2)
#: Extra-Wide-Lane wavelengths.
LAMBDA_EWL_15 = _C / (F1 - F5)
LAMBDA_EWL_25 = _C / (F2 - F5)


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


def fix_iono_free_ambiguity(
    p1_m: np.ndarray,
    p2_m: np.ndarray,
    phi1_cycles: np.ndarray,
    phi2_cycles: np.ndarray,
    float_iono_free_ambig_m: float,
    *,
    f1: float = F1,
    f2: float = F2,
    sigma_wl_cycles: float = 0.25,
    sigma_nl_cycles: float = 0.15,
) -> dict:
    """Resolve integer L1, L2 ambiguities given a PPP float iono-free ambiguity.

    Implements the standard PPP Wide-Lane / Narrow-Lane decomposition
    (Ge et al. 2008, Geng et al. 2010):

    1. Average the Melbourne-Wuebbena combination across all valid
       epochs in the arc to get a float ``N_WL = N1 - N2`` estimate.
       Round to the nearest integer; reject the fix if the fractional
       residual exceeds ``sigma_wl_cycles``.
    2. With ``N_WL`` held at its integer, derive the float ``N_NL =
       N1 + N2`` from the PPP float iono-free phase ambiguity in meters:

           B_IF = (lambda_WL/2) * N_WL + (lambda_NL/2) * N_NL

       so

           N_NL_float = 2 * (B_IF - (lambda_WL/2) * N_WL) / lambda_NL

    3. Round ``N_NL_float`` to the nearest integer; reject the fix if
       the fractional residual exceeds ``sigma_nl_cycles`` OR if
       ``N_NL`` and ``N_WL`` have different parities (the parity
       constraint enforces ``N1`` and ``N2`` being integer).
    4. Reconstruct ``N1 = (N_WL + N_NL) / 2`` and ``N2 = (N_NL - N_WL) / 2``.

    Parameters
    ----------
    p1_m, p2_m:
        Per-epoch code pseudoranges on L1 and L2, in meters.
    phi1_cycles, phi2_cycles:
        Per-epoch carrier phase on L1 and L2, in cycles. Must match
        ``p1_m`` / ``p2_m`` in length.
    float_iono_free_ambig_m:
        The PPP-derived float iono-free phase ambiguity, in meters,
        for this satellite. Typically from
        :func:`rinexpy.positioning.ppp_solve_static_batch`'s output.
    f1, f2:
        Carrier frequencies in Hz. Defaults are GPS L1, L2.
    sigma_wl_cycles, sigma_nl_cycles:
        Maximum fractional-cycle residual from the rounded integer.
        Defaults: 0.25 cycles for WL (~22 cm at lambda_WL = 86 cm),
        0.15 cycles for NL (~16 mm at lambda_NL = 10.7 cm).

    Returns
    -------
    dict
        ``{"n_wl": int | None, "n_nl": int | None, "n1": int | None,
        "n2": int | None, "wl_fixed": bool, "nl_fixed": bool,
        "parity_ok": bool, "fixed_iono_free_ambig_m": float | None,
        "float_n_wl": float, "float_n_nl": float}``.
        Integer fields are ``None`` if the fix isn't accepted; the
        ``float_*`` fields are always populated for diagnostics.
    """
    wl_fix = resolve_wide_lane(
        phi1_cycles, phi2_cycles, p1_m, p2_m, sigma_threshold=sigma_wl_cycles
    )
    # resolve_wide_lane operates per-element; for a single arc we average
    # the per-epoch MW values via the existing fixed_mask. Use the
    # median for robustness to outliers.
    mw = melbourne_wubbena(phi1_cycles, phi2_cycles, p1_m, p2_m)
    finite = np.isfinite(mw)
    if not finite.any():
        return {
            "n_wl": None, "n_nl": None, "n1": None, "n2": None,
            "wl_fixed": False, "nl_fixed": False, "parity_ok": False,
            "fixed_iono_free_ambig_m": None,
            "float_n_wl": float("nan"), "float_n_nl": float("nan"),
        }
    float_wl = float(np.median(mw[finite]) / LAMBDA_WL)
    n_wl_int = int(round(float_wl))
    wl_resid = abs(float_wl - n_wl_int)
    wl_fixed = wl_resid < sigma_wl_cycles

    lam_wl = _C / (f1 - f2)
    lam_nl = _C / (f1 + f2)
    float_nl = 2.0 * (float_iono_free_ambig_m - 0.5 * lam_wl * n_wl_int) / lam_nl
    n_nl_int = int(round(float_nl))
    nl_resid = abs(float_nl - n_nl_int)
    nl_fixed = nl_resid < sigma_nl_cycles
    parity_ok = (n_wl_int + n_nl_int) % 2 == 0

    if wl_fixed and nl_fixed and parity_ok:
        n1 = (n_wl_int + n_nl_int) // 2
        n2 = (n_nl_int - n_wl_int) // 2
        fixed_b_if = 0.5 * lam_wl * n_wl_int + 0.5 * lam_nl * n_nl_int
        return {
            "n_wl": n_wl_int,
            "n_nl": n_nl_int,
            "n1": n1,
            "n2": n2,
            "wl_fixed": True,
            "nl_fixed": True,
            "parity_ok": True,
            "fixed_iono_free_ambig_m": float(fixed_b_if),
            "float_n_wl": float_wl,
            "float_n_nl": float_nl,
        }
    return {
        "n_wl": n_wl_int if wl_fixed else None,
        "n_nl": None,
        "n1": None,
        "n2": None,
        "wl_fixed": wl_fixed,
        "nl_fixed": nl_fixed,
        "parity_ok": parity_ok,
        "fixed_iono_free_ambig_m": None,
        "float_n_wl": float_wl,
        "float_n_nl": float_nl,
    }


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


def extra_wide_lane_phase(
    phi2_cycles: np.ndarray, phi5_cycles: np.ndarray
) -> np.ndarray:
    """Extra-Wide-Lane (L2 - L5) carrier-phase combination in EWL cycles.

    The EWL wavelength is ~5.86 m, which makes the integer
    ``N_EWL = N2 - N5`` essentially trivial to resolve: one MW
    sample is enough at any reasonable noise level.
    """
    return np.asarray(phi2_cycles) - np.asarray(phi5_cycles)


def melbourne_wubbena_ewl(
    phi2_cycles: np.ndarray,
    phi5_cycles: np.ndarray,
    p2_m: np.ndarray,
    p5_m: np.ndarray,
    *,
    f2: float = F2,
    f5: float = F5,
) -> np.ndarray:
    """L2/L5 Melbourne-Wuebbena combination in EWL cycles."""
    lam2 = _C / f2
    lam5 = _C / f5
    phi_diff = np.asarray(phi2_cycles) - np.asarray(phi5_cycles)
    code_term = (
        (f2 - f5) / (f2 + f5)
        * (np.asarray(p2_m) / lam2 + np.asarray(p5_m) / lam5)
    )
    return phi_diff - code_term


def resolve_extra_wide_lane(
    mw_ewl_cycles: np.ndarray,
    *,
    threshold_cycles: float = 0.25,
) -> dict[str, np.ndarray]:
    """Round the EWL MW combination to integer N_EWL and validate.

    Returns dict with ``N_EWL`` (int array), ``fixed_mask``, and
    ``fraction_fixed``. SVs whose |float - round| exceeds
    ``threshold_cycles`` are flagged as not-fixed.
    """
    mw = np.asarray(mw_ewl_cycles, dtype=float)
    n_ewl = np.round(np.where(np.isnan(mw), 0.0, mw)).astype(int)
    fixed = np.isfinite(mw) & (np.abs(mw - n_ewl) <= threshold_cycles)
    return {
        "N_EWL": n_ewl,
        "fixed_mask": fixed,
        "fraction_fixed": float(np.mean(fixed)) if fixed.size else 0.0,
    }


def tcar_resolve(
    phi1_cycles: np.ndarray,
    phi2_cycles: np.ndarray,
    phi5_cycles: np.ndarray,
    p1_m: np.ndarray,
    p2_m: np.ndarray,
    p5_m: np.ndarray,
    *,
    f1: float = F1,
    f2: float = F2,
    f5: float = F5,
    threshold_ewl: float = 0.25,
    threshold_wl: float = 0.25,
) -> dict[str, np.ndarray]:
    """Three-Carrier Ambiguity Resolution (Forssell / Vollath 1997).

    Cascade:

    1. EWL (L2-L5) via MW: ~5.86 m wavelength, single-epoch fix.
    2. WL (L1-L2) via MW: ~0.86 m wavelength.
    3. NL bootstrap to recover N_L1, N_L2, N_L5 integer triples.

    Returns dict with all three integer ambiguity vectors plus
    per-stage and combined fixed masks.
    """
    mw_ewl = melbourne_wubbena_ewl(phi2_cycles, phi5_cycles, p2_m, p5_m, f2=f2, f5=f5)
    ewl = resolve_extra_wide_lane(mw_ewl, threshold_cycles=threshold_ewl)
    n_ewl = ewl["N_EWL"]
    fixed_ewl = ewl["fixed_mask"]

    # melbourne_wubbena() in this module returns meters; convert to WL
    # cycles by dividing by LAMBDA_WL before rounding.
    mw_wl_m = melbourne_wubbena(phi1_cycles, phi2_cycles, p1_m, p2_m)
    mw_wl_cycles = mw_wl_m / LAMBDA_WL
    n_wl = np.round(np.where(np.isnan(mw_wl_cycles), 0.0, mw_wl_cycles)).astype(int)
    fixed_wl = np.isfinite(mw_wl_cycles) & (np.abs(mw_wl_cycles - n_wl) <= threshold_wl)

    nl = narrow_lane_phase(phi1_cycles, phi2_cycles)
    n_nl = np.round(np.where(np.isnan(nl), 0.0, nl)).astype(int)
    # { N1 - N2 = N_WL,  N1 + N2 = N_NL } -> N1 = (N_NL + N_WL)/2
    n_l1 = (n_nl + n_wl) // 2
    n_l2 = n_l1 - n_wl
    n_l5 = n_l2 - n_ewl

    fixed = fixed_ewl & fixed_wl
    return {
        "N_EWL": n_ewl,
        "N_WL": n_wl,
        "N_L1": n_l1,
        "N_L2": n_l2,
        "N_L5": n_l5,
        "fixed_mask_ewl": fixed_ewl,
        "fixed_mask_wl": fixed_wl,
        "fixed_mask": fixed,
        "fraction_fixed": float(np.mean(fixed)) if fixed.size else 0.0,
    }


__all__ = [
    "F1",
    "F2",
    "F5",
    "LAMBDA_EWL_15",
    "LAMBDA_EWL_25",
    "LAMBDA_L1",
    "LAMBDA_L2",
    "LAMBDA_L5",
    "LAMBDA_NL",
    "LAMBDA_WL",
    "extra_wide_lane_phase",
    "fix_iono_free_ambiguity",
    "lambda_dual_freq",
    "melbourne_wubbena",
    "melbourne_wubbena_ewl",
    "narrow_lane_phase",
    "resolve_extra_wide_lane",
    "resolve_wide_lane",
    "split_wl_into_l1_l2",
    "tcar_resolve",
    "wide_lane_phase",
]
