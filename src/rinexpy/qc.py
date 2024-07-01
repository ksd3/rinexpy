"""Quality control: cycle slip detection on RINEX OBS observations.

Three detection methods, roughly in increasing data requirement:

1. ``detect_slips_phase_only(phi)``. Second time difference of a
   single phase signal. Cheap, but conflates slips with rapid
   ionospheric variation; best for short receivers and dense data.

2. ``detect_slips_geometry_free(phi1, phi2, ...)``. The GF
   combination ``lambda1 * phi1 - lambda2 * phi2`` removes geometry
   and the receiver/SV clocks. Slips show up as step discontinuities
   in the first time difference.

3. ``detect_slips_mw(phi1, phi2, p1, p2, ...)``. The Melbourne-
   Wuebbena combination is roughly constant between slips (about
   0.5-cycle RMS noise from the code term). A jump larger than the
   threshold flags a slip. Needs dual-frequency code AND carrier,
   and is the most reliable when available.

``detect_slips(obs)`` picks the best available method per SV based
on which variables are in the dataset.
"""

from __future__ import annotations

import numpy as np

_C = 299_792_458.0
_F_L1 = 1575.42e6
_F_L2 = 1227.60e6
_LAMBDA_L1 = _C / _F_L1
_LAMBDA_L2 = _C / _F_L2


def detect_slips_phase_only(
    phi_cycles: np.ndarray,
    *,
    threshold_cycles: float = 1.0,
) -> np.ndarray:
    """Flag cycle slips from a single-frequency carrier-phase series.

    Uses the second time difference: a step of 1 cycle in ``phi``
    appears as a step of 1 cycle in the first difference and as a
    delta of 1 cycle in the second difference. For clean data sampled
    densely enough that range rate barely changes between epochs,
    the second difference stays well below 1 cycle.

    Parameters
    ----------
    phi_cycles:
        ``(n_epoch,)`` carrier-phase in cycles. NaN where there's no
        measurement.
    threshold_cycles:
        Detection threshold on the absolute second difference.
        Default 1.0 catches obvious slips and lets quiet noise pass.

    Returns
    -------
    ndarray of bool, shape ``(n_epoch,)``
        True at epoch indices where a slip is detected.
    """
    phi = np.asarray(phi_cycles, dtype=float)
    n = len(phi)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out
    d2 = phi[2:] - 2.0 * phi[1:-1] + phi[:-2]
    out[2:] = np.isfinite(d2) & (np.abs(d2) > threshold_cycles)
    return out


def detect_slips_geometry_free(
    phi1_cycles: np.ndarray,
    phi2_cycles: np.ndarray,
    *,
    lambda1: float = _LAMBDA_L1,
    lambda2: float = _LAMBDA_L2,
    threshold_m: float = 0.05,
) -> np.ndarray:
    """Flag cycle slips via the geometry-free phase combination.

    ``GF = lambda1 * phi1 - lambda2 * phi2``. Removes geometry and
    the clocks; leaves the ionospheric delay difference plus
    integer-ambiguity terms. Between slips the ionospheric variation
    is slow (millimetres per second at most), so any per-epoch jump
    above ``threshold_m`` is suspicious.

    Parameters
    ----------
    phi1_cycles, phi2_cycles:
        ``(n_epoch,)`` carrier-phase in cycles on the two frequencies.
    lambda1, lambda2:
        Carrier wavelengths in meters. Defaults are GPS L1 and L2.
    threshold_m:
        First-difference threshold in meters. Default 0.05 m, well
        above typical ionospheric rates.

    Returns
    -------
    ndarray of bool, shape ``(n_epoch,)``
        True where a slip is detected.
    """
    phi1 = np.asarray(phi1_cycles, dtype=float)
    phi2 = np.asarray(phi2_cycles, dtype=float)
    gf = lambda1 * phi1 - lambda2 * phi2
    n = len(gf)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    dgf = np.diff(gf)
    out[1:] = np.isfinite(dgf) & (np.abs(dgf) > threshold_m)
    return out


def detect_slips_mw(
    phi1_cycles: np.ndarray,
    phi2_cycles: np.ndarray,
    p1_m: np.ndarray,
    p2_m: np.ndarray,
    *,
    f1: float = _F_L1,
    f2: float = _F_L2,
    threshold_cycles: float = 0.5,
) -> np.ndarray:
    """Flag cycle slips via the Melbourne-Wuebbena combination.

    In cycles of the wide-lane wavelength:

    ``MW = (phi1 - phi2) - (f1 - f2)/(f1 + f2) * (p1/lambda1 + p2/lambda2)``

    cancels geometry and most of the ionosphere, leaving
    ``N_WL = N1 - N2`` plus noise dominated by the code term. The
    first-difference noise scales linearly with code RMS: a 0.05 m
    geodetic receiver gives ~0.05 cycles RMS, while a 0.3 m consumer
    receiver gives ~0.65 cycles RMS and benefits from running-mean
    smoothing of MW before this function is called.

    Parameters
    ----------
    phi1_cycles, phi2_cycles:
        ``(n_epoch,)`` carrier-phase in cycles on the two frequencies.
    p1_m, p2_m:
        ``(n_epoch,)`` code pseudorange in meters on the same
        frequencies.
    f1, f2:
        Carrier frequencies in Hz. Defaults are GPS L1 and L2.
    threshold_cycles:
        First-difference threshold in wide-lane cycles. Default 0.5.

    Returns
    -------
    ndarray of bool, shape ``(n_epoch,)``
        True where a slip is detected.
    """
    phi1 = np.asarray(phi1_cycles, dtype=float)
    phi2 = np.asarray(phi2_cycles, dtype=float)
    p1 = np.asarray(p1_m, dtype=float)
    p2 = np.asarray(p2_m, dtype=float)
    lam1 = _C / f1
    lam2 = _C / f2
    mw = (phi1 - phi2) - (f1 - f2) / (f1 + f2) * (p1 / lam1 + p2 / lam2)
    n = len(mw)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    dmw = np.diff(mw)
    out[1:] = np.isfinite(dmw) & (np.abs(dmw) > threshold_cycles)
    return out


def _first_present(obs, candidates):
    for c in candidates:
        if c in obs:
            return c
    return None


def detect_slips(
    obs,
    *,
    threshold_cycles_mw: float = 0.5,
    threshold_m_gf: float = 0.05,
    threshold_cycles_phase: float = 1.0,
) -> dict:
    """Detect cycle slips across every SV in an OBS dataset.

    For each SV, picks the highest-quality method available given
    the observations present:

    - dual-freq code + carrier: MW
    - dual-freq carrier only: GF
    - single-freq carrier: phase-only second difference

    Parameters
    ----------
    obs:
        ``xarray.Dataset`` as returned by ``rinexobs``. Looks for the
        standard RINEX 3 measurement names (C1C, C2C, L1C, L2C) and
        the RINEX 2 fallbacks (C1, C2, P1, P2, L1, L2).
    threshold_cycles_mw, threshold_m_gf, threshold_cycles_phase:
        Per-method detection thresholds.

    Returns
    -------
    dict
        ``{"slips_by_sv": {sv: [epoch_idx, ...]}, "methods_by_sv":
        {sv: "mw" | "gf" | "phase"}}``.

    Raises
    ------
    ValueError
        If the dataset has no L1 carrier-phase variable.
    """
    l1_name = _first_present(obs, ("L1C", "L1"))
    l2_name = _first_present(obs, ("L2C", "L2W", "L2P", "L2"))
    p1_name = _first_present(obs, ("C1C", "C1", "P1"))
    p2_name = _first_present(obs, ("C2C", "C2W", "C2P", "P2"))

    if l1_name is None:
        raise ValueError(
            "detect_slips: no L1 carrier-phase variable in obs "
            "(looked for L1C, L1)"
        )

    has_sv = "sv" in obs[l1_name].dims
    sv_iter = obs[l1_name].sv.values if has_sv else [None]

    slips_by_sv: dict[str, list[int]] = {}
    methods_by_sv: dict[str, str] = {}

    for sv in sv_iter:
        def _get(name):
            if name is None:
                return None
            arr = obs[name]
            if has_sv:
                arr = arr.sel(sv=sv)
            return arr.values

        phi1 = _get(l1_name)
        phi2 = _get(l2_name)
        p1 = _get(p1_name)
        p2 = _get(p2_name)

        if phi2 is not None and p1 is not None and p2 is not None:
            mask = detect_slips_mw(
                phi1, phi2, p1, p2,
                threshold_cycles=threshold_cycles_mw,
            )
            method = "mw"
        elif phi2 is not None:
            mask = detect_slips_geometry_free(
                phi1, phi2,
                threshold_m=threshold_m_gf,
            )
            method = "gf"
        else:
            mask = detect_slips_phase_only(
                phi1,
                threshold_cycles=threshold_cycles_phase,
            )
            method = "phase"

        label = str(sv) if sv is not None else "<single>"
        slips_by_sv[label] = np.where(mask)[0].tolist()
        methods_by_sv[label] = method

    return {"slips_by_sv": slips_by_sv, "methods_by_sv": methods_by_sv}


__all__ = [
    "detect_slips",
    "detect_slips_geometry_free",
    "detect_slips_mw",
    "detect_slips_phase_only",
]
