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

from . import _native

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


def mp1(
    p1_m: np.ndarray,
    l1_m: np.ndarray,
    l2_m: np.ndarray,
    *,
    f1: float = _F_L1,
    f2: float = _F_L2,
) -> np.ndarray:
    """TEQC-style MP1 multipath combination on the primary frequency.

    ``MP1 = P1 - (1 + 2/(alpha - 1)) * L1 + (2/(alpha - 1)) * L2``

    with ``alpha = (f1/f2)^2``. Cancels the geometric range, the
    troposphere, and the satellite and receiver clocks. What's left is
    the carrier-phase ambiguity (a constant between slips), a slow
    ionospheric drift, and the code multipath + code noise.

    Parameters
    ----------
    p1_m:
        ``(n_epoch,)`` primary-frequency pseudorange in meters.
    l1_m, l2_m:
        ``(n_epoch,)`` carrier phase on the two frequencies, in meters
        (cycles times the wavelength).
    f1, f2:
        Frequencies in Hz. Defaults are GPS L1, L2.

    Returns
    -------
    ndarray
        ``(n_epoch,)`` MP1 series, in meters.
    """
    alpha = (f1 / f2) ** 2
    k = 2.0 / (alpha - 1.0)
    return np.asarray(p1_m, float) - (1.0 + k) * np.asarray(l1_m, float) + k * np.asarray(l2_m, float)


def mp2(
    p2_m: np.ndarray,
    l1_m: np.ndarray,
    l2_m: np.ndarray,
    *,
    f1: float = _F_L1,
    f2: float = _F_L2,
) -> np.ndarray:
    """TEQC-style MP2 multipath combination on the secondary frequency.

    ``MP2 = P2 - (2*alpha/(alpha - 1)) * L1 + (2*alpha/(alpha - 1) - 1) * L2``

    with ``alpha = (f1/f2)^2``. Same cancellation properties as
    :func:`mp1` but on the second-frequency code.
    """
    alpha = (f1 / f2) ** 2
    k = 2.0 * alpha / (alpha - 1.0)
    return np.asarray(p2_m, float) - k * np.asarray(l1_m, float) + (k - 1.0) * np.asarray(l2_m, float)


def multipath_rms(mp_m: np.ndarray, slips: np.ndarray | None = None) -> float:
    """Per-arc RMS of an MP series, averaged across arcs.

    Splits ``mp_m`` at the ``slips`` indices (or treats the whole series
    as a single arc when ``slips`` is None), subtracts each arc's mean,
    and returns the average RMS across arcs. NaN samples are dropped.
    Arcs shorter than 2 valid samples are skipped.

    Parameters
    ----------
    mp_m:
        ``(n_epoch,)`` MP series from :func:`mp1` or :func:`mp2`.
    slips:
        Optional ``(n_epoch,)`` bool mask marking cycle-slip epochs;
        each True splits the series into a new arc.

    Returns
    -------
    float
        Arc-averaged RMS in meters. NaN if no arc has enough samples.
    """
    mp = np.asarray(mp_m, float)
    n = len(mp)
    if slips is None:
        boundaries = [0, n]
    else:
        slip_idx = np.flatnonzero(np.asarray(slips, bool)).tolist()
        boundaries = [0, *slip_idx, n]
    rms: list[float] = []
    for i in range(len(boundaries) - 1):
        arc = mp[boundaries[i] : boundaries[i + 1]]
        arc = arc[np.isfinite(arc)]
        if arc.size < 2:
            continue
        rms.append(float(np.std(arc - arc.mean())))
    if not rms:
        return float("nan")
    return float(np.mean(rms))


def hatch_filter(
    pr_m: np.ndarray,
    phase_m: np.ndarray,
    *,
    window: int = 100,
    slips: np.ndarray | None = None,
) -> np.ndarray:
    """Carrier-smooth a code pseudorange series via the Hatch filter.

    The recursion is

        P_s[k] = (P[k] + (m - 1) * (P_s[k-1] + (phi[k] - phi[k-1]))) / m

    where ``m`` ramps from 1 up to ``window`` as samples accumulate.
    The carrier phase tracks range cleanly between slips, so the
    smoothed pseudorange inherits carrier-grade noise while keeping
    the code's absolute scale (no integer ambiguity).

    Parameters
    ----------
    pr_m:
        ``(n_epoch,)`` raw code pseudorange in meters. NaN where
        there's no observation.
    phase_m:
        ``(n_epoch,)`` carrier phase in meters (cycles times the
        wavelength). NaN where there's no observation.
    window:
        Maximum smoothing window in epochs. Default 100 (a common
        choice; larger windows give smoother output but more lag if
        the ionosphere drifts).
    slips:
        Optional ``(n_epoch,)`` bool mask. ``True`` at any epoch
        resets the filter to take the raw ``pr_m`` value (i.e. as
        if the SV had just been reacquired). Pair with the output
        of :func:`detect_slips_mw` or similar.

    Returns
    -------
    ndarray
        ``(n_epoch,)`` smoothed pseudorange in meters. NaN epochs in
        either input map to NaN in the output and reset the filter.

    Raises
    ------
    ValueError
        If ``pr_m`` and ``phase_m`` differ in shape, or ``window < 1``.
    """
    pr = np.ascontiguousarray(pr_m, dtype=float)
    phi = np.ascontiguousarray(phase_m, dtype=float)
    if pr.shape != phi.shape:
        raise ValueError(
            f"pr_m and phase_m must match shape; got {pr.shape} vs {phi.shape}"
        )
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if _native.have_hatch_filter():
        if slips is None:
            slips_u8 = np.empty(0, dtype=np.uint8)
        else:
            slips_u8 = np.ascontiguousarray(
                np.asarray(slips).astype(bool), dtype=np.uint8,
            )
        return np.asarray(_native.hatch_filter(pr, phi, slips_u8, window))

    n = len(pr)
    out = np.full(n, np.nan)
    m = 0
    prev_phi = np.nan
    for k in range(n):
        if not np.isfinite(pr[k]) or not np.isfinite(phi[k]):
            m = 0
            prev_phi = np.nan
            continue
        slip_here = slips is not None and bool(slips[k])
        if m == 0 or slip_here or not np.isfinite(prev_phi):
            out[k] = pr[k]
            m = 1
        else:
            m = min(m + 1, window)
            out[k] = (pr[k] + (m - 1) * (out[k - 1] + (phi[k] - prev_phi))) / m
        prev_phi = phi[k]
    return out


def repair_slips(
    phase_cycles: np.ndarray,
    slips: np.ndarray,
    *,
    fit_window: int = 5,
) -> np.ndarray:
    """Repair integer cycle slips in a single-signal phase series.

    For each slip flagged in ``slips``, a least-squares line is fit to
    the last ``fit_window`` pre-slip epochs, the expected phase at the
    slip epoch is predicted by extrapolation, and the rounded integer
    residual (in cycles) is subtracted from all subsequent phase
    samples. The detector decides where the slips are; this routine
    is just the integer-jump correction.

    Reliability notes:

    - The linear-extrapolation model breaks down at low elevation where
      the ionosphere accelerates. For long-arc repair, prefer a dual-
      frequency MW+GF estimate (see :func:`repair_slips_dual`).
    - Slips closer together than ``fit_window`` epochs are repaired
      against the already-repaired segment, which works only if the
      first slip was estimated correctly.

    Parameters
    ----------
    phase_cycles:
        ``(n_epoch,)`` carrier phase in cycles. NaN where there's no
        observation.
    slips:
        ``(n_epoch,)`` bool mask. ``True`` at every epoch where a slip
        was detected (typically the output of :func:`detect_slips_mw`
        or similar).
    fit_window:
        Number of pre-slip epochs used for the extrapolation fit.
        Default 5; use larger for cleaner data, smaller for fast-
        varying conditions.

    Returns
    -------
    ndarray
        ``(n_epoch,)`` repaired phase in cycles. Same shape and NaN
        pattern as the input; finite epochs after a detected slip are
        shifted by an integer.
    """
    phi = np.asarray(phase_cycles, dtype=float).copy()
    if phi.size == 0:
        return phi
    slips = np.asarray(slips, dtype=bool)
    slip_indices = np.flatnonzero(slips)
    for k in slip_indices:
        if k < 2:
            continue
        pre_end = k - 1
        pre_start = max(0, pre_end - fit_window + 1)
        x = np.arange(pre_start, pre_end + 1, dtype=float)
        y = phi[pre_start : pre_end + 1]
        finite = np.isfinite(y)
        if finite.sum() < 2:
            continue
        slope, intercept = np.polyfit(x[finite], y[finite], 1)
        if not np.isfinite(phi[k]):
            continue
        predicted = slope * k + intercept
        delta = round(phi[k] - predicted)
        if delta != 0:
            tail = np.isfinite(phi[k:])
            phi[k:] = np.where(tail, phi[k:] - delta, phi[k:])
    return phi


def repair_slips_dual(
    phi1_cycles: np.ndarray,
    phi2_cycles: np.ndarray,
    p1_m: np.ndarray,
    p2_m: np.ndarray,
    slips: np.ndarray,
    *,
    f1: float = _F_L1,
    f2: float = _F_L2,
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-frequency cycle slip repair via MW + GF jumps.

    At each flagged slip the joint integer pair ``(dN1, dN2)`` is
    solved from

        dN_WL = round(MW[k] - MW[k-1])          # N1 - N2 jump
        dN_GF = round((GF[k] - GF[k-1]) / lambda_pseudo)

    where the GF combination ``lambda1*phi1 - lambda2*phi2`` gives a
    second constraint that distinguishes ``(dN1, dN2)`` pairs sharing
    the same wide-lane difference. The local pseudo-wavelength here
    is the GF wavelength ``lambda1 - lambda2`` taken as a proxy.

    Reliability: works well when both MW and GF noise levels are at
    the few-tenths-of-a-cycle scale; ambiguous when the slip is very
    small (sub-cycle drift) or both signals slipped by very large
    integers in opposite directions.

    Parameters
    ----------
    phi1_cycles, phi2_cycles:
        ``(n_epoch,)`` carrier phases on the two frequencies.
    p1_m, p2_m:
        ``(n_epoch,)`` code pseudoranges on the same two frequencies.
    slips:
        ``(n_epoch,)`` bool slip mask.
    f1, f2:
        Carrier frequencies in Hz. Defaults are GPS L1 and L2.

    Returns
    -------
    (phi1_repaired, phi2_repaired):
        Both shaped ``(n_epoch,)`` and unit-matched to the inputs.
    """
    phi1 = np.asarray(phi1_cycles, dtype=float).copy()
    phi2 = np.asarray(phi2_cycles, dtype=float).copy()
    p1 = np.asarray(p1_m, dtype=float)
    p2 = np.asarray(p2_m, dtype=float)
    slips = np.asarray(slips, dtype=bool)
    lam1 = _C / f1
    lam2 = _C / f2
    slip_indices = np.flatnonzero(slips)
    for k in slip_indices:
        if k < 1:
            continue
        # MW at epoch k and k-1 (using current phi1/phi2 so previous
        # repairs propagate).
        mw_k = (phi1[k] - phi2[k]) - (f1 - f2) / (f1 + f2) * (p1[k] / lam1 + p2[k] / lam2)
        mw_p = (phi1[k - 1] - phi2[k - 1]) - (f1 - f2) / (f1 + f2) * (
            p1[k - 1] / lam1 + p2[k - 1] / lam2
        )
        gf_k = lam1 * phi1[k] - lam2 * phi2[k]
        gf_p = lam1 * phi1[k - 1] - lam2 * phi2[k - 1]
        if not (np.isfinite(mw_k) and np.isfinite(mw_p) and np.isfinite(gf_k) and np.isfinite(gf_p)):
            continue
        dN_WL = round(mw_k - mw_p)
        # GF jump in cycles of the GF "wavelength" (lam1 - lam2). This is
        # an approximation; the exact GF integer mapping depends on
        # ionospheric drift between the bracket epochs.
        dN_GF_m = gf_k - gf_p
        # Joint linear inversion for (dN1, dN2):
        #   dN_WL    = dN1 - dN2
        #   dN_GF_m  = lam1 * dN1 - lam2 * dN2
        # Substituting dN2 = dN1 - dN_WL into the second eq:
        #   dN_GF_m = (lam1 - lam2) * dN1 + lam2 * dN_WL
        # =>  dN1 = (dN_GF_m - lam2 * dN_WL) / (lam1 - lam2)
        #     dN2 = dN1 - dN_WL
        dN1 = round((dN_GF_m - lam2 * dN_WL) / (lam1 - lam2))
        dN2 = dN1 - dN_WL
        if dN1 != 0:
            tail = np.isfinite(phi1[k:])
            phi1[k:] = np.where(tail, phi1[k:] - dN1, phi1[k:])
        if dN2 != 0:
            tail = np.isfinite(phi2[k:])
            phi2[k:] = np.where(tail, phi2[k:] - dN2, phi2[k:])
    return phi1, phi2


__all__ = [
    "detect_slips",
    "detect_slips_geometry_free",
    "detect_slips_mw",
    "detect_slips_phase_only",
    "hatch_filter",
    "mp1",
    "mp2",
    "multipath_rms",
    "repair_slips",
    "repair_slips_dual",
]
