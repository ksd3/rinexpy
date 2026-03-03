"""Network RTK Virtual Reference Station (VRS) synthesis.

A VRS combines observations from a network of physical bases into a
synthesized observation set at a *virtual* receiver co-located with
the rover. The rover then runs ordinary single-baseline RTK against
this VRS as if it were a real nearby base. The benefit is short-baseline
performance (kilometres) without operators having to set up a real
nearby base.

The minimum-viable VRS for one SV is:

    pr_VRS = pr_master - |sv - master| + |sv - rover| + correction

i.e. we take the master station's measured pseudorange, subtract its
geometric range (cancelling clock + atmosphere terms that the master
sees), add the geometric range from the SV to the rover, and add a
spatial correction interpolated from the *residuals*

    residual_b = pr_b - |sv - base_b|

across the network of bases. With ``n_bases >= 3`` we fit a planar
model ``r(lat, lon) = a + b*lat + c*lon`` to the per-base residuals and
evaluate it at the rover's lat/lon. With fewer bases the master's
residual is used as-is (zeroth-order VRS).

The carrier phase synthesis follows the same form, in cycles:

    phase_VRS = phase_master + (|sv - rover| - |sv - master|) / lambda

Reference: Wanninger, L. (2002). *Virtual Reference Stations (VRS).*
GPS Solutions 5 (3): 22-29.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geodesy import ecef_to_lla


def _plane_fit_residual(
    base_lats: np.ndarray, base_lons: np.ndarray, residuals: np.ndarray,
    rover_lat: float, rover_lon: float,
) -> float:
    """Fit residual = a + b*lat + c*lon to (n_bases) samples and evaluate
    at the rover. If fewer than 3 bases, returns the first base's
    residual unchanged (zeroth-order)."""
    n = base_lats.size
    if n < 3:
        return float(residuals[0])
    A = np.column_stack([np.ones(n), base_lats, base_lons])
    try:
        coeffs, *_ = np.linalg.lstsq(A, residuals, rcond=None)
    except np.linalg.LinAlgError:
        return float(residuals[0])
    return float(coeffs[0] + coeffs[1] * rover_lat + coeffs[2] * rover_lon)


def synthesize_vrs(
    bases: list[dict[str, Any]],
    rover_approx_pos: tuple[float, float, float],
    *,
    wavelength: float,
) -> dict[str, Any]:
    """Synthesize a VRS observation set at the rover's approximate
    position from a network of base stations.

    Parameters
    ----------
    bases:
        List of per-base dicts. The first base is treated as the
        *master* and its pseudoranges + phases are the VRS anchor.
        Every base dict must contain:

        - ``base_position``: ``(3,)`` ECEF position of the base (m).
        - ``sv_positions``: ``(n_sv, 3)`` satellite ECEF positions for
          the SVs common to this base. **Must be the same SV ordering
          on every base** so per-SV interpolation is well-defined.
        - ``pr``: ``(n_sv,)`` pseudorange in meters at this base.
        - ``phase``: ``(n_sv,)`` carrier phase in cycles at this base.
    rover_approx_pos:
        ``(x, y, z)`` ECEF approximate rover position (m). Typically
        ~10 km accuracy is enough.
    wavelength:
        Carrier wavelength in meters.

    Returns
    -------
    dict
        ``{"base_position": rover_approx_pos, "sv_positions": ndarray,
        "pr": ndarray, "phase": ndarray}`` - a baseline-block suitable
        for handing to :func:`rinexpy.rtk.double_difference_solve` or
        :func:`rinexpy.rtk.rtk_fix`.
    """
    if not bases:
        raise ValueError("synthesize_vrs needs >= 1 base")
    rover = np.asarray(rover_approx_pos, dtype=float)
    master = bases[0]
    sv = np.asarray(master["sv_positions"], dtype=float)
    n_sv = sv.shape[0]
    pr_master = np.asarray(master["pr"], dtype=float)
    phase_master = np.asarray(master["phase"], dtype=float)
    base_positions = np.array([b["base_position"] for b in bases], dtype=float)
    if any(np.asarray(b["sv_positions"]).shape != sv.shape for b in bases):
        raise ValueError("all bases must share the same SV ordering and count")

    # Per-base lat/lon for the plane fit.
    lats = np.empty(len(bases))
    lons = np.empty(len(bases))
    for i, bp in enumerate(base_positions):
        lat, lon, _ = ecef_to_lla(float(bp[0]), float(bp[1]), float(bp[2]))
        lats[i] = lat
        lons[i] = lon
    rover_lat, rover_lon, _ = ecef_to_lla(float(rover[0]), float(rover[1]), float(rover[2]))

    pr_vrs = np.zeros(n_sv)
    phase_vrs = np.zeros(n_sv)
    rho_rover = np.linalg.norm(sv - rover, axis=1)
    for i in range(n_sv):
        rho_per_base = np.linalg.norm(
            np.asarray([b["sv_positions"] for b in bases])[:, i, :] - base_positions, axis=1
        )
        pr_residuals = np.array([b["pr"][i] - rho_per_base[k] for k, b in enumerate(bases)])
        phase_residuals = np.array([
            b["phase"][i] - rho_per_base[k] / wavelength for k, b in enumerate(bases)
        ])
        r_pr = _plane_fit_residual(lats, lons, pr_residuals, rover_lat, rover_lon)
        r_ph = _plane_fit_residual(lats, lons, phase_residuals, rover_lat, rover_lon)
        pr_vrs[i] = rho_rover[i] + r_pr
        phase_vrs[i] = rho_rover[i] / wavelength + r_ph

    return {
        "base_position": tuple(float(x) for x in rover),
        "sv_positions": sv.copy(),
        "pr": pr_vrs,
        "phase": phase_vrs,
    }


__all__ = ["synthesize_vrs"]
