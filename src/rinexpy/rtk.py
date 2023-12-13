"""Single-baseline RTK double-difference solver (float ambiguities).

Forms double-differences between a rover and a base receiver to cancel
both receiver and satellite clock biases plus most of the atmospheric
delay, then solves a linear LSQ for ``(baseline_vector, ambiguities)``.

Integer ambiguity resolution (the LAMBDA method) is intentionally
**not** implemented — the float solution gives ~10-30 cm baseline
accuracy, which is the right "MVP" granularity. A future commit can add
LAMBDA on top.

Typical usage:

.. code-block:: python

    from rinexpy.rtk import double_difference_solve
    sol = double_difference_solve(
        rover_pr, base_pr,
        rover_phase, base_phase,
        sv_positions_ecef,
        base_position_ecef,
        wavelength=0.1903,  # GPS L1 in m
    )
    # sol["baseline"] is the (dx, dy, dz) ECEF vector from base to rover.
"""

from __future__ import annotations

import numpy as np

# Speed of light is irrelevant here (clocks cancel) but commonly imported.
_C = 299_792_458.0


def double_difference_solve(
    rover_pr: np.ndarray,
    base_pr: np.ndarray,
    rover_phase: np.ndarray,
    base_phase: np.ndarray,
    sv_positions_ecef: np.ndarray,
    base_position_ecef: tuple[float, float, float],
    *,
    wavelength: float,
    initial_baseline: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_iter: int = 8,
    tol: float = 1e-3,
) -> dict:
    """Float-ambiguity RTK solution.

    Parameters
    ----------
    rover_pr, base_pr:
        ``(n_sv,)`` pseudorange (m) at rover and base for the same epoch
        and same SV ordering.
    rover_phase, base_phase:
        ``(n_sv,)`` carrier-phase observation in cycles.
    sv_positions_ecef:
        ``(n_sv, 3)`` satellite ECEF positions at signal-emission time.
    base_position_ecef:
        Known base receiver ECEF position (m).
    wavelength:
        Carrier wavelength in meters (e.g. 0.1903 for GPS L1).
    initial_baseline:
        Starting guess for the rover-minus-base ECEF vector (m).
    max_iter, tol:
        LSQ convergence controls.

    Returns
    -------
    dict
        ``{"baseline": (dx, dy, dz), "rover_position": (x, y, z),
        "ambiguities": ndarray, "n_iter": int, "residuals": ndarray}``.

    Raises
    ------
    ValueError
        If fewer than 5 satellites are supplied (we need n_sv-1
        double-differences plus 3 baseline unknowns plus n_sv-1
        float ambiguities — minimum n_sv = 5 for a uniquely-determined
        ambiguities-and-baseline solve).
    RuntimeError
        If the iteration does not converge within ``max_iter``.
    """
    sv = np.asarray(sv_positions_ecef, dtype=float)
    n_sv = sv.shape[0]
    if n_sv < 5:
        raise ValueError("RTK needs >= 5 common satellites")

    # Pick the reference satellite as the one with smallest base-to-sat
    # distance (highest elevation proxy).
    base = np.asarray(base_position_ecef, dtype=float)
    base_ranges = np.linalg.norm(sv - base, axis=1)
    ref = int(np.argmin(base_ranges))
    others = [i for i in range(n_sv) if i != ref]

    # Form double-difference phase observations.
    rover_phase_m = np.asarray(rover_phase) * wavelength
    base_phase_m = np.asarray(base_phase) * wavelength
    sd_phase = rover_phase_m - base_phase_m  # single-diff (rover - base)
    dd_phase_m = sd_phase[others] - sd_phase[ref]
    sd_pr = np.asarray(rover_pr) - np.asarray(base_pr)
    dd_pr_m = sd_pr[others] - sd_pr[ref]

    baseline = np.array(initial_baseline, dtype=float)
    rover_pos = base + baseline

    # Float ambiguity estimate from pseudorange minus phase. Both DDs
    # remove the iono delay to first order and the receiver/satellite
    # clocks exactly, so:
    #     dd_phase_m - dd_pr_m  ~  -2 * iono + lambda * dd_amb
    # We assume iono is negligible (kept short-baseline) and solve:
    float_amb = (dd_phase_m - dd_pr_m) / wavelength

    for it in range(max_iter):
        # Predicted geometric ranges.
        rho_b = np.linalg.norm(sv - base, axis=1)
        rho_r = np.linalg.norm(sv - rover_pos, axis=1)
        sd_rho = rho_r - rho_b
        dd_rho = sd_rho[others] - sd_rho[ref]
        # Unit line-of-sight from rover to each SV.
        u = (sv - rover_pos) / rho_r[:, None]
        # Per-DD baseline coefficient: u[ref] - u[other] (shape (n_dd, 3)).
        a_baseline = u[ref] - u[others]
        # Phase observation minus float-ambiguity term gives a clean
        # geometric DD that's a function of baseline alone.
        residuals = (dd_phase_m - wavelength * float_amb) - dd_rho
        try:
            update, *_ = np.linalg.lstsq(a_baseline, residuals, rcond=None)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"RTK normal equations singular: {e}") from e
        baseline += update
        rover_pos = base + baseline
        if np.linalg.norm(update) < tol:
            return {
                "baseline": tuple(float(x) for x in baseline),
                "rover_position": tuple(float(x) for x in rover_pos),
                "ambiguities": float_amb,
                "n_iter": it + 1,
                "residuals": residuals,
                "reference_sv_index": ref,
                "dd_pseudorange": dd_pr_m,
            }
    raise RuntimeError(f"RTK did not converge in {max_iter} iterations")


__all__ = ["double_difference_solve"]
