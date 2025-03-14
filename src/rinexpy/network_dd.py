"""Network double-difference solver.

Single-baseline RTK (``rinexpy.rtk.double_difference_solve``) cancels
the receiver / satellite clocks and most of the atmosphere in the two-
station double-difference. A network solution generalizes this to
multiple base stations: every base contributes its own DD observations
and its own ambiguities, but they all share one rover position. This
is the textbook approach used by NRTK / VRS systems and by GAMIT-class
network adjustment software.

Implementation:

1. For each (rover, base_i) baseline, pick a reference satellite (the
   highest-elevation common SV at the base) and form DD pseudorange
   and DD carrier phase residuals.
2. Build a block design matrix:

       residuals = [DD_phase_1; DD_phase_2; ...; DD_phase_N]
       A = [G_1  L_1  0    ...
            G_2  0    L_2  ...
            ...
            G_N  0    0    ... L_N]

   where ``G_i`` is the (n_i-1, 3) geometry matrix for baseline ``i``
   and ``L_i = wavelength * I`` is the ambiguity coefficient block for
   that baseline.
3. Solve the stacked LSQ for the rover position update and the float
   ambiguities of every baseline simultaneously.
4. Iterate the linearization around the current rover position estimate.

This solver returns a float-ambiguity solution. Integer ambiguity
resolution can be applied on top per baseline via :mod:`rinexpy.lambda_ar`
on the corresponding block of the float ambiguity vector and its
covariance.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def network_dd_solve(
    baselines: list[dict[str, Any]],
    *,
    wavelength: float,
    initial_rover: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_iter: int = 12,
    tol: float = 1e-3,
) -> dict[str, Any]:
    """Joint-baseline float-ambiguity network RTK solution.

    Parameters
    ----------
    baselines:
        List of per-baseline dicts. Each dict must contain:

        - ``base_position``: ``(3,)`` ECEF position of the base station.
        - ``sv_positions``: ``(n_sv_i, 3)`` satellite ECEF positions for
          the SVs common to rover and base on this baseline.
        - ``rover_pr``: ``(n_sv_i,)`` rover pseudorange in meters.
        - ``base_pr``: ``(n_sv_i,)`` base pseudorange in meters.
        - ``rover_phase``: ``(n_sv_i,)`` rover carrier phase in cycles.
        - ``base_phase``: ``(n_sv_i,)`` base carrier phase in cycles.
    wavelength:
        Carrier wavelength in meters (e.g. 0.1903 for GPS L1).
    initial_rover:
        ECEF starting guess for the rover position. Default is Earth's
        center, which converges from anywhere on the surface within
        a handful of iterations once at least one baseline has been
        provided.
    max_iter:
        Iteration cap. Raises ``RuntimeError`` if not converged in time.
    tol:
        Convergence tolerance on rover-position update norm, in meters.

    Returns
    -------
    dict
        ``{"rover_position": (x, y, z), "ambiguities": [array_i, ...],
        "n_iter": int, "residuals": (n_total,), "reference_sv":
        [int, ...]}``.

    Raises
    ------
    ValueError
        If fewer than 1 baseline is supplied, or any baseline has fewer
        than 5 common satellites (5 are needed: 4 DDs + 1 reference SV
        with 3 baseline unknowns + (n-1) ambiguities for that baseline).
    RuntimeError
        If the iteration does not converge within ``max_iter``.
    """
    if len(baselines) < 1:
        raise ValueError("network_dd_solve needs >= 1 baseline")
    # Validate and snapshot each baseline; pick reference satellite once.
    blocks: list[dict[str, Any]] = []
    n_dd_total = 0
    for k, b in enumerate(baselines):
        sv = np.asarray(b["sv_positions"], dtype=float)
        n_sv = sv.shape[0]
        if n_sv < 5:
            raise ValueError(
                f"baseline {k} has only {n_sv} common SVs; need >= 5"
            )
        base = np.asarray(b["base_position"], dtype=float)
        base_ranges = np.linalg.norm(sv - base, axis=1)
        ref = int(np.argmin(base_ranges))
        others = [i for i in range(n_sv) if i != ref]
        rover_phase_m = np.asarray(b["rover_phase"], dtype=float) * wavelength
        base_phase_m = np.asarray(b["base_phase"], dtype=float) * wavelength
        sd_phase = rover_phase_m - base_phase_m
        dd_phase = sd_phase[others] - sd_phase[ref]
        sd_pr = np.asarray(b["rover_pr"], dtype=float) - np.asarray(b["base_pr"], dtype=float)
        dd_pr = sd_pr[others] - sd_pr[ref]
        float_amb = (dd_phase - dd_pr) / wavelength
        n_dd = n_sv - 1
        blocks.append({
            "sv": sv,
            "base": base,
            "ref": ref,
            "others": np.array(others, dtype=int),
            "dd_phase": dd_phase,
            "dd_pr": dd_pr,
            "n_dd": n_dd,
            "float_amb": float_amb.copy(),
        })
        n_dd_total += n_dd

    # Float ambiguities are fixed at the pre-iteration estimate
    # ``(dd_phase - dd_pr) / wavelength``, which already cancels the
    # geometric range in the noiseless case. The Gauss-Newton iteration
    # then only updates the rover position (3 unknowns) against the
    # stacked DD-phase residuals across every baseline.
    rover_pos = np.array(initial_rover, dtype=float)
    for it in range(max_iter):
        rows: list[np.ndarray] = []
        rhs: list[np.ndarray] = []
        for b in blocks:
            sv = b["sv"]
            ref = b["ref"]
            others = b["others"]
            rho_b = np.linalg.norm(sv - b["base"], axis=1)
            rho_r = np.linalg.norm(sv - rover_pos, axis=1)
            dd_rho = (rho_r[others] - rho_b[others]) - (rho_r[ref] - rho_b[ref])
            u = (sv - rover_pos) / rho_r[:, None]
            g_block = u[ref] - u[others]
            rows.append(g_block)
            rhs.append(b["dd_phase"] - wavelength * b["float_amb"] - dd_rho)
        a = np.vstack(rows)
        b_rhs = np.concatenate(rhs)
        try:
            update, *_ = np.linalg.lstsq(a, b_rhs, rcond=None)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"network DD normal equations singular: {e}") from e
        rover_pos += update
        if np.linalg.norm(update) < tol:
            return {
                "rover_position": tuple(float(x) for x in rover_pos),
                "ambiguities": [bk["float_amb"].copy() for bk in blocks],
                "n_iter": it + 1,
                "residuals": b_rhs,
                "reference_sv": [bk["ref"] for bk in blocks],
            }
    raise RuntimeError(f"network DD did not converge in {max_iter} iterations")


def network_dd_solve_ar(
    baselines: list[dict[str, Any]],
    *,
    wavelength: float,
    sigma_pr_m: float = 1.0,
    sigma_phase_cycles: float = 0.005,
    ratio_threshold: float = 3.0,
    initial_rover: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_iter: int = 12,
    tol: float = 1e-3,
) -> dict[str, Any]:
    """Network DD with per-baseline LAMBDA integer ambiguity resolution.

    Two-phase solution:

    1. **Float**: jointly solve the weighted-LSQ system that stacks every
       baseline's phase DDs and pseudorange DDs, with phase weighted by
       ``1/sigma_phase_m**2`` and pseudoranges by ``1/sigma_pr_m**2``.
       Returns float ambiguities for every baseline plus the joint
       covariance matrix; per-baseline ambiguity covariance blocks are
       extracted from the diagonal blocks of the inverse normal matrix.
    2. **Fix**: for each baseline, run :func:`rinexpy.lambda_ar.lambda_resolve`
       on ``(float_amb_i, Q_a_i)``. Pass-the-ratio-test baselines lock
       their integer ambiguities; if **all** baselines pass, the rover
       position is refined by substituting the integer ambiguities and
       re-solving the phase DDs for position alone.

    Parameters
    ----------
    baselines:
        Same per-baseline dict format as :func:`network_dd_solve`.
    wavelength:
        Carrier wavelength in meters.
    sigma_pr_m:
        Pseudorange noise (1-sigma, meters). Default 1 m, a typical
        geodetic-receiver code residual.
    sigma_phase_cycles:
        Carrier-phase noise (1-sigma, cycles). Default 0.005 cycles
        (~1 mm on L1).
    ratio_threshold:
        LAMBDA ratio-test threshold (best vs second-best squared error).
        Default 3.0 (standard tight setting).
    initial_rover, max_iter, tol:
        Same role as in :func:`network_dd_solve`.

    Returns
    -------
    dict
        ``{"float_rover_position": (x, y, z),
        "fixed_rover_position": (x, y, z) | None,
        "float_ambiguities": [array, ...],
        "integer_ambiguities": [array, ...],
        "ratios": [float, ...],
        "accepted": [bool, ...],
        "all_fixed": bool,
        "reference_sv": [int, ...],
        "n_iter": int}``.

        ``fixed_rover_position`` is ``None`` when at least one baseline
        failed the ratio test (the float position is then the operational
        answer).

    Raises
    ------
    ValueError
        Same as :func:`network_dd_solve`.
    RuntimeError
        If the joint LSQ does not converge in ``max_iter`` iterations or
        the normal-equation matrix becomes singular.
    """
    from .lambda_ar import lambda_resolve

    if len(baselines) < 1:
        raise ValueError("network_dd_solve_ar needs >= 1 baseline")

    blocks: list[dict[str, Any]] = []
    for k, b in enumerate(baselines):
        sv = np.asarray(b["sv_positions"], dtype=float)
        n_sv = sv.shape[0]
        if n_sv < 5:
            raise ValueError(f"baseline {k} has only {n_sv} common SVs; need >= 5")
        base = np.asarray(b["base_position"], dtype=float)
        base_ranges = np.linalg.norm(sv - base, axis=1)
        ref = int(np.argmin(base_ranges))
        others = [i for i in range(n_sv) if i != ref]
        rover_phase_m = np.asarray(b["rover_phase"], dtype=float) * wavelength
        base_phase_m = np.asarray(b["base_phase"], dtype=float) * wavelength
        sd_phase = rover_phase_m - base_phase_m
        dd_phase = sd_phase[others] - sd_phase[ref]
        sd_pr = (
            np.asarray(b["rover_pr"], dtype=float)
            - np.asarray(b["base_pr"], dtype=float)
        )
        dd_pr = sd_pr[others] - sd_pr[ref]
        blocks.append({
            "sv": sv,
            "base": base,
            "ref": ref,
            "others": np.array(others, dtype=int),
            "dd_phase": dd_phase,
            "dd_pr": dd_pr,
            "n_dd": n_sv - 1,
        })

    n_amb = sum(b["n_dd"] for b in blocks)
    rover_pos = np.array(initial_rover, dtype=float)
    float_amb = np.zeros(n_amb)
    # Floor the phase sigma at 0.0005 cycles (~ 0.1 mm) to keep the
    # ratio of phase / pseudorange weights manageable. A smaller-than-
    # floor noise level is physically implausible and just makes the
    # normal-equations matrix singular without buying any precision.
    sigma_phase_eff = max(sigma_phase_cycles, 5e-4)
    sigma_phase_m = sigma_phase_eff * wavelength
    w_phase = 1.0 / sigma_phase_m ** 2
    w_pr = 1.0 / sigma_pr_m ** 2

    cov_full = None
    for it in range(max_iter):
        rows: list[np.ndarray] = []
        rhs: list[np.ndarray] = []
        weights: list[float] = []
        amb_off = 0
        for b in blocks:
            sv = b["sv"]
            ref = b["ref"]
            others = b["others"]
            rho_b = np.linalg.norm(sv - b["base"], axis=1)
            rho_r = np.linalg.norm(sv - rover_pos, axis=1)
            dd_rho = (rho_r[others] - rho_b[others]) - (rho_r[ref] - rho_b[ref])
            u = (sv - rover_pos) / rho_r[:, None]
            g_block = u[ref] - u[others]
            n = b["n_dd"]
            # Phase DD rows
            row_p = np.zeros((n, 3 + n_amb))
            row_p[:, :3] = g_block
            row_p[:, 3 + amb_off : 3 + amb_off + n] = wavelength * np.eye(n)
            rows.append(row_p)
            rhs.append(b["dd_phase"] - dd_rho - wavelength * float_amb[amb_off : amb_off + n])
            weights.extend([w_phase] * n)
            # Pseudo DD rows
            row_r = np.zeros((n, 3 + n_amb))
            row_r[:, :3] = g_block
            rows.append(row_r)
            rhs.append(b["dd_pr"] - dd_rho)
            weights.extend([w_pr] * n)
            amb_off += n
        a = np.vstack(rows)
        b_rhs = np.concatenate(rhs)
        w_diag = np.array(weights)
        n_normal = a.T @ (a * w_diag[:, None])
        y_normal = a.T @ (w_diag * b_rhs)
        try:
            cov_full = np.linalg.inv(n_normal)
            update = cov_full @ y_normal
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"network DD AR normal matrix singular: {e}") from e
        rover_pos += update[:3]
        float_amb += update[3:]
        if np.linalg.norm(update[:3]) < tol:
            break
    else:
        raise RuntimeError(f"network DD AR did not converge in {max_iter} iterations")

    # Per-baseline ambiguity blocks + LAMBDA per baseline.
    ar_results: list[dict[str, Any]] = []
    integer_amb: list[np.ndarray] = []
    float_amb_blocks: list[np.ndarray] = []
    amb_off = 0
    for b in blocks:
        n = b["n_dd"]
        fa = float_amb[amb_off : amb_off + n]
        qa = cov_full[3 + amb_off : 3 + amb_off + n, 3 + amb_off : 3 + amb_off + n]
        res = lambda_resolve(fa, qa, ratio_threshold=ratio_threshold)
        ar_results.append(res)
        integer_amb.append(res["a_int"])
        float_amb_blocks.append(fa.copy())
        amb_off += n

    all_fixed = all(r["accepted"] for r in ar_results)
    fixed_rover_position: tuple[float, float, float] | None = None
    if all_fixed:
        # Refine position with fixed ambiguities: re-solve phase DDs alone
        # for delta_pos.
        rover_fixed = rover_pos.copy()
        for it in range(max_iter):
            rows: list[np.ndarray] = []
            rhs: list[np.ndarray] = []
            amb_off = 0
            for i, b in enumerate(blocks):
                sv = b["sv"]
                ref = b["ref"]
                others = b["others"]
                rho_b = np.linalg.norm(sv - b["base"], axis=1)
                rho_r = np.linalg.norm(sv - rover_fixed, axis=1)
                dd_rho = (rho_r[others] - rho_b[others]) - (rho_r[ref] - rho_b[ref])
                u = (sv - rover_fixed) / rho_r[:, None]
                g_block = u[ref] - u[others]
                rows.append(g_block)
                rhs.append(b["dd_phase"] - wavelength * integer_amb[i] - dd_rho)
                amb_off += b["n_dd"]
            a = np.vstack(rows)
            b_rhs = np.concatenate(rhs)
            upd, *_ = np.linalg.lstsq(a, b_rhs, rcond=None)
            rover_fixed += upd
            if np.linalg.norm(upd) < tol:
                break
        fixed_rover_position = tuple(float(x) for x in rover_fixed)

    return {
        "float_rover_position": tuple(float(x) for x in rover_pos),
        "fixed_rover_position": fixed_rover_position,
        "float_ambiguities": float_amb_blocks,
        "integer_ambiguities": integer_amb,
        "ratios": [r["ratio"] for r in ar_results],
        "accepted": [r["accepted"] for r in ar_results],
        "all_fixed": all_fixed,
        "reference_sv": [b["ref"] for b in blocks],
        "n_iter": it + 1,
    }


__all__ = ["network_dd_solve", "network_dd_solve_ar"]
