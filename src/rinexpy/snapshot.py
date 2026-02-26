"""Snapshot positioning: code-phase-only short-data fix.

Snapshot receivers (used in IoT / asset-tracking devices) record well
under one second of raw IF samples, so they cannot track satellites
long enough to demodulate the navigation message or extract a full
pseudorange. What they *can* measure is the *fractional* code phase
of each visible SV - the position of the C/A code chip boundary within
the snapshot window.

Each fractional code phase resolves the pseudorange modulo one C/A
code period: ``1 ms = c/1000 = 299792.458 m``. The integer ms count
``K`` is unknown. If the receiver has a coarse position prior (often
from cell-tower geolocation, ~30 km), it can fix ``K`` per SV by
choosing the integer that makes the resulting pseudorange consistent
with the prior. Then a standard 4-unknown LSQ recovers position and
time bias.

This module implements the minimum-viable code-phase-only snapshot
solver of van Diggelen ("A-GPS", 2009).

Notes
-----
- The initial position must be within < 150 km of truth to disambiguate
  K reliably (one-half of the 300 km ms-rollover).
- Doppler measurements (when available) tighten the time-of-week
  ambiguity further; that path is not implemented here - the receiver
  is assumed to have GPS-week + approximate-second-of-week known
  externally.
- This is a coarse-time receiver. Typical accuracy is ~10 m horizontal
  with a good geometry and clean code-phase measurements.
"""

from __future__ import annotations

import numpy as np

from .geodesy import ecef_to_lla

_C_M_PER_S = 299_792_458.0
_CODE_PERIOD_S = 1.0e-3            # GPS L1 C/A is 1 ms long
_CODE_CHIPS = 1023
_CHIP_LEN_M = _C_M_PER_S / 1.023e6  # one C/A chip = ~293 m
_PERIOD_LEN_M = _C_M_PER_S * _CODE_PERIOD_S  # one ms = ~299.8 km


def snapshot_positioning(
    code_phase_chips: np.ndarray,
    sv_positions_ecef: np.ndarray,
    initial_position_ecef: tuple[float, float, float],
    *,
    max_iter: int = 20,
    tol: float = 1.0,
) -> dict:
    """Code-phase-only snapshot SPP.

    Parameters
    ----------
    code_phase_chips:
        ``(n_sv,)`` fractional chip offsets for each SV in the snapshot
        window. Values in ``[0, 1023)``.
    sv_positions_ecef:
        ``(n_sv, 3)`` satellite ECEF positions at the (approximate)
        signal-emission epoch.
    initial_position_ecef:
        ``(x, y, z)`` coarse receiver position prior (m). Must be
        within ~150 km of the true position so the ms-rollover integer
        per SV is unique.
    max_iter, tol:
        Iteration limit and position-update tolerance (m) for the LSQ
        convergence.

    Returns
    -------
    dict
        ``{"position_ecef", "lla", "time_bias_s", "pseudoranges_m",
        "K_integer_ms", "n_iter"}``. ``lla`` is ``(lat_deg, lon_deg,
        alt_m)``.
    """
    sv = np.asarray(sv_positions_ecef, dtype=float)
    n = sv.shape[0]
    if n < 4:
        raise ValueError("snapshot SPP needs >= 4 satellites")

    frac_pr_m = np.asarray(code_phase_chips, dtype=float) * _CHIP_LEN_M
    pos = np.array(initial_position_ecef, dtype=float)
    bias_s = 0.0

    pseudoranges = np.zeros(n)
    K = np.zeros(n, dtype=int)

    for it in range(max_iter):
        expected_range = np.linalg.norm(sv - pos, axis=1) + bias_s * _C_M_PER_S
        K = np.round((expected_range - frac_pr_m) / _PERIOD_LEN_M).astype(int)
        pseudoranges = K * _PERIOD_LEN_M + frac_pr_m

        rho = np.linalg.norm(sv - pos, axis=1)
        u = (pos - sv) / rho[:, None]
        # Geometry matrix: [ux, uy, uz, 1] - the last column is the
        # clock-bias partial (cdt).
        H = np.hstack([u, np.ones((n, 1))])
        b = pseudoranges - rho - bias_s * _C_M_PER_S
        try:
            delta, *_ = np.linalg.lstsq(H, b, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"snapshot LSQ singular: {exc}") from exc

        pos += delta[:3]
        bias_s += delta[3] / _C_M_PER_S
        if np.linalg.norm(delta[:3]) < tol:
            n_iter = it + 1
            break
    else:
        n_iter = max_iter

    lla = ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))
    return {
        "position_ecef": tuple(float(x) for x in pos),
        "lla": lla,
        "time_bias_s": bias_s,
        "pseudoranges_m": pseudoranges,
        "K_integer_ms": K,
        "n_iter": n_iter,
    }


__all__ = ["snapshot_positioning"]
