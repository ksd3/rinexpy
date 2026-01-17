"""Spoofing / jamming detection heuristics for GNSS observation streams.

Out-of-the-box checks per the receiver-autonomous-integrity literature:

- **SNR / CNR consistency** (``check_snr_uniformity``): legitimate
  multi-SV signals show SNR variance by elevation; a uniform high SNR
  across many SVs at low elevation suggests a spoofer broadcasting all
  signals from a single antenna.
- **Position-jump detection** (``check_position_jumps``): epoch-to-
  epoch ECEF jumps faster than physically plausible flag spoofing /
  meaconing onset.
- **Clock-drift sanity** (``check_clock_drift``): receiver clock
  drift rate consistent with a typical TCXO / OCXO; jumps > a few
  microseconds without an explanation indicate clock-bias attacks.
- **AGC / SNR anomaly** (``check_agc``): static-receiver AGC ramping
  up indicates an interferer; the receiver gain control compensates
  for added power.

All checks return boolean flags or per-epoch scores; thresholds are
caller-tunable. Pair with :class:`rinexpy.qc.detect_slips` for a full
integrity layer.
"""

from __future__ import annotations

import numpy as np


def check_snr_uniformity(
    snr_dbhz: np.ndarray,
    elevation_deg: np.ndarray,
    *,
    sigma_threshold: float = 1.5,
) -> dict:
    """Flag epochs where SNR is suspiciously uniform across SVs.

    For each epoch we compute the residual of SNR after subtracting a
    linear elevation-fit. A legitimate sky shows residual std of a few
    dB-Hz; a single-antenna spoofer typically gives residuals well
    under 1 dB-Hz because every "SV" is the same signal.

    Parameters
    ----------
    snr_dbhz:
        ``(n_epoch, n_sv)`` SNR/CNR in dB-Hz; NaN for missing.
    elevation_deg:
        ``(n_epoch, n_sv)`` elevation; NaN for missing.
    sigma_threshold:
        Per-epoch residual std (dB-Hz) below which we flag suspicion.
        Default 1.5 (legitimate skies show 3-6 dB-Hz residual std).

    Returns
    -------
    dict
        ``{"flagged_epochs": ndarray of bool (n_epoch,), "residual_std":
        ndarray float (n_epoch,)}``.
    """
    snr = np.asarray(snr_dbhz, dtype=float)
    el = np.asarray(elevation_deg, dtype=float)
    if snr.shape != el.shape:
        raise ValueError("snr and elevation shape mismatch")
    n_epoch = snr.shape[0]
    flagged = np.zeros(n_epoch, dtype=bool)
    resid_std = np.full(n_epoch, np.nan)
    for k in range(n_epoch):
        mask = np.isfinite(snr[k]) & np.isfinite(el[k])
        if mask.sum() < 4:
            continue
        # Linear fit SNR ~ a + b * sin(el) -- elevation effect.
        sin_el = np.sin(np.radians(el[k][mask]))
        slope_design = np.column_stack([np.ones_like(sin_el), sin_el])
        coefs, *_ = np.linalg.lstsq(slope_design, snr[k][mask], rcond=None)
        pred = slope_design @ coefs
        resid = snr[k][mask] - pred
        std = float(np.std(resid))
        resid_std[k] = std
        flagged[k] = std < sigma_threshold
    return {"flagged_epochs": flagged, "residual_std": resid_std}


def check_position_jumps(
    positions_ecef: np.ndarray,
    times_s: np.ndarray,
    *,
    max_speed_m_per_s: float = 300.0,
) -> dict:
    """Flag epochs where the receiver position jumps faster than
    ``max_speed_m_per_s``.

    For a static or pedestrian receiver this is a strong meaconing
    signal: legitimate jumps are limited by GNSS noise (a few cm)
    while a spoofer can pull the receiver tens to thousands of meters
    in a single epoch.

    Parameters
    ----------
    positions_ecef:
        ``(n_epoch, 3)`` ECEF positions in meters.
    times_s:
        ``(n_epoch,)`` epoch times in seconds (relative, monotonic).
    max_speed_m_per_s:
        Above this implied epoch-to-epoch speed, an epoch is flagged.
        Default 300 m/s (Mach 1) — well above pedestrian / vehicle
        speeds but well below a meaconing pull.

    Returns
    -------
    dict
        ``{"flagged_epochs": ndarray (n_epoch,) bool, "speed_mps":
        ndarray (n_epoch,) float}``. The first epoch has speed=0.
    """
    pos = np.asarray(positions_ecef, dtype=float)
    t = np.asarray(times_s, dtype=float)
    n = pos.shape[0]
    flagged = np.zeros(n, dtype=bool)
    speed = np.zeros(n)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        if dt <= 0:
            continue
        dx = np.linalg.norm(pos[k] - pos[k - 1])
        speed[k] = dx / dt
        if speed[k] > max_speed_m_per_s:
            flagged[k] = True
    return {"flagged_epochs": flagged, "speed_mps": speed}


def check_clock_drift(
    clock_bias_s: np.ndarray,
    times_s: np.ndarray,
    *,
    max_drift_rate: float = 1e-6,
    max_jump_s: float = 1e-5,
) -> dict:
    """Flag epochs whose receiver-clock bias jumps or drifts too fast.

    A typical TCXO ages at ~1 ppm and an OCXO at ~1 ppb. A sudden
    multi-microsecond jump in the receiver clock bias without a
    documented receiver reset suggests a clock-bias spoofing attempt.

    Parameters
    ----------
    clock_bias_s:
        ``(n_epoch,)`` receiver clock bias in seconds.
    times_s:
        ``(n_epoch,)`` corresponding epoch times.
    max_drift_rate:
        Maximum legitimate clock drift in seconds/second. Default
        1e-6 (1 ppm).
    max_jump_s:
        Maximum legitimate per-epoch clock jump in seconds. Default
        1e-5 (10 us) -- below typical receiver-reset behaviour.

    Returns
    -------
    dict
        ``{"flagged_epochs": ndarray bool, "drift_rate":
        ndarray float, "jumps": ndarray float}``.
    """
    bias = np.asarray(clock_bias_s, dtype=float)
    t = np.asarray(times_s, dtype=float)
    n = bias.shape[0]
    flagged = np.zeros(n, dtype=bool)
    drift = np.zeros(n)
    jumps = np.zeros(n)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        if dt <= 0:
            continue
        db = bias[k] - bias[k - 1]
        jumps[k] = db
        drift[k] = db / dt
        if abs(db) > max_jump_s or abs(drift[k]) > max_drift_rate:
            flagged[k] = True
    return {"flagged_epochs": flagged, "drift_rate": drift, "jumps": jumps}


def check_agc(
    agc_db: np.ndarray,
    *,
    max_jump_db: float = 6.0,
) -> dict:
    """Flag epochs where the receiver automatic gain control (AGC)
    moves by more than ``max_jump_db`` dB between epochs.

    AGC ramps up when the receiver detects added in-band power, which
    is the signature of a jammer or a strong spoofer. Many UBX / SBF
    receivers expose AGC on UBX-NAV-AGC / SBF AGCData.

    Parameters
    ----------
    agc_db:
        ``(n_epoch,)`` AGC (relative dB) over time.
    max_jump_db:
        Threshold for flagging an AGC jump. Default 6 dB (a 2x power
        change).

    Returns
    -------
    dict
        ``{"flagged_epochs": ndarray bool, "jumps": ndarray float}``.
    """
    agc = np.asarray(agc_db, dtype=float)
    n = agc.shape[0]
    flagged = np.zeros(n, dtype=bool)
    jumps = np.zeros(n)
    for k in range(1, n):
        d = agc[k] - agc[k - 1]
        jumps[k] = d
        if abs(d) > max_jump_db:
            flagged[k] = True
    return {"flagged_epochs": flagged, "jumps": jumps}


__all__ = [
    "check_agc",
    "check_clock_drift",
    "check_position_jumps",
    "check_snr_uniformity",
]
