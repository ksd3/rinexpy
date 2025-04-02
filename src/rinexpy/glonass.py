"""GLONASS frequency-division multiple access (FDMA) helpers.

Unlike GPS / Galileo / BeiDou, GLONASS satellites transmit on slightly
different carrier frequencies indexed by the per-satellite channel
number (also called frequency slot, FCN). The channel number ranges
from -7 to +6 on L1OF / L2OF and is fixed per spacecraft by ground
allocation (the same channel can be reused by two antipodal SVs).

Carrier frequencies (per GLONASS ICD L1 L2 v5.1):

    f_L1OF(k) = 1602.0 MHz + k * 0.5625 MHz
    f_L2OF(k) = 1246.0 MHz + k * 0.4375 MHz

with k the channel number. Wavelengths follow ``c / f``. Operationally
this means that a single ``wavelength`` constant is *wrong* for GLONASS
-- a Melbourne-Wuebbena combination over two GLONASS SVs with different
channel numbers has to use each SV's own wavelength.

This module ships the constants table and the small helpers that take
a channel number and return frequencies / wavelengths / iono-free
combinations for GLONASS observations.
"""

from __future__ import annotations

import numpy as np

C_M_PER_S = 299_792_458.0

#: GLONASS L1OF base frequency in Hz (channel 0).
F_L1OF_BASE_HZ = 1_602_000_000.0
#: GLONASS L2OF base frequency in Hz (channel 0).
F_L2OF_BASE_HZ = 1_246_000_000.0
#: L1OF channel step in Hz.
F_L1OF_STEP_HZ = 562_500.0
#: L2OF channel step in Hz.
F_L2OF_STEP_HZ = 437_500.0

#: Valid GLONASS channel numbers (inclusive).
CHANNEL_MIN = -7
CHANNEL_MAX = 6


def _check_channel(k: int) -> None:
    if not (CHANNEL_MIN <= int(k) <= CHANNEL_MAX):
        raise ValueError(
            f"GLONASS channel number must be in [{CHANNEL_MIN}, {CHANNEL_MAX}], got {k}"
        )


def l1_frequency_hz(channel: int) -> float:
    """L1OF carrier frequency in Hz for one GLONASS channel number."""
    _check_channel(channel)
    return F_L1OF_BASE_HZ + int(channel) * F_L1OF_STEP_HZ


def l2_frequency_hz(channel: int) -> float:
    """L2OF carrier frequency in Hz for one GLONASS channel number."""
    _check_channel(channel)
    return F_L2OF_BASE_HZ + int(channel) * F_L2OF_STEP_HZ


def l1_wavelength_m(channel: int) -> float:
    """L1OF carrier wavelength in meters."""
    return C_M_PER_S / l1_frequency_hz(channel)


def l2_wavelength_m(channel: int) -> float:
    """L2OF carrier wavelength in meters."""
    return C_M_PER_S / l2_frequency_hz(channel)


def frequencies_array(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``(f_L1, f_L2)`` arrays for a list of channels.

    Parameters
    ----------
    channels:
        ``(n,)`` channel numbers (int).

    Returns
    -------
    f_l1, f_l2 : ndarray
        ``(n,)`` arrays of L1OF and L2OF frequencies in Hz.
    """
    k = np.asarray(channels, dtype=int)
    if np.any(k < CHANNEL_MIN) or np.any(k > CHANNEL_MAX):
        raise ValueError(
            f"channel numbers must be in [{CHANNEL_MIN}, {CHANNEL_MAX}]"
        )
    return (
        F_L1OF_BASE_HZ + k * F_L1OF_STEP_HZ,
        F_L2OF_BASE_HZ + k * F_L2OF_STEP_HZ,
    )


def iono_free_pseudorange(
    p1_m: np.ndarray,
    p2_m: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    """Per-SV iono-free code combination for GLONASS observations.

    The standard iono-free combination

        IF = (f1^2 * P1 - f2^2 * P2) / (f1^2 - f2^2)

    uses each SV's own ``(f1, f2)`` because GLONASS frequencies depend
    on the channel number. This function broadcasts the right pair
    for each row.

    Parameters
    ----------
    p1_m, p2_m:
        ``(n_sv,)`` L1OF and L2OF code pseudoranges in meters.
    channels:
        ``(n_sv,)`` GLONASS channel numbers.

    Returns
    -------
    ndarray
        ``(n_sv,)`` iono-free pseudoranges in meters.
    """
    f1, f2 = frequencies_array(channels)
    f1sq = f1 * f1
    f2sq = f2 * f2
    p1 = np.asarray(p1_m, dtype=float)
    p2 = np.asarray(p2_m, dtype=float)
    return (f1sq * p1 - f2sq * p2) / (f1sq - f2sq)


def iono_free_phase(
    l1_m: np.ndarray,
    l2_m: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    """Per-SV iono-free carrier phase combination for GLONASS.

    Inputs are phase in meters (cycles times the per-SV wavelength).
    """
    f1, f2 = frequencies_array(channels)
    f1sq = f1 * f1
    f2sq = f2 * f2
    l1 = np.asarray(l1_m, dtype=float)
    l2 = np.asarray(l2_m, dtype=float)
    return (f1sq * l1 - f2sq * l2) / (f1sq - f2sq)


def phase_cycles_to_meters(
    phi_cycles: np.ndarray,
    channels: np.ndarray,
    *,
    band: str = "L1",
) -> np.ndarray:
    """Convert per-SV phase from cycles to meters using each SV's wavelength.

    Parameters
    ----------
    phi_cycles:
        ``(n_sv,)`` carrier phase in cycles.
    channels:
        ``(n_sv,)`` GLONASS channel numbers.
    band:
        ``"L1"`` or ``"L2"``. Default ``"L1"``.
    """
    f = (
        frequencies_array(channels)[0]
        if band.upper() == "L1"
        else frequencies_array(channels)[1]
    )
    lam = C_M_PER_S / f
    return np.asarray(phi_cycles, dtype=float) * lam


__all__ = [
    "CHANNEL_MAX",
    "CHANNEL_MIN",
    "C_M_PER_S",
    "F_L1OF_BASE_HZ",
    "F_L1OF_STEP_HZ",
    "F_L2OF_BASE_HZ",
    "F_L2OF_STEP_HZ",
    "frequencies_array",
    "iono_free_phase",
    "iono_free_pseudorange",
    "l1_frequency_hz",
    "l1_wavelength_m",
    "l2_frequency_hz",
    "l2_wavelength_m",
    "phase_cycles_to_meters",
]
