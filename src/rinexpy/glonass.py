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

from typing import Any

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


# ---------------------------------------------------------------------------
# Raw string decoders (GLONASS ICD §4.4)
# ---------------------------------------------------------------------------
#
# Each GLONASS string is 100 bits / 2 s; the first 85 are data, the
# trailing 15 are an idle gap. ICD numbering counts bit 85 first
# (idle, =0) and bit 1 last (LSB of the 8-bit Hamming KX).
#
# The decoders below take the 85 data bits packed MSB-first into 11
# bytes (the last byte is left-shifted by 3 zero pad bits). That is
# the layout produced by u-blox RXM-SFRBX after byte-swapping into
# big-endian.
#
# IMPORTANT: GLONASS numeric fields use **sign-magnitude** encoding,
# not two's complement. The MSB is a sign bit; the remaining (n-1)
# bits are the unsigned magnitude.

KM_TO_M = 1000.0


def _bits(data: bytes, start: int, n: int) -> int:
    """MSB-first unsigned bit extraction (matches gps_cnav._bits)."""
    try:
        from . import _native
        if _native.have_read_bits():
            return _native.read_bits(bytes(data), start, n, False)
    except Exception:
        pass
    value = 0
    for i in range(n):
        byte_idx, bit_idx = divmod(start + i, 8)
        value = (value << 1) | ((data[byte_idx] >> (7 - bit_idx)) & 1)
    return value


def _sign_magnitude(raw: int, n: int) -> int:
    """GLONASS sign-magnitude decode: MSB is the sign, low (n-1) bits
    are the unsigned magnitude. (Sign bit set → negative.)"""
    sign = (raw >> (n - 1)) & 1
    mag = raw & ((1 << (n - 1)) - 1)
    return -mag if sign else mag


def _icd_bit_offset(icd_bit: int) -> int:
    """Translate ICD bit number (1..85, MSB=85) to our 0-based MSB-first
    offset within the 85-data-bit buffer."""
    return 85 - icd_bit


def _string_number(payload: bytes) -> int:
    # ICD bits 84..81 -> offsets 1..4 (4 bits) holds string number m.
    return _bits(payload, 1, 4)


def decode_glonass_string1(payload: bytes) -> dict[str, Any]:
    """Decode GLONASS NAV String 1 (ephemeris X + X_dot + X_dot_dot).

    Per ICD §4.4 Table 4.5. ``payload`` must be the 85 data bits packed
    MSB-first into 11 bytes (the 8th bit of the 11th byte is the LSB of
    the 8-bit Hamming code; trailing 3 bits are pad).
    """
    m = _string_number(payload)
    if m != 1:
        raise ValueError(f"expected GLONASS string 1, got m={m}")
    # ICD bits 78-77 = P1
    p1 = _bits(payload, _icd_bit_offset(78), 2)
    # ICD bits 76-65 = t_k (12 bits, time at start of frame, encoded as
    # 5+6+1 = hours/min/30s subgroups)
    t_k_raw = _bits(payload, _icd_bit_offset(76), 12)
    t_k_h = (t_k_raw >> 7) & 0x1F
    t_k_m = (t_k_raw >> 1) & 0x3F
    t_k_30s = t_k_raw & 1
    t_k_s = t_k_h * 3600 + t_k_m * 60 + t_k_30s * 30
    # ICD bits 64-41 = X_dot (24 bits, sign-magnitude, 2^-20 km/s)
    xdot_raw = _bits(payload, _icd_bit_offset(64), 24)
    xdot_km_s = _sign_magnitude(xdot_raw, 24) * 2 ** -20
    # ICD bits 40-36 = X_dot_dot (5 bits, sign-magnitude, 2^-30 km/s^2)
    xddot_raw = _bits(payload, _icd_bit_offset(40), 5)
    xddot_km_s2 = _sign_magnitude(xddot_raw, 5) * 2 ** -30
    # ICD bits 35-9 = X (27 bits, sign-magnitude, 2^-11 km)
    x_raw = _bits(payload, _icd_bit_offset(35), 27)
    x_km = _sign_magnitude(x_raw, 27) * 2 ** -11
    return {
        "string": 1,
        "P1": p1,
        "t_k_s": t_k_s,
        "x_m": x_km * KM_TO_M,
        "x_dot_m_s": xdot_km_s * KM_TO_M,
        "x_dot_dot_m_s2": xddot_km_s2 * KM_TO_M,
    }


def decode_glonass_string2(payload: bytes) -> dict[str, Any]:
    """Decode GLONASS NAV String 2 (ephemeris Y + Y_dot + Y_dot_dot,
    plus B_n / P2 / t_b)."""
    m = _string_number(payload)
    if m != 2:
        raise ValueError(f"expected GLONASS string 2, got m={m}")
    # ICD bits 80-78 = B_n
    b_n = _bits(payload, _icd_bit_offset(80), 3)
    # ICD bit 77 = P2
    p2 = _bits(payload, _icd_bit_offset(77), 1)
    # ICD bits 76-70 = t_b (7 bits, 15-minute increments, 1..96)
    t_b = _bits(payload, _icd_bit_offset(76), 7)
    # ICD bits 64-41 = Y_dot (sign-magnitude, 2^-20 km/s)
    ydot_raw = _bits(payload, _icd_bit_offset(64), 24)
    ydot_km_s = _sign_magnitude(ydot_raw, 24) * 2 ** -20
    # ICD bits 40-36 = Y_dot_dot
    yddot_raw = _bits(payload, _icd_bit_offset(40), 5)
    yddot_km_s2 = _sign_magnitude(yddot_raw, 5) * 2 ** -30
    # ICD bits 35-9 = Y
    y_raw = _bits(payload, _icd_bit_offset(35), 27)
    y_km = _sign_magnitude(y_raw, 27) * 2 ** -11
    return {
        "string": 2,
        "B_n": b_n,
        "P2": p2,
        "t_b_s": t_b * 15 * 60,
        "y_m": y_km * KM_TO_M,
        "y_dot_m_s": ydot_km_s * KM_TO_M,
        "y_dot_dot_m_s2": yddot_km_s2 * KM_TO_M,
    }


def decode_glonass_string3(payload: bytes) -> dict[str, Any]:
    """Decode GLONASS NAV String 3 (ephemeris Z + Z_dot + Z_dot_dot,
    clock relative-frequency gamma_n, P3, l_n)."""
    m = _string_number(payload)
    if m != 3:
        raise ValueError(f"expected GLONASS string 3, got m={m}")
    # ICD bit 80 = P3
    p3 = _bits(payload, _icd_bit_offset(80), 1)
    # ICD bits 79-69 = gamma_n (11 bits, sign-magnitude, 2^-40)
    gamma_raw = _bits(payload, _icd_bit_offset(79), 11)
    gamma_n = _sign_magnitude(gamma_raw, 11) * 2 ** -40
    # ICD bits 67-66 = P (mode flag, 2 bits)
    p_mode = _bits(payload, _icd_bit_offset(67), 2)
    # ICD bit 65 = l_n (SV health for string-3-only check)
    l_n = _bits(payload, _icd_bit_offset(65), 1)
    # ICD bits 64-41 = Z_dot
    zdot_raw = _bits(payload, _icd_bit_offset(64), 24)
    zdot_km_s = _sign_magnitude(zdot_raw, 24) * 2 ** -20
    # ICD bits 40-36 = Z_dot_dot
    zddot_raw = _bits(payload, _icd_bit_offset(40), 5)
    zddot_km_s2 = _sign_magnitude(zddot_raw, 5) * 2 ** -30
    # ICD bits 35-9 = Z
    z_raw = _bits(payload, _icd_bit_offset(35), 27)
    z_km = _sign_magnitude(z_raw, 27) * 2 ** -11
    return {
        "string": 3,
        "P3": p3,
        "gamma_n": gamma_n,
        "P": p_mode,
        "l_n": l_n,
        "z_m": z_km * KM_TO_M,
        "z_dot_m_s": zdot_km_s * KM_TO_M,
        "z_dot_dot_m_s2": zddot_km_s2 * KM_TO_M,
    }


def decode_glonass_string(payload: bytes) -> dict[str, Any]:
    """Dispatch to the matching string decoder by reading the 4-bit
    string number. Strings outside 1..3 return ``{"string": m, "raw":
    bytes(...)}``."""
    m = _string_number(payload)
    if m == 1:
        return decode_glonass_string1(payload)
    if m == 2:
        return decode_glonass_string2(payload)
    if m == 3:
        return decode_glonass_string3(payload)
    return {"string": m, "raw": bytes(payload)}


__all__ = [
    "CHANNEL_MAX",
    "CHANNEL_MIN",
    "C_M_PER_S",
    "F_L1OF_BASE_HZ",
    "F_L1OF_STEP_HZ",
    "F_L2OF_BASE_HZ",
    "F_L2OF_STEP_HZ",
    "decode_glonass_string",
    "decode_glonass_string1",
    "decode_glonass_string2",
    "decode_glonass_string3",
    "frequencies_array",
    "iono_free_phase",
    "iono_free_pseudorange",
    "l1_frequency_hz",
    "l1_wavelength_m",
    "l2_frequency_hz",
    "l2_wavelength_m",
    "phase_cycles_to_meters",
]
