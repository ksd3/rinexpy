"""GPS CNAV-2 (L1C) navigation message decoder.

CNAV-2 is the modernized navigation message broadcast on the L1C
civilian signal. The frame structure (per IS-GPS-800) is:

    Frame = Subframe 1 (TOI, 9 bits) + Subframe 2 (1200 bits, clock +
    ephemeris) + Subframe 3 (274 bits, various page types).

Subframe 2 always carries: TOW, WN, ITOW, plus the full Keplerian
ephemeris in the same field set as CNAV MT 10 + MT 11 combined.

Subframe 3 is paginated by page-ID:
    Page 1: UTC parameters
    Page 2: GGTO (GPS-GNSS Time Offset)
    Page 3: Reduced almanac
    Page 4: Midi almanac
    Page 5: Differential corrections
    Page 6: Text message

This module decodes Subframe 2 (the ephemeris); Subframe 3 pages
return the raw payload.

Reference: IS-GPS-800.
"""

from __future__ import annotations

from typing import Any

from .gps_cnav import _bits


def decode_cnav2_subframe2(payload: bytes) -> dict[str, Any]:
    """Decode CNAV-2 Subframe 2 (1200 bits = 150 bytes).

    Returns a dict with TOW, WN, the same Keplerian fields as the
    CNAV MT 10 + MT 11 combined set, plus health and ISC corrections.
    """
    bit = 0
    wn = _bits(payload, bit, 13); bit += 13
    itow = _bits(payload, bit, 8); bit += 8
    t_op_s = _bits(payload, bit, 11) * 300; bit += 11
    health = _bits(payload, bit, 1); bit += 1
    ura_index = _bits(payload, bit, 5, signed=True); bit += 5
    t_oe_s = _bits(payload, bit, 11) * 300; bit += 11
    delta_a_m = _bits(payload, bit, 26, signed=True) * 2 ** -9; bit += 26
    Adot_m_s = _bits(payload, bit, 25, signed=True) * 2 ** -21; bit += 25
    delta_n0 = _bits(payload, bit, 17, signed=True) * 2 ** -44; bit += 17
    delta_n0_dot = _bits(payload, bit, 23, signed=True) * 2 ** -57; bit += 23
    m0_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    e_n = _bits(payload, bit, 33) * 2 ** -34; bit += 33
    omega_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    omega0_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    i0_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    delta_omega_dot = _bits(payload, bit, 17, signed=True) * 2 ** -44; bit += 17
    idot = _bits(payload, bit, 15, signed=True) * 2 ** -44; bit += 15
    cis = _bits(payload, bit, 16, signed=True) * 2 ** -30; bit += 16
    cic = _bits(payload, bit, 16, signed=True) * 2 ** -30; bit += 16
    crs = _bits(payload, bit, 24, signed=True) * 2 ** -8; bit += 24
    crc = _bits(payload, bit, 24, signed=True) * 2 ** -8; bit += 24
    cus = _bits(payload, bit, 21, signed=True) * 2 ** -30; bit += 21
    cuc = _bits(payload, bit, 21, signed=True) * 2 ** -30; bit += 21
    af0 = _bits(payload, bit, 26, signed=True) * 2 ** -35; bit += 26
    af1 = _bits(payload, bit, 20, signed=True) * 2 ** -48; bit += 20
    af2 = _bits(payload, bit, 10, signed=True) * 2 ** -60; bit += 10
    tgd = _bits(payload, bit, 13, signed=True) * 2 ** -35; bit += 13
    return {
        "week": wn,
        "ITOW": itow,
        "t_op_s": t_op_s,
        "SV_health": health,
        "URA_index": ura_index,
        "t_oe_s": t_oe_s,
        "delta_A_m": delta_a_m,
        "Adot_m_per_s": Adot_m_s,
        "delta_n0_semicircles_per_s": delta_n0,
        "delta_n0_dot_semicircles_per_s2": delta_n0_dot,
        "M_0_n_semicircles": m0_n,
        "e_n": e_n,
        "omega_n_semicircles": omega_n,
        "Omega_0_n_semicircles": omega0_n,
        "i_0_n_semicircles": i0_n,
        "delta_Omega_dot_semicircles_per_s": delta_omega_dot,
        "IDOT_semicircles_per_s": idot,
        "C_is_rad": cis, "C_ic_rad": cic,
        "C_rs_m": crs, "C_rc_m": crc,
        "C_us_rad": cus, "C_uc_rad": cuc,
        "a_f0_s": af0, "a_f1_s_per_s": af1, "a_f2_s_per_s2": af2,
        "T_GD_s": tgd,
    }


__all__ = ["decode_cnav2_subframe2"]
