"""NavIC / IRNSS L5 + S navigation message decoder.

Per IRNSS-SIS-ICD-SPS v1.1 §5, every NavIC subframe is 600 bits / 12 s.
The on-air layout is:

    [16-bit sync (0xEB90)] [292-bit data] [24-bit CRC] [6 tail bits]
    \\___________________ 600 bits _____________________/

Then half-rate FEC doubles symbol count on-air; this module assumes
the application has already stripped FEC, CRC, and the sync word so
that ``payload`` is the 292 data bits packed MSB-first into 37 bytes
(the last byte holds 4 data bits + 4 zero pad bits).

Subframes 1 and 2 carry the broadcast ephemeris in disjoint halves;
subframes 3 and 4 carry paginated almanac / ionosphere / UTC data,
keyed by a 6-bit Message Type ID. This module decodes SF1 and SF2
in full; SF3/SF4 return ``{"sf_id", "message_id", "raw"}`` so callers
can dispatch further per Table 16/17 of the ICD.

Reference: IS-IRNSS-ICD-SPS v1.1 §5.2 (Frame structure) + Table 13
(SF1 fields) + Table 14 (SF2 fields).
"""

from __future__ import annotations

from typing import Any

from .gps_cnav import _bits

#: NavIC sync word (16 bits, transmitted at the start of every subframe).
SYNC = 0xEB90


def _common_header(payload: bytes) -> dict[str, Any]:
    """Decode the 22 leading bits common to every NavIC subframe:
    TOWC, alert, autonav, SF-ID, spare."""
    bit = 0
    towc = _bits(payload, bit, 17); bit += 17
    alert = _bits(payload, bit, 1); bit += 1
    autonav = _bits(payload, bit, 1); bit += 1
    sf_id = _bits(payload, bit, 2); bit += 2
    # bit 22 is a reserved/spare bit
    return {
        "TOWC": towc,
        "alert": bool(alert),
        "autonav": bool(autonav),
        "sf_id": sf_id + 1,  # ICD uses 1..4
    }


def decode_navic_subframe1(payload: bytes) -> dict[str, Any]:
    """Decode NavIC Subframe 1 (clock + ephemeris part 1, 292 data bits).

    Returns the common header fields plus WN, a_f0/1/2, URA, t_oc,
    T_GD, delta_n, IODEC, signal health flags, C_uc/C_us/C_ic/C_is/
    C_rc/C_rs harmonic correction coefficients, and IDOT.
    """
    out = _common_header(payload)
    bit = 22
    wn = _bits(payload, bit, 10); bit += 10
    af0 = _bits(payload, bit, 22, signed=True) * 2 ** -31; bit += 22
    af1 = _bits(payload, bit, 16, signed=True) * 2 ** -43; bit += 16
    af2 = _bits(payload, bit, 8, signed=True) * 2 ** -55; bit += 8
    ura = _bits(payload, bit, 4); bit += 4
    t_oc = _bits(payload, bit, 16) * 16; bit += 16
    tgd = _bits(payload, bit, 8, signed=True) * 2 ** -31; bit += 8
    delta_n = _bits(payload, bit, 22, signed=True) * 2 ** -41; bit += 22
    iodec = _bits(payload, bit, 8); bit += 8
    bit += 10  # reserved
    health_l5 = _bits(payload, bit, 1); bit += 1
    health_s = _bits(payload, bit, 1); bit += 1
    c_uc = _bits(payload, bit, 15, signed=True) * 2 ** -28; bit += 15
    c_us = _bits(payload, bit, 15, signed=True) * 2 ** -28; bit += 15
    c_ic = _bits(payload, bit, 15, signed=True) * 2 ** -28; bit += 15
    c_is = _bits(payload, bit, 15, signed=True) * 2 ** -28; bit += 15
    c_rc = _bits(payload, bit, 15, signed=True) * 2 ** -4; bit += 15
    c_rs = _bits(payload, bit, 15, signed=True) * 2 ** -4; bit += 15
    idot = _bits(payload, bit, 14, signed=True) * 2 ** -43; bit += 14
    out.update({
        "week": wn,
        "a_f0_s": af0,
        "a_f1_s_per_s": af1,
        "a_f2_s_per_s2": af2,
        "URA": ura,
        "t_oc_s": t_oc,
        "T_GD_s": tgd,
        "delta_n_semicircles_per_s": delta_n,
        "IODEC": iodec,
        "SV_health_L5": health_l5,
        "SV_health_S": health_s,
        "C_uc_rad": c_uc,
        "C_us_rad": c_us,
        "C_ic_rad": c_ic,
        "C_is_rad": c_is,
        "C_rc_m": c_rc,
        "C_rs_m": c_rs,
        "IDOT_semicircles_per_s": idot,
    })
    return out


def decode_navic_subframe2(payload: bytes) -> dict[str, Any]:
    """Decode NavIC Subframe 2 (ephemeris part 2, 292 data bits).

    Returns the common header fields plus M_0, t_oe, e, sqrt_A,
    Omega_0, omega, Omega_dot, i_0.
    """
    out = _common_header(payload)
    bit = 22
    m0 = _bits(payload, bit, 32, signed=True) * 2 ** -31; bit += 32
    t_oe = _bits(payload, bit, 16) * 16; bit += 16
    e = _bits(payload, bit, 32) * 2 ** -33; bit += 32
    sqrt_a = _bits(payload, bit, 32) * 2 ** -19; bit += 32
    omega_0 = _bits(payload, bit, 32, signed=True) * 2 ** -31; bit += 32
    omega = _bits(payload, bit, 32, signed=True) * 2 ** -31; bit += 32
    omega_dot = _bits(payload, bit, 22, signed=True) * 2 ** -41; bit += 22
    i_0 = _bits(payload, bit, 32, signed=True) * 2 ** -31; bit += 32
    out.update({
        "M_0_semicircles": m0,
        "t_oe_s": t_oe,
        "e": e,
        "sqrt_A_root_m": sqrt_a,
        "Omega_0_semicircles": omega_0,
        "omega_semicircles": omega,
        "Omega_dot_semicircles_per_s": omega_dot,
        "i_0_semicircles": i_0,
    })
    return out


def decode_navic_subframe34(payload: bytes) -> dict[str, Any]:
    """Decode NavIC Subframe 3 or 4 (paginated almanac / iono / UTC).

    Returns ``{"sf_id", "message_id", "raw"}`` - the 6-bit Message Type
    ID is read at the start of the subframe-specific payload and the
    rest of the data is returned untouched so callers can dispatch
    per ICD Table 16 / 17.
    """
    out = _common_header(payload)
    bit = 22
    message_id = _bits(payload, bit, 6)
    out["message_id"] = message_id
    out["raw"] = bytes(payload)
    return out


def decode_navic_subframe(payload: bytes) -> dict[str, Any]:
    """Dispatch a NavIC subframe to the right decoder by reading the
    2-bit SF-ID field. SF3/SF4 return raw payload + message_id."""
    sf_id = _bits(payload, 19, 2) + 1
    if sf_id == 1:
        return decode_navic_subframe1(payload)
    if sf_id == 2:
        return decode_navic_subframe2(payload)
    return decode_navic_subframe34(payload)


__all__ = [
    "SYNC",
    "decode_navic_subframe",
    "decode_navic_subframe1",
    "decode_navic_subframe2",
    "decode_navic_subframe34",
]
