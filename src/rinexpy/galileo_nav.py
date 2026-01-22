"""Galileo F-NAV (E5a) and I-NAV (E1B / E5b) raw subframe decoders.

Per the Galileo Open Service ICD:

- **F-NAV** is transmitted on E5a, 25 sps, paginated in pages of 244
  bits. Each page carries one of MT 1..7. Pages 1, 2, 3 carry the
  clock + ephemeris.
- **I-NAV** is transmitted on E1B and E5b, 250 sps, paginated in
  words of 128 bits. Each subframe is 720 bits. Word types of
  interest:
    - Word 1: SVID + IODnav + ephemeris1
    - Word 2: ephemeris2
    - Word 3: ephemeris3
    - Word 4: clock correction
    - Word 5: ionospheric model + BGD + signal health
    - Word 6: GST + UTC parameters

Reference: Galileo OS SIS ICD §4.3 (F-NAV) / §4.2 (I-NAV).
"""

from __future__ import annotations

from typing import Any

from .gps_cnav import _bits

_PI = 3.1415926535898


def decode_fnav_page1(payload: bytes) -> dict[str, Any]:
    """Decode an F-NAV page-type 1 (clock + ephemeris part 1)."""
    bit = 0
    page_type = _bits(payload, bit, 6); bit += 6
    if page_type != 1:
        raise ValueError(f"expected F-NAV page type 1, got {page_type}")
    svid = _bits(payload, bit, 6); bit += 6
    iodnav = _bits(payload, bit, 10); bit += 10
    t_0c = _bits(payload, bit, 14) * 60; bit += 14
    af0 = _bits(payload, bit, 31, signed=True) * 2 ** -34; bit += 31
    af1 = _bits(payload, bit, 21, signed=True) * 2 ** -46; bit += 21
    af2 = _bits(payload, bit, 6, signed=True) * 2 ** -59; bit += 6
    sisa = _bits(payload, bit, 8); bit += 8
    ai0 = _bits(payload, bit, 11) * 2 ** -2; bit += 11
    ai1 = _bits(payload, bit, 11, signed=True) * 2 ** -8; bit += 11
    ai2 = _bits(payload, bit, 14, signed=True) * 2 ** -15; bit += 14
    return {
        "svid": svid,
        "IODnav": iodnav,
        "t_oc_s": t_0c,
        "a_f0_s": af0,
        "a_f1_s_per_s": af1,
        "a_f2_s_per_s2": af2,
        "SISA": sisa,
        "ai0": ai0, "ai1": ai1, "ai2": ai2,
    }


def decode_fnav_page2(payload: bytes) -> dict[str, Any]:
    """Decode an F-NAV page-type 2 (ephemeris part 2)."""
    bit = 0
    page_type = _bits(payload, bit, 6); bit += 6
    if page_type != 2:
        raise ValueError(f"expected F-NAV page type 2, got {page_type}")
    iodnav = _bits(payload, bit, 10); bit += 10
    m0 = _bits(payload, bit, 32, signed=True) * 2 ** -31 * _PI; bit += 32
    omega_dot = _bits(payload, bit, 24, signed=True) * 2 ** -43 * _PI; bit += 24
    e = _bits(payload, bit, 32) * 2 ** -33; bit += 32
    sqrt_a = _bits(payload, bit, 32) * 2 ** -19; bit += 32
    omega0 = _bits(payload, bit, 32, signed=True) * 2 ** -31 * _PI; bit += 32
    idot = _bits(payload, bit, 14, signed=True) * 2 ** -43 * _PI; bit += 14
    t_oe = _bits(payload, bit, 14) * 60; bit += 14
    return {
        "IODnav": iodnav,
        "M_0_rad": m0,
        "Omega_dot_rad_s": omega_dot,
        "e": e,
        "sqrt_A_root_m": sqrt_a,
        "Omega_0_rad": omega0,
        "IDOT_rad_s": idot,
        "t_oe_s": t_oe,
    }


def decode_inav_word1(payload: bytes) -> dict[str, Any]:
    """I-NAV Word type 1 (Ephemeris part 1 - IODnav, t_oe, M0, e, sqrtA)."""
    bit = 0
    wt = _bits(payload, bit, 6); bit += 6
    if wt != 1:
        raise ValueError(f"expected I-NAV word type 1, got {wt}")
    iodnav = _bits(payload, bit, 10); bit += 10
    t_oe = _bits(payload, bit, 14) * 60; bit += 14
    m0 = _bits(payload, bit, 32, signed=True) * 2 ** -31 * _PI; bit += 32
    e = _bits(payload, bit, 32) * 2 ** -33; bit += 32
    sqrt_a = _bits(payload, bit, 32) * 2 ** -19; bit += 32
    return {
        "word_type": 1,
        "IODnav": iodnav,
        "t_oe_s": t_oe,
        "M_0_rad": m0,
        "e": e,
        "sqrt_A_root_m": sqrt_a,
    }


def decode_inav_word4(payload: bytes) -> dict[str, Any]:
    """I-NAV Word type 4 (clock correction + a_f0 / a_f1 / a_f2)."""
    bit = 0
    wt = _bits(payload, bit, 6); bit += 6
    if wt != 4:
        raise ValueError(f"expected I-NAV word type 4, got {wt}")
    iodnav = _bits(payload, bit, 10); bit += 10
    svid = _bits(payload, bit, 6); bit += 6
    cic = _bits(payload, bit, 16, signed=True) * 2 ** -29; bit += 16
    cis = _bits(payload, bit, 16, signed=True) * 2 ** -29; bit += 16
    t_oc = _bits(payload, bit, 14) * 60; bit += 14
    af0 = _bits(payload, bit, 31, signed=True) * 2 ** -34; bit += 31
    af1 = _bits(payload, bit, 21, signed=True) * 2 ** -46; bit += 21
    af2 = _bits(payload, bit, 6, signed=True) * 2 ** -59; bit += 6
    return {
        "word_type": 4,
        "IODnav": iodnav,
        "svid": svid,
        "C_ic_rad": cic, "C_is_rad": cis,
        "t_oc_s": t_oc,
        "a_f0_s": af0,
        "a_f1_s_per_s": af1,
        "a_f2_s_per_s2": af2,
    }


__all__ = [
    "decode_fnav_page1",
    "decode_fnav_page2",
    "decode_inav_word1",
    "decode_inav_word4",
]
