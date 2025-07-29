"""GPS LNAV navigation message subframe decoder.

LNAV (Legacy NAV) is the original GPS broadcast navigation message, 50
bps, transmitted on L1 C/A and L2 P(Y). A frame is 30 s of 5 subframes;
each subframe is 6 s of 300 bits (10 words by 30 bits, 24 data + 6
parity per word).

Subframe layout:
    1: clock parameters, T_GD, week, SV health, IODC.
    2: ephemeris part 1 (IODE, M_0, e, sqrt(A), C_uc, C_us, dn, t_oe).
    3: ephemeris part 2 (i_0, omega, Omega_0, Omega_dot, IDOT,
       C_ic, C_is, C_rc).
    4, 5: almanac, ionospheric model, UTC parameters
          (page-paginated; not decoded here).

Reference: ICD-GPS-200 sections 20.3.3.3 / 20.3.3.4 (Tables 20-I,
20-II, 20-III).

These decoders take pre-extracted 30-bit words (typically from a
receiver firmware or RTCM3 1019 / UBX RXM-SFRBX) and return the
structured fields. Parity validation isn't performed; the receiver
firmware that produced the words has already verified it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _native

#: GPS LNAV 8-bit preamble (1000 1011).
PREAMBLE = 0x8B

#: Pi exactly as ICD-GPS-200 specifies.
_PI = 3.1415926535898


def _strip_parity(words: list[int]) -> str:
    """Drop 6 trailing parity bits per word; return a 240-bit data string."""
    if len(words) < 10:
        raise ValueError("LNAV subframe needs 10 words")
    return "".join(f"{((w >> 6) & 0xFFFFFF):024b}" for w in words)


def _bits(data: str, start: int, n: int, *, signed: bool = False) -> int:
    """Read ``n`` bits from ``data`` starting at ``start`` (MSB-first)."""
    chunk = data[start : start + n]
    value = int(chunk, 2)
    if signed and chunk[0] == "1":
        value -= 1 << n
    return value


def _split_uint(data: str, msb_start: int, msb_n: int, lsb_start: int, lsb_n: int) -> int:
    """Concatenate two unsigned bitfields into a single integer."""
    return (_bits(data, msb_start, msb_n) << lsb_n) | _bits(data, lsb_start, lsb_n)


def _split_int(data: str, msb_start: int, msb_n: int, lsb_start: int, lsb_n: int) -> int:
    """Concatenate two bitfields into a signed (two's-complement) integer."""
    combined = _split_uint(data, msb_start, msb_n, lsb_start, lsb_n)
    total = msb_n + lsb_n
    if combined & (1 << (total - 1)):
        combined -= 1 << total
    return combined


def _check_preamble_and_id(data: str, expected: int) -> int:
    pre = _bits(data, 0, 8)
    if pre != PREAMBLE:
        raise ValueError(f"bad GPS LNAV preamble: {pre:#04x} != {PREAMBLE:#04x}")
    # Word 2 (HOW): bits 20-22 of the word = data positions 19-21 within
    # the word, i.e. positions 24+19 .. 24+21 = 43-45 in the 240-bit stream.
    sf = _bits(data, 43, 3)
    if sf != expected:
        raise ValueError(f"expected subframe {expected}, got {sf}")
    return sf


def _tow(data: str) -> int:
    """The 17-bit TOW count from the HOW word, in 1.5 s units, mod 1 week."""
    return _bits(data, 24, 17)


def _dispatch_native(words: list[int], expected_id: int) -> dict[str, Any]:
    """Forward a subframe through the C++ kernel; mirror the Python
    error-path messages so callers can't tell which path executed."""
    try:
        return _native.decode_lnav_subframe(
            np.asarray(words, dtype=np.uint32), expected_id
        )
    except Exception as e:  # nanobind raises std::invalid_argument as ValueError
        msg = str(e)
        if "preamble" in msg.lower():
            raise ValueError(f"bad GPS LNAV preamble") from None
        if "id mismatch" in msg.lower() or "expected" in msg.lower():
            raise ValueError(
                f"expected subframe {expected_id}, got mismatch"
            ) from None
        raise


def decode_lnav_subframe1(words: list[int]) -> dict[str, Any]:
    """Decode LNAV subframe 1 (clock, T_GD, week, IODC).

    Parameters
    ----------
    words:
        Ten 30-bit ``int``s, one per word of the subframe (parity bits
        included; we strip them).

    Returns
    -------
    dict
        Clock and signal-quality fields in SI units per ICD-GPS-200
        Table 20-I.

    Raises
    ------
    ValueError
        If the preamble or subframe ID don't match.
    """
    if len(words) < 10:
        raise ValueError("LNAV subframe needs 10 words")
    if _native.have_decode_lnav_subframe():
        return _dispatch_native(words, 1)
    data = _strip_parity(words)
    sf = _check_preamble_and_id(data, 1)

    # Word 3 (data 48..71): WN(10) CA/P(2) URA(4) Health(6) IODC_msb(2)
    week = _bits(data, 48, 10)
    ca_or_p_on_l2 = _bits(data, 58, 2)
    ura = _bits(data, 60, 4)
    health = _bits(data, 64, 6)
    iodc_msb = _bits(data, 70, 2)
    # Word 4 (data 72..95): L2 P data flag (1), reserved (23)
    l2_p_data_flag = _bits(data, 72, 1)
    # Word 7 (data 144..167): reserved (16) + T_GD(8 signed)
    tgd = _bits(data, 160, 8, signed=True) * 2**-31
    # Word 8 (data 168..191): IODC_lsb(8) + t_oc(16)
    iodc_lsb = _bits(data, 168, 8)
    toc = _bits(data, 176, 16) * 16
    # Word 9 (data 192..215): a_f2(8 signed) + a_f1(16 signed)
    af2 = _bits(data, 192, 8, signed=True) * 2**-55
    af1 = _bits(data, 200, 16, signed=True) * 2**-43
    # Word 10 (data 216..237): a_f0(22 signed) + 2 parity-helper bits
    af0 = _bits(data, 216, 22, signed=True) * 2**-31

    return {
        "subframe_id": sf,
        "tow_count": _tow(data),
        "week": week,
        "ca_or_p_on_l2": ca_or_p_on_l2,
        "URA": ura,
        "SV_health": health,
        "IODC": (iodc_msb << 8) | iodc_lsb,
        "L2_P_data_flag": l2_p_data_flag,
        "T_GD_s": tgd,
        "t_oc_s": toc,
        "a_f0_s": af0,
        "a_f1_s_per_s": af1,
        "a_f2_s_per_s2": af2,
    }


def decode_lnav_subframe2(words: list[int]) -> dict[str, Any]:
    """Decode LNAV subframe 2 (ephemeris part 1).

    Returns IODE, M_0, e, sqrt(A), C_uc, C_us, delta_n, t_oe in SI
    units per ICD-GPS-200 Table 20-II.
    """
    if len(words) < 10:
        raise ValueError("LNAV subframe needs 10 words")
    if _native.have_decode_lnav_subframe():
        return _dispatch_native(words, 2)
    data = _strip_parity(words)
    sf = _check_preamble_and_id(data, 2)

    # Word 3 (data 48..71): IODE(8) + C_rs(16 signed)
    iode = _bits(data, 48, 8)
    c_rs = _bits(data, 56, 16, signed=True) * 2**-5
    # Word 4 (data 72..95): dn(16 signed) + M_0_msb(8)
    delta_n_sc = _bits(data, 72, 16, signed=True) * 2**-43
    delta_n = delta_n_sc * _PI
    # Word 5 (data 96..119): M_0_lsb(24); combined 32-bit signed
    m0_sc = _split_int(data, 88, 8, 96, 24) * 2**-31
    m0 = m0_sc * _PI
    # Word 6 (data 120..143): C_uc(16 signed) + e_msb(8)
    c_uc = _bits(data, 120, 16, signed=True) * 2**-29
    # Word 7 (data 144..167): e_lsb(24); combined 32-bit unsigned
    e = _split_uint(data, 136, 8, 144, 24) * 2**-33
    # Word 8 (data 168..191): C_us(16 signed) + sqrt(A)_msb(8)
    c_us = _bits(data, 168, 16, signed=True) * 2**-29
    # Word 9 (data 192..215): sqrt(A)_lsb(24); combined 32-bit unsigned
    sqrt_a = _split_uint(data, 184, 8, 192, 24) * 2**-19
    # Word 10 (data 216..239): t_oe(16) + fit_interval(1) + AODO(5) + 2 unused
    toe = _bits(data, 216, 16) * 16
    fit_interval_flag = _bits(data, 232, 1)
    aodo = _bits(data, 233, 5)

    return {
        "subframe_id": sf,
        "tow_count": _tow(data),
        "IODE": iode,
        "C_rs_m": c_rs,
        "delta_n_rad_s": delta_n,
        "M_0_rad": m0,
        "C_uc_rad": c_uc,
        "e": e,
        "C_us_rad": c_us,
        "sqrt_A_root_m": sqrt_a,
        "t_oe_s": toe,
        "fit_interval_flag": fit_interval_flag,
        "AODO": aodo,
    }


def decode_lnav_subframe3(words: list[int]) -> dict[str, Any]:
    """Decode LNAV subframe 3 (ephemeris part 2).

    Returns C_ic, Omega_0, C_is, i_0, C_rc, omega, Omega_dot, IODE,
    IDOT in SI units per ICD-GPS-200 Table 20-III.
    """
    if len(words) < 10:
        raise ValueError("LNAV subframe needs 10 words")
    if _native.have_decode_lnav_subframe():
        return _dispatch_native(words, 3)
    data = _strip_parity(words)
    sf = _check_preamble_and_id(data, 3)

    # Word 3 (data 48..71): C_ic(16 signed) + Omega_0_msb(8)
    c_ic = _bits(data, 48, 16, signed=True) * 2**-29
    # Word 4 (data 72..95): Omega_0_lsb(24); combined 32-bit signed
    omega0_sc = _split_int(data, 64, 8, 72, 24) * 2**-31
    omega0 = omega0_sc * _PI
    # Word 5 (data 96..119): C_is(16 signed) + i_0_msb(8)
    c_is = _bits(data, 96, 16, signed=True) * 2**-29
    # Word 6 (data 120..143): i_0_lsb(24); combined 32-bit signed
    i0_sc = _split_int(data, 112, 8, 120, 24) * 2**-31
    i0 = i0_sc * _PI
    # Word 7 (data 144..167): C_rc(16 signed) + omega_msb(8)
    c_rc = _bits(data, 144, 16, signed=True) * 2**-5
    # Word 8 (data 168..191): omega_lsb(24); combined 32-bit signed
    omega_sc = _split_int(data, 160, 8, 168, 24) * 2**-31
    omega = omega_sc * _PI
    # Word 9 (data 192..215): Omega_dot(24 signed)
    omega_dot_sc = _bits(data, 192, 24, signed=True) * 2**-43
    omega_dot = omega_dot_sc * _PI
    # Word 10 (data 216..239): IODE(8) + IDOT(14 signed) + 2 unused
    iode = _bits(data, 216, 8)
    idot_sc = _bits(data, 224, 14, signed=True) * 2**-43
    idot = idot_sc * _PI

    return {
        "subframe_id": sf,
        "tow_count": _tow(data),
        "C_ic_rad": c_ic,
        "Omega_0_rad": omega0,
        "C_is_rad": c_is,
        "i_0_rad": i0,
        "C_rc_m": c_rc,
        "omega_rad": omega,
        "Omega_dot_rad_s": omega_dot,
        "IODE": iode,
        "IDOT_rad_s": idot,
    }


def encode_lnav_words(field_specs: list[tuple[int, int]]) -> list[int]:
    """Helper for tests: pack ``[(value, n_bits), ...]`` into 10 30-bit words.

    Treats the list as a stream of data bits, MSB-first, packed 24 bits
    per word. The 6 parity bits per word are written as zeros; the
    decoders strip parity before reading, so this round-trips without
    computing parity.
    """
    bits = ""
    for value, n in field_specs:
        if n <= 0:
            continue
        bits += f"{value & ((1 << n) - 1):0{n}b}"
    pad = max(0, 240 - len(bits))
    bits = (bits + "0" * pad)[:240]
    return [int(bits[w * 24 : (w + 1) * 24], 2) << 6 for w in range(10)]


__all__ = [
    "PREAMBLE",
    "decode_lnav_subframe1",
    "decode_lnav_subframe2",
    "decode_lnav_subframe3",
    "encode_lnav_words",
]
