"""Synthetic round-trip tests for the Galileo F-NAV / I-NAV decoders.

Per Galileo OS SIS ICD §4.2 / §4.3, every page or word begins with a
6-bit type indicator. These tests build a page/word bit-pattern from
known field values and verify each field decodes with the documented
scaling.
"""

from __future__ import annotations

from pytest import approx

from rinexpy.galileo_nav import (
    decode_fnav_page1,
    decode_fnav_page2,
    decode_inav_word1,
    decode_inav_word4,
)

_PI = 3.1415926535898


def _pack(field_specs: list[tuple[int, int]]) -> bytes:
    bits = "".join(
        f"{value & ((1 << n) - 1):0{n}b}" for value, n in field_specs if n > 0
    )
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def test_fnav_page1_round_trip():
    payload = _pack(
        [
            (1, 6),       # page type
            (12, 6),      # SVID
            (200, 10),    # IODnav
            (5, 14),      # t_0c = 300 s
            (-1234, 31),  # a_f0
            (7, 21),      # a_f1
            (-3, 6),      # a_f2
            (160, 8),     # SISA
            (50, 11),     # ai0
            (-7, 11),     # ai1
            (3, 14),      # ai2
        ]
    )
    out = decode_fnav_page1(payload)
    assert out["svid"] == 12
    assert out["IODnav"] == 200
    assert out["t_oc_s"] == 300
    assert out["a_f0_s"] == approx(-1234 * 2 ** -34)
    assert out["a_f1_s_per_s"] == approx(7 * 2 ** -46)
    assert out["a_f2_s_per_s2"] == approx(-3 * 2 ** -59)
    assert out["SISA"] == 160
    assert out["ai0"] == approx(50 * 2 ** -2)
    assert out["ai1"] == approx(-7 * 2 ** -8)
    assert out["ai2"] == approx(3 * 2 ** -15)


def test_fnav_page2_round_trip():
    payload = _pack(
        [
            (2, 6),
            (200, 10),       # IODnav
            (-100, 32),      # M_0
            (5, 24),         # Omega_dot
            (12345, 32),     # e (unsigned)
            (54321, 32),     # sqrtA
            (-7, 32),        # Omega_0
            (3, 14),         # IDOT
            (8, 14),         # t_oe = 480 s
        ]
    )
    out = decode_fnav_page2(payload)
    assert out["IODnav"] == 200
    assert out["M_0_rad"] == approx(-100 * 2 ** -31 * _PI)
    assert out["Omega_dot_rad_s"] == approx(5 * 2 ** -43 * _PI)
    assert out["e"] == approx(12345 * 2 ** -33)
    assert out["sqrt_A_root_m"] == approx(54321 * 2 ** -19)
    assert out["IDOT_rad_s"] == approx(3 * 2 ** -43 * _PI)
    assert out["t_oe_s"] == 480


def test_inav_word1_round_trip():
    payload = _pack(
        [
            (1, 6),
            (50, 10),        # IODnav
            (7, 14),         # t_oe = 420 s
            (-9, 32),        # M_0
            (12345, 32),     # e
            (54321, 32),     # sqrtA
        ]
    )
    out = decode_inav_word1(payload)
    assert out["word_type"] == 1
    assert out["IODnav"] == 50
    assert out["t_oe_s"] == 420
    assert out["e"] == approx(12345 * 2 ** -33)


def test_inav_word4_round_trip():
    payload = _pack(
        [
            (4, 6),
            (99, 10),       # IODnav
            (15, 6),        # SVID
            (-100, 16),     # Cic
            (200, 16),      # Cis
            (3, 14),        # t_oc = 180 s
            (-1234, 31),    # a_f0
            (5, 21),        # a_f1
            (-1, 6),        # a_f2
        ]
    )
    out = decode_inav_word4(payload)
    assert out["word_type"] == 4
    assert out["IODnav"] == 99
    assert out["svid"] == 15
    assert out["C_ic_rad"] == approx(-100 * 2 ** -29)
    assert out["C_is_rad"] == approx(200 * 2 ** -29)
    assert out["t_oc_s"] == 180
    assert out["a_f0_s"] == approx(-1234 * 2 ** -34)
