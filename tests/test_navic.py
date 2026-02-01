"""Synthetic round-trip tests for the NavIC / IRNSS subframe decoder.

No public NavIC RXM-SFRBX fixture is shipped in tests/data, so these
tests build a 292-bit data payload from known field values and verify
each field round-trips with the scale documented in IRNSS-SIS-ICD-SPS
v1.1 Table 13 (SF1) and Table 14 (SF2).
"""

from __future__ import annotations

from pytest import approx

from rinexpy.navic import (
    decode_navic_subframe,
    decode_navic_subframe1,
    decode_navic_subframe2,
    decode_navic_subframe34,
)


def _pack(field_specs: list[tuple[int, int]]) -> bytes:
    bits = "".join(
        f"{value & ((1 << n) - 1):0{n}b}" for value, n in field_specs if n > 0
    )
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def _header(sf_id: int, towc: int = 100, alert: int = 0, autonav: int = 0):
    # sf_id is 1..4 in the ICD; the field on-wire is sf_id - 1.
    return [
        (towc, 17),
        (alert, 1),
        (autonav, 1),
        (sf_id - 1, 2),
        (0, 1),    # spare
    ]


def test_navic_subframe1_round_trip():
    payload = _pack(
        _header(sf_id=1, towc=12345)
        + [
            (789, 10),       # WN
            (-1234, 22),     # a_f0
            (5, 16),         # a_f1
            (-3, 8),         # a_f2
            (8, 4),          # URA
            (1000, 16),      # t_oc -> 16000 s
            (-9, 8),         # T_GD
            (200, 22),       # delta_n
            (123, 8),        # IODEC
            (0, 10),         # reserved
            (1, 1), (0, 1),  # L5 health, S health
            (-7, 15), (8, 15),   # C_uc, C_us
            (-1, 15), (2, 15),   # C_ic, C_is
            (-3, 15), (4, 15),   # C_rc, C_rs
            (-5, 14),            # IDOT
        ]
    )
    out = decode_navic_subframe1(payload)
    assert out["sf_id"] == 1
    assert out["TOWC"] == 12345
    assert out["week"] == 789
    assert out["a_f0_s"] == approx(-1234 * 2 ** -31)
    assert out["a_f1_s_per_s"] == approx(5 * 2 ** -43)
    assert out["URA"] == 8
    assert out["t_oc_s"] == 16000
    assert out["T_GD_s"] == approx(-9 * 2 ** -31)
    assert out["IODEC"] == 123
    assert out["SV_health_L5"] == 1
    assert out["C_uc_rad"] == approx(-7 * 2 ** -28)
    assert out["C_rc_m"] == approx(-3 * 2 ** -4)
    assert out["IDOT_semicircles_per_s"] == approx(-5 * 2 ** -43)


def test_navic_subframe2_round_trip():
    payload = _pack(
        _header(sf_id=2, towc=200, alert=1)
        + [
            (-100, 32),       # M_0
            (500, 16),        # t_oe -> 8000 s
            (12345, 32),      # e
            (54321, 32),      # sqrt_A
            (-7, 32),         # Omega_0
            (3, 32),          # omega
            (-1, 22),         # Omega_dot
            (200, 32),        # i_0
        ]
    )
    out = decode_navic_subframe2(payload)
    assert out["sf_id"] == 2
    assert out["TOWC"] == 200
    assert out["alert"] is True
    assert out["M_0_semicircles"] == approx(-100 * 2 ** -31)
    assert out["t_oe_s"] == 8000
    assert out["e"] == approx(12345 * 2 ** -33)
    assert out["sqrt_A_root_m"] == approx(54321 * 2 ** -19)
    assert out["Omega_0_semicircles"] == approx(-7 * 2 ** -31)
    assert out["Omega_dot_semicircles_per_s"] == approx(-1 * 2 ** -41)


def test_navic_subframe34_returns_message_id_and_raw():
    payload = _pack(
        _header(sf_id=3)
        + [
            (9, 6),          # Message Type ID
            (0xABCDEF, 24),  # payload-after-message-id
        ]
    )
    out = decode_navic_subframe34(payload)
    assert out["sf_id"] == 3
    assert out["message_id"] == 9
    assert isinstance(out["raw"], bytes)


def test_navic_dispatch_routes_by_sf_id():
    payload1 = _pack(_header(sf_id=1) + [(0, 250)])
    payload4 = _pack(_header(sf_id=4) + [(11, 6)])
    assert decode_navic_subframe(payload1)["sf_id"] == 1
    out4 = decode_navic_subframe(payload4)
    assert out4["sf_id"] == 4
    assert out4["message_id"] == 11
