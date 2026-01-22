"""Synthetic round-trip tests for the GPS CNAV (L2C / L5) decoders.

There is no public RXM-SFRBX CNAV capture in tests/data, so these
tests build a CNAV message bit-pattern from known field values, feed
it through the decoder, and assert each field comes back with the
expected scaling. Cross-checks against IS-GPS-200 §30.3 field tables.
"""

from __future__ import annotations

from pytest import approx

from rinexpy.gps_cnav import (
    PREAMBLE,
    decode_cnav_message,
    decode_cnav_mt10,
    decode_cnav_mt11,
)


def _pack(field_specs: list[tuple[int, int]]) -> bytes:
    """Pack [(value, n_bits), ...] MSB-first into bytes (zero-padded
    to the next byte boundary)."""
    bits = "".join(
        f"{value & ((1 << n) - 1):0{n}b}" for value, n in field_specs if n > 0
    )
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def _header(prn: int, msg_id: int, tow_6s: int = 100, alert: int = 0):
    return [
        (PREAMBLE, 8),
        (prn, 6),
        (msg_id, 6),
        (tow_6s, 17),
        (alert, 1),
    ]


def test_cnav_mt10_round_trip():
    payload = _pack(
        _header(prn=5, msg_id=10, tow_6s=200)
        + [
            (2096, 13),     # week
            (0, 1), (0, 1), (0, 1),  # L1/L2/L5 health
            (3, 5),         # URA index (signed)
            (4, 11),        # t_op = 4 * 300 s
            (5, 11),        # t_oe = 5 * 300 s
            (-1234, 26),    # delta A (signed)
            (7, 25),        # Adot
            (-9, 17),       # delta_n0
            (3, 23),        # delta_n0_dot
            (-555, 33),     # M_0
            (12345, 33),    # e (unsigned)
            (777, 33),      # omega
        ]
    )
    out = decode_cnav_mt10(payload)
    assert out["prn"] == 5
    assert out["msg_id"] == 10
    assert out["tow_count_s"] == 1200
    assert out["week"] == 2096
    assert out["URA_index"] == 3
    assert out["t_op_s"] == 1200
    assert out["t_oe_s"] == 1500
    assert out["delta_A_m"] == approx(-1234 * 2 ** -9)
    assert out["Adot_m_per_s"] == approx(7 * 2 ** -21)
    assert out["delta_n0_semicircles_per_s"] == approx(-9 * 2 ** -44)
    assert out["M_0_n_semicircles"] == approx(-555 * 2 ** -32)
    assert out["e_n"] == approx(12345 * 2 ** -34)


def test_cnav_mt11_round_trip():
    payload = _pack(
        _header(prn=12, msg_id=11)
        + [
            (0, 11),         # t_oe (skipped by decoder)
            (-1000, 33),     # Omega_0
            (200000, 33),    # i_0
            (-7, 17),        # delta_Omega_dot
            (3, 15),         # IDOT
            (-100, 16),      # Cis
            (50, 16),        # Cic
            (-1234, 24),     # Crs
            (5678, 24),      # Crc
            (-77, 21),       # Cus
            (88, 21),        # Cuc
        ]
    )
    out = decode_cnav_mt11(payload)
    assert out["msg_id"] == 11
    assert out["Omega_0_n_semicircles"] == approx(-1000 * 2 ** -32)
    assert out["i_0_n_semicircles"] == approx(200000 * 2 ** -32)
    assert out["delta_Omega_dot_semicircles_per_s"] == approx(-7 * 2 ** -44)
    assert out["IDOT_semicircles_per_s"] == approx(3 * 2 ** -44)
    assert out["C_is_rad"] == approx(-100 * 2 ** -30)
    assert out["C_rs_m"] == approx(-1234 * 2 ** -8)
    assert out["C_us_rad"] == approx(-77 * 2 ** -30)


def test_cnav_dispatch_unknown_msg_id_returns_raw():
    payload = _pack(_header(prn=1, msg_id=37) + [(0, 200)])
    out = decode_cnav_message(payload)
    assert "header" in out
    assert out["header"]["msg_id"] == 37
    assert isinstance(out["raw"], bytes)
