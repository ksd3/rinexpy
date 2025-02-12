"""Tests for the RTCM3 SSR (State Space Representation) decoders.

Messages 1057 (GPS orbit corrections) and 1058 (GPS clock corrections)
are encoded synthetically and round-tripped through ``decode_message``
to verify both the header parse and the per-satellite bit-packing.
"""

from __future__ import annotations

import pytest

from rinexpy.rtcm3 import decode_message


def _set_bits(buf: bytearray, start_bit: int, n_bits: int, value: int, *, signed: bool = False) -> None:
    """Write ``value`` into ``buf`` starting at bit ``start_bit``."""
    if signed and value < 0:
        value = (1 << n_bits) + value
    for i in range(n_bits):
        bit_val = (value >> (n_bits - 1 - i)) & 1
        byte_idx, bit_idx = divmod(start_bit + i, 8)
        while byte_idx >= len(buf):
            buf.append(0)
        buf[byte_idx] |= bit_val << (7 - bit_idx)


def _encode_ssr_header(buf: bytearray, *, msg_id: int, epoch_s: int, iod: int,
                       provider: int, solution: int, n_sats: int,
                       has_datum: bool, ref_datum: int = 0) -> int:
    """Write the common SSR header. Returns the bit cursor after the header."""
    bit = 0
    _set_bits(buf, bit, 12, msg_id); bit += 12
    _set_bits(buf, bit, 20, epoch_s); bit += 20
    _set_bits(buf, bit, 4, 0); bit += 4   # update interval index
    _set_bits(buf, bit, 1, 0); bit += 1   # multiple message indicator
    _set_bits(buf, bit, 4, iod); bit += 4
    _set_bits(buf, bit, 16, provider); bit += 16
    _set_bits(buf, bit, 4, solution); bit += 4
    if has_datum:
        _set_bits(buf, bit, 1, ref_datum); bit += 1
    _set_bits(buf, bit, 6, n_sats); bit += 6
    return bit


def test_decode_1057_single_satellite_round_trip():
    """Build a 1057 with one GPS satellite, decode, verify all fields."""
    body = bytearray()
    bit = _encode_ssr_header(
        body, msg_id=1057, epoch_s=12345, iod=3, provider=99,
        solution=1, n_sats=1, has_datum=True, ref_datum=0,
    )
    # PRN 17, IODE 42; orbit corrections in raw LSB units.
    _set_bits(body, bit, 6, 17); bit += 6      # PRN
    _set_bits(body, bit, 8, 42); bit += 8      # IODE
    _set_bits(body, bit, 22,  1234, signed=True); bit += 22   # d_radial   (0.1 mm)
    _set_bits(body, bit, 20,  5678, signed=True); bit += 20   # d_along    (0.4 mm)
    _set_bits(body, bit, 20, -9012, signed=True); bit += 20   # d_cross    (0.4 mm)
    _set_bits(body, bit, 21,  -100, signed=True); bit += 21   # dot_radial (1 um/s)
    _set_bits(body, bit, 19,   200, signed=True); bit += 19   # dot_along  (4 um/s)
    _set_bits(body, bit, 19,    -3, signed=True); bit += 19   # dot_cross  (4 um/s)

    msg = decode_message(1057, bytes(body))

    assert msg["msg_id"] == 1057
    assert msg["header"]["n_sats"] == 1
    assert msg["header"]["iod_ssr"] == 3
    assert msg["header"]["provider_id"] == 99
    assert msg["header"]["solution_id"] == 1
    assert msg["header"]["epoch_time_s"] == 12345
    sat = msg["satellites"][0]
    assert sat["prn"] == 17
    assert sat["iode"] == 42
    assert sat["delta_radial_m"] == pytest.approx(1234 * 1e-4)
    assert sat["delta_along_track_m"] == pytest.approx(5678 * 4e-4)
    assert sat["delta_cross_track_m"] == pytest.approx(-9012 * 4e-4)
    assert sat["dot_delta_radial_m_per_s"] == pytest.approx(-100 * 1e-6)
    assert sat["dot_delta_along_track_m_per_s"] == pytest.approx(200 * 4e-6)
    assert sat["dot_delta_cross_track_m_per_s"] == pytest.approx(-3 * 4e-6)


def test_decode_1057_three_satellites():
    body = bytearray()
    bit = _encode_ssr_header(
        body, msg_id=1057, epoch_s=0, iod=0, provider=0,
        solution=0, n_sats=3, has_datum=True, ref_datum=1,
    )
    for prn, iode in ((1, 11), (5, 50), (32, 200)):
        _set_bits(body, bit, 6, prn); bit += 6
        _set_bits(body, bit, 8, iode); bit += 8
        for nbits in (22, 20, 20, 21, 19, 19):
            _set_bits(body, bit, nbits, 0, signed=True); bit += nbits
    msg = decode_message(1057, bytes(body))
    assert [s["prn"] for s in msg["satellites"]] == [1, 5, 32]
    assert [s["iode"] for s in msg["satellites"]] == [11, 50, 200]
    assert msg["header"]["ref_datum"] == 1


def test_decode_1058_round_trip():
    body = bytearray()
    bit = _encode_ssr_header(
        body, msg_id=1058, epoch_s=999, iod=2, provider=42,
        solution=3, n_sats=2, has_datum=False,
    )
    # First satellite: PRN 7, c0/c1/c2 in raw LSB.
    _set_bits(body, bit, 6, 7); bit += 6
    _set_bits(body, bit, 22, -1500, signed=True); bit += 22   # c0 (0.1 mm)
    _set_bits(body, bit, 21,    50, signed=True); bit += 21   # c1 (1 um/s)
    _set_bits(body, bit, 27,   -10, signed=True); bit += 27   # c2 (20 nm/s^2)
    # Second satellite: PRN 22.
    _set_bits(body, bit, 6, 22); bit += 6
    _set_bits(body, bit, 22,  2222, signed=True); bit += 22
    _set_bits(body, bit, 21,     0, signed=True); bit += 21
    _set_bits(body, bit, 27,     0, signed=True); bit += 27

    msg = decode_message(1058, bytes(body))

    assert msg["msg_id"] == 1058
    assert msg["header"]["ref_datum"] is None   # not present for clock messages
    assert msg["header"]["n_sats"] == 2
    s0, s1 = msg["satellites"]
    assert s0["prn"] == 7
    assert s0["c0_m"] == pytest.approx(-1500 * 1e-4)
    assert s0["c1_m_per_s"] == pytest.approx(50 * 1e-6)
    assert s0["c2_m_per_s2"] == pytest.approx(-10 * 2e-8)
    assert s1["prn"] == 22
    assert s1["c0_m"] == pytest.approx(2222 * 1e-4)
