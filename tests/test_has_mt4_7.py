"""Tests for HAS MT 4 (clock subset), MT 5 (code biases), MT 6 (phase
biases), and MT 7 (URA)."""

from __future__ import annotations

import pytest

from rinexpy.has import (
    decode_has_message,
)


def _set_bits(buf: bytearray, start_bit: int, n_bits: int, value: int, *, signed: bool = False) -> None:
    if signed and value < 0:
        value = (1 << n_bits) + value
    for i in range(n_bits):
        bit_val = (value >> (n_bits - 1 - i)) & 1
        byte_idx, bit_idx = divmod(start_bit + i, 8)
        while byte_idx >= len(buf):
            buf.append(0)
        buf[byte_idx] |= bit_val << (7 - bit_idx)


def _build_header(buf: bytearray, *, mt: int, mid: int = 0, page_count: int = 1,
                  page_id: int = 0, mask_id: int = 7, iod_set: int = 3,
                  status: int = 1) -> int:
    bit = 0
    _set_bits(buf, bit, 2, status); bit += 2
    _set_bits(buf, bit, 2, 0); bit += 2
    _set_bits(buf, bit, 4, mt); bit += 4
    _set_bits(buf, bit, 5, mid); bit += 5
    _set_bits(buf, bit, 5, page_count); bit += 5
    _set_bits(buf, bit, 5, page_id); bit += 5
    _set_bits(buf, bit, 5, mask_id); bit += 5
    _set_bits(buf, bit, 4, iod_set); bit += 4
    return bit


def _build_mask_message(*, gnss_id: int, prns: list[int], signals: list[int],
                        validity_idx: int = 4, cell_mask: list[list[int]] | None = None) -> bytes:
    body = bytearray()
    bit = _build_header(body, mt=1)
    _set_bits(body, bit, 4, validity_idx); bit += 4
    gnss_mask = 1 << (3 - gnss_id)
    _set_bits(body, bit, 4, gnss_mask); bit += 4
    sat_mask = 0
    for prn in prns:
        sat_mask |= 1 << (40 - prn)
    _set_bits(body, bit, 40, sat_mask); bit += 40
    sig_mask = 0
    for s in signals:
        sig_mask |= 1 << (15 - s)
    _set_bits(body, bit, 16, sig_mask); bit += 16
    if cell_mask is None:
        _set_bits(body, bit, 1, 0); bit += 1
    else:
        _set_bits(body, bit, 1, 1); bit += 1
        for row in cell_mask:
            for v in row:
                _set_bits(body, bit, 1, v); bit += 1
    _set_bits(body, bit, 3, 0); bit += 3
    return bytes(body)


def test_mt4_clock_subset_with_multiplier():
    """Two Galileo SVs, multiplier 4 -> LSB is 0.01 m."""
    body = bytearray()
    bit = _build_header(body, mt=4)
    _set_bits(body, bit, 4, 3); bit += 4         # validity index
    # GNSSSubsetMask: only Galileo (gnss_id=2 -> bit position 1 in 4-bit mask).
    _set_bits(body, bit, 4, 0b0010); bit += 4
    # Multiplier index 2 -> multiplier value 4.
    _set_bits(body, bit, 2, 2); bit += 2
    # SatelliteSubsetMask: SV 5 and SV 17.
    sat_mask = (1 << (40 - 5)) | (1 << (40 - 17))
    _set_bits(body, bit, 40, sat_mask); bit += 40
    # c0 raw values
    _set_bits(body, bit, 13, 25, signed=True); bit += 13     # 25 * 0.0025 * 4 = 0.25 m
    _set_bits(body, bit, 13, -50, signed=True); bit += 13    # -50 * 0.0025 * 4 = -0.50 m

    msg = decode_has_message(bytes(body))
    assert msg["header"]["message_type"] == 4
    payload = msg["payload"]
    assert len(payload["satellites"]) == 2
    s5, s17 = payload["satellites"]
    assert s5["prn"] == 5
    assert s5["multiplier"] == 4
    assert s5["delta_clock_c0_m"] == pytest.approx(0.25)
    assert s17["prn"] == 17
    assert s17["delta_clock_c0_m"] == pytest.approx(-0.50)


def test_mt5_code_bias_three_cells_no_cellmask():
    """Cell mask absent -> every (sat, signal) combination is a cell."""
    mask_bytes = _build_mask_message(gnss_id=2, prns=[5, 12], signals=[0, 1])
    mask = decode_has_message(mask_bytes)["payload"]

    body = bytearray()
    bit = _build_header(body, mt=5)
    _set_bits(body, bit, 4, 5); bit += 4    # validity index
    # 4 cells = 2 sats * 2 signals. Raw values in 0.01 m / LSB.
    for raw in (50, -25, 100, -10):
        _set_bits(body, bit, 11, raw, signed=True); bit += 11

    msg = decode_has_message(bytes(body), mask=mask)
    assert msg["header"]["message_type"] == 5
    biases = msg["payload"]["biases"]
    assert len(biases) == 4
    # Iteration order: sat 5 sig 0, sat 5 sig 1, sat 12 sig 0, sat 12 sig 1
    assert (biases[0]["prn"], biases[0]["signal_index"]) == (5, 0)
    assert biases[0]["code_bias_m"] == pytest.approx(0.50)
    assert biases[3]["code_bias_m"] == pytest.approx(-0.10)


def test_mt5_code_bias_with_cellmask():
    """Cell mask gates which (sat, signal) cells carry data."""
    # 2 SVs x 2 signals; enable (sat0,sig0), (sat0,sig1), disable (sat1,sig0),
    # enable (sat1,sig1) -> 3 active cells.
    cm = [[1, 1], [0, 1]]
    mask_bytes = _build_mask_message(
        gnss_id=2, prns=[5, 12], signals=[0, 1], cell_mask=cm,
    )
    mask = decode_has_message(mask_bytes)["payload"]

    body = bytearray()
    bit = _build_header(body, mt=5)
    _set_bits(body, bit, 4, 5); bit += 4
    for raw in (10, 20, 30):
        _set_bits(body, bit, 11, raw, signed=True); bit += 11

    msg = decode_has_message(bytes(body), mask=mask)
    biases = msg["payload"]["biases"]
    assert len(biases) == 3
    assert [(b["prn"], b["signal_index"]) for b in biases] == [
        (5, 0), (5, 1), (12, 1),
    ]


def test_mt6_phase_bias_with_discontinuity():
    mask_bytes = _build_mask_message(gnss_id=2, prns=[5], signals=[0])
    mask = decode_has_message(mask_bytes)["payload"]

    body = bytearray()
    bit = _build_header(body, mt=6)
    _set_bits(body, bit, 4, 5); bit += 4
    # One cell only.
    _set_bits(body, bit, 11, 75, signed=True); bit += 11    # bias 75 * 0.01 = 0.75 cyc
    _set_bits(body, bit, 2, 3); bit += 2                    # discontinuity = 3

    msg = decode_has_message(bytes(body), mask=mask)
    assert msg["header"]["message_type"] == 6
    biases = msg["payload"]["biases"]
    assert biases[0]["phase_bias_cycles"] == pytest.approx(0.75)
    assert biases[0]["discontinuity_indicator"] == 3


def test_mt7_ura():
    mask_bytes = _build_mask_message(gnss_id=2, prns=[5, 12, 27], signals=[0])
    mask = decode_has_message(mask_bytes)["payload"]

    body = bytearray()
    bit = _build_header(body, mt=7)
    _set_bits(body, bit, 4, 5); bit += 4
    for u in (1, 4, 15):
        _set_bits(body, bit, 4, u); bit += 4

    msg = decode_has_message(bytes(body), mask=mask)
    assert msg["header"]["message_type"] == 7
    sats = msg["payload"]["satellites"]
    assert [s["ura_index"] for s in sats] == [1, 4, 15]
    assert [s["prn"] for s in sats] == [5, 12, 27]


def test_mt5_without_mask_raises():
    body = bytearray()
    _build_header(body, mt=5)
    with pytest.raises(ValueError, match="MT-1"):
        decode_has_message(bytes(body))


def test_mt4_works_without_master_mask():
    """MT 4 carries its own subset mask, so it can be decoded even
    without a prior MT-1 decoded."""
    body = bytearray()
    bit = _build_header(body, mt=4)
    _set_bits(body, bit, 4, 5); bit += 4
    _set_bits(body, bit, 4, 0b0010); bit += 4      # only Galileo
    _set_bits(body, bit, 2, 0); bit += 2           # multiplier index 0 -> 1
    sat_mask = 1 << (40 - 9)
    _set_bits(body, bit, 40, sat_mask); bit += 40
    _set_bits(body, bit, 13, 400, signed=True); bit += 13

    msg = decode_has_message(bytes(body))   # no mask passed
    s = msg["payload"]["satellites"]
    assert len(s) == 1
    assert s[0]["prn"] == 9
    assert s[0]["delta_clock_c0_m"] == pytest.approx(400 * 0.0025)
