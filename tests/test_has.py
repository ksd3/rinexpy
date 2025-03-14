"""Tests for the Galileo HAS message decoder (HAS SDD v1.0)."""

from __future__ import annotations

import pytest

from rinexpy.has import (
    HAS_VALIDITY_S,
    decode_has_clock_full,
    decode_has_header,
    decode_has_mask,
    decode_has_message,
    decode_has_orbit,
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
    """Write the 32-bit HAS header. Returns the bit cursor afterward."""
    bit = 0
    _set_bits(buf, bit, 2, status); bit += 2
    _set_bits(buf, bit, 2, 0); bit += 2          # reserved
    _set_bits(buf, bit, 4, mt); bit += 4
    _set_bits(buf, bit, 5, mid); bit += 5
    _set_bits(buf, bit, 5, page_count); bit += 5
    _set_bits(buf, bit, 5, page_id); bit += 5
    _set_bits(buf, bit, 5, mask_id); bit += 5
    _set_bits(buf, bit, 4, iod_set); bit += 4
    return bit


def test_header_round_trip():
    body = bytearray()
    _build_header(body, mt=1, mid=12, page_count=4, page_id=2, mask_id=21,
                  iod_set=5, status=1)
    h, bit = decode_has_header(bytes(body))
    assert h == {
        "status": 1,
        "message_type": 1,
        "message_id": 12,
        "page_count": 4,
        "page_id": 2,
        "mask_id": 21,
        "iod_set": 5,
    }
    assert bit == 32


def _build_mask_message(*, gnss_id: int, prns: list[int], signals: list[int],
                        validity_idx: int = 4) -> bytes:
    """Build an MT-1 HAS Mask body for a single GNSS, no cell mask."""
    body = bytearray()
    bit = _build_header(body, mt=1)
    _set_bits(body, bit, 4, validity_idx); bit += 4
    # GNSSMask: one bit per system, MSB first (bits 3..0 = systems 0..3).
    gnss_mask = 1 << (3 - gnss_id)
    _set_bits(body, bit, 4, gnss_mask); bit += 4
    # SatelliteMask (40 bits, MSB = PRN 1).
    sat_mask = 0
    for prn in prns:
        sat_mask |= 1 << (40 - prn)
    _set_bits(body, bit, 40, sat_mask); bit += 40
    # SignalMask (16 bits).
    sig_mask = 0
    for s in signals:
        sig_mask |= 1 << (15 - s)
    _set_bits(body, bit, 16, sig_mask); bit += 16
    # CellMaskFlag = 0 (no per-cell mask).
    _set_bits(body, bit, 1, 0); bit += 1
    # NavMessage = 0 (default broadcast nav).
    _set_bits(body, bit, 3, 0); bit += 3
    return bytes(body)


def test_decode_has_mask_single_gnss():
    body = _build_mask_message(gnss_id=2, prns=[5, 12, 27], signals=[0, 3], validity_idx=4)
    msg = decode_has_message(body)
    assert msg["header"]["message_type"] == 1
    payload = msg["payload"]
    assert payload["validity_interval_s"] == HAS_VALIDITY_S[4]
    assert len(payload["gnss_entries"]) == 1
    e = payload["gnss_entries"][0]
    assert e["gnss_id"] == 2
    assert e["gnss_name"] == "Galileo"
    assert e["satellites"] == [5, 12, 27]
    assert e["signals"] == [0, 3]
    assert e["cell_mask"] is None


def test_decode_has_orbit_against_known_mask():
    """Build mask then orbit; cross-check values for two Galileo SVs."""
    mask_bytes = _build_mask_message(gnss_id=2, prns=[5, 12], signals=[0], validity_idx=5)
    mask_msg = decode_has_message(mask_bytes)
    mask = mask_msg["payload"]

    # Now build an MT-2 message for the same two SVs.
    body = bytearray()
    bit = _build_header(body, mt=2, mask_id=7, iod_set=3)
    _set_bits(body, bit, 4, 5); bit += 4   # validity index
    # SV 5: IOD 17, deltas
    _set_bits(body, bit, 8, 17); bit += 8
    _set_bits(body, bit, 13, 400, signed=True); bit += 13     # radial   0.0025 m -> 1.0 m
    _set_bits(body, bit, 12, -100, signed=True); bit += 12    # in-track 0.0080 m -> -0.8 m
    _set_bits(body, bit, 12, 50, signed=True); bit += 12      # cross    0.0080 m -> 0.4 m
    # SV 12: IOD 99
    _set_bits(body, bit, 8, 99); bit += 8
    _set_bits(body, bit, 13, -200, signed=True); bit += 13
    _set_bits(body, bit, 12, 250, signed=True); bit += 12
    _set_bits(body, bit, 12, -125, signed=True); bit += 12

    msg = decode_has_message(bytes(body), mask=mask)
    assert msg["header"]["message_type"] == 2
    sats = msg["payload"]["satellites"]
    assert len(sats) == 2
    assert sats[0]["prn"] == 5
    assert sats[0]["iod"] == 17
    assert sats[0]["delta_radial_m"] == pytest.approx(1.0)
    assert sats[0]["delta_along_track_m"] == pytest.approx(-0.8)
    assert sats[0]["delta_cross_track_m"] == pytest.approx(0.4)
    assert sats[1]["prn"] == 12
    assert sats[1]["iod"] == 99
    assert sats[1]["delta_radial_m"] == pytest.approx(-0.5)


def test_decode_has_clock_with_multiplier():
    """Two Galileo SVs, clock multiplier set to 2 -> LSB doubles."""
    mask_bytes = _build_mask_message(gnss_id=2, prns=[5, 12], signals=[0], validity_idx=3)
    mask = decode_has_message(mask_bytes)["payload"]

    body = bytearray()
    bit = _build_header(body, mt=3, mask_id=7)
    _set_bits(body, bit, 4, 3); bit += 4   # validity index
    # Multiplier index 1 -> multiplier value 2.
    _set_bits(body, bit, 2, 1); bit += 2
    # SV 5: c0 = 100 raw LSB -> 100 * 0.0025 * 2 = 0.5 m
    _set_bits(body, bit, 13, 100, signed=True); bit += 13
    # SV 12: c0 = -50 raw -> -50 * 0.0025 * 2 = -0.25 m
    _set_bits(body, bit, 13, -50, signed=True); bit += 13

    msg = decode_has_message(bytes(body), mask=mask)
    assert msg["header"]["message_type"] == 3
    payload = msg["payload"]
    assert payload["gnss_multipliers"][2] == 2
    sats = payload["satellites"]
    assert sats[0]["prn"] == 5
    assert sats[0]["delta_clock_c0_m"] == pytest.approx(0.5)
    assert sats[1]["prn"] == 12
    assert sats[1]["delta_clock_c0_m"] == pytest.approx(-0.25)


def test_orbit_without_mask_raises():
    body = bytearray()
    _build_header(body, mt=2)
    with pytest.raises(ValueError, match="MT-1"):
        decode_has_message(bytes(body))


def test_unsupported_mt_returns_marker():
    """MT 0 is not defined in the SDD; the dispatcher should return a
    marker dict rather than raise."""
    body = bytearray()
    _build_header(body, mt=0)
    msg = decode_has_message(bytes(body))
    assert msg["payload"]["unsupported_message_type"] == 0


def test_mask_with_cell_mask_flag():
    """Set CellMaskFlag=1 and provide the per-cell bits."""
    body = bytearray()
    bit = _build_header(body, mt=1)
    _set_bits(body, bit, 4, 5); bit += 4   # validity index 5 -> 60 s
    _set_bits(body, bit, 4, 0b0010); bit += 4   # only GNSS 2 (Galileo)
    sat_mask = (1 << (40 - 5)) | (1 << (40 - 12))
    _set_bits(body, bit, 40, sat_mask); bit += 40
    sig_mask = (1 << 15) | (1 << 13)   # signals 0 and 2
    _set_bits(body, bit, 16, sig_mask); bit += 16
    _set_bits(body, bit, 1, 1); bit += 1   # CellMaskFlag = 1
    # 2 sats * 2 signals = 4 bits: enable (sat0,sig0), (sat0,sig1),
    # disable (sat1,sig0), enable (sat1,sig1).
    for v in (1, 1, 0, 1):
        _set_bits(body, bit, 1, v); bit += 1
    _set_bits(body, bit, 3, 0); bit += 3   # nav message

    msg = decode_has_message(bytes(body))
    e = msg["payload"]["gnss_entries"][0]
    assert e["cell_mask"] == [[1, 1], [0, 1]]
