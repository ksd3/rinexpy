"""Tests for the Septentrio SBF decoder."""

from __future__ import annotations

import io
import math
import struct

from rinexpy.sbf import SYNC, crc_ccitt, iter_blocks


def _build_block(block_id: int, body: bytes) -> bytes:
    """Wrap a body in a valid SBF block.

    ``length`` field includes the 8-byte header (sync + crc + id + len).
    ``id`` here is the full 16-bit field including a zero revision.
    """
    length = 8 + len(body)
    head = struct.pack("<HH", block_id, length)  # id, length (CRC prefix below)
    crc = crc_ccitt(head + body)
    return SYNC + struct.pack("<H", crc) + head + body


def test_iter_yields_unknown_blocks():
    body = struct.pack("<IH", 1234, 56) + b"\x00" * 4
    frame = _build_block(0x0123, body)
    msgs = list(iter_blocks(io.BytesIO(frame)))
    assert len(msgs) == 1
    assert msgs[0]["block_id"] == 0x0123
    assert msgs[0]["tow_ms"] == 1234
    assert msgs[0]["wnc"] == 56


def test_iter_skips_garbage_before_sync():
    body = struct.pack("<IH", 0, 0) + b"\x00" * 2
    frame = _build_block(0x0001, body)
    msgs = list(iter_blocks(io.BytesIO(b"junk" + frame)))
    assert len(msgs) == 1


def test_crc_rejects_corrupt_block():
    body = struct.pack("<IH", 0, 0) + b"\x00" * 4
    frame = bytearray(_build_block(0x0001, body))
    frame[-1] ^= 0xFF
    msgs = list(iter_blocks(io.BytesIO(bytes(frame)), check_crc=True))
    assert msgs == []


def test_decode_pvt_geodetic_round_trip():
    """Build a PVTGeodetic payload and verify decoded geodetic fields."""
    payload = bytearray(80)
    payload[0] = 4              # fix_mode (RTK fixed)
    payload[1] = 0              # error_code (OK)
    struct.pack_into("<d", payload, 2, math.radians(40.0))   # lat
    struct.pack_into("<d", payload, 10, math.radians(-3.0))  # lon
    struct.pack_into("<d", payload, 18, 100.0)               # height
    payload[60] = 9             # n_sv
    body = struct.pack("<IH", 100_000, 1982) + bytes(payload)
    frame = _build_block(4007, body)
    msg = next(iter(iter_blocks(io.BytesIO(frame))))
    assert msg["block_id"] == 4007
    assert msg["fix_mode"] == 4
    assert msg["lat_rad"] == math.radians(40.0)
    assert msg["lon_rad"] == math.radians(-3.0)
    assert msg["height_m"] == 100.0
    assert msg["n_sv"] == 9


def test_crc_ccitt_known_vector():
    """Spot-check CRC-CCITT against a small known input."""
    # CRC-CCITT (poly 0x1021, init 0) of "123456789" = 0x31C3
    assert crc_ccitt(b"123456789") == 0x31C3
