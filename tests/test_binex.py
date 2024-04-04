"""Tests for the BINEX decoder."""

from __future__ import annotations

import io

from rinexpy.binex import (
    SYNC,
    crc16_ccitt,
    encode_ubnxi,
    iter_records,
    read_ubnxi,
    xor_checksum,
)


def _build_record(record_id: int, body: bytes) -> bytes:
    """Build a forward-byte-order BINEX record with the right checksum."""
    rid_bytes = encode_ubnxi(record_id)
    len_bytes = encode_ubnxi(len(body))
    if len(body) <= 127:
        cs = xor_checksum(bytes([SYNC]) + rid_bytes + len_bytes + body)
        cs_bytes = bytes([cs])
    elif len(body) <= 4095:
        cs = crc16_ccitt(bytes([SYNC]) + rid_bytes + len_bytes + body)
        cs_bytes = bytes([(cs >> 8) & 0xFF, cs & 0xFF])
    else:
        cs_bytes = b"\x00\x00\x00\x00"  # placeholder for the 32-bit branch
    return bytes([SYNC]) + rid_bytes + len_bytes + body + cs_bytes


def test_ubnxi_round_trip():
    for v in (0, 1, 0x7F, 0x80, 0x3FFF, 0x4000, 0x1FFFFF):
        encoded = encode_ubnxi(v)
        decoded = read_ubnxi(io.BytesIO(encoded))
        assert decoded == v


def test_xor_checksum_known():
    assert xor_checksum(b"\x01\x02\x03") == 0x01 ^ 0x02 ^ 0x03


def test_iter_records_unknown_id():
    body = b"hello"
    frame = _build_record(0x42, body)
    msgs = list(iter_records(io.BytesIO(frame)))
    assert len(msgs) == 1
    assert msgs[0]["record_id"] == 0x42
    assert msgs[0]["body_bytes"] == body


def test_iter_records_skips_garbage_before_sync():
    body = b"abc"
    frame = _build_record(0x01, body)
    stream = io.BytesIO(b"junkbytes" + frame)
    msgs = list(iter_records(stream))
    assert len(msgs) == 1


def test_crc_rejects_corrupt_short_record():
    body = b"\x00\x01\x02"
    frame = bytearray(_build_record(0x01, body))
    frame[-1] ^= 0xFF
    msgs = list(iter_records(io.BytesIO(bytes(frame)), check_crc=True))
    assert msgs == []


def test_iter_records_long_body_uses_crc16():
    """Bodies of 128+ bytes use a 2-byte CRC-16/CCITT instead of XOR."""
    body = b"\x00" * 200
    frame = _build_record(0x10, body)
    msgs = list(iter_records(io.BytesIO(frame)))
    assert len(msgs) == 1
    assert msgs[0]["length"] == 200


def test_crc16_ccitt_known_vector():
    """Spot-check CRC-16/CCITT against a published reference."""
    assert crc16_ccitt(b"123456789") == 0x31C3
