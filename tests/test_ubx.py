"""Tests for the u-blox UBX decoder."""

from __future__ import annotations

import io
import struct

from rinexpy.ubx import SYNC1, SYNC2, decode_message, fletcher_checksum, iter_messages


def _build_frame(msg_class: int, msg_id: int, payload: bytes) -> bytes:
    """Wrap a payload in a valid UBX frame with correct Fletcher checksum."""
    head = bytes([msg_class, msg_id]) + struct.pack("<H", len(payload))
    ck_a, ck_b = fletcher_checksum(head + payload)
    return bytes([SYNC1, SYNC2]) + head + payload + bytes([ck_a, ck_b])


def test_iter_yields_unknown_messages():
    frame = _build_frame(0x0A, 0x04, b"\x00" * 8)  # MON-VER (header only)
    msgs = list(iter_messages(io.BytesIO(frame)))
    assert len(msgs) == 1
    assert msgs[0]["msg_class"] == 0x0A
    assert msgs[0]["msg_id"] == 0x04


def test_iter_skips_garbage_before_sync():
    frame = _build_frame(0x0A, 0x04, b"\x00" * 4)
    stream = io.BytesIO(b"junk_bytes_here" + frame)
    msgs = list(iter_messages(stream))
    assert len(msgs) == 1


def test_crc_rejects_corrupt_frame():
    frame = bytearray(_build_frame(0x0A, 0x04, b"\x00" * 4))
    frame[-1] ^= 0xFF
    msgs = list(iter_messages(io.BytesIO(bytes(frame)), check_crc=True))
    assert msgs == []


def test_decode_nav_pvt_round_trip():
    """Build a 92-byte NAV-PVT payload and verify decoded fields.

    We construct the payload as a ``bytearray`` and only set the fields
    we assert against — everything else stays zero.
    """
    payload = bytearray(92)
    struct.pack_into("<I", payload, 0, 100)         # itow
    struct.pack_into("<H", payload, 4, 2024)        # year
    payload[6] = 3                                  # month
    payload[7] = 14                                 # day
    payload[8] = 1                                  # hour
    payload[9] = 2                                  # minute
    payload[10] = 3                                 # second
    payload[20] = 3                                 # fix_type (3D)
    payload[23] = 7                                 # n_sat
    struct.pack_into("<i", payload, 24, -30_000_000)   # lon * 1e-7 -> -3 deg
    struct.pack_into("<i", payload, 28, 400_000_000)   # lat * 1e-7 -> 40 deg
    struct.pack_into("<i", payload, 32, 100_000)       # height (mm)
    struct.pack_into("<i", payload, 36, 100_000)       # h_msl
    struct.pack_into("<I", payload, 40, 500)           # h_acc
    struct.pack_into("<I", payload, 44, 1000)          # v_acc
    struct.pack_into("<H", payload, 76, 150)           # pdop * 0.01 -> 1.50

    frame = _build_frame(0x01, 0x07, bytes(payload))
    msg = next(iter(iter_messages(io.BytesIO(frame))))
    assert msg["fix_type"] == 3
    assert msg["lat_deg"] == 40.0
    assert msg["lon_deg"] == -3.0
    assert msg["n_sat"] == 7


def test_decode_nav_sat():
    """One-satellite NAV-SAT decodes the per-SV block."""
    header = struct.pack("<IBBBB", 1000, 1, 1, 0, 0)
    sat = struct.pack("<BBBbhhI", 0, 5, 42, 30, 180, 0, 0)
    payload = header + sat
    frame = _build_frame(0x01, 0x35, payload)
    msg = next(iter(iter_messages(io.BytesIO(frame))))
    assert msg["n_sat"] == 1
    assert msg["satellites"][0]["cno_dbhz"] == 42
    assert msg["satellites"][0]["elevation_deg"] == 30
    assert msg["satellites"][0]["azimuth_deg"] == 180


def test_decode_message_dispatch_unknown():
    out = decode_message(0xAA, 0xBB, b"hello")
    assert out["msg_class"] == 0xAA
    assert out["msg_id"] == 0xBB
    assert out["payload_bytes"] == b"hello"


def test_fletcher_checksum_known():
    # Spot-check Fletcher against a known input.
    a, b = fletcher_checksum(b"\x01\x02\x03")
    assert (a, b) == ((1 + 2 + 3) % 256, (1 + (1 + 2) + (1 + 2 + 3)) % 256)
