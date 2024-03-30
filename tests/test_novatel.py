"""Tests for the NovAtel OEM binary decoder."""

from __future__ import annotations

import io
import struct

from rinexpy.novatel import SYNC, crc32, iter_messages


def _build_msg(
    msg_id: int,
    body: bytes,
    *,
    msg_type: int = 0,
    week: int = 1982,
    gps_ms: int = 100_000,
) -> bytes:
    """Wrap a body in a valid NovAtel OEM message with correct CRC32."""
    header_length = 28
    header = bytearray(header_length)
    header[0:3] = SYNC
    header[3] = header_length
    struct.pack_into("<H", header, 4, msg_id)
    header[6] = msg_type
    header[7] = 0
    struct.pack_into("<H", header, 8, len(body))
    struct.pack_into("<H", header, 10, 0)         # sequence
    struct.pack_into("<H", header, 14, week)
    struct.pack_into("<I", header, 16, gps_ms)
    full = bytes(header) + body
    cs = crc32(full)
    return full + struct.pack("<I", cs)


def test_iter_yields_unknown_messages():
    body = b"hello world!" + b"\x00" * 4  # 16 bytes
    frame = _build_msg(0x9999, body)
    msgs = list(iter_messages(io.BytesIO(frame)))
    assert len(msgs) == 1
    assert msgs[0]["msg_id"] == 0x9999
    assert msgs[0]["body_bytes"] == body


def test_iter_skips_garbage_before_sync():
    body = b"\x00" * 4
    frame = _build_msg(0x1234, body)
    msgs = list(iter_messages(io.BytesIO(b"junk_bytes_here" + frame)))
    assert len(msgs) == 1


def test_crc_rejects_corrupt_message():
    body = b"\x00" * 4
    frame = bytearray(_build_msg(0x1234, body))
    frame[-1] ^= 0xFF
    msgs = list(iter_messages(io.BytesIO(bytes(frame)), check_crc=True))
    assert msgs == []


def test_decode_bestpos_round_trip():
    """Build a BESTPOS body and verify decoded geodetic fields."""
    body = bytearray(72)
    struct.pack_into("<I", body, 0, 0)                 # sol_status: SOL_COMPUTED
    struct.pack_into("<I", body, 4, 16)                # pos_type: SINGLE
    struct.pack_into("<d", body, 8, 40.0)              # lat
    struct.pack_into("<d", body, 16, -3.0)             # lon
    struct.pack_into("<d", body, 24, 100.0)            # height
    body[64] = 12                                      # n_obs
    body[65] = 11                                      # n_sv_used
    body[66] = 11                                      # n_sv_above_mask
    frame = _build_msg(42, bytes(body))
    msg = next(iter(iter_messages(io.BytesIO(frame))))
    assert msg["msg_id"] == 42
    assert msg["lat_deg"] == 40.0
    assert msg["lon_deg"] == -3.0
    assert msg["height_m"] == 100.0
    assert msg["n_sv_used"] == 11


def test_crc32_matches_zlib():
    """The NovAtel CRC is just the standard zlib CRC32."""
    import zlib as _zlib
    assert crc32(b"hello") == _zlib.crc32(b"hello")
