"""Tests for the RTCM3 framing + message decoder."""

from __future__ import annotations

import io
import struct

from rinexpy.rtcm3 import PREAMBLE, crc24q, decode_message, iter_messages


def _build_frame(msg_id: int, body_bits: list[tuple[int, int]]) -> bytes:
    """Build a valid RTCM3 frame around a payload.

    ``body_bits`` is a list of ``(value, n_bits)`` tuples to pack at the
    start of the payload. Message ID is prepended automatically.
    """
    bits: list[tuple[int, int]] = [(msg_id, 12)] + body_bits
    total_bits = sum(n for _, n in bits)
    pad = (-total_bits) % 8
    bits.append((0, pad))

    body_int = 0
    for v, n in bits:
        body_int = (body_int << n) | (v & ((1 << n) - 1))
    body_len = (total_bits + pad) // 8
    body = body_int.to_bytes(body_len, "big")
    head = bytes([PREAMBLE, body_len >> 8, body_len & 0xFF])
    crc = crc24q(head + body)
    return head + body + struct.pack(">I", crc)[1:]


def test_iter_yields_unknown_messages():
    frame = _build_frame(3000, [(0, 8)])
    msgs = list(iter_messages(io.BytesIO(frame)))
    assert len(msgs) == 1
    assert msgs[0]["msg_id"] == 3000


def test_iter_skips_garbage_before_preamble():
    frame = _build_frame(3000, [(0, 8)])
    msgs = list(iter_messages(io.BytesIO(b"garbage" + frame)))
    assert len(msgs) == 1


def test_crc_validation_rejects_corrupt_frame():
    frame = bytearray(_build_frame(3000, [(0, 8)]))
    frame[-1] ^= 0xFF
    msgs = list(iter_messages(io.BytesIO(bytes(frame)), check_crc=True))
    assert msgs == []


def test_decode_1005_position():
    # Build a 1005 with x=y=z=0 at station id 12.
    bits = [
        (12, 12),  # station_id
        (0, 6),  # ITRF year
        (0, 4),  # indicators
        (0, 38),  # x
        (0, 1),  # single rx
        (0, 1),  # reserved
        (0, 38),  # y
        (0, 2),  # quarter cycle
        (0, 38),  # z
    ]
    frame = _build_frame(1005, bits)
    msgs = list(iter_messages(io.BytesIO(frame)))
    assert msgs[0]["msg_id"] == 1005
    assert msgs[0]["station_id"] == 12
    assert msgs[0]["position"] == (0.0, 0.0, 0.0)


def test_decode_message_dispatch_unknown():
    # decode_message itself doesn't constrain msg_id range — only the wire
    # format does (12 bits = 0..4095).
    out = decode_message(7777, b"\x00" * 4)
    assert out["msg_id"] == 7777
    assert "payload_bytes" in out
