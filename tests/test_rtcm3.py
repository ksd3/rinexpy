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


def test_decode_msm7_header_only():
    """MSM7 header masks decode correctly even with no cell payload.

    When the body is too short for the cell/sat/obs blocks we just set
    ``payload_truncated=True`` and return what we've parsed so far.
    """
    bits = [
        (123, 12),
        (456_000, 30),
        (0, 1),
        (0, 3),
        (0, 7),
        (0, 2),
        (0, 2),
        (0, 1),
        (0, 3),
        (0xFFFFFFFF, 32),
        (0x00000000, 32),
        (0b11, 32),
    ]
    frame = _build_frame(1077, bits)
    msgs = list(iter_messages(io.BytesIO(frame)))
    assert msgs[0]["msg_id"] == 1077
    assert msgs[0]["station_id"] == 123
    assert msgs[0]["n_sv"] == 32
    assert msgs[0]["n_sig"] == 2


def test_decode_msm4_full_with_one_cell():
    """A 1-SV / 1-signal / 1-cell MSM4 (1074) decodes per-cell observation."""
    sv_mask = 1 << 63  # SV index 0 -> G01
    sig_mask = 1 << 31
    bits = [
        (123, 12),
        (456_000, 30),
        (0, 1),
        (0, 3),
        (0, 7),
        (0, 2),
        (0, 2),
        (0, 1),
        (0, 3),
        (sv_mask >> 32, 32),
        (sv_mask & 0xFFFFFFFF, 32),
        (sig_mask, 32),
        (1, 1),  # cell mask: 1 cell present
        # Per-satellite MSM4 layout (22 bits): rough_int_ms(8) +
        # ext_info(4) + rough_mod_1ms(10). MSM4 does NOT carry
        # rough_doppler (that's the MSM5/MSM7 layout).
        (10, 8),
        (0, 4),
        (512, 10),
        # Per-cell MSM4 (15+22+4+1+6 = 48 bits).
        (1000, 15),  # fine PR (signed)
        (2000, 22),  # fine phase
        (5, 4),  # lock
        (0, 1),  # half-cycle
        (45, 6),  # CNR raw (1 dB-Hz scale) -> 45 dB-Hz
    ]
    frame = _build_frame(1074, bits)
    msgs = list(iter_messages(io.BytesIO(frame)))
    msg = msgs[0]
    assert msg["msg_id"] == 1074
    assert msg["satellites"][0]["sv"] == "G01"
    assert len(msg["observations"]) == 1
    obs = msg["observations"][0]
    assert obs["cnr_dbhz"] == 45.0
    # MSM4 has no Doppler.
    import math

    assert math.isnan(obs["doppler_mps"])
    # Pseudorange ~ rough 10.5 ms * c
    assert 3.1e6 < obs["pseudorange_m"] < 3.2e6


def test_decode_msm7_full_with_one_cell():
    """A 1-SV / 1-signal / 1-cell MSM7 fully decodes per-cell observation."""
    sv_mask = 1 << 63  # SV index 0 -> G01
    sig_mask = 1 << 31  # signal index 0
    bits = [
        (123, 12),  # station id
        (456_000, 30),  # tow_ms
        (0, 1),  # sync
        (0, 3),  # iod
        (0, 7),  # session
        (0, 2),  # clock steering
        (0, 2),  # external clock
        (0, 1),  # smooth indicator
        (0, 3),  # smooth interval
        (sv_mask >> 32, 32),
        (sv_mask & 0xFFFFFFFF, 32),
        (sig_mask, 32),
        (1, 1),  # cell mask: present
        # Per-satellite block (8 + 4 + 10 + 14 = 36 bits).
        (10, 8),  # rough int ms = 10
        (0, 4),  # ext info
        (512, 10),  # rough mod 1 ms (mid-range)
        (0, 14),  # rough doppler
        # Per-cell block (80 bits).
        (1000, 20),  # fine PR
        (2000, 24),  # fine phase
        (5, 10),  # lock time indicator
        (0, 1),  # half-cycle ambiguity
        (480, 10),  # CNR raw -> 30.0 dB-Hz
        (100, 15),  # fine doppler
    ]
    frame = _build_frame(1077, bits)
    msgs = list(iter_messages(io.BytesIO(frame)))
    msg = msgs[0]
    assert msg["n_sv"] == 1
    assert msg["n_sig"] == 1
    assert msg["satellites"][0]["sv"] == "G01"
    assert len(msg["observations"]) == 1
    obs = msg["observations"][0]
    assert obs["sv"] == "G01"
    assert obs["cnr_dbhz"] == 30.0
    # rough_ms = 10 + 512/1024 = 10.5 -> ~3.15e6 m
    assert 3.1e6 < obs["pseudorange_m"] < 3.2e6
    assert 3.1e6 < obs["phase_m"] < 3.2e6


def test_decode_1033_strings():
    """1033 round-trips short ASCII strings."""

    def _str_bits(s: str) -> list[tuple[int, int]]:
        out = [(len(s), 8)]
        out += [(ord(c), 8) for c in s]
        return out

    bits = [
        (10, 12),  # station id
        *_str_bits("ANT_X"),
        (0, 8),  # antenna setup ID
        *_str_bits("S123"),
        *_str_bits("RX_Y"),
        *_str_bits("V1.0"),
        *_str_bits("R456"),
    ]
    frame = _build_frame(1033, bits)
    msgs = list(iter_messages(io.BytesIO(frame)))
    out = msgs[0]
    assert out["msg_id"] == 1033
    assert out["antenna_descriptor"] == "ANT_X"
    assert out["receiver_type"] == "RX_Y"
    assert out["receiver_serial"] == "R456"
