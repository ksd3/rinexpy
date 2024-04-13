"""Tests for the RTCM 2.x decoder."""

from __future__ import annotations

import io

from rinexpy.rtcm2 import PREAMBLE, extract_data_bits, iter_messages


def _bits_to_wire(bitstring: str) -> bytes:
    """Pack a '0'/'1' string (MSB-first) into wire bytes (6 data bits per byte).

    The high two bits of each wire byte are zero (we don't apply a
    parity model in the test fixtures).
    """
    # Pad to a multiple of 6 bits.
    pad = (-len(bitstring)) % 6
    bitstring = bitstring + "0" * pad
    out = bytearray()
    for i in range(0, len(bitstring), 6):
        out.append(int(bitstring[i : i + 6], 2))
    return bytes(out)


def _build_frame(
    msg_type: int,
    station_id: int,
    z_count: int,
    sequence: int,
    data_words: list[int],
    *,
    health: int = 0,
) -> bytes:
    """Build a minimal RTCM 2.x frame with the requested data words.

    Word 1 layout: preamble (8) | msg_type (6) | station_id (10) | parity-pad (6)
    Word 2 layout: z_count (13) | sequence (3) | n_words (5) | health (3) | parity-pad (6)
    Each data word: 24 data bits + 6 parity-pad bits.
    """
    n_words = len(data_words)
    word1 = (
        f"{PREAMBLE:08b}"
        + f"{msg_type:06b}"
        + f"{station_id:010b}"
        + "0" * 6
    )
    word2 = (
        f"{z_count:013b}"
        + f"{sequence:03b}"
        + f"{n_words:05b}"
        + f"{health:03b}"
        + "0" * 6
    )
    bits = word1 + word2
    for w in data_words:
        bits += f"{w:024b}" + "0" * 6
    return _bits_to_wire(bits)


def test_extract_data_bits_strips_high_bits():
    raw = bytes([0b00111111, 0b01010101, 0b10101010])
    bits = extract_data_bits(raw)
    assert bits == "111111" + "010101" + "101010"


def test_iter_yields_unknown_message_with_data_words():
    frame = _build_frame(
        msg_type=42,
        station_id=12,
        z_count=1234,
        sequence=3,
        data_words=[0x123456, 0xABCDEF],
    )
    msgs = list(iter_messages(io.BytesIO(frame)))
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["msg_type"] == 42
    assert msg["station_id"] == 12
    assert msg["z_count"] == 1234
    assert msg["sequence"] == 3
    assert msg["n_words"] == 2
    assert msg["data_words"] == [0x123456, 0xABCDEF]


def test_decode_type3_recovers_ecef():
    """Type 3 station coords: round-trip a known (X, Y, Z) in cm."""
    # X=4.886e6 m, Y=-256068.5 m, Z=4.078e6 m -> centimetres.
    x_cm = 488607881
    y_cm = -25606854
    z_cm = 407804985
    flat = (
        (f"{x_cm & 0xFFFFFFFF:032b}")
        + (f"{y_cm & 0xFFFFFFFF:032b}")
        + (f"{z_cm & 0xFFFFFFFF:032b}")
    )
    # Pack into 24-bit data words.
    data_words = [int(flat[i : i + 24], 2) for i in range(0, len(flat), 24)]
    frame = _build_frame(3, 1, 0, 0, data_words)
    msg = next(iter(iter_messages(io.BytesIO(frame))))
    assert msg["msg_type"] == 3
    assert abs(msg["x_m"] - 4886078.81) < 1
    assert abs(msg["y_m"] - -256068.54) < 1
    assert abs(msg["z_m"] - 4078049.85) < 1


def test_iter_skips_garbage_before_preamble():
    frame = _build_frame(42, 1, 0, 0, [0])
    stream = io.BytesIO(b"\x00\x00\x00\x00" + frame)  # zero bytes before
    msgs = list(iter_messages(stream))
    assert len(msgs) >= 1
    # The first decoded msg should be msg_type 42.
    assert any(m["msg_type"] == 42 for m in msgs)


def test_decode_type1_corrections_present():
    """Type 1: synthesise one correction set + verify the decoded fields."""
    # 40-bit correction: scale=0 | udre=0 | sat_id=5 | PRC=100 (=2.0 m) |
    # RRC=10 (=0.02 m/s) | IODE=42
    chunk = "0" + "00" + "00101" + f"{100:016b}" + f"{10:08b}" + f"{42:08b}"
    # Pack into 24-bit words.
    flat = chunk + "0" * ((24 - len(chunk) % 24) % 24)
    data_words = [int(flat[i : i + 24], 2) for i in range(0, len(flat), 24)]
    frame = _build_frame(1, 1, 0, 0, data_words)
    msg = next(iter(iter_messages(io.BytesIO(frame))))
    corrections = msg.get("corrections", [])
    assert len(corrections) >= 1
    c = corrections[0]
    assert c["sat_id"] == 5
    assert c["prc_m"] == 2.0
    assert c["rrc_m_s"] == 0.02
    assert c["iode"] == 42
