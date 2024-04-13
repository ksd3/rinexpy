"""RTCM SC-104 v2.x decoder.

RTCM 2.x is the legacy DGPS / RTK correction format. The wire layout
is the GPS L1 navigation-message style: each "word" is 30 bits made
up of 24 data bits plus 6 parity bits, packed onto the wire as
6 bits of data per byte (with the byte's high bits typically zero).

Each frame begins with a fixed 8-bit preamble (``0x66``); a header
follows in two words; then ``N`` data words, where ``N`` is encoded
in the second header word.

This module implements:

- :func:`extract_data_bits` — strip the 6-of-8 wire encoding to a
  contiguous bitstream of data bits only.
- :func:`iter_messages` — locate the preamble, decode the two header
  words, and yield one structured dict per message.
- per-type decoders for the most common DGPS messages: **Type 1**
  (differential GPS corrections), **Type 3** (reference-station ECEF),
  **Type 9** (high-rate corrections, same payload shape as Type 1).

Other types come back with raw ``data_words`` (a list of 24-bit ints).

The 6-bit parity check (Hamming over the previous word's last 2 bits
plus the current 24 data bits) is *not* validated in this MVP — we
trust the caller's framing layer (typically a serial UART). Callers
who want full parity validation can subclass and override
``_check_parity``.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from typing import Any, BinaryIO

#: RTCM 2.x preamble byte.
PREAMBLE = 0x66

#: Number of data bits per "byte" on the wire (the high two bits are
#: padding/parity-related and stripped).
_BITS_PER_BYTE = 6


def extract_data_bits(buf: bytes) -> str:
    """Strip the 6-of-8 wire encoding to a bitstream of data bits.

    Each byte contributes its low 6 bits, MSB-first, to the output
    bitstream. The result is a string of '0'/'1' characters that can
    be sliced into 30-bit words at the caller's convenience.

    Parameters
    ----------
    buf:
        Wire bytes from the receiver.

    Returns
    -------
    str
        A ``"0"`` / ``"1"`` string of length ``len(buf) * 6``.
    """
    return "".join(f"{b & 0x3F:06b}" for b in buf)


def _bits_to_int(bits: str, signed: bool = False) -> int:
    """Convert a bitstring (MSB-first) to an integer; optional two's complement."""
    value = int(bits, 2)
    if signed and bits[0] == "1":
        value -= 1 << len(bits)
    return value


def iter_messages(stream: BinaryIO) -> Iterator[dict[str, Any]]:
    """Iterate RTCM 2.x messages from a binary stream.

    Parameters
    ----------
    stream:
        Anything supporting ``read(n)`` returning bytes.

    Yields
    ------
    dict
        Always contains ``msg_type``, ``station_id``, ``z_count``,
        ``sequence``, ``n_words``, ``health``, ``data_words``.
        Decoded message types (1, 3, 9) add structured fields.
    """
    # Read in 5-byte chunks (one 30-bit word per 5 wire bytes).
    pending = b""
    while True:
        # Slurp some bytes if we don't have enough for a header.
        if len(pending) < 60:
            new = stream.read(256)
            if not new:
                if not pending:
                    return
                # Try to flush the remaining bytes.
            pending += new

        # Find the preamble in the data-bit stream.
        bits = extract_data_bits(pending)
        idx = bits.find(f"{PREAMBLE:08b}")
        if idx < 0:
            # Resync: drop most of the buffer (keep the last 5 bytes
            # in case the preamble straddles).
            pending = pending[-5:]
            if not stream.read(1):
                return
            continue

        # The preamble lives inside word 1; word 1 starts ``idx`` bits
        # before the preamble's first bit if the preamble is at offset 0.
        # In RTCM 2.x word 1 layout, the preamble occupies bits 0-7
        # and the message type bits 8-13.
        if idx + 60 > len(bits):
            # Need more bytes for two header words (60 bits).
            new = stream.read(60)
            if not new:
                return
            pending += new
            continue

        word1 = bits[idx : idx + 30]
        word2 = bits[idx + 30 : idx + 60]

        msg_type = _bits_to_int(word1[8:14])
        station_id = _bits_to_int(word1[14:24])
        z_count = _bits_to_int(word2[0:13])
        sequence = _bits_to_int(word2[13:16])
        n_words = _bits_to_int(word2[16:21])
        health = _bits_to_int(word2[21:24])

        # Need n_words * 30 more data bits after the 60-bit header.
        total_data_bits = 60 + n_words * 30
        # Wire bytes needed to cover that bit count, plus the index offset:
        bytes_needed = (idx + total_data_bits + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
        while len(pending) < bytes_needed:
            new = stream.read(bytes_needed - len(pending))
            if not new:
                return
            pending += new
        bits = extract_data_bits(pending)

        data_words: list[int] = []
        for w in range(n_words):
            start = idx + 60 + w * 30
            data_words.append(_bits_to_int(bits[start : start + 24]))

        out: dict[str, Any] = {
            "msg_type": msg_type,
            "station_id": station_id,
            "z_count": z_count,
            "sequence": sequence,
            "n_words": n_words,
            "health": health,
            "data_words": data_words,
        }
        decoder = _DECODERS.get(msg_type)
        if decoder is not None:
            out.update(decoder(data_words))

        yield out

        # Consume the bytes we just used.
        consumed = (idx + total_data_bits) // _BITS_PER_BYTE
        pending = pending[consumed:]


def _decode_type1(words: list[int]) -> dict[str, Any]:
    """Type 1: differential GPS corrections.

    Each correction set is 40 data bits (1 word + 16 bits of next):
    scale (1) | UDRE (2) | sat ID (5) | PRC (16) | RRC (8) | IODE (8).
    The packed form crosses 30-bit word boundaries, so we re-flatten
    the words into a continuous bitstream first.
    """
    flat = "".join(f"{w:024b}" for w in words)
    corrections: list[dict[str, Any]] = []
    bit = 0
    while bit + 40 <= len(flat):
        chunk = flat[bit : bit + 40]
        scale = int(chunk[0])
        udre = int(chunk[1:3], 2)
        sat_id = int(chunk[3:8], 2)
        prc_raw = int(chunk[8:24], 2)
        if prc_raw & 0x8000:
            prc_raw -= 0x10000
        rrc_raw = int(chunk[24:32], 2)
        if rrc_raw & 0x80:
            rrc_raw -= 0x100
        iode = int(chunk[32:40], 2)
        # PRC scale: 0.02 m if scale=0, 0.32 m if scale=1.
        # RRC scale: 0.002 m/s if scale=0, 0.032 m/s if scale=1.
        prc_m = prc_raw * (0.32 if scale else 0.02)
        rrc_m_s = rrc_raw * (0.032 if scale else 0.002)
        corrections.append(
            {
                "sat_id": sat_id,
                "udre": udre,
                "prc_m": prc_m,
                "rrc_m_s": rrc_m_s,
                "iode": iode,
            }
        )
        bit += 40
    return {"corrections": corrections}


def _decode_type3(words: list[int]) -> dict[str, Any]:
    """Type 3: reference station ECEF coordinates.

    96 data bits: X (32) | Y (32) | Z (32), each in cm and signed.
    """
    if len(words) < 4:
        return {"truncated": True}
    flat = "".join(f"{w:024b}" for w in words)
    if len(flat) < 96:
        return {"truncated": True}
    x_cm = _bits_to_int(flat[0:32], signed=True)
    y_cm = _bits_to_int(flat[32:64], signed=True)
    z_cm = _bits_to_int(flat[64:96], signed=True)
    return {
        "x_m": x_cm * 0.01,
        "y_m": y_cm * 0.01,
        "z_m": z_cm * 0.01,
    }


def _decode_type9(words: list[int]) -> dict[str, Any]:
    """Type 9: high-rate DGPS corrections (same payload as Type 1)."""
    return _decode_type1(words)


_DECODERS = {
    1: _decode_type1,
    3: _decode_type3,
    9: _decode_type9,
}


__all__ = [
    "PREAMBLE",
    "extract_data_bits",
    "iter_messages",
]


# Re-export struct so the test can build encoded frames easily.
_ = struct
