"""BINEX (BINary EXchange) decoder.

BINEX is UNAVCO's binary archival format for GNSS data. The full spec
supports four byte-order/record-orientation combinations; we currently
decode the most common one — forward byte order, normal records —
which uses sync byte ``0xC2``.

Each record is laid out as

    0xC2 | record_id (ubnxi) | length (ubnxi) | body | checksum

where ``ubnxi`` is BINEX's variable-length unsigned integer encoding
(7 bits per byte, top bit set = "more bytes follow").

The checksum width depends on the record length:

- ≤ 127 body bytes: 1-byte XOR over the entire record (sync through body)
- 128-4095 body bytes: 2-byte CRC-16/CCITT
- ≥ 4096 body bytes: 4-byte CRC-32

Decoded record types in this MVP are limited to the framing — the
body comes back as raw ``body_bytes``. The most common record IDs
(per the UNAVCO catalog) are:

- ``0x00`` site / monument information
- ``0x01`` decoded GPS ephemeris
- ``0x02`` decoded GLONASS ephemeris
- ``0x7E`` trace / positioning info
- ``0x7F`` test record

Reference: https://www.unavco.org/data/gps-gnss/data-formats/binex/
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, BinaryIO

#: BINEX sync byte for forward byte order, normal records.
SYNC = 0xC2


def read_ubnxi(stream: BinaryIO) -> int | None:
    """Read one ubnxi (BINEX variable-length integer) from ``stream``.

    Returns ``None`` at end-of-stream.
    """
    value = 0
    shift = 0
    for _ in range(8):                  # at most 8 bytes per ubnxi
        b = stream.read(1)
        if not b:
            return None
        byte = b[0]
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value
        shift += 7
    return value


def encode_ubnxi(value: int) -> bytes:
    """Inverse of :func:`read_ubnxi`. Raises ``ValueError`` on negatives."""
    if value < 0:
        raise ValueError("ubnxi cannot encode negative integers")
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def xor_checksum(data: bytes) -> int:
    """1-byte XOR checksum over ``data`` (used for body length ≤ 127)."""
    cs = 0
    for byte in data:
        cs ^= byte
    return cs


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT (poly 0x1021, init 0) — for body length 128-4095."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def iter_records(stream: BinaryIO, *, check_crc: bool = True) -> Iterator[dict[str, Any]]:
    """Iterate forward-byte-order BINEX records from ``stream``.

    Parameters
    ----------
    stream:
        Anything supporting ``read(n)`` returning bytes.
    check_crc:
        Validate the trailing checksum. Default True.

    Yields
    ------
    dict
        Always contains ``record_id``, ``length``, ``body_bytes``.
    """
    while True:
        b = stream.read(1)
        if not b:
            return
        if b[0] != SYNC:
            continue
        record_id = read_ubnxi(stream)
        if record_id is None:
            return
        length = read_ubnxi(stream)
        if length is None:
            return
        body = stream.read(length)
        if len(body) < length:
            return
        if length <= 127:
            cs_bytes = stream.read(1)
            if len(cs_bytes) < 1:
                return
            if check_crc:
                # XOR over sync + record_id_bytes + length_bytes + body.
                rid_bytes = encode_ubnxi(record_id)
                len_bytes = encode_ubnxi(length)
                if xor_checksum(bytes([SYNC]) + rid_bytes + len_bytes + body) != cs_bytes[0]:
                    continue
        elif length <= 4095:
            cs_bytes = stream.read(2)
            if len(cs_bytes) < 2:
                return
            if check_crc:
                rid_bytes = encode_ubnxi(record_id)
                len_bytes = encode_ubnxi(length)
                expected = (cs_bytes[0] << 8) | cs_bytes[1]
                if crc16_ccitt(bytes([SYNC]) + rid_bytes + len_bytes + body) != expected:
                    continue
        else:
            # 32-bit CRC variant: skipped in MVP, just consume 4 bytes.
            _ = stream.read(4)

        yield {"record_id": record_id, "length": length, "body_bytes": body}


__all__ = [
    "SYNC",
    "crc16_ccitt",
    "encode_ubnxi",
    "iter_records",
    "read_ubnxi",
    "xor_checksum",
]
