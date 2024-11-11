"""Furuno GW-10 raw-message framer + SBAS L1 extractor.

The Furuno GW-10 is a single-frequency GPS+SBAS receiver. Its output
binary format is documented (Furuno GW-10 III manual, July 2004) and
RTKLIB ships a parser at src/rcv/gw10.c which is the reference we
followed here.

A GW-10 frame is

    0x8B (sync) + ID (1 byte) + payload (varies by ID) + checksum (1 byte)

with total length looked up from the ID. The checksum is the sum of
every byte except the sync byte and itself.

The receiver wraps several payload types; the one this module
specifically exposes is ID 0x03 (SBAS L1 message), which carries the
GPS time of week, the GEO PRN, and 29 bytes of the SBAS 250-bit
message. With the message type and preamble exposed, downstream code
can dispatch to per-MT SBAS decoders.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, BinaryIO

#: GW-10 frame sync byte.
SYNC = 0x8B

#: Total frame length per message ID, from the GW-10 III manual / RTKLIB.
_MSG_LENGTHS = {
    0x02: 48,    # GPS subframe message
    0x03: 40,    # SBAS L1 message
    0x06: 21,    # DGPS
    0x07: 22,    # DGPS reference info
    0x08: 379,   # raw observations
    0x20: 227,   # solution
    0x22: 17,    # satellite health
    0x23: 67,    # satellite orbit
    0x24: 68,    # ephemeris
    0x25: 39,    # almanac
    0x26: 32,    # iono / UTC correction
    0x27: 98,    # raw ephemeris
}


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    out = stream.read(n)
    if len(out) < n:
        return b""
    return out


def iter_frames(
    stream: BinaryIO, *, check_checksum: bool = True
) -> Iterator[dict[str, Any]]:
    """Yield one ``dict`` per recognised GW-10 frame in the stream.

    The state machine resynchronises after a bad byte or a frame whose
    checksum fails.

    Yields
    ------
    dict
        ``{"id": int, "length": int, "payload": bytes, "checksum_ok": bool}``.
        The ``payload`` field excludes the sync, ID, and checksum bytes
        (i.e. it's the body the per-message decoders consume).
    """
    while True:
        b = stream.read(1)
        if not b:
            return
        if b[0] != SYNC:
            continue
        id_b = stream.read(1)
        if not id_b:
            return
        msg_id = id_b[0]
        n = _MSG_LENGTHS.get(msg_id)
        if n is None:
            # Unknown ID; resync on the next 0x8B.
            continue
        # Remaining bytes after sync + id is (n - 2): payload (n - 3) + cksum (1).
        rest = _read_exact(stream, n - 2)
        if not rest:
            return
        body = rest[:-1]
        cksum = rest[-1]
        # Checksum is the unsigned sum of bytes from index 1 to n-2 (inclusive)
        # in the full frame, mod 256. With our slicing that's id + body.
        cs = (msg_id + sum(body)) & 0xFF
        ok = cs == cksum
        if check_checksum and not ok:
            continue
        yield {
            "id": msg_id,
            "length": n,
            "payload": body,
            "checksum_ok": ok,
        }


def decode_sbas(payload: bytes) -> dict[str, Any]:
    """Decode the GW-10 SBAS L1 (ID 0x03) payload.

    Layout: 4-byte TOW (in ms, big-endian) + 1-byte PRN + 29-byte SBAS
    L1 message + 3 reserved bytes.

    Returns
    -------
    dict
        ``tow_s`` (float, seconds-of-week), ``prn`` (int, SBAS GEO PRN
        120-158), ``sbas_message`` (29 bytes of the L1 message),
        ``message_type`` (int, 0-63 from bits 8-13 of the message,
        immediately after the 8-bit preamble), ``preamble`` (int,
        the 8-bit preamble at the start of the message; one of
        0x53, 0x9A, 0xC6).
    """
    if len(payload) < 5 + 29:
        return {"truncated": True}
    tow_ms = int.from_bytes(payload[0:4], "big", signed=False)
    prn = payload[4]
    msg = bytes(payload[5 : 5 + 29])
    preamble = msg[0]
    # Message type is the next 6 bits after the preamble (i.e. bits 8-13).
    # That's the top 6 bits of msg[1].
    message_type = (msg[1] >> 2) & 0x3F
    return {
        "tow_s": tow_ms / 1000.0,
        "prn": prn,
        "sbas_message": msg,
        "preamble": preamble,
        "message_type": message_type,
    }


def iter_sbas_messages(
    stream: BinaryIO, *, check_checksum: bool = True
) -> Iterator[dict[str, Any]]:
    """High-level helper: walk a GW-10 stream, yield only SBAS L1 frames.

    Convenience wrapper around :func:`iter_frames` that filters to
    ``id == 0x03`` and decodes the SBAS payload.
    """
    for frame in iter_frames(stream, check_checksum=check_checksum):
        if frame["id"] != 0x03:
            continue
        out = decode_sbas(frame["payload"])
        out["checksum_ok"] = frame["checksum_ok"]
        yield out


__all__ = [
    "SYNC",
    "decode_sbas",
    "iter_frames",
    "iter_sbas_messages",
]
