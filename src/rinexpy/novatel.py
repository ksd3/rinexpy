"""NovAtel OEM binary log decoder.

NovAtel binary logs use a 28-byte long header preceded by a 3-byte
sync sequence and followed by a 32-bit CRC. Layout (little-endian):

    0xAA 0x44 0x12 | header_length | msg_id (u16) | msg_type (u8) |
    port (u8) | msg_length (u16) | sequence (u16) | idle_time (u8) |
    time_status (u8) | week (u16) | gps_ms (u32) | rx_status (u32) |
    reserved (u16) | rx_sw_ver (u16)
    | <body, msg_length bytes>
    | CRC32 (u32)

The 32-bit CRC uses the standard IEEE-802.3 polynomial 0xEDB88320 over
everything from the first sync byte through the end of the body
(excluding the CRC itself).

Decoded message IDs:

- ``BESTPOS`` (42) — best-available position solution
- ``BESTXYZ`` (241) — same as ECEF + velocity
- ``RAWEPHEM`` (41) — broadcast ephemeris subframes (raw 30-bit words)

Other IDs come back with the raw ``body_bytes``.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from typing import Any, BinaryIO

#: NovAtel OEM sync sequence.
SYNC = b"\xaa\x44\x12"


def crc32(data: bytes) -> int:
    """NovAtel-style 32-bit CRC (matches Python's ``zlib.crc32``)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def iter_messages(stream: BinaryIO, *, check_crc: bool = True) -> Iterator[dict[str, Any]]:
    """Iterate NovAtel OEM messages from a binary stream.

    Parameters
    ----------
    stream:
        Anything supporting ``read(n)`` returning bytes.
    check_crc:
        Validate the trailing CRC32 on every message (default True).

    Yields
    ------
    dict
        Always contains ``msg_id``, ``msg_type``, ``msg_length``,
        ``week``, ``gps_ms``, ``body_bytes``. Decoded message types
        add structured fields.
    """
    while True:
        # Re-sync on the 3-byte sync sequence.
        b = stream.read(1)
        if not b:
            return
        if b[0] != SYNC[0]:
            continue
        b2 = stream.read(1)
        if not b2 or b2[0] != SYNC[1]:
            continue
        b3 = stream.read(1)
        if not b3 or b3[0] != SYNC[2]:
            continue
        # The header length lives at offset 3 (one byte after the sync).
        # We've consumed 3 sync bytes; read header_len byte + the rest.
        hdr_len_byte = stream.read(1)
        if not hdr_len_byte:
            return
        header_length = hdr_len_byte[0]
        if header_length < 28 or header_length > 64:
            continue  # implausible
        # Read the rest of the header.
        rest_of_header = stream.read(header_length - 4)
        if len(rest_of_header) < header_length - 4:
            return
        full_header = SYNC + bytes([header_length]) + rest_of_header

        # Field offsets within the header (per NovAtel spec).
        msg_id = struct.unpack_from("<H", full_header, 4)[0]
        msg_type = full_header[6]
        port = full_header[7]
        msg_length = struct.unpack_from("<H", full_header, 8)[0]
        sequence = struct.unpack_from("<H", full_header, 10)[0]
        week = struct.unpack_from("<H", full_header, 14)[0]
        gps_ms = struct.unpack_from("<I", full_header, 16)[0]

        body = stream.read(msg_length)
        crc_bytes = stream.read(4)
        if len(body) < msg_length or len(crc_bytes) < 4:
            return
        if check_crc:
            (recv_crc,) = struct.unpack("<I", crc_bytes)
            if crc32(full_header + body) != recv_crc:
                continue

        out = {
            "msg_id": msg_id,
            "msg_type": msg_type,
            "port": port,
            "msg_length": msg_length,
            "sequence": sequence,
            "week": week,
            "gps_ms": gps_ms,
            "body_bytes": body,
        }
        decoder = _DECODERS.get(msg_id)
        if decoder is not None:
            out.update(decoder(body))
        yield out


def _decode_bestpos(p: bytes) -> dict[str, Any]:
    """BESTPOS (42): best-available geodetic position fix.

    Spec: NovAtel OEM7 commands & logs reference, BESTPOS log.
    """
    if len(p) < 72:
        return {"truncated": True}
    return {
        "sol_status": struct.unpack_from("<I", p, 0)[0],
        "pos_type": struct.unpack_from("<I", p, 4)[0],
        "lat_deg": struct.unpack_from("<d", p, 8)[0],
        "lon_deg": struct.unpack_from("<d", p, 16)[0],
        "height_m": struct.unpack_from("<d", p, 24)[0],
        "undulation_m": struct.unpack_from("<f", p, 32)[0],
        "datum_id": struct.unpack_from("<I", p, 36)[0],
        "lat_stdev_m": struct.unpack_from("<f", p, 40)[0],
        "lon_stdev_m": struct.unpack_from("<f", p, 44)[0],
        "height_stdev_m": struct.unpack_from("<f", p, 48)[0],
        "n_obs": p[64],
        "n_sv_used": p[65],
        "n_sv_above_mask": p[66],
    }


def _decode_bestxyz(p: bytes) -> dict[str, Any]:
    """BESTXYZ (241): best-available ECEF position + velocity."""
    if len(p) < 112:
        return {"truncated": True}
    return {
        "p_sol_status": struct.unpack_from("<I", p, 0)[0],
        "pos_type": struct.unpack_from("<I", p, 4)[0],
        "x_m": struct.unpack_from("<d", p, 8)[0],
        "y_m": struct.unpack_from("<d", p, 16)[0],
        "z_m": struct.unpack_from("<d", p, 24)[0],
        "x_stdev_m": struct.unpack_from("<f", p, 32)[0],
        "y_stdev_m": struct.unpack_from("<f", p, 36)[0],
        "z_stdev_m": struct.unpack_from("<f", p, 40)[0],
        "v_sol_status": struct.unpack_from("<I", p, 52)[0],
        "vel_type": struct.unpack_from("<I", p, 56)[0],
        "vx_m_s": struct.unpack_from("<d", p, 60)[0],
        "vy_m_s": struct.unpack_from("<d", p, 68)[0],
        "vz_m_s": struct.unpack_from("<d", p, 76)[0],
    }


def _decode_rawephem(p: bytes) -> dict[str, Any]:
    """RAWEPHEM (41): raw GPS LNAV subframes 1, 2, 3 (30 bytes each)."""
    if len(p) < 102:
        return {"truncated": True}
    return {
        "prn": struct.unpack_from("<I", p, 0)[0],
        "ref_week": struct.unpack_from("<I", p, 4)[0],
        "ref_seconds": struct.unpack_from("<I", p, 8)[0],
        "subframe1_bytes": p[12:42],
        "subframe2_bytes": p[42:72],
        "subframe3_bytes": p[72:102],
    }


_DECODERS = {
    41: _decode_rawephem,
    42: _decode_bestpos,
    241: _decode_bestxyz,
}


__all__ = ["SYNC", "crc32", "iter_messages"]
