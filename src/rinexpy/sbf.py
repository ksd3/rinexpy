"""Septentrio Binary Format (SBF) decoder.

SBF block layout (little-endian):

    0x24 0x40 (sync)
    | CRC (u16) | ID (u16) | length (u16) | TOW (u32) | WNc (u16) | body...

The CRC is computed over everything from the ID through the end of
the block (excluding the sync and the CRC field itself), using the
CRC-CCITT polynomial 0x1021 with init 0.

Decoded block IDs (a small but practical subset):

- ``PVTGeodetic`` (4007) — full geodetic position/velocity/time fix
- ``MeasEpoch`` (4027) — header for a per-channel measurement set
- ``GPSNav`` (5891) — GPS LNAV broadcast ephemeris

Other IDs come back with the raw ``payload_bytes`` so callers can
dispatch them.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from typing import Any, BinaryIO

#: SBF sync bytes (ASCII '$' '@').
SYNC = b"\x24\x40"


def crc_ccitt(data: bytes, *, init: int = 0) -> int:
    """SBF CRC: CRC-CCITT (polynomial 0x1021, init 0, no final XOR)."""
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def iter_blocks(stream: BinaryIO, *, check_crc: bool = True) -> Iterator[dict[str, Any]]:
    """Iterate SBF blocks from a binary stream.

    Parameters
    ----------
    stream:
        Anything supporting ``read(n)`` returning bytes.
    check_crc:
        Validate the CRC-CCITT checksum on every block (default ``True``).

    Yields
    ------
    dict
        Always contains ``block_id``, ``block_revision``, ``length``,
        ``tow_ms``, ``wnc``, ``payload_bytes``. Decoded block types add
        structured fields.
    """
    while True:
        b = stream.read(1)
        if not b:
            return
        if b[0] != SYNC[0]:
            continue
        b2 = stream.read(1)
        if not b2 or b2[0] != SYNC[1]:
            continue
        head = stream.read(6)
        if len(head) < 6:
            return
        crc_recv, id_field, length = struct.unpack("<HHH", head)
        # The "length" includes the 8-byte header (sync + crc + id + len);
        # the body is therefore length - 8 bytes.
        body_len = length - 8
        if body_len < 0 or body_len > 65536:
            continue  # pathological; resync.
        body = stream.read(body_len)
        if len(body) < body_len:
            return
        if check_crc:
            calc = crc_ccitt(head[2:] + body)  # exclude sync + crc field
            if calc != crc_recv:
                continue
        block_id = id_field & 0x1FFF
        block_revision = (id_field >> 13) & 0x07

        if body_len < 6:
            yield {
                "block_id": block_id,
                "block_revision": block_revision,
                "length": length,
                "tow_ms": None,
                "wnc": None,
                "payload_bytes": body,
            }
            continue

        tow_ms, wnc = struct.unpack_from("<IH", body, 0)
        payload = body[6:]
        out = {
            "block_id": block_id,
            "block_revision": block_revision,
            "length": length,
            "tow_ms": tow_ms,
            "wnc": wnc,
            "payload_bytes": payload,
        }
        decoder = _DECODERS.get(block_id)
        if decoder is not None:
            out.update(decoder(payload))
        yield out


def _decode_pvt_geodetic(p: bytes) -> dict[str, Any]:
    """PVTGeodetic (4007): geodetic lat/lon/height + velocity + clock bias.

    Spec: Septentrio SBF Reference Guide §3.4.1.
    """
    if len(p) < 80:
        return {"truncated": True}
    return {
        "fix_mode": p[0],
        "error_code": p[1],
        "lat_rad": struct.unpack_from("<d", p, 2)[0],
        "lon_rad": struct.unpack_from("<d", p, 10)[0],
        "height_m": struct.unpack_from("<d", p, 18)[0],
        "undulation_m": struct.unpack_from("<f", p, 26)[0],
        "vn_m_s": struct.unpack_from("<f", p, 30)[0],
        "ve_m_s": struct.unpack_from("<f", p, 34)[0],
        "vu_m_s": struct.unpack_from("<f", p, 38)[0],
        "cog_deg": struct.unpack_from("<f", p, 42)[0],
        "rx_clock_bias_ms": struct.unpack_from("<d", p, 46)[0],
        "rx_clock_drift_ppm": struct.unpack_from("<f", p, 54)[0],
        "time_system": p[58],
        "datum": p[59],
        "n_sv": p[60],
        "wa_corr_info": p[61],
    }


def _decode_meas_epoch(p: bytes) -> dict[str, Any]:
    """MeasEpoch (4027): per-channel measurement set header.

    We expose the header fields; per-channel sub-blocks vary by
    revision and are returned as raw bytes for callers that need them.
    """
    if len(p) < 6:
        return {"truncated": True}
    return {
        "n_channels": p[0],
        "sb1_length": p[1],
        "sb2_length": p[2],
        "common_flags": p[3],
        "cum_clock_jumps": p[4],
        "channel_data": p[6:],
    }


def _decode_gps_nav(p: bytes) -> dict[str, Any]:
    """GPSNav (5891): broadcast LNAV ephemeris (selected fields)."""
    if len(p) < 32:
        return {"truncated": True}
    return {
        "prn": p[0],
        "reserved": p[1],
        "wn": struct.unpack_from("<H", p, 2)[0],
        "ca_or_p_on_l2": p[4],
        "ura_index": p[5],
        "health": p[6],
        "l2_p_data_flag": p[7],
        "iodc": struct.unpack_from("<H", p, 8)[0],
        "iode2": p[10],
        "iode3": p[11],
        "fit_int_flag": p[12],
        "t_gd_s": struct.unpack_from("<f", p, 14)[0],
        "t_oc_s": struct.unpack_from("<I", p, 18)[0],
        "a_f2": struct.unpack_from("<f", p, 22)[0],
        "a_f1": struct.unpack_from("<f", p, 26)[0],
    }


_DECODERS = {
    4007: _decode_pvt_geodetic,
    4027: _decode_meas_epoch,
    5891: _decode_gps_nav,
}


__all__ = ["SYNC", "crc_ccitt", "iter_blocks"]
