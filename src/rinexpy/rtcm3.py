"""RTCM 3.x message decoder (minimum viable).

Reference: RTCM Standard 10403.x. RTCM3 frames every message as

    0xD3 (preamble) | 6 bits reserved | 10 bits length | length bytes | 24 bits CRC-24Q

The payload's first 12 bits are the message number. This module decodes
the framing and a small set of common message types:

- **1005**: Stationary RTK reference station ARP (station ECEF position)
- **1006**: Same as 1005 plus antenna height
- **1019**: GPS broadcast ephemeris
- **1020**: GLONASS broadcast ephemeris

Other messages parse as ``{"msg_id": N, "payload_bytes": b}`` and the
caller can dispatch them.

CRC-24Q is *not* validated by default (see :func:`iter_messages` ``check_crc=``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, BinaryIO

#: RTCM3 sync byte.
PREAMBLE = 0xD3


def _bits(buf: bytes, start_bit: int, n_bits: int, *, signed: bool = False) -> int:
    """Read ``n_bits`` from ``buf`` starting at bit ``start_bit`` (MSB-first).

    The RTCM3 spec packs every field bit-aligned to make the wire format
    compact. This is a slow-but-clear extraction helper.
    """
    value = 0
    for i in range(n_bits):
        byte_idx, bit_idx = divmod(start_bit + i, 8)
        bit = (buf[byte_idx] >> (7 - bit_idx)) & 1
        value = (value << 1) | bit
    if signed and (value >> (n_bits - 1)) & 1:
        value -= 1 << n_bits
    return value


def crc24q(data: bytes) -> int:
    """Compute the RTCM3 CRC-24Q checksum over ``data``.

    Polynomial 0x1864CFB, initial value 0. The CRC trails the message body.
    """
    crc = 0
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
    return crc & 0xFFFFFF


def iter_messages(stream: BinaryIO, *, check_crc: bool = False) -> Iterator[dict[str, Any]]:
    """Iterate RTCM3 messages from a binary stream.

    Parameters
    ----------
    stream:
        Anything supporting ``read(n)`` returning bytes. Typically a
        ``socket.makefile("rb")`` from an NTRIP feed, or a ``BytesIO``.
    check_crc:
        If True, verify CRC-24Q on every frame and skip frames that
        fail. Default False (the framing layer above us usually verifies).

    Yields
    ------
    dict
        Always contains ``msg_id`` and ``payload_bytes``. Decoded message
        types add structured fields (see module docstring).
    """
    while True:
        b = stream.read(1)
        if not b:
            return
        if b[0] != PREAMBLE:
            continue  # resync
        head = stream.read(2)
        if len(head) < 2:
            return
        # 6 reserved bits + 10 length bits.
        length = ((head[0] & 0x03) << 8) | head[1]
        body = stream.read(length)
        crc = stream.read(3)
        if len(body) < length or len(crc) < 3:
            return
        if check_crc:
            calc = crc24q(b + head + body)
            recv = (crc[0] << 16) | (crc[1] << 8) | crc[2]
            if calc != recv:
                continue
        if length < 2:
            continue
        msg_id = (body[0] << 4) | (body[1] >> 4)
        out = decode_message(msg_id, body)
        yield out


def decode_message(msg_id: int, body: bytes) -> dict[str, Any]:
    """Dispatch ``body`` to the right decoder for ``msg_id``.

    Unknown message types come back as
    ``{"msg_id": N, "payload_bytes": body}``.
    """
    decoders = {
        1004: _decode_1004,
        1005: _decode_1005,
        1006: _decode_1006,
        1019: _decode_1019,
        1020: _decode_1020,
        1033: _decode_1033,
    }
    if msg_id in decoders:
        return decoders[msg_id](body)
    # MSM7 messages: 1077=GPS, 1087=GLO, 1097=GAL, 1107=SBAS, 1117=QZSS,
    # 1127=BDS, 1137=NavIC. All share the same MSM header layout.
    if msg_id in {1077, 1087, 1097, 1107, 1117, 1127, 1137}:
        return _decode_msm_header(msg_id, body)
    return {"msg_id": msg_id, "payload_bytes": body}


def _decode_1005(body: bytes) -> dict[str, Any]:
    """Stationary RTK reference station ARP (no antenna height)."""
    # Layout per RTCM 3.x table 3.5-15. We start the bit cursor at 12
    # (past the 12-bit message number).
    bit = 12
    sta_id = _bits(body, bit, 12)
    bit += 12
    bit += 6  # ITRF realization year
    bit += 4  # GPS/GLO/Galileo indicator + reference-station indicator + sync
    x = _bits(body, bit, 38, signed=True) * 1e-4
    bit += 38
    bit += 1  # single-receiver-oscillator indicator
    bit += 1  # reserved
    y = _bits(body, bit, 38, signed=True) * 1e-4
    bit += 38
    bit += 2  # quarter-cycle indicator
    z = _bits(body, bit, 38, signed=True) * 1e-4
    bit += 38
    return {"msg_id": 1005, "station_id": sta_id, "position": (x, y, z)}


def _decode_1006(body: bytes) -> dict[str, Any]:
    """1005 + 16-bit antenna height in meters."""
    out = _decode_1005(body)
    out["msg_id"] = 1006
    # Antenna height is the last 16 bits.
    height = _bits(body, 12 + 12 + 6 + 4 + 38 + 1 + 1 + 38 + 2 + 38, 16) * 1e-4
    out["antenna_height"] = height
    return out


def _decode_1019(body: bytes) -> dict[str, Any]:
    """GPS broadcast ephemeris (selected fields).

    The full 1019 message has ~30 fields. We parse the most commonly-used
    subset: SV id, week, IODE, IODC, sqrtA, eccentricity, M0, Toe.
    """
    bit = 12
    sat = _bits(body, bit, 6)
    bit += 6
    week = _bits(body, bit, 10)
    bit += 10
    bit += 4  # SV accuracy
    bit += 2  # CA/P on L2
    bit += 14  # IDOT
    iode = _bits(body, bit, 8)
    bit += 8
    toc = _bits(body, bit, 16) * 16
    bit += 16
    bit += 8 + 16 + 32  # af2, af1, af0
    iodc = _bits(body, bit, 10)
    bit += 10
    bit += 16 + 16 + 32  # Crs, Delta_n, M0
    bit += 16 + 32 + 16  # Cuc, Eccentricity, Cus
    sqrt_a_raw = _bits(body, bit, 32)
    bit += 32
    toe = _bits(body, bit, 16) * 16
    bit += 16
    return {
        "msg_id": 1019,
        "sv": f"G{sat:02d}",
        "week": week,
        "iode": iode,
        "iodc": iodc,
        "toc": toc,
        "toe": toe,
        "sqrtA": sqrt_a_raw * 2**-19,
    }


def _decode_1020(body: bytes) -> dict[str, Any]:
    """GLONASS broadcast ephemeris (selected fields)."""
    bit = 12
    slot = _bits(body, bit, 6)
    bit += 6
    chan = _bits(body, bit, 5, signed=True) - 7
    bit += 5
    return {"msg_id": 1020, "sv": f"R{slot:02d}", "freq_channel": chan}


def _decode_1004(body: bytes) -> dict[str, Any]:
    """Extended L1+L2 GPS RTK observations.

    We decode the message header (station, epoch time, sync, n_sat,
    smoothed/divergence-free indicators) and the per-satellite L1/L2
    fields (sat id, code, pseudorange, phase, lock time, ambiguity, CNR).

    Returns a dict ``{"msg_id": 1004, "station_id", "tow_ms", "n_sat",
    "satellites": [...]}``. Each satellite dict has ``sv``, ``L1_pr``,
    ``L1_phase``, ``L2_pr``, ``L2_phase``, etc. in standard SI units.
    """
    bit = 12
    sta_id = _bits(body, bit, 12)
    bit += 12
    tow_ms = _bits(body, bit, 30)
    bit += 30
    sync = _bits(body, bit, 1)
    bit += 1
    n_sat = _bits(body, bit, 5)
    bit += 5
    smooth = _bits(body, bit, 1)
    bit += 1
    smooth_iv = _bits(body, bit, 3)
    bit += 3

    sats = []
    for _ in range(n_sat):
        sat_id = _bits(body, bit, 6)
        bit += 6
        l1_code_ind = _bits(body, bit, 1)
        bit += 1
        l1_pr_raw = _bits(body, bit, 24)
        bit += 24
        l1_phase_diff = _bits(body, bit, 20, signed=True)
        bit += 20
        l1_lock = _bits(body, bit, 7)
        bit += 7
        l1_amb = _bits(body, bit, 8)
        bit += 8
        l1_cnr = _bits(body, bit, 8) * 0.25
        bit += 8
        l2_code_ind = _bits(body, bit, 2)
        bit += 2
        l2_pr_diff = _bits(body, bit, 14, signed=True)
        bit += 14
        l2_phase_diff = _bits(body, bit, 20, signed=True)
        bit += 20
        l2_lock = _bits(body, bit, 7)
        bit += 7
        l2_cnr = _bits(body, bit, 8) * 0.25
        bit += 8

        # Convert raw to SI per RTCM 3.x §3.5-3:
        # L1 pseudorange = (l1_pr_raw * 0.02 + l1_amb * 299792.458) m
        l1_pr_m = l1_pr_raw * 0.02 + l1_amb * 299_792.458
        l1_phase_m = l1_pr_m + l1_phase_diff * 0.0005
        l2_pr_m = l1_pr_m + l2_pr_diff * 0.02
        l2_phase_m = l1_pr_m + l2_phase_diff * 0.0005

        sats.append(
            {
                "sv": f"G{sat_id:02d}",
                "L1_code_ind": l1_code_ind,
                "L1_pr": l1_pr_m,
                "L1_phase": l1_phase_m,
                "L1_lock_time": l1_lock,
                "L1_cnr_dbhz": l1_cnr,
                "L2_code_ind": l2_code_ind,
                "L2_pr": l2_pr_m,
                "L2_phase": l2_phase_m,
                "L2_lock_time": l2_lock,
                "L2_cnr_dbhz": l2_cnr,
            }
        )

    return {
        "msg_id": 1004,
        "station_id": sta_id,
        "tow_ms": tow_ms,
        "sync": sync,
        "n_sat": n_sat,
        "smoothing_indicator": smooth,
        "smoothing_interval": smooth_iv,
        "satellites": sats,
    }


def _decode_1033(body: bytes) -> dict[str, Any]:
    """Receiver and antenna descriptor strings.

    Six length-prefixed ASCII strings: antenna descriptor, antenna
    serial, receiver type, receiver firmware, receiver serial.
    """
    bit = 12
    sta_id = _bits(body, bit, 12)
    bit += 12
    out = {"msg_id": 1033, "station_id": sta_id}

    def _read_str(field: str) -> None:
        nonlocal bit
        n = _bits(body, bit, 8)
        bit += 8
        chars = bytearray()
        for _ in range(n):
            chars.append(_bits(body, bit, 8))
            bit += 8
        out[field] = chars.decode("ascii", errors="ignore")

    _read_str("antenna_descriptor")
    bit += 8  # antenna setup ID
    _read_str("antenna_serial")
    _read_str("receiver_type")
    _read_str("receiver_firmware")
    _read_str("receiver_serial")
    return out


def _decode_msm_header(msg_id: int, body: bytes) -> dict[str, Any]:
    """Decode the MSM (Multiple Signal Message) header.

    All MSM7 message types (1077/1087/1097/1107/1117/1127/1137) share
    this header layout, then differ in the per-satellite signal block.
    We expose the SV mask and signal mask so the caller knows which
    constellations + signals were present.
    """
    bit = 12
    sta_id = _bits(body, bit, 12)
    bit += 12
    tow_ms = _bits(body, bit, 30)
    bit += 30
    sync = _bits(body, bit, 1)
    bit += 1
    iod = _bits(body, bit, 3)
    bit += 3
    bit += 7  # session time (reserved in MSM)
    bit += 2  # clock steering
    bit += 2  # external clock
    smooth = _bits(body, bit, 1)
    bit += 1
    smooth_iv = _bits(body, bit, 3)
    bit += 3
    # 64-bit satellite mask
    sv_mask_hi = _bits(body, bit, 32)
    bit += 32
    sv_mask_lo = _bits(body, bit, 32)
    bit += 32
    sv_mask = (sv_mask_hi << 32) | sv_mask_lo
    # 32-bit signal mask
    sig_mask = _bits(body, bit, 32)
    bit += 32
    n_sv = bin(sv_mask).count("1")
    n_sig = bin(sig_mask).count("1")

    return {
        "msg_id": msg_id,
        "station_id": sta_id,
        "tow_ms": tow_ms,
        "sync": sync,
        "iod": iod,
        "smoothing_indicator": smooth,
        "smoothing_interval": smooth_iv,
        "sv_mask": sv_mask,
        "signal_mask": sig_mask,
        "n_sv": n_sv,
        "n_sig": n_sig,
    }


__all__ = ["PREAMBLE", "crc24q", "decode_message", "iter_messages"]
