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

from . import _native

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

    Dispatches to the optional :mod:`rinexpy_native` C++ kernel (~150x
    faster than this fallback) when the extension is importable; otherwise
    uses the pure-Python loop below. The numerical contract is identical.
    """
    if _native.have_crc24q():
        return _native.crc24q(bytes(data))
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
        1029: _decode_1029,
        1033: _decode_1033,
        1042: _decode_1042,
        1045: _decode_1045,
        1046: _decode_1046,
        1230: _decode_1230,
        1057: _decode_1057,
        1058: _decode_1058,
    }
    if msg_id in decoders:
        return decoders[msg_id](body)
    # MSM4 messages: 1074=GPS, 1084=GLO, 1094=GAL, 1104=SBAS, 1114=QZSS,
    # 1124=BDS, 1134=NavIC. Reduced-precision per-cell layout (no Doppler).
    if msg_id in {1074, 1084, 1094, 1104, 1114, 1124, 1134}:
        return _decode_msm_header(msg_id, body, msm_kind=4)
    # MSM7 messages: 1077=GPS, 1087=GLO, 1097=GAL, 1107=SBAS, 1117=QZSS,
    # 1127=BDS, 1137=NavIC. Full-precision per-cell layout.
    if msg_id in {1077, 1087, 1097, 1107, 1117, 1127, 1137}:
        return _decode_msm_header(msg_id, body, msm_kind=7)
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


_PI = 3.1415926535898


def _decode_1042(body: bytes) -> dict[str, Any]:
    """BeiDou broadcast ephemeris (selected fields).

    RTCM 10403.3 message type 1042. We extract the standard set most
    downstream Keplerian-to-ECEF code needs (sv, week, AODE, Toc, Toe,
    M_0, e, sqrt(A), C_rs / C_uc / C_us / etc.) plus the TGD pair.
    """
    bit = 12
    sat = _bits(body, bit, 6); bit += 6
    week = _bits(body, bit, 13); bit += 13
    urai = _bits(body, bit, 4); bit += 4
    idot = _bits(body, bit, 14, signed=True) * 2**-43 * _PI
    bit += 14
    aode = _bits(body, bit, 5); bit += 5
    toc = _bits(body, bit, 17) * 8; bit += 17
    a2 = _bits(body, bit, 11, signed=True) * 2**-66; bit += 11
    a1 = _bits(body, bit, 22, signed=True) * 2**-50; bit += 22
    a0 = _bits(body, bit, 24, signed=True) * 2**-33; bit += 24
    aodc = _bits(body, bit, 5); bit += 5
    crs = _bits(body, bit, 18, signed=True) * 2**-6; bit += 18
    delta_n = _bits(body, bit, 16, signed=True) * 2**-43 * _PI; bit += 16
    m0 = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    cuc = _bits(body, bit, 18, signed=True) * 2**-31; bit += 18
    e = _bits(body, bit, 32) * 2**-33; bit += 32
    cus = _bits(body, bit, 18, signed=True) * 2**-31; bit += 18
    sqrt_a = _bits(body, bit, 32) * 2**-19; bit += 32
    toe = _bits(body, bit, 17) * 8; bit += 17
    cic = _bits(body, bit, 18, signed=True) * 2**-31; bit += 18
    omega0 = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    cis = _bits(body, bit, 18, signed=True) * 2**-31; bit += 18
    i0 = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    crc = _bits(body, bit, 18, signed=True) * 2**-6; bit += 18
    omega = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    omega_dot = _bits(body, bit, 24, signed=True) * 2**-43 * _PI; bit += 24
    tgd1 = _bits(body, bit, 10, signed=True) * 0.1e-9; bit += 10
    tgd2 = _bits(body, bit, 10, signed=True) * 0.1e-9; bit += 10
    sv_health = _bits(body, bit, 1); bit += 1

    return {
        "msg_id": 1042,
        "sv": f"C{sat:02d}",
        "week": week,
        "URAI": urai,
        "AODE": aode,
        "AODC": aodc,
        "t_oc_s": toc,
        "t_oe_s": toe,
        "a_f0_s": a0,
        "a_f1_s_per_s": a1,
        "a_f2_s_per_s2": a2,
        "C_rs_m": crs,
        "C_rc_m": crc,
        "C_uc_rad": cuc,
        "C_us_rad": cus,
        "C_ic_rad": cic,
        "C_is_rad": cis,
        "delta_n_rad_s": delta_n,
        "M_0_rad": m0,
        "e": e,
        "sqrt_A_root_m": sqrt_a,
        "Omega_0_rad": omega0,
        "i_0_rad": i0,
        "omega_rad": omega,
        "Omega_dot_rad_s": omega_dot,
        "IDOT_rad_s": idot,
        "TGD1_s": tgd1,
        "TGD2_s": tgd2,
        "SV_health": sv_health,
    }


def _decode_galileo_eph_common(body: bytes, msg_id: int, sv_letter: str) -> dict[str, Any]:
    """Shared decoder for RTCM 1045 (Galileo F/NAV) and 1046 (Galileo I/NAV).

    The two messages have almost identical layouts; the F/NAV variant
    has BGD_E1E5a and an OSHS / OSDVS flag, while the I/NAV variant has
    BGD_E1E5a, BGD_E1E5b, and the E5b / E1B data validity / signal
    health bits.
    """
    bit = 12
    sat = _bits(body, bit, 6); bit += 6
    week = _bits(body, bit, 12); bit += 12
    iodnav = _bits(body, bit, 10); bit += 10
    sisa = _bits(body, bit, 8); bit += 8
    idot = _bits(body, bit, 14, signed=True) * 2**-43 * _PI; bit += 14
    toc = _bits(body, bit, 14) * 60; bit += 14
    af2 = _bits(body, bit, 6, signed=True) * 2**-59; bit += 6
    af1 = _bits(body, bit, 21, signed=True) * 2**-46; bit += 21
    af0 = _bits(body, bit, 31, signed=True) * 2**-34; bit += 31
    crs = _bits(body, bit, 16, signed=True) * 2**-5; bit += 16
    delta_n = _bits(body, bit, 16, signed=True) * 2**-43 * _PI; bit += 16
    m0 = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    cuc = _bits(body, bit, 16, signed=True) * 2**-29; bit += 16
    e = _bits(body, bit, 32) * 2**-33; bit += 32
    cus = _bits(body, bit, 16, signed=True) * 2**-29; bit += 16
    sqrt_a = _bits(body, bit, 32) * 2**-19; bit += 32
    toe = _bits(body, bit, 14) * 60; bit += 14
    cic = _bits(body, bit, 16, signed=True) * 2**-29; bit += 16
    omega0 = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    cis = _bits(body, bit, 16, signed=True) * 2**-29; bit += 16
    i0 = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    crc = _bits(body, bit, 16, signed=True) * 2**-5; bit += 16
    omega = _bits(body, bit, 32, signed=True) * 2**-31 * _PI; bit += 32
    omega_dot = _bits(body, bit, 24, signed=True) * 2**-43 * _PI; bit += 24
    bgd_e1e5a = _bits(body, bit, 10, signed=True) * 2**-32; bit += 10

    out: dict[str, Any] = {
        "msg_id": msg_id,
        "sv": f"{sv_letter}{sat:02d}",
        "week": week,
        "IODnav": iodnav,
        "SISA": sisa,
        "t_oc_s": toc,
        "t_oe_s": toe,
        "a_f0_s": af0,
        "a_f1_s_per_s": af1,
        "a_f2_s_per_s2": af2,
        "C_rs_m": crs,
        "C_rc_m": crc,
        "C_uc_rad": cuc,
        "C_us_rad": cus,
        "C_ic_rad": cic,
        "C_is_rad": cis,
        "delta_n_rad_s": delta_n,
        "M_0_rad": m0,
        "e": e,
        "sqrt_A_root_m": sqrt_a,
        "Omega_0_rad": omega0,
        "i_0_rad": i0,
        "omega_rad": omega,
        "Omega_dot_rad_s": omega_dot,
        "IDOT_rad_s": idot,
        "BGD_E1E5a_s": bgd_e1e5a,
    }
    if msg_id == 1046:
        # I/NAV adds a second BGD plus the E5b/E1B health flags.
        bgd_e1e5b = _bits(body, bit, 10, signed=True) * 2**-32; bit += 10
        out["BGD_E1E5b_s"] = bgd_e1e5b
        out["E5b_DVS"] = _bits(body, bit, 1); bit += 1
        out["E5b_HS"] = _bits(body, bit, 2); bit += 2
        out["E1B_DVS"] = _bits(body, bit, 1); bit += 1
        out["E1B_HS"] = _bits(body, bit, 2); bit += 2
    else:
        out["OSHS"] = _bits(body, bit, 2); bit += 2
        out["OSDVS"] = _bits(body, bit, 1); bit += 1
    return out


def _decode_1045(body: bytes) -> dict[str, Any]:
    """Galileo F/NAV broadcast ephemeris (E5a-based)."""
    return _decode_galileo_eph_common(body, 1045, "E")


def _decode_1046(body: bytes) -> dict[str, Any]:
    """Galileo I/NAV broadcast ephemeris (E1B + E5b)."""
    return _decode_galileo_eph_common(body, 1046, "E")


def _decode_1029(body: bytes) -> dict[str, Any]:
    """Unicode (UTF-8) text message.

    Free-form station-to-rover text, sent at arbitrary cadence.
    """
    bit = 12
    sta_id = _bits(body, bit, 12)
    bit += 12
    mjd = _bits(body, bit, 16)
    bit += 16
    sod = _bits(body, bit, 17)
    bit += 17
    n_chars = _bits(body, bit, 7)
    bit += 7
    n_bytes = _bits(body, bit, 8)
    bit += 8
    raw = bytearray()
    for _ in range(n_bytes):
        raw.append(_bits(body, bit, 8))
        bit += 8
    return {
        "msg_id": 1029,
        "station_id": sta_id,
        "mjd": mjd,
        "sod_s": sod,
        "n_chars": n_chars,
        "n_bytes": n_bytes,
        "text": raw.decode("utf-8", errors="replace"),
    }


def _decode_1230(body: bytes) -> dict[str, Any]:
    """GLONASS L1/L2 code-phase biases.

    Aligns GLONASS code measurements between receivers from different
    manufacturers, which is needed before a multi-vendor RTK fix on
    GLONASS will close. Decodes the 4-bit signal mask and the per-signal
    16-bit signed bias scaled by 0.02 m.
    """
    bit = 12
    sta_id = _bits(body, bit, 12)
    bit += 12
    bias_indicator = _bits(body, bit, 1)
    bit += 1
    bit += 3  # reserved
    mask = _bits(body, bit, 4)
    bit += 4

    biases_m: dict[str, float] = {}
    if mask & 0b1000:
        biases_m["L1_CA"] = _bits(body, bit, 16, signed=True) * 0.02
        bit += 16
    if mask & 0b0100:
        biases_m["L1_P"] = _bits(body, bit, 16, signed=True) * 0.02
        bit += 16
    if mask & 0b0010:
        biases_m["L2_CA"] = _bits(body, bit, 16, signed=True) * 0.02
        bit += 16
    if mask & 0b0001:
        biases_m["L2_P"] = _bits(body, bit, 16, signed=True) * 0.02
        bit += 16

    return {
        "msg_id": 1230,
        "station_id": sta_id,
        "bias_indicator": bias_indicator,
        "signal_mask": mask,
        "biases_m": biases_m,
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


_MSM_C = 299_792.458  # speed of light in m/ms


def _decode_msm_header(msg_id: int, body: bytes, *, msm_kind: int = 7) -> dict[str, Any]:
    """Decode an MSM4 or MSM7 message: header + per-satellite + per-cell blocks.

    All MSM message types share the header. MSM4 (1074/1084/.../1134)
    uses a reduced-precision per-cell layout (15+22+4+1+6 = 48 bits per
    cell, no Doppler). MSM7 (1077/1087/.../1137) uses the full-precision
    layout (20+24+10+1+10+15 = 80 bits per cell).

    The output dict has the header fields plus a ``satellites`` list
    (one dict per SV in the SV mask) and an ``observations`` list (one
    dict per cell present in the cell mask). All observations are in
    SI units regardless of MSM kind.
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
    bit += 7  # session time
    bit += 2  # clock steering
    bit += 2  # external clock
    smooth = _bits(body, bit, 1)
    bit += 1
    smooth_iv = _bits(body, bit, 3)
    bit += 3
    sv_mask_hi = _bits(body, bit, 32)
    bit += 32
    sv_mask_lo = _bits(body, bit, 32)
    bit += 32
    sv_mask = (sv_mask_hi << 32) | sv_mask_lo
    sig_mask = _bits(body, bit, 32)
    bit += 32

    sv_indices = [i for i in range(64) if (sv_mask >> (63 - i)) & 1]
    sig_indices = [i for i in range(32) if (sig_mask >> (31 - i)) & 1]
    n_sv = len(sv_indices)
    n_sig = len(sig_indices)

    out: dict[str, Any] = {
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
        "sv_indices": sv_indices,
        "signal_indices": sig_indices,
    }

    # Cell mask follows: n_sv * n_sig bits.
    n_cells = n_sv * n_sig
    if bit + n_cells > 8 * len(body):
        out["payload_truncated"] = True
        return out
    cell_mask_bits = [_bits(body, bit + i, 1) for i in range(n_cells)]
    bit += n_cells
    out["cell_mask"] = cell_mask_bits

    # Per-satellite block (MSM7): for each SV, 8+4+10+14 = 36 bits.
    sats: list[dict[str, Any]] = []
    sat_letter = _MSM_SYSTEM_LETTER.get(msg_id, "?")
    if bit + 36 * n_sv > 8 * len(body):
        out["payload_truncated"] = True
        return out
    for sv_idx in sv_indices:
        rough_int_ms = _bits(body, bit, 8)
        bit += 8
        ext_info = _bits(body, bit, 4)
        bit += 4
        rough_mod_1ms = _bits(body, bit, 10)
        bit += 10
        rough_doppler = _bits(body, bit, 14, signed=True)
        bit += 14
        sats.append(
            {
                "sv": f"{sat_letter}{sv_idx + 1:02d}",
                "rough_range_ms": rough_int_ms + rough_mod_1ms / 1024.0,
                "extended_info": ext_info,
                "rough_doppler_mps": rough_doppler,
            }
        )
    out["satellites"] = sats

    # Per-cell signal block.
    bits_per_cell = 80 if msm_kind == 7 else 48
    n_present = sum(cell_mask_bits)
    if bit + bits_per_cell * n_present > 8 * len(body):
        out["payload_truncated"] = True
        return out

    observations: list[dict[str, Any]] = []
    for cell_idx, present in enumerate(cell_mask_bits):
        if not present:
            continue
        sv_k = cell_idx // n_sig
        sig_k = cell_idx % n_sig
        sv_label = sats[sv_k]["sv"]
        rough_ms = sats[sv_k]["rough_range_ms"]

        if msm_kind == 7:
            fine_pr = _bits(body, bit, 20, signed=True)
            bit += 20
            fine_phase = _bits(body, bit, 24, signed=True)
            bit += 24
            lock = _bits(body, bit, 10)
            bit += 10
            halfcyc = _bits(body, bit, 1)
            bit += 1
            cnr = _bits(body, bit, 10) / 16.0
            bit += 10
            fine_dop = _bits(body, bit, 15, signed=True)
            bit += 15
            # Fine PR scale: 2^-29 ms; fine phase scale: 2^-31 ms.
            pr_m = (rough_ms + fine_pr * 2**-29) * _MSM_C
            phase_m = (rough_ms + fine_phase * 2**-31) * _MSM_C
            doppler_mps = fine_dop * 1e-4
        else:
            # MSM4: 15+22+4+1+6 bits, no fine Doppler.
            fine_pr = _bits(body, bit, 15, signed=True)
            bit += 15
            fine_phase = _bits(body, bit, 22, signed=True)
            bit += 22
            lock = _bits(body, bit, 4)
            bit += 4
            halfcyc = _bits(body, bit, 1)
            bit += 1
            cnr = float(_bits(body, bit, 6))
            bit += 6
            # MSM4 PR scale: 2^-24 ms; phase scale: 2^-29 ms.
            pr_m = (rough_ms + fine_pr * 2**-24) * _MSM_C
            phase_m = (rough_ms + fine_phase * 2**-29) * _MSM_C
            doppler_mps = float("nan")

        observations.append(
            {
                "sv": sv_label,
                "signal_index": sig_indices[sig_k],
                "pseudorange_m": pr_m,
                "phase_m": phase_m,
                "lock_time": lock,
                "half_cycle_ambiguity": halfcyc,
                "cnr_dbhz": cnr,
                "doppler_mps": doppler_mps,
            }
        )
    out["observations"] = observations
    return out


_MSM_SYSTEM_LETTER: dict[int, str] = {
    # MSM4
    1074: "G",
    1084: "R",
    1094: "E",
    1104: "S",
    1114: "J",
    1124: "C",
    1134: "I",
    # MSM7
    1077: "G",
    1087: "R",
    1097: "E",
    1107: "S",
    1117: "J",
    1127: "C",
    1137: "I",
}


def _decode_ssr_header(body: bytes, *, has_datum: bool) -> tuple[dict[str, Any], int, int]:
    """Decode the common SSR header. Returns (header, bit_cursor, n_sats).

    The 1057-style orbit message has an extra 1-bit ``satellite reference
    datum`` field after the IOD SSR; clock-only / bias-only messages
    omit it. The shared layout is: DF002 (12 bits, already passed as
    msg_id), DF385 epoch time (20 bits), DF391 update interval (4 bits),
    DF388 multiple-message indicator (1 bit), DF394 IOD SSR (4 bits),
    DF395 provider ID (16 bits), DF396 solution ID (4 bits), then the
    optional 1-bit reference datum, then DF387 number of satellites
    (6 bits).
    """
    bit = 12
    epoch_time = _bits(body, bit, 20); bit += 20
    update_interval = _bits(body, bit, 4); bit += 4
    multiple_msg = _bits(body, bit, 1); bit += 1
    iod_ssr = _bits(body, bit, 4); bit += 4
    provider_id = _bits(body, bit, 16); bit += 16
    solution_id = _bits(body, bit, 4); bit += 4
    if has_datum:
        ref_datum = _bits(body, bit, 1); bit += 1
    else:
        ref_datum = None
    n_sats = _bits(body, bit, 6); bit += 6
    header = {
        "epoch_time_s": epoch_time,
        "update_interval_index": update_interval,
        "multiple_message": bool(multiple_msg),
        "iod_ssr": iod_ssr,
        "provider_id": provider_id,
        "solution_id": solution_id,
        "ref_datum": ref_datum,
        "n_sats": n_sats,
    }
    return header, bit, n_sats


def _decode_1057(body: bytes) -> dict[str, Any]:
    """SSR GPS orbit corrections (RTCM 3.x message 1057).

    Returns the common SSR header plus a per-satellite list of:

    - ``prn`` (1-32 GPS PRN)
    - ``iode``: GPS IODE this correction applies to
    - ``delta_radial_m``, ``delta_along_track_m``, ``delta_cross_track_m``
    - ``dot_delta_radial_m_per_s``, ``dot_delta_along_track_m_per_s``,
      ``dot_delta_cross_track_m_per_s``

    Scaling per the RTCM 3.x spec: delta radial is 0.1 mm/LSB (22 bits
    signed), the two transverse deltas are 0.4 mm/LSB (20 bits signed),
    and their rate counterparts are 0.001 mm/s and 0.004 mm/s
    respectively.
    """
    header, bit, n_sats = _decode_ssr_header(body, has_datum=True)
    sats: list[dict[str, Any]] = []
    for _ in range(n_sats):
        prn = _bits(body, bit, 6); bit += 6
        iode = _bits(body, bit, 8); bit += 8
        d_radial = _bits(body, bit, 22, signed=True); bit += 22
        d_along = _bits(body, bit, 20, signed=True); bit += 20
        d_cross = _bits(body, bit, 20, signed=True); bit += 20
        dot_radial = _bits(body, bit, 21, signed=True); bit += 21
        dot_along = _bits(body, bit, 19, signed=True); bit += 19
        dot_cross = _bits(body, bit, 19, signed=True); bit += 19
        sats.append({
            "prn": prn,
            "iode": iode,
            "delta_radial_m": d_radial * 1e-4,
            "delta_along_track_m": d_along * 4e-4,
            "delta_cross_track_m": d_cross * 4e-4,
            "dot_delta_radial_m_per_s": dot_radial * 1e-6,
            "dot_delta_along_track_m_per_s": dot_along * 4e-6,
            "dot_delta_cross_track_m_per_s": dot_cross * 4e-6,
        })
    return {"msg_id": 1057, "header": header, "satellites": sats}


def _decode_1058(body: bytes) -> dict[str, Any]:
    """SSR GPS clock corrections (RTCM 3.x message 1058).

    Returns the common SSR header plus a per-satellite list of:

    - ``prn``
    - ``c0_m``: constant clock correction in meters (22 bits signed,
      0.1 mm/LSB)
    - ``c1_m_per_s``: linear rate (21 bits signed, 0.001 mm/s/LSB)
    - ``c2_m_per_s2``: quadratic acceleration (27 bits signed,
      2e-5 mm/s^2 LSB)

    The polynomial gives the satellite-clock correction at time t as
    ``c0 + c1 * (t - t_0) + c2 * (t - t_0)^2`` (in meters of range).
    """
    header, bit, n_sats = _decode_ssr_header(body, has_datum=False)
    sats: list[dict[str, Any]] = []
    for _ in range(n_sats):
        prn = _bits(body, bit, 6); bit += 6
        c0 = _bits(body, bit, 22, signed=True); bit += 22
        c1 = _bits(body, bit, 21, signed=True); bit += 21
        c2 = _bits(body, bit, 27, signed=True); bit += 27
        sats.append({
            "prn": prn,
            "c0_m": c0 * 1e-4,
            "c1_m_per_s": c1 * 1e-6,
            "c2_m_per_s2": c2 * 2e-8,
        })
    return {"msg_id": 1058, "header": header, "satellites": sats}


__all__ = ["PREAMBLE", "crc24q", "decode_message", "iter_messages"]
