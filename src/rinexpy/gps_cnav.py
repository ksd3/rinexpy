"""GPS CNAV (L2C / L5) navigation message decoder.

CNAV (Civil NAV) is the modernized GPS broadcast format used on L2C
and L5. It transmits at 50 sps (L2C) / 100 sps (L5) and carries the
ephemeris in *messages* rather than the LNAV-style subframes:

Message types of interest (per ICD-GPS-200 §30.3):
- MT 10: clock + ephemeris (preamble + sat health + ephemeris part 1)
- MT 11: ephemeris part 2
- MT 30: clock + iono / UTC
- MT 31: clock + reduced almanac
- MT 32-35: ground correction, UTC, EOP, differential corrections
- MT 36: text message
- MT 37: midi almanac
- MT 0: dummy

Each CNAV message is 300 bits / 6 seconds; 276 bits of payload + 24
bits of CRC-24Q. The first 24 bits are the message header: 8-bit
preamble (0x8B), 6-bit PRN, 6-bit msg type ID, 17-bit TOW (in 6s
units), 1-bit alert. After that, a per-message-type payload.

This module implements the ephemeris (MT 10 + MT 11) decode used by
RXM-SFRBX / SBF / similar capture streams. Other MTs return raw
payload bytes for now.

Reference: IS-GPS-200 §30.3 / §40 (CNAV / CNAV-2).
"""

from __future__ import annotations

from typing import Any

#: CNAV preamble byte.
PREAMBLE = 0x8B


def _bits(data: bytes, start: int, n: int, *, signed: bool = False) -> int:
    """MSB-first bit extraction (delegates to the C++ kernel when
    rinexpy_native is present, else a Python fallback)."""
    try:
        from . import _native
        if _native.have_read_bits():
            return _native.read_bits(bytes(data), start, n, signed)
    except Exception:
        pass
    value = 0
    for i in range(n):
        byte_idx, bit_idx = divmod(start + i, 8)
        value = (value << 1) | ((data[byte_idx] >> (7 - bit_idx)) & 1)
    if signed and (value >> (n - 1)) & 1:
        value -= 1 << n
    return value


def _decode_cnav_header(payload: bytes) -> dict[str, Any]:
    """Decode the common 24-bit CNAV header at the start of every
    CNAV message body."""
    pre = _bits(payload, 0, 8)
    if pre != PREAMBLE:
        raise ValueError(f"bad CNAV preamble: {pre:#04x}")
    prn = _bits(payload, 8, 6)
    msg_id = _bits(payload, 14, 6)
    tow_count_6s = _bits(payload, 20, 17)
    alert = _bits(payload, 37, 1)
    return {
        "prn": prn,
        "msg_id": msg_id,
        "tow_count_s": tow_count_6s * 6,
        "alert": bool(alert),
    }


def decode_cnav_mt10(payload: bytes) -> dict[str, Any]:
    """Decode CNAV MT 10 — clock + ephemeris part 1.

    Returns a dict with: week, ura_index, sv health, t_op, t_oe,
    Delta_A, Adot, Delta_n0, Delta_n0_dot, M0_n, e_n, omega_n, plus
    the inherited header fields.
    """
    out = _decode_cnav_header(payload)
    bit = 38
    week = _bits(payload, bit, 13); bit += 13
    health_l1 = _bits(payload, bit, 1); bit += 1
    health_l2 = _bits(payload, bit, 1); bit += 1
    health_l5 = _bits(payload, bit, 1); bit += 1
    ura_index = _bits(payload, bit, 5, signed=True); bit += 5
    t_op_s = _bits(payload, bit, 11) * 300; bit += 11
    t_oe_s = _bits(payload, bit, 11) * 300; bit += 11
    # Delta_A from semi-major axis reference (a_ref = 26559710 m).
    delta_a_m = _bits(payload, bit, 26, signed=True) * 2 ** -9; bit += 26
    Adot_m_s = _bits(payload, bit, 25, signed=True) * 2 ** -21; bit += 25
    delta_n0 = _bits(payload, bit, 17, signed=True) * 2 ** -44; bit += 17
    delta_n0_dot = _bits(payload, bit, 23, signed=True) * 2 ** -57; bit += 23
    m0_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    e_n = _bits(payload, bit, 33) * 2 ** -34; bit += 33
    omega_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    out.update({
        "week": week,
        "SV_health_L1": health_l1,
        "SV_health_L2": health_l2,
        "SV_health_L5": health_l5,
        "URA_index": ura_index,
        "t_op_s": t_op_s,
        "t_oe_s": t_oe_s,
        "delta_A_m": delta_a_m,
        "Adot_m_per_s": Adot_m_s,
        "delta_n0_semicircles_per_s": delta_n0,
        "delta_n0_dot_semicircles_per_s2": delta_n0_dot,
        "M_0_n_semicircles": m0_n,
        "e_n": e_n,
        "omega_n_semicircles": omega_n,
    })
    return out


def decode_cnav_mt11(payload: bytes) -> dict[str, Any]:
    """Decode CNAV MT 11 — ephemeris part 2 (Omega0, i0, Delta-Omega-
    dot, etc.)."""
    out = _decode_cnav_header(payload)
    bit = 38
    bit += 11  # t_oe (redundant with MT10)
    omega_0_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    i_0_n = _bits(payload, bit, 33, signed=True) * 2 ** -32; bit += 33
    delta_omega_dot = _bits(payload, bit, 17, signed=True) * 2 ** -44; bit += 17
    idot = _bits(payload, bit, 15, signed=True) * 2 ** -44; bit += 15
    cis = _bits(payload, bit, 16, signed=True) * 2 ** -30; bit += 16
    cic = _bits(payload, bit, 16, signed=True) * 2 ** -30; bit += 16
    crs = _bits(payload, bit, 24, signed=True) * 2 ** -8; bit += 24
    crc = _bits(payload, bit, 24, signed=True) * 2 ** -8; bit += 24
    cus = _bits(payload, bit, 21, signed=True) * 2 ** -30; bit += 21
    cuc = _bits(payload, bit, 21, signed=True) * 2 ** -30; bit += 21
    out.update({
        "Omega_0_n_semicircles": omega_0_n,
        "i_0_n_semicircles": i_0_n,
        "delta_Omega_dot_semicircles_per_s": delta_omega_dot,
        "IDOT_semicircles_per_s": idot,
        "C_is_rad": cis,
        "C_ic_rad": cic,
        "C_rs_m": crs,
        "C_rc_m": crc,
        "C_us_rad": cus,
        "C_uc_rad": cuc,
    })
    return out


def decode_cnav_message(payload: bytes) -> dict[str, Any]:
    """Dispatch a CNAV message to the appropriate decoder by header
    msg type. Unknown types come back as ``{"header": ..., "raw":
    <bytes>}``."""
    hdr = _decode_cnav_header(payload)
    mid = hdr["msg_id"]
    if mid == 10:
        return decode_cnav_mt10(payload)
    if mid == 11:
        return decode_cnav_mt11(payload)
    return {"header": hdr, "raw": bytes(payload[5:])}


__all__ = [
    "PREAMBLE",
    "decode_cnav_message",
    "decode_cnav_mt10",
    "decode_cnav_mt11",
]
