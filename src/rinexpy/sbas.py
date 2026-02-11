"""SBAS L1 (WAAS / EGNOS / MSAS / GAGAN) message decoder.

Each L1 SBAS message is 250 bits / 1 second (250 sps after FEC, 500 sps
on-air before FEC). The on-air layout is:

    [8-bit preamble] [6-bit MT] [212-bit payload] [24-bit CRC-24Q]
    \\_____________________ 250 bits _______________________/

The 8-bit preamble cycles through 0x53, 0x9A, 0xC6 across consecutive
messages, forming a 24-bit sync pattern across each 3-second block.

This module decodes the bit-fields for the dominant SBAS message types
(MT 1, 2-5, 6, 7, 9, 17, 18, 24, 25, 26). Unknown types come back with
``{"header", "raw"}`` so callers can dispatch further.

The decoder takes ``payload`` as the 250-bit message packed MSB-first
into 32 bytes (last byte holds 2 message bits + 6 zero pad). Most
real-world SBAS capture sources (NovAtel, u-blox RXM-SFRBX) deliver
the bits in this layout.

Reference: RTCA DO-229E §A.4 (Message Types and Formats).
"""

from __future__ import annotations

from typing import Any

from .gps_cnav import _bits

#: SBAS preamble bytes cycle through these three values.
PREAMBLES = (0x53, 0x9A, 0xC6)


def _decode_header(payload: bytes) -> dict[str, Any]:
    pre = _bits(payload, 0, 8)
    mt = _bits(payload, 8, 6)
    return {"preamble": pre, "msg_type": mt}


def decode_sbas_mt1(payload: bytes) -> dict[str, Any]:
    """MT 1: PRN mask assignment. Returns the active 210-bit mask plus
    the 2-bit IODP."""
    out = _decode_header(payload)
    bit = 14
    mask: list[int] = []
    for i in range(210):
        mask.append(_bits(payload, bit + i, 1))
    bit += 210
    iodp = _bits(payload, bit, 2)
    out["mask"] = mask
    out["IODP"] = iodp
    out["active_prns"] = [i + 1 for i, b in enumerate(mask) if b]
    return out


def decode_sbas_mt2_5(payload: bytes) -> dict[str, Any]:
    """MT 2-5: Fast pseudorange corrections (13 PRCs + 13 UDREIs)."""
    out = _decode_header(payload)
    bit = 14
    iodf_i = _bits(payload, bit, 2); bit += 2
    iodp = _bits(payload, bit, 2); bit += 2
    prc = []
    for _ in range(13):
        prc.append(_bits(payload, bit, 12, signed=True) * 0.125)
        bit += 12
    udrei = []
    for _ in range(13):
        udrei.append(_bits(payload, bit, 4)); bit += 4
    out.update({
        "IODF_i": iodf_i,
        "IODP": iodp,
        "PRC_m": prc,
        "UDREI": udrei,
    })
    return out


def decode_sbas_mt6(payload: bytes) -> dict[str, Any]:
    """MT 6: Integrity information - UDREIs for the 51 currently masked
    fast-corrected SVs, plus the 4 IODF_j tags."""
    out = _decode_header(payload)
    bit = 14
    iodfs = [_bits(payload, bit + 2 * j, 2) for j in range(4)]
    bit += 8
    udrei = []
    for _ in range(51):
        udrei.append(_bits(payload, bit, 4))
        bit += 4
    out["IODF"] = iodfs
    out["UDREI"] = udrei
    return out


def decode_sbas_mt7(payload: bytes) -> dict[str, Any]:
    """MT 7: Fast correction degradation factors (51 × 4-bit a_i_index)."""
    out = _decode_header(payload)
    bit = 14
    sys_latency = _bits(payload, bit, 4); bit += 4
    iodp = _bits(payload, bit, 2); bit += 2
    bit += 2  # spare
    ai = [_bits(payload, bit + 4 * j, 4) for j in range(51)]
    out.update({
        "system_latency_s": sys_latency,
        "IODP": iodp,
        "ai_index": ai,
    })
    return out


def decode_sbas_mt9(payload: bytes) -> dict[str, Any]:
    """MT 9: GEO navigation message - ranging function for the GEO SV.
    Returns t_0, URA, ECEF position, velocity, acceleration, clock
    offset a_Gf0 and drift a_Gf1."""
    out = _decode_header(payload)
    bit = 14
    bit += 8  # reserved data ID
    t_0 = _bits(payload, bit, 13) * 16; bit += 13
    ura = _bits(payload, bit, 4); bit += 4
    x = _bits(payload, bit, 30, signed=True) * 0.08; bit += 30
    y = _bits(payload, bit, 30, signed=True) * 0.08; bit += 30
    z = _bits(payload, bit, 25, signed=True) * 0.4; bit += 25
    xv = _bits(payload, bit, 17, signed=True) * 0.000625; bit += 17
    yv = _bits(payload, bit, 17, signed=True) * 0.000625; bit += 17
    zv = _bits(payload, bit, 18, signed=True) * 0.004; bit += 18
    xa = _bits(payload, bit, 10, signed=True) * 0.0000125; bit += 10
    ya = _bits(payload, bit, 10, signed=True) * 0.0000125; bit += 10
    za = _bits(payload, bit, 10, signed=True) * 0.0000625; bit += 10
    a_gf0 = _bits(payload, bit, 12, signed=True) * 2 ** -31; bit += 12
    a_gf1 = _bits(payload, bit, 8, signed=True) * 2 ** -40; bit += 8
    out.update({
        "t_0_s": t_0,
        "URA": ura,
        "x_m": x, "y_m": y, "z_m": z,
        "x_dot_m_s": xv, "y_dot_m_s": yv, "z_dot_m_s": zv,
        "x_ddot_m_s2": xa, "y_ddot_m_s2": ya, "z_ddot_m_s2": za,
        "a_Gf0_s": a_gf0,
        "a_Gf1_s_per_s": a_gf1,
    })
    return out


def decode_sbas_mt17(payload: bytes) -> dict[str, Any]:
    """MT 17: GEO satellite almanacs (up to 3 GEO entries + t_0)."""
    out = _decode_header(payload)
    bit = 14
    entries = []
    for _ in range(3):
        bit += 2          # reserved data ID
        prn = _bits(payload, bit, 8); bit += 8
        health = _bits(payload, bit, 8); bit += 8
        x_g = _bits(payload, bit, 15, signed=True) * 2600.0; bit += 15
        y_g = _bits(payload, bit, 15, signed=True) * 2600.0; bit += 15
        z_g = _bits(payload, bit, 9, signed=True) * 26000.0; bit += 9
        xd = _bits(payload, bit, 3, signed=True) * 10.0; bit += 3
        yd = _bits(payload, bit, 3, signed=True) * 10.0; bit += 3
        zd = _bits(payload, bit, 4, signed=True) * 60.0; bit += 4
        entries.append({
            "PRN": prn,
            "health": health,
            "x_m": x_g, "y_m": y_g, "z_m": z_g,
            "x_dot_m_s": xd, "y_dot_m_s": yd, "z_dot_m_s": zd,
        })
    t_0 = _bits(payload, bit, 11) * 64
    out["entries"] = entries
    out["t_0_s"] = t_0
    return out


def decode_sbas_mt18(payload: bytes) -> dict[str, Any]:
    """MT 18: Ionospheric grid point mask for one band."""
    out = _decode_header(payload)
    bit = 14
    n_bands = _bits(payload, bit, 4); bit += 4
    band = _bits(payload, bit, 4); bit += 4
    iodi = _bits(payload, bit, 2); bit += 2
    mask = [_bits(payload, bit + i, 1) for i in range(201)]
    out.update({
        "N_bands": n_bands,
        "band": band,
        "IODI": iodi,
        "mask": mask,
        "active_igps": [i + 1 for i, b in enumerate(mask) if b],
    })
    return out


def decode_sbas_mt24(payload: bytes) -> dict[str, Any]:
    """MT 24: Mixed fast + long-term corrections.

    Carries 6 PRCs (instead of MT 2-5's 13), then a 106-bit MT 25 half
    block which is left as a raw bitstring for downstream MT-25 reuse.
    """
    out = _decode_header(payload)
    bit = 14
    prc = []
    for _ in range(6):
        prc.append(_bits(payload, bit, 12, signed=True) * 0.125)
        bit += 12
    udrei = []
    for _ in range(6):
        udrei.append(_bits(payload, bit, 4)); bit += 4
    iodp = _bits(payload, bit, 2); bit += 2
    block_id = _bits(payload, bit, 2); bit += 2
    iodf_j = _bits(payload, bit, 2); bit += 2
    bit += 4  # spare
    # 106-bit MT 25 half-block - keep as integer for caller dispatch.
    half = _bits(payload, bit, 106)
    out.update({
        "PRC_m": prc,
        "UDREI": udrei,
        "IODP": iodp,
        "block_id": block_id,
        "IODF_j": iodf_j,
        "long_term_half_106b": half,
    })
    return out


def _decode_long_term_half(half_bits: int) -> dict[str, Any]:
    """Decode one 106-bit MT 25 half-block. The leading bit is the
    velocity code; the rest of the layout depends on it.

    velocity code 0: 2 SVs × (mask-prn 6b + IODE 8b + delta-X/Y/Z 9b
    signed + delta-a-f0 10b signed) + IODP 2b + spare.
    velocity code 1: 1 SV with positions + velocities + clock + IODE +
    t_0 + IODP.
    """
    def take(start: int, n: int, *, signed: bool = False) -> int:
        v = (half_bits >> (106 - start - n)) & ((1 << n) - 1)
        if signed and (v >> (n - 1)) & 1:
            v -= 1 << n
        return v
    vel_code = take(0, 1)
    if vel_code == 0:
        out = {"velocity_code": 0, "entries": []}
        bit = 1
        for _ in range(2):
            mask_prn = take(bit, 6); bit += 6
            iode = take(bit, 8); bit += 8
            dx = take(bit, 9, signed=True) * 0.125; bit += 9
            dy = take(bit, 9, signed=True) * 0.125; bit += 9
            dz = take(bit, 9, signed=True) * 0.125; bit += 9
            d_af0 = take(bit, 10, signed=True) * 2 ** -31; bit += 10
            out["entries"].append({
                "mask_prn_slot": mask_prn,
                "IODE": iode,
                "delta_x_m": dx, "delta_y_m": dy, "delta_z_m": dz,
                "delta_a_f0_s": d_af0,
            })
        out["IODP"] = take(bit, 2)
        return out
    # velocity_code = 1
    bit = 1
    mask_prn = take(bit, 6); bit += 6
    iode = take(bit, 8); bit += 8
    dx = take(bit, 11, signed=True) * 0.125; bit += 11
    dy = take(bit, 11, signed=True) * 0.125; bit += 11
    dz = take(bit, 11, signed=True) * 0.125; bit += 11
    d_af0 = take(bit, 11, signed=True) * 2 ** -31; bit += 11
    dx_v = take(bit, 8, signed=True) * 2 ** -11; bit += 8
    dy_v = take(bit, 8, signed=True) * 2 ** -11; bit += 8
    dz_v = take(bit, 8, signed=True) * 2 ** -11; bit += 8
    d_af1 = take(bit, 8, signed=True) * 2 ** -39; bit += 8
    t_0 = take(bit, 13) * 16; bit += 13
    iodp = take(bit, 2)
    return {
        "velocity_code": 1,
        "mask_prn_slot": mask_prn,
        "IODE": iode,
        "delta_x_m": dx, "delta_y_m": dy, "delta_z_m": dz,
        "delta_x_dot_m_s": dx_v, "delta_y_dot_m_s": dy_v, "delta_z_dot_m_s": dz_v,
        "delta_a_f0_s": d_af0,
        "delta_a_f1_s_per_s": d_af1,
        "t_0_s": t_0,
        "IODP": iodp,
    }


def decode_sbas_mt25(payload: bytes) -> dict[str, Any]:
    """MT 25: Long-term satellite corrections (two 106-bit halves)."""
    out = _decode_header(payload)
    half1 = _bits(payload, 14, 106)
    half2 = _bits(payload, 14 + 106, 106)
    out["halves"] = [_decode_long_term_half(half1), _decode_long_term_half(half2)]
    return out


def decode_sbas_mt26(payload: bytes) -> dict[str, Any]:
    """MT 26: Ionospheric delays for 15 grid points within a band."""
    out = _decode_header(payload)
    bit = 14
    band_number = _bits(payload, bit, 4); bit += 4
    block_id = _bits(payload, bit, 4); bit += 4
    entries = []
    for _ in range(15):
        igvd = _bits(payload, bit, 9) * 0.125; bit += 9
        givei = _bits(payload, bit, 4); bit += 4
        entries.append({"IGVD_m": igvd, "GIVEI": givei})
    iodi = _bits(payload, bit, 2)
    out.update({
        "band_number": band_number,
        "block_id": block_id,
        "entries": entries,
        "IODI": iodi,
    })
    return out


_DISPATCH = {
    1: decode_sbas_mt1,
    2: decode_sbas_mt2_5,
    3: decode_sbas_mt2_5,
    4: decode_sbas_mt2_5,
    5: decode_sbas_mt2_5,
    6: decode_sbas_mt6,
    7: decode_sbas_mt7,
    9: decode_sbas_mt9,
    17: decode_sbas_mt17,
    18: decode_sbas_mt18,
    24: decode_sbas_mt24,
    25: decode_sbas_mt25,
    26: decode_sbas_mt26,
}


def decode_sbas_message(payload: bytes) -> dict[str, Any]:
    """Decode an SBAS L1 message by dispatching on the 6-bit MT field.

    Unknown / unimplemented MTs return ``{"preamble", "msg_type",
    "raw"}``.
    """
    hdr = _decode_header(payload)
    mt = hdr["msg_type"]
    fn = _DISPATCH.get(mt)
    if fn is None:
        return {**hdr, "raw": bytes(payload)}
    return fn(payload)


__all__ = [
    "PREAMBLES",
    "decode_sbas_message",
    "decode_sbas_mt1",
    "decode_sbas_mt2_5",
    "decode_sbas_mt6",
    "decode_sbas_mt7",
    "decode_sbas_mt9",
    "decode_sbas_mt17",
    "decode_sbas_mt18",
    "decode_sbas_mt24",
    "decode_sbas_mt25",
    "decode_sbas_mt26",
]
