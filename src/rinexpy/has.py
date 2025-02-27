"""Galileo High Accuracy Service (HAS) message decoder.

The Galileo HAS service broadcasts SSR-style precise orbit and clock
corrections free-to-air on the Galileo E6-B signal (and is also
encapsulated in the RTCM 3.x experimental message 4076). HAS Messages
(HASMs) are reassembled from E6-B HAS Pages by the receiver and decoded
into one of seven message types (MT):

  MT 1   Mask                   -- which (gnss, sat, signal) cells the
                                  subsequent messages reference.
  MT 2   Orbit corrections      -- per-satellite radial / in-track /
                                  cross-track deltas.
  MT 3   Clock full-set         -- per-satellite C0 clock deltas.
  MT 4   Clock subset           -- subset of MT 3 (optional).
  MT 5   Code biases            -- per-(sat, signal) code corrections.
  MT 6   Phase biases           -- per-(sat, signal) phase corrections.
  MT 7   URA                    -- per-satellite User Range Accuracy.

Reference: Galileo HAS Service Definition Document v1.0, Section 8.2.

This module ships the header decoder, the mask decoder, and the orbit
+ clock-full-set decoders -- the SSR core needed to apply HAS in a PPP
filter. Code / phase bias and URA decoders follow the same pattern and
can be added on demand; the per-(sat, signal) layout is documented in
the SDD section 8.2.4 / 8.2.5.

The decoders consume bytes directly. The bit-cursor helper is shared
with the RTCM3 module.
"""

from __future__ import annotations

from typing import Any

from .rtcm3 import _bits


HAS_GNSS_NAMES = {0: "GPS", 1: "GLONASS", 2: "Galileo", 3: "BDS", 4: "SBAS", 5: "QZSS", 6: "NavIC"}

# Validity-interval lookup (SDD Table 13): index -> seconds.
HAS_VALIDITY_S = {
    0: 5, 1: 10, 2: 15, 3: 20, 4: 30, 5: 60, 6: 90, 7: 120,
    8: 180, 9: 240, 10: 300, 11: 600, 12: 900, 13: 1800, 14: 3600,
    15: 0,    # 'undefined / use default'
}


def decode_has_header(body: bytes) -> tuple[dict[str, Any], int]:
    """Decode the 32-bit HAS message header.

    Layout (HAS SDD v1.0 Section 8.2.1):

        Status (2 bits)        -- 0=test, 1=operational, 2,3=reserved
        Reserved (2 bits)
        Message type (4 bits)  -- MT 1..7
        Message ID (5 bits)    -- rolling identifier for retransmission
        Page count (5 bits)    -- total pages this HASM spans
        Page ID (5 bits)       -- index of this page in the HASM
        Mask ID (5 bits)       -- which mask MT-1 this message references
        IOD set (4 bits)       -- rolling mask-set identifier

    Returns
    -------
    (header_dict, bit_cursor):
        Header and the bit position immediately past the header so the
        per-MT payload decoder can pick up.
    """
    bit = 0
    status = _bits(body, bit, 2); bit += 2
    bit += 2   # reserved
    mt = _bits(body, bit, 4); bit += 4
    mid = _bits(body, bit, 5); bit += 5
    page_count = _bits(body, bit, 5); bit += 5
    page_id = _bits(body, bit, 5); bit += 5
    mask_id = _bits(body, bit, 5); bit += 5
    iod_set = _bits(body, bit, 4); bit += 4
    return {
        "status": status,
        "message_type": mt,
        "message_id": mid,
        "page_count": page_count,
        "page_id": page_id,
        "mask_id": mask_id,
        "iod_set": iod_set,
    }, bit


def decode_has_mask(body: bytes, bit: int) -> dict[str, Any]:
    """Decode an MT-1 HAS Mask body starting at bit cursor ``bit``.

    Layout (HAS SDD v1.0 Section 8.2.2):

        ValidityIntervalIndex (4 bits)
        GNSSMask (4 bits)               -- which GNSS systems follow
        For each masked GNSS:
            SatelliteMask (40 bits)     -- bit per PRN, MSB = PRN 1
            SignalMask (16 bits)        -- bit per signal slot
            CellMaskFlag (1 bit)
            If CellMaskFlag == 1:
                CellMask (n_sat * n_signal bits)
            NavMessage (3 bits)         -- which broadcast nav source

    Returns a dict listing the masked GNSS systems and their selected
    PRNs / signals / cells.
    """
    valid_idx = _bits(body, bit, 4); bit += 4
    gnss_mask = _bits(body, bit, 4); bit += 4
    gnss_entries = []
    for gnss_bit in range(4):
        if not (gnss_mask >> (3 - gnss_bit)) & 1:
            continue
        sat_mask = _bits(body, bit, 40); bit += 40
        sig_mask = _bits(body, bit, 16); bit += 16
        cell_flag = _bits(body, bit, 1); bit += 1
        sats = [i + 1 for i in range(40) if (sat_mask >> (39 - i)) & 1]
        sigs = [i for i in range(16) if (sig_mask >> (15 - i)) & 1]
        cells = None
        if cell_flag:
            cells = []
            for _ in range(len(sats)):
                row = []
                for _s in range(len(sigs)):
                    row.append(_bits(body, bit, 1)); bit += 1
                cells.append(row)
        nav_msg = _bits(body, bit, 3); bit += 3
        gnss_entries.append({
            "gnss_id": gnss_bit,
            "gnss_name": HAS_GNSS_NAMES.get(gnss_bit, f"GNSS_{gnss_bit}"),
            "satellite_mask": sat_mask,
            "signal_mask": sig_mask,
            "satellites": sats,
            "signals": sigs,
            "cell_mask": cells,
            "nav_message_type": nav_msg,
        })
    return {
        "validity_interval_s": HAS_VALIDITY_S.get(valid_idx, 0),
        "validity_interval_index": valid_idx,
        "gnss_entries": gnss_entries,
    }


def decode_has_orbit(body: bytes, bit: int, mask: dict[str, Any]) -> dict[str, Any]:
    """Decode an MT-2 HAS Orbit corrections body.

    Layout (HAS SDD v1.0 Section 8.2.3):

        ValidityIntervalIndex (4 bits)
        For each masked satellite (in mask order):
            GNSS_IOD (8 or 10 bits, depending on GNSS)
            DeltaRadial    (13 bits signed, 0.0025 m / LSB)
            DeltaInTrack   (12 bits signed, 0.0080 m / LSB)
            DeltaCrossTrack (12 bits signed, 0.0080 m / LSB)

    The IOD width is 8 bits for GPS / Galileo and 10 bits for BeiDou
    (SDD Table 17). We honor that distinction.
    """
    valid_idx = _bits(body, bit, 4); bit += 4
    out: list[dict[str, Any]] = []
    for ge in mask["gnss_entries"]:
        gnss = ge["gnss_id"]
        iod_bits = 10 if gnss == 3 else 8
        for sat in ge["satellites"]:
            iod = _bits(body, bit, iod_bits); bit += iod_bits
            d_radial = _bits(body, bit, 13, signed=True); bit += 13
            d_along = _bits(body, bit, 12, signed=True); bit += 12
            d_cross = _bits(body, bit, 12, signed=True); bit += 12
            out.append({
                "gnss_id": gnss,
                "gnss_name": ge["gnss_name"],
                "prn": sat,
                "iod": iod,
                "delta_radial_m": d_radial * 0.0025,
                "delta_along_track_m": d_along * 0.0080,
                "delta_cross_track_m": d_cross * 0.0080,
            })
    return {
        "validity_interval_s": HAS_VALIDITY_S.get(valid_idx, 0),
        "satellites": out,
    }


def decode_has_clock_full(body: bytes, bit: int, mask: dict[str, Any]) -> dict[str, Any]:
    """Decode an MT-3 HAS Clock Full-Set body.

    Layout (HAS SDD v1.0 Section 8.2.4):

        ValidityIntervalIndex (4 bits)
        For each masked GNSS:
            DeltaClockMultiplier (2 bits)   -- 1, 2, 4, or 8
        For each masked satellite (in mask order):
            DeltaClockC0 (13 bits signed, base 0.0025 m / LSB)

    The multiplier scales the raw 0.0025 m LSB so the full HAS dynamic
    range can represent clocks up to ~80 m at 0.02 m resolution when
    needed.
    """
    valid_idx = _bits(body, bit, 4); bit += 4
    multipliers: dict[int, int] = {}
    for ge in mask["gnss_entries"]:
        m = _bits(body, bit, 2); bit += 2
        multipliers[ge["gnss_id"]] = [1, 2, 4, 8][m]
    out: list[dict[str, Any]] = []
    for ge in mask["gnss_entries"]:
        mult = multipliers[ge["gnss_id"]]
        for sat in ge["satellites"]:
            c0 = _bits(body, bit, 13, signed=True); bit += 13
            out.append({
                "gnss_id": ge["gnss_id"],
                "gnss_name": ge["gnss_name"],
                "prn": sat,
                "delta_clock_c0_m": c0 * 0.0025 * mult,
                "multiplier": mult,
            })
    return {
        "validity_interval_s": HAS_VALIDITY_S.get(valid_idx, 0),
        "satellites": out,
        "gnss_multipliers": multipliers,
    }


def decode_has_message(body: bytes, mask: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decode a complete HAS Message: header + MT-specific payload.

    The mask (decoded from a prior MT-1 message with a matching
    ``mask_id``) is required to interpret orbit / clock / bias messages;
    pass ``None`` if you're decoding the mask itself.
    """
    header, bit = decode_has_header(body)
    mt = header["message_type"]
    payload: dict[str, Any]
    if mt == 1:
        payload = decode_has_mask(body, bit)
    elif mt == 2:
        if mask is None:
            raise ValueError("MT-2 (orbit) requires a decoded mask (MT-1) first")
        payload = decode_has_orbit(body, bit, mask)
    elif mt == 3:
        if mask is None:
            raise ValueError("MT-3 (clock) requires a decoded mask (MT-1) first")
        payload = decode_has_clock_full(body, bit, mask)
    else:
        payload = {"unsupported_message_type": mt}
    return {"header": header, "payload": payload}


__all__ = [
    "HAS_GNSS_NAMES",
    "HAS_VALIDITY_S",
    "decode_has_clock_full",
    "decode_has_header",
    "decode_has_mask",
    "decode_has_message",
    "decode_has_orbit",
]
