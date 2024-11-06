"""u-blox UBX binary protocol decoder.

UBX frames are little-endian binary with the layout

    0xB5 0x62 | class | id | length (LE u16) | payload | ck_a | ck_b

where the two checksum bytes are the 8-bit Fletcher checksum over
``class | id | length | payload``.

Decoded message classes (a small but practical subset):

- ``NAV-PVT`` (0x01 0x07) — full position/velocity/time fix
- ``NAV-SAT`` (0x01 0x35) — per-satellite info: prn / cno / elev / az
- ``RXM-RAWX`` (0x02 0x15) — raw measurements (pseudorange, phase, Doppler)
- ``RXM-SFRBX`` (0x02 0x13) — broadcast subframe data words

Other classes/IDs come back with ``payload_bytes`` so callers can
dispatch them.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from typing import Any, BinaryIO

#: UBX sync bytes.
SYNC1 = 0xB5
SYNC2 = 0x62


def fletcher_checksum(data: bytes) -> tuple[int, int]:
    """Compute UBX's 8-bit Fletcher (CK_A, CK_B) checksum over ``data``."""
    a = 0
    b = 0
    for byte in data:
        a = (a + byte) & 0xFF
        b = (b + a) & 0xFF
    return a, b


def iter_messages(stream: BinaryIO, *, check_crc: bool = True) -> Iterator[dict[str, Any]]:
    """Iterate UBX messages from a binary stream.

    Parameters
    ----------
    stream:
        Anything supporting ``read(n)`` returning bytes.
    check_crc:
        Validate the trailing Fletcher checksum on every frame.
        Default ``True``.

    Yields
    ------
    dict
        Always contains ``msg_class``, ``msg_id``, ``length``,
        ``payload_bytes``. Decoded message types add structured fields.
    """
    while True:
        # Re-sync on the 2-byte preamble.
        b = stream.read(1)
        if not b:
            return
        if b[0] != SYNC1:
            continue
        b2 = stream.read(1)
        if not b2 or b2[0] != SYNC2:
            continue
        head = stream.read(4)
        if len(head) < 4:
            return
        msg_class, msg_id, length = head[0], head[1], struct.unpack_from("<H", head, 2)[0]
        payload = stream.read(length)
        ck = stream.read(2)
        if len(payload) < length or len(ck) < 2:
            return
        if check_crc:
            ck_a, ck_b = fletcher_checksum(head + payload)
            if (ck_a, ck_b) != (ck[0], ck[1]):
                continue
        out = decode_message(msg_class, msg_id, payload)
        yield out


def decode_message(msg_class: int, msg_id: int, payload: bytes) -> dict[str, Any]:
    """Dispatch one UBX payload to the right decoder.

    Unknown class/ID combinations come back with the raw payload.
    """
    decoders = {
        (0x01, 0x07): _decode_nav_pvt,
        (0x01, 0x22): _decode_nav_clock,
        (0x01, 0x04): _decode_nav_dop,
        (0x01, 0x12): _decode_nav_velned,
        (0x01, 0x21): _decode_nav_timeutc,
        (0x01, 0x35): _decode_nav_sat,
        (0x02, 0x15): _decode_rxm_rawx,
        (0x02, 0x13): _decode_rxm_sfrbx,
    }
    fn = decoders.get((msg_class, msg_id))
    base = {
        "msg_class": msg_class,
        "msg_id": msg_id,
        "length": len(payload),
        "payload_bytes": payload,
    }
    if fn is not None:
        base.update(fn(payload))
    return base


def _decode_nav_pvt(p: bytes) -> dict[str, Any]:
    """NAV-PVT: 92-byte position/velocity/time fix message.

    Spec offsets are from u-blox M8 receiver protocol §32.17.16. We
    decode each field at its absolute offset rather than via one big
    format string so the layout is easy to audit against the manual.
    """
    if len(p) < 92:
        return {"truncated": True}
    return {
        "itow": struct.unpack_from("<I", p, 0)[0],
        "year": struct.unpack_from("<H", p, 4)[0],
        "month": p[6],
        "day": p[7],
        "hour": p[8],
        "minute": p[9],
        "second": p[10],
        "valid_flags": p[11],
        "t_acc_ns": struct.unpack_from("<I", p, 12)[0],
        "nano": struct.unpack_from("<i", p, 16)[0],
        "fix_type": p[20],
        "flags": p[21],
        "flags2": p[22],
        "n_sat": p[23],
        "lon_deg": struct.unpack_from("<i", p, 24)[0] * 1e-7,
        "lat_deg": struct.unpack_from("<i", p, 28)[0] * 1e-7,
        "height_mm": struct.unpack_from("<i", p, 32)[0],
        "h_msl_mm": struct.unpack_from("<i", p, 36)[0],
        "h_acc_mm": struct.unpack_from("<I", p, 40)[0],
        "v_acc_mm": struct.unpack_from("<I", p, 44)[0],
        "vel_n_mm_s": struct.unpack_from("<i", p, 48)[0],
        "vel_e_mm_s": struct.unpack_from("<i", p, 52)[0],
        "vel_d_mm_s": struct.unpack_from("<i", p, 56)[0],
        "g_speed_mm_s": struct.unpack_from("<i", p, 60)[0],
        "head_motion_deg": struct.unpack_from("<i", p, 64)[0] * 1e-5,
        "s_acc_mm_s": struct.unpack_from("<I", p, 68)[0],
        "head_acc_deg": struct.unpack_from("<I", p, 72)[0] * 1e-5,
        "p_dop": struct.unpack_from("<H", p, 76)[0] * 0.01,
    }


def _decode_nav_sat(p: bytes) -> dict[str, Any]:
    """NAV-SAT: header (8 bytes) + 12 bytes per satellite."""
    if len(p) < 8:
        return {"truncated": True}
    itow, version, n_sat, _, _ = struct.unpack_from("<IBBBB", p)
    sats: list[dict[str, Any]] = []
    bit = 8
    for _ in range(n_sat):
        if bit + 12 > len(p):
            break
        gnss_id, sv_id, cno, elev, azim, pr_res, flags = struct.unpack_from(
            "<BBBbhhI", p, bit
        )
        sats.append(
            {
                "gnss_id": gnss_id,
                "sv_id": sv_id,
                "cno_dbhz": cno,
                "elevation_deg": elev,
                "azimuth_deg": azim,
                "pseudorange_residual_dm": pr_res,
                "flags": flags,
            }
        )
        bit += 12
    return {
        "itow": itow,
        "version": version,
        "n_sat": n_sat,
        "satellites": sats,
    }


def _decode_rxm_rawx(p: bytes) -> dict[str, Any]:
    """RXM-RAWX: header (16 bytes) + 32 bytes per measurement."""
    if len(p) < 16:
        return {"truncated": True}
    rcv_tow, week, leap_s, n_meas, rec_stat = struct.unpack_from("<dHbBB", p)
    meas: list[dict[str, Any]] = []
    bit = 16
    for _ in range(n_meas):
        if bit + 32 > len(p):
            break
        (
            pr_m, cp_cycles, doppler, gnss_id, sv_id, sig_id, freq_id,
            lock_ms, cno, pr_stdev, cp_stdev, do_stdev, trk_stat, _res,
        ) = struct.unpack_from("<ddfBBBBHBBBBBB", p, bit)
        meas.append(
            {
                "pseudorange_m": pr_m,
                "carrier_phase_cycles": cp_cycles,
                "doppler_hz": doppler,
                "gnss_id": gnss_id,
                "sv_id": sv_id,
                "sig_id": sig_id,
                "freq_id": freq_id,
                "lock_time_ms": lock_ms,
                "cno_dbhz": cno,
                "pr_stdev_m": 0.01 * (2 ** (pr_stdev & 0x0F)),
                "cp_stdev_cycles": 0.004 * (cp_stdev & 0x0F),
                "doppler_stdev_hz": 0.002 * (2 ** (do_stdev & 0x0F)),
                "trk_status": trk_stat,
            }
        )
        bit += 32
    return {
        "rcv_tow": rcv_tow,
        "week": week,
        "leap_seconds": leap_s,
        "n_meas": n_meas,
        "rec_status": rec_stat,
        "measurements": meas,
    }


def _decode_rxm_sfrbx(p: bytes) -> dict[str, Any]:
    """RXM-SFRBX: header + n_words * 4 bytes of subframe data."""
    if len(p) < 8:
        return {"truncated": True}
    gnss_id, sv_id, _res1, freq_id, n_words, chn, version, _res2 = struct.unpack_from(
        "<BBBBBBBB", p
    )
    words: list[int] = []
    bit = 8
    for _ in range(n_words):
        if bit + 4 > len(p):
            break
        (w,) = struct.unpack_from("<I", p, bit)
        words.append(w)
        bit += 4
    return {
        "gnss_id": gnss_id,
        "sv_id": sv_id,
        "freq_id": freq_id,
        "n_words": n_words,
        "channel": chn,
        "version": version,
        "subframe_words": words,
    }


def _decode_nav_velned(p: bytes) -> dict[str, Any]:
    """NAV-VELNED: 36-byte velocity in North/East/Down frame.

    Spec offsets from u-blox M8 protocol section 32.17.21. Velocities
    are stored as int32 cm/s; this returns them in m/s.
    """
    if len(p) < 36:
        return {"truncated": True}
    itow, velN, velE, velD, speed, gSpeed, heading, sAcc, cAcc = struct.unpack_from(
        "<I i i i I I i I I", p
    )
    return {
        "itow": itow,
        "velN_m_s": velN * 1e-2,
        "velE_m_s": velE * 1e-2,
        "velD_m_s": velD * 1e-2,
        "speed_3d_m_s": speed * 1e-2,
        "speed_2d_m_s": gSpeed * 1e-2,
        "heading_deg": heading * 1e-5,
        "speed_accuracy_m_s": sAcc * 1e-2,
        "heading_accuracy_deg": cAcc * 1e-5,
    }


def _decode_nav_timeutc(p: bytes) -> dict[str, Any]:
    """NAV-TIMEUTC: 20-byte UTC time + accuracy.

    Spec offsets from u-blox M8 protocol section 32.17.27.
    """
    if len(p) < 20:
        return {"truncated": True}
    itow, tAcc, nano, year, month, day, hour, minute, second, valid = (
        struct.unpack_from("<I I i H B B B B B B", p)
    )
    return {
        "itow": itow,
        "time_accuracy_ns": tAcc,
        "nano_offset_ns": nano,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "valid_flags": valid,
        "valid_tow": bool(valid & 0x01),
        "valid_wkn": bool(valid & 0x02),
        "valid_utc": bool(valid & 0x04),
    }


def _decode_nav_clock(p: bytes) -> dict[str, Any]:
    """NAV-CLOCK: 20-byte receiver clock state.

    Spec offsets are from u-blox M8 receiver protocol section 32.17.7.
    Reports the receiver's current clock bias and drift estimates plus
    their accuracy.
    """
    if len(p) < 20:
        return {"truncated": True}
    itow, clk_bias_ns, clk_drift_ns_per_s, t_acc_ns, f_acc_ps_per_s = struct.unpack_from(
        "<I i i I I", p
    )
    return {
        "itow": itow,
        "clock_bias_s": clk_bias_ns * 1e-9,
        "clock_drift_s_per_s": clk_drift_ns_per_s * 1e-9,
        "time_accuracy_s": t_acc_ns * 1e-9,
        "frequency_accuracy_s_per_s": f_acc_ps_per_s * 1e-12,
    }


def _decode_nav_dop(p: bytes) -> dict[str, Any]:
    """NAV-DOP: 18-byte dilution-of-precision values.

    Spec offsets are from u-blox M8 protocol section 32.17.10. All DOPs are
    transmitted as uint16 in units of 0.01 (so 100 -> DOP of 1.00).
    """
    if len(p) < 18:
        return {"truncated": True}
    itow, gdop, pdop, tdop, vdop, hdop, ndop, edop = struct.unpack_from(
        "<I H H H H H H H", p
    )
    return {
        "itow": itow,
        "GDOP": gdop * 0.01,
        "PDOP": pdop * 0.01,
        "TDOP": tdop * 0.01,
        "VDOP": vdop * 0.01,
        "HDOP": hdop * 0.01,
        "NDOP": ndop * 0.01,
        "EDOP": edop * 0.01,
    }


__all__ = [
    "SYNC1",
    "SYNC2",
    "decode_message",
    "fletcher_checksum",
    "iter_messages",
]
