"""Tests for the newly-added UBX NAV-CLOCK and NAV-DOP decoders."""

from __future__ import annotations

import struct

from pytest import approx

from rinexpy.ubx import decode_message


def test_nav_clock_round_trips_signed_fields():
    """Pack NAV-CLOCK with known fields and verify the unit conversions."""
    itow = 123_456_789
    clk_bias_ns = -42_000        # -42 us
    clk_drift_ns_per_s = 1_500   # +1.5 us/s
    t_acc_ns = 25
    f_acc_ps_per_s = 7
    payload = struct.pack("<I i i I I", itow, clk_bias_ns,
                          clk_drift_ns_per_s, t_acc_ns, f_acc_ps_per_s)
    out = decode_message(0x01, 0x22, payload)
    assert out["itow"] == itow
    assert out["clock_bias_s"] == approx(-42e-6, abs=1e-12)
    assert out["clock_drift_s_per_s"] == approx(1.5e-6, abs=1e-12)
    assert out["time_accuracy_s"] == approx(25e-9, abs=1e-15)
    assert out["frequency_accuracy_s_per_s"] == approx(7e-12, abs=1e-18)


def test_nav_clock_truncated_payload():
    """A short payload is flagged rather than crashing."""
    out = decode_message(0x01, 0x22, b"\x00" * 10)
    assert out.get("truncated") is True


def test_nav_dop_unit_conversion():
    """All DOP values come back as float scaled by 0.01."""
    itow = 100
    gdop = 250   # 2.50
    pdop = 200   # 2.00
    tdop = 100   # 1.00
    vdop = 150   # 1.50
    hdop = 150   # 1.50
    ndop = 120   # 1.20
    edop = 90    # 0.90
    payload = struct.pack("<I H H H H H H H", itow,
                          gdop, pdop, tdop, vdop, hdop, ndop, edop)
    out = decode_message(0x01, 0x04, payload)
    assert out["itow"] == itow
    assert out["GDOP"] == 2.50
    assert out["PDOP"] == 2.00
    assert out["TDOP"] == 1.00
    assert out["VDOP"] == 1.50
    assert out["HDOP"] == 1.50
    assert out["NDOP"] == 1.20
    assert out["EDOP"] == 0.90


def test_nav_dop_truncated_payload():
    out = decode_message(0x01, 0x04, b"\x00" * 5)
    assert out.get("truncated") is True


def test_nav_velned_round_trips_velocities():
    """Velocity components and accuracy fields decode in m/s and degrees."""
    payload = struct.pack(
        "<I i i i I I i I I",
        500_000,    # iTOW
        100,        # velN: 1.00 m/s
        -200,       # velE: -2.00 m/s
        50,         # velD: 0.50 m/s
        225,        # speed_3d: 2.25 m/s
        220,        # speed_2d: 2.20 m/s
        12345678,   # heading: 123.45678 deg
        10,         # sAcc: 0.10 m/s
        20000,      # cAcc: 0.20000 deg
    )
    out = decode_message(0x01, 0x12, payload)
    assert out["itow"] == 500_000
    assert out["velN_m_s"] == approx(1.00)
    assert out["velE_m_s"] == approx(-2.00)
    assert out["velD_m_s"] == approx(0.50)
    assert out["speed_3d_m_s"] == approx(2.25)
    assert out["speed_2d_m_s"] == approx(2.20)
    assert out["heading_deg"] == approx(123.45678, abs=1e-5)
    assert out["speed_accuracy_m_s"] == approx(0.10)
    assert out["heading_accuracy_deg"] == approx(0.20, abs=1e-5)


def test_nav_velned_truncated_payload():
    out = decode_message(0x01, 0x12, b"\x00" * 10)
    assert out.get("truncated") is True


def test_nav_timeutc_decodes_utc_components():
    """NAV-TIMEUTC reads year/month/day/h/m/s and the valid-flag bits."""
    payload = struct.pack(
        "<I I i H B B B B B B",
        100_000_000,    # iTOW
        50,             # tAcc 50 ns
        -200,           # nano: -200 ns offset
        2024,
        6,              # month
        21,             # day
        14, 30, 45,
        0x07,           # valid: TOW + WKN + UTC bits set
    )
    out = decode_message(0x01, 0x21, payload)
    assert out["year"] == 2024
    assert out["month"] == 6
    assert out["day"] == 21
    assert out["hour"] == 14 and out["minute"] == 30 and out["second"] == 45
    assert out["time_accuracy_ns"] == 50
    assert out["nano_offset_ns"] == -200
    assert out["valid_tow"] is True
    assert out["valid_wkn"] is True
    assert out["valid_utc"] is True


def test_nav_timeutc_truncated_payload():
    out = decode_message(0x01, 0x21, b"\x00" * 5)
    assert out.get("truncated") is True


def test_unknown_class_id_falls_back_to_raw():
    """Unknown msg_class / msg_id combinations still produce the base envelope."""
    out = decode_message(0xFF, 0xFF, b"\x01\x02\x03")
    assert out["msg_class"] == 0xFF
    assert out["msg_id"] == 0xFF
    assert out["payload_bytes"] == b"\x01\x02\x03"
