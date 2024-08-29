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


def test_unknown_class_id_falls_back_to_raw():
    """Unknown msg_class / msg_id combinations still produce the base envelope."""
    out = decode_message(0xFF, 0xFF, b"\x01\x02\x03")
    assert out["msg_class"] == 0xFF
    assert out["msg_id"] == 0xFF
    assert out["payload_bytes"] == b"\x01\x02\x03"
