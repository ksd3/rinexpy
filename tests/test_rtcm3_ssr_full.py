"""Parity tests for the full SSR family (RTCM 10403.3 messages 1057-1068
and 1240-1263) across GPS, GLONASS, Galileo, QZSS, SBAS, BeiDou.

The legacy GPS-only decoders (1057, 1058) lived in test_rtcm3_ssr.py;
this file covers the generic per-system dispatcher.
"""

from __future__ import annotations

import pytest

from rinexpy.rtcm3 import decode_message


def pack_bits(*fields: tuple[int, int]) -> bytes:
    s = ""
    for v, n in fields:
        if v < 0:
            v = (1 << n) + v
        s += f"{v & ((1 << n) - 1):0{n}b}"
    if len(s) % 8:
        s += "0" * (8 - len(s) % 8)
    return bytes(int(s[i:i + 8], 2) for i in range(0, len(s), 8))


SSR_HEADER_FIELDS = [
    (123456, 20),  # epoch_time_s
    (3, 4),         # update_interval_index
    (0, 1),         # multiple_message
    (5, 4),         # iod_ssr
    (42, 16),       # provider_id
    (1, 4),         # solution_id
]


# (msg_id, system, prn_bits, iode_bits)
ORBIT_SYSTEMS = [
    (1057, "G", 6, 8),
    (1063, "R", 5, 8),
    (1240, "E", 6, 10),
    (1246, "J", 4, 8),
    (1252, "S", 6, 9),
    (1258, "C", 6, 10),
]


@pytest.mark.parametrize("msg_id, system, prn_bits, iode_bits", ORBIT_SYSTEMS)
def test_ssr_orbit_per_system(msg_id, system, prn_bits, iode_bits):
    """1057/1063/1240/1246/1252/1258 — SSR orbit RAC corrections."""
    body = pack_bits(
        (msg_id, 12),
        *SSR_HEADER_FIELDS,
        (0, 1),         # ref_datum
        (1, 6),         # n_sats
        (3, prn_bits),  # PRN 3
        (77, iode_bits),
        (100, 22), (200, 20), (300, 20),
        (50, 21), (60, 19), (70, 19),
    )
    out = decode_message(msg_id, body)
    assert out["msg_id"] == msg_id
    assert out["system"] == system
    sat = out["satellites"][0]
    assert sat["prn"] == 3
    assert sat["iode"] == 77
    assert sat["delta_radial_m"] == pytest.approx(100 * 1e-4)
    assert sat["delta_along_track_m"] == pytest.approx(200 * 4e-4)
    assert sat["delta_cross_track_m"] == pytest.approx(300 * 4e-4)


CLOCK_SYSTEMS = [
    (1058, "G", 6),
    (1064, "R", 5),
    (1241, "E", 6),
    (1247, "J", 4),
    (1253, "S", 6),
    (1259, "C", 6),
]


@pytest.mark.parametrize("msg_id, system, prn_bits", CLOCK_SYSTEMS)
def test_ssr_clock_per_system(msg_id, system, prn_bits):
    """1058/1064/1241/1247/1253/1259 — SSR clock polynomial."""
    body = pack_bits(
        (msg_id, 12),
        *SSR_HEADER_FIELDS,
        (1, 6),         # n_sats
        (5, prn_bits),
        (500, 22), (10, 21), (5, 27),
    )
    out = decode_message(msg_id, body)
    assert out["system"] == system
    sat = out["satellites"][0]
    assert sat["prn"] == 5
    assert sat["c0_m"] == pytest.approx(500 * 1e-4)
    assert sat["c1_m_per_s"] == pytest.approx(10 * 1e-6)


CODE_BIAS_SYSTEMS = [
    (1059, "G", 6),
    (1065, "R", 5),
    (1242, "E", 6),
    (1248, "J", 4),
    (1254, "S", 6),
    (1260, "C", 6),
]


@pytest.mark.parametrize("msg_id, system, prn_bits", CODE_BIAS_SYSTEMS)
def test_ssr_code_bias_per_system(msg_id, system, prn_bits):
    """1059/1065/1242/1248/1254/1260 — SSR per-signal code biases."""
    body = pack_bits(
        (msg_id, 12),
        *SSR_HEADER_FIELDS,
        (1, 6),         # n_sats
        (2, prn_bits),  # PRN 2
        (2, 5),         # 2 signals
        (4, 5), (1500, 14),
        (8, 5), (-500, 14),
    )
    out = decode_message(msg_id, body)
    assert out["system"] == system
    sat = out["satellites"][0]
    assert sat["n_signals"] == 2
    assert sat["signals"][0]["signal_id"] == 4
    assert sat["signals"][0]["bias_m"] == pytest.approx(15.0)
    assert sat["signals"][1]["bias_m"] == pytest.approx(-5.0)


URA_SYSTEMS = [
    (1061, "G", 6),
    (1067, "R", 5),
    (1244, "E", 6),
    (1250, "J", 4),
    (1256, "S", 6),
    (1262, "C", 6),
]


@pytest.mark.parametrize("msg_id, system, prn_bits", URA_SYSTEMS)
def test_ssr_ura_per_system(msg_id, system, prn_bits):
    body = pack_bits(
        (msg_id, 12), *SSR_HEADER_FIELDS,
        (1, 6),
        (1, prn_bits), (12, 6),
    )
    out = decode_message(msg_id, body)
    assert out["system"] == system
    assert out["satellites"][0]["ura_index"] == 12


HR_CLOCK_SYSTEMS = [
    (1062, "G", 6),
    (1068, "R", 5),
    (1245, "E", 6),
    (1251, "J", 4),
    (1257, "S", 6),
    (1263, "C", 6),
]


@pytest.mark.parametrize("msg_id, system, prn_bits", HR_CLOCK_SYSTEMS)
def test_ssr_hr_clock_per_system(msg_id, system, prn_bits):
    body = pack_bits(
        (msg_id, 12), *SSR_HEADER_FIELDS,
        (1, 6),
        (1, prn_bits), (1234, 22),
    )
    out = decode_message(msg_id, body)
    assert out["system"] == system
    assert out["satellites"][0]["hr_clock_m"] == pytest.approx(1234 * 1e-4)


COMBINED_SYSTEMS = [
    (1060, "G", 6, 8),
    (1066, "R", 5, 8),
    (1243, "E", 6, 10),
    (1249, "J", 4, 8),
    (1255, "S", 6, 9),
    (1261, "C", 6, 10),
]


@pytest.mark.parametrize("msg_id, system, prn_bits, iode_bits", COMBINED_SYSTEMS)
def test_ssr_combined_per_system(msg_id, system, prn_bits, iode_bits):
    body = pack_bits(
        (msg_id, 12), *SSR_HEADER_FIELDS,
        (0, 1),         # ref_datum
        (1, 6),         # n_sats
        (1, prn_bits), (7, iode_bits),
        (100, 22), (200, 20), (300, 20),
        (50, 21), (60, 19), (70, 19),
        (500, 22), (10, 21), (5, 27),
    )
    out = decode_message(msg_id, body)
    assert out["system"] == system
    sat = out["satellites"][0]
    assert sat["iode"] == 7
    assert sat["c0_m"] == pytest.approx(500 * 1e-4)
    assert sat["delta_radial_m"] == pytest.approx(100 * 1e-4)
