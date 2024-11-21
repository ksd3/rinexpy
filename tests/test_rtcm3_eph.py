"""Tests for the RTCM3 BeiDou / Galileo ephemeris decoders (1042, 1045, 1046)."""

from __future__ import annotations

import math

from pytest import approx

from rinexpy.rtcm3 import decode_message

_PI = 3.1415926535898


def _pack(field_specs, msg_id):
    bits = f"{msg_id:012b}"
    for value, n in field_specs:
        if n <= 0:
            continue
        bits += f"{value & ((1 << n) - 1):0{n}b}"
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def _signed(value, n_bits):
    """Two's-complement-encode `value` into n_bits."""
    return value & ((1 << n_bits) - 1)


def test_decode_1042_round_trips_known_fields():
    """Pack synthetic BeiDou ephemeris, decode, verify scaling and SV id."""
    specs = [
        (7, 6),                            # SV id 7
        (700, 13),                         # week
        (2, 4),                            # URAI
        (_signed(50, 14), 14),             # IDOT raw
        (10, 5),                           # AODE
        (1000, 17),                        # toc raw (-> 8000 s)
        (_signed(-3, 11), 11),             # a2 raw
        (_signed(40, 22), 22),             # a1 raw
        (_signed(-1234, 24), 24),          # a0 raw
        (10, 5),                           # AODC
        (_signed(20, 18), 18),             # Crs raw
        (_signed(15, 16), 16),             # delta_n raw
        (_signed(-100, 32), 32),           # M0 raw
        (_signed(60, 18), 18),             # Cuc raw
        (0x10000000, 32),                  # e raw
        (_signed(-60, 18), 18),            # Cus raw
        (0x1ABCDEF0, 32),                  # sqrt(A) raw
        (1100, 17),                        # toe raw
        (_signed(12, 18), 18),             # Cic raw
        (_signed(-22222, 32), 32),         # Omega_0 raw
        (_signed(-12, 18), 18),            # Cis raw
        (_signed(50000, 32), 32),          # i_0 raw
        (_signed(-30, 18), 18),            # Crc raw
        (_signed(99999, 32), 32),          # omega raw
        (_signed(-50, 24), 24),            # Omega_dot raw
        (_signed(7, 10), 10),              # TGD1 raw
        (_signed(-7, 10), 10),             # TGD2 raw
        (1, 1),                            # SV health
    ]
    body = _pack(specs, 1042)
    out = decode_message(1042, body)

    assert out["msg_id"] == 1042
    assert out["sv"] == "C07"
    assert out["week"] == 700
    assert out["URAI"] == 2
    assert out["AODE"] == 10
    assert out["AODC"] == 10
    assert out["t_oc_s"] == 1000 * 8
    assert out["t_oe_s"] == 1100 * 8
    assert out["a_f0_s"] == approx(-1234 * 2**-33)
    assert out["a_f1_s_per_s"] == approx(40 * 2**-50)
    assert out["a_f2_s_per_s2"] == approx(-3 * 2**-66)
    assert out["delta_n_rad_s"] == approx(15 * 2**-43 * _PI)
    assert out["IDOT_rad_s"] == approx(50 * 2**-43 * _PI)
    assert out["e"] == approx(0x10000000 * 2**-33)
    assert out["sqrt_A_root_m"] == approx(0x1ABCDEF0 * 2**-19)
    assert out["TGD1_s"] == approx(7 * 0.1e-9)
    assert out["TGD2_s"] == approx(-7 * 0.1e-9)
    assert out["SV_health"] == 1


def test_decode_1045_galileo_fnav_round_trips():
    """Galileo F/NAV (E5a) ephemeris decode."""
    specs = [
        (12, 6),                           # SV
        (1234, 12),                        # week
        (50, 10),                          # IODnav
        (100, 8),                          # SISA
        (_signed(20, 14), 14),             # IDOT
        (500, 14),                         # toc raw
        (_signed(2, 6), 6),                # af2
        (_signed(-100, 21), 21),           # af1
        (_signed(50000, 31), 31),          # af0
        (_signed(15, 16), 16),             # Crs
        (_signed(20, 16), 16),             # delta_n
        (_signed(-12345, 32), 32),         # M0
        (_signed(7, 16), 16),              # Cuc
        (0x08000000, 32),                  # e
        (_signed(-7, 16), 16),             # Cus
        (0x1ABCDEF0, 32),                  # sqrt(A)
        (600, 14),                         # toe
        (_signed(3, 16), 16),              # Cic
        (_signed(-9876, 32), 32),          # Omega_0
        (_signed(-4, 16), 16),             # Cis
        (_signed(54321, 32), 32),          # i0
        (_signed(-25, 16), 16),            # Crc
        (_signed(88888, 32), 32),          # omega
        (_signed(-40, 24), 24),            # Omega_dot
        (_signed(5, 10), 10),              # BGD_E1E5a
        (1, 2),                            # OSHS
        (0, 1),                            # OSDVS
    ]
    body = _pack(specs, 1045)
    out = decode_message(1045, body)

    assert out["msg_id"] == 1045
    assert out["sv"] == "E12"
    assert out["week"] == 1234
    assert out["IODnav"] == 50
    assert out["SISA"] == 100
    assert out["t_oc_s"] == 500 * 60
    assert out["t_oe_s"] == 600 * 60
    assert out["BGD_E1E5a_s"] == approx(5 * 2**-32)
    assert out["OSHS"] == 1
    assert out["OSDVS"] == 0
    # 1045 should NOT have BGD_E1E5b.
    assert "BGD_E1E5b_s" not in out


def test_decode_1046_galileo_inav_has_second_bgd_and_flags():
    """Galileo I/NAV (E1B/E5b) ephemeris carries BGD_E1E5b plus the
    per-signal health/DVS flags."""
    specs = [
        (5, 6), (2000, 12), (100, 10), (50, 8),
        (_signed(0, 14), 14),
        (300, 14),
        (_signed(0, 6), 6),
        (_signed(0, 21), 21),
        (_signed(0, 31), 31),
        (_signed(0, 16), 16),
        (_signed(0, 16), 16),
        (_signed(0, 32), 32),
        (_signed(0, 16), 16),
        (0, 32),
        (_signed(0, 16), 16),
        (0, 32),
        (400, 14),
        (_signed(0, 16), 16),
        (_signed(0, 32), 32),
        (_signed(0, 16), 16),
        (_signed(0, 32), 32),
        (_signed(0, 16), 16),
        (_signed(0, 32), 32),
        (_signed(0, 24), 24),
        (_signed(10, 10), 10),    # BGD_E1E5a
        (_signed(-10, 10), 10),   # BGD_E1E5b
        (1, 1),                   # E5b DVS
        (2, 2),                   # E5b HS
        (0, 1),                   # E1B DVS
        (1, 2),                   # E1B HS
    ]
    body = _pack(specs, 1046)
    out = decode_message(1046, body)

    assert out["msg_id"] == 1046
    assert out["sv"] == "E05"
    assert out["BGD_E1E5a_s"] == approx(10 * 2**-32)
    assert out["BGD_E1E5b_s"] == approx(-10 * 2**-32)
    assert out["E5b_DVS"] == 1
    assert out["E5b_HS"] == 2
    assert out["E1B_DVS"] == 0
    assert out["E1B_HS"] == 1
    # 1046 should NOT have the F/NAV-only flags.
    assert "OSHS" not in out
    assert "OSDVS" not in out
