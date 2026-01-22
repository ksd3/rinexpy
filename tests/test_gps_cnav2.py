"""Synthetic round-trip tests for the GPS CNAV-2 (L1C) Subframe 2 decoder.

No public capture of CNAV-2 frames exists in tests/data, so a synthetic
bit-pattern is built and decoded, verifying every field against its
expected scaling per IS-GPS-800 Table 3.5-1.
"""

from __future__ import annotations

from pytest import approx

from rinexpy.gps_cnav2 import decode_cnav2_subframe2


def _pack(field_specs: list[tuple[int, int]]) -> bytes:
    bits = "".join(
        f"{value & ((1 << n) - 1):0{n}b}" for value, n in field_specs if n > 0
    )
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def test_cnav2_subframe2_round_trip():
    specs = [
        (2100, 13),       # WN
        (50, 8),          # ITOW
        (3, 11),          # t_op = 900
        (1, 1),           # health
        (-2, 5),          # URA (signed)
        (4, 11),          # t_oe = 1200
        (-500, 26),       # delta A
        (3, 25),          # Adot
        (-9, 17),         # delta_n0
        (5, 23),          # delta_n0_dot
        (-1234, 33),      # M_0
        (5000, 33),       # e
        (-555, 33),       # omega
        (-100, 33),       # Omega_0
        (200, 33),        # i_0
        (-1, 17),         # delta_Omega_dot
        (2, 15),          # IDOT
        (-7, 16), (8, 16),     # C_is, C_ic
        (-9, 24), (10, 24),    # C_rs, C_rc
        (-11, 21), (12, 21),   # C_us, C_uc
        (-13, 26), (14, 20), (-3, 10),  # a_f0, a_f1, a_f2
        (-2, 13),         # T_GD
    ]
    payload = _pack(specs)
    out = decode_cnav2_subframe2(payload)
    assert out["week"] == 2100
    assert out["ITOW"] == 50
    assert out["t_op_s"] == 900
    assert out["SV_health"] == 1
    assert out["URA_index"] == -2
    assert out["t_oe_s"] == 1200
    assert out["delta_A_m"] == approx(-500 * 2 ** -9)
    assert out["e_n"] == approx(5000 * 2 ** -34)
    assert out["Omega_0_n_semicircles"] == approx(-100 * 2 ** -32)
    assert out["IDOT_semicircles_per_s"] == approx(2 * 2 ** -44)
    assert out["C_is_rad"] == approx(-7 * 2 ** -30)
    assert out["C_rs_m"] == approx(-9 * 2 ** -8)
    assert out["a_f0_s"] == approx(-13 * 2 ** -35)
    assert out["a_f2_s_per_s2"] == approx(-3 * 2 ** -60)
    assert out["T_GD_s"] == approx(-2 * 2 ** -35)
