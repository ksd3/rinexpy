"""Synthetic round-trip tests for the SBAS L1 message decoder.

No public SBAS L1 capture is shipped in tests/data, so each test builds
a 250-bit message bit-pattern from known field values and verifies the
decoder returns them with the documented scale per RTCA DO-229E §A.4.
"""

from __future__ import annotations

from pytest import approx

from rinexpy.sbas import (
    PREAMBLES,
    decode_sbas_message,
    decode_sbas_mt1,
    decode_sbas_mt2_5,
    decode_sbas_mt9,
    decode_sbas_mt17,
    decode_sbas_mt18,
    decode_sbas_mt25,
    decode_sbas_mt26,
)


def _pack(field_specs: list[tuple[int, int]]) -> bytes:
    bits = "".join(
        f"{value & ((1 << n) - 1):0{n}b}" for value, n in field_specs if n > 0
    )
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def _h(mt: int):
    return [(PREAMBLES[0], 8), (mt, 6)]


def test_mt1_prn_mask_round_trip():
    mask = [1 if i in (0, 4, 10, 119) else 0 for i in range(210)]
    payload = _pack(_h(1) + [(b, 1) for b in mask] + [(2, 2)])
    out = decode_sbas_mt1(payload)
    assert out["msg_type"] == 1
    assert out["IODP"] == 2
    assert out["active_prns"] == [1, 5, 11, 120]


def test_mt2_5_fast_corrections_round_trip():
    prc_raw = [10, -20, 100, -100, 0, 50, -50, 200, -200, 1000, -1000, 5, -5]
    udrei = list(range(13))
    specs = _h(3) + [(0, 2), (1, 2)]
    specs += [(v, 12) for v in prc_raw]
    specs += [(v, 4) for v in udrei]
    out = decode_sbas_mt2_5(_pack(specs))
    assert out["msg_type"] == 3
    assert out["IODF_i"] == 0
    assert out["IODP"] == 1
    assert out["PRC_m"][0] == approx(10 * 0.125)
    assert out["PRC_m"][1] == approx(-20 * 0.125)
    assert out["UDREI"] == udrei


def test_mt9_geo_navigation_round_trip():
    specs = _h(9) + [
        (0, 8),         # reserved data ID
        (100, 13),      # t_0 -> 1600 s
        (5, 4),         # URA
        (1000, 30),     # X -> 80 m
        (-2000, 30),    # Y -> -160 m
        (3000, 25),     # Z -> 1200 m
        (10, 17),       # X_dot
        (-20, 17),      # Y_dot
        (30, 18),       # Z_dot
        (1, 10),        # X_acc
        (-2, 10),       # Y_acc
        (3, 10),        # Z_acc
        (-100, 12),     # a_Gf0
        (5, 8),         # a_Gf1
    ]
    out = decode_sbas_mt9(_pack(specs))
    assert out["t_0_s"] == 1600
    assert out["URA"] == 5
    assert out["x_m"] == approx(1000 * 0.08)
    assert out["y_m"] == approx(-2000 * 0.08)
    assert out["z_m"] == approx(3000 * 0.4)
    assert out["x_dot_m_s"] == approx(10 * 0.000625)
    assert out["z_ddot_m_s2"] == approx(3 * 0.0000625)
    assert out["a_Gf0_s"] == approx(-100 * 2 ** -31)


def test_mt17_geo_almanac_round_trip():
    specs = _h(17)
    for k in range(3):
        specs += [
            (0, 2),                 # reserved data ID
            (120 + k, 8),           # PRN
            (0, 8),                 # health
            (100 + k, 15),          # X_G
            (-(50 + k), 15),        # Y_G
            (10 + k, 9),            # Z_G
            (1, 3),                 # X_dot
            (-1, 3),                # Y_dot
            (2, 4),                 # Z_dot
        ]
    specs += [(5, 11)]              # t_0 -> 320 s
    out = decode_sbas_mt17(_pack(specs))
    assert len(out["entries"]) == 3
    assert out["entries"][0]["PRN"] == 120
    assert out["entries"][1]["PRN"] == 121
    assert out["entries"][2]["PRN"] == 122
    assert out["entries"][0]["x_m"] == approx(100 * 2600.0)
    assert out["entries"][0]["y_m"] == approx(-50 * 2600.0)
    assert out["entries"][0]["x_dot_m_s"] == approx(10.0)
    assert out["t_0_s"] == 5 * 64


def test_mt18_iono_grid_mask_round_trip():
    mask = [1 if i in (0, 50, 200) else 0 for i in range(201)]
    specs = _h(18) + [(9, 4), (2, 4), (3, 2)] + [(b, 1) for b in mask] + [(0, 1)]
    out = decode_sbas_mt18(_pack(specs))
    assert out["N_bands"] == 9
    assert out["band"] == 2
    assert out["IODI"] == 3
    assert out["active_igps"] == [1, 51, 201]


def test_mt25_long_term_velocity_code_zero():
    # Two entries per half, two halves.
    half_a = [
        (0, 1),        # velocity code
        (5, 6), (123, 8), (10, 9, True), (-20, 9, True), (30, 9, True), (-100, 10, True),
        (7, 6), (200, 8), (-5, 9, True), (15, 9, True), (-25, 9, True), (50, 10, True),
        (2, 2),        # IODP
    ]
    half_b = [
        (0, 1),
        (1, 6), (1, 8), (1, 9, True), (1, 9, True), (1, 9, True), (1, 10, True),
        (2, 6), (2, 8), (-1, 9, True), (-1, 9, True), (-1, 9, True), (-1, 10, True),
        (1, 2),
    ]
    # Pad each half to 106 bits.
    def total_bits(fs):
        return sum(n for *_, n in (((v, n) if len(t) == 2 else (v, n)) for t in fs for v, n in [(t[0], t[1])]))
    # Build linear specs (n, value, signed?) flat.
    def normalize(fs):
        return [(t[0], t[1]) for t in fs]
    half_a_specs = normalize(half_a)
    half_b_specs = normalize(half_b)
    pad_a = max(0, 106 - sum(n for _, n in half_a_specs))
    pad_b = max(0, 106 - sum(n for _, n in half_b_specs))
    specs = _h(25) + half_a_specs + [(0, pad_a)] + half_b_specs + [(0, pad_b)]
    out = decode_sbas_mt25(_pack(specs))
    assert out["msg_type"] == 25
    assert out["halves"][0]["velocity_code"] == 0
    assert out["halves"][0]["entries"][0]["mask_prn_slot"] == 5
    assert out["halves"][0]["entries"][0]["IODE"] == 123
    assert out["halves"][0]["entries"][0]["delta_x_m"] == approx(10 * 0.125)
    assert out["halves"][0]["entries"][0]["delta_a_f0_s"] == approx(-100 * 2 ** -31)
    assert out["halves"][0]["IODP"] == 2


def test_mt26_iono_delays_round_trip():
    specs = _h(26) + [(3, 4), (7, 4)]
    for k in range(15):
        specs += [(100 + k, 9), (k, 4)]
    specs += [(2, 2)]
    out = decode_sbas_mt26(_pack(specs))
    assert out["band_number"] == 3
    assert out["block_id"] == 7
    assert out["IODI"] == 2
    assert out["entries"][0]["IGVD_m"] == approx(100 * 0.125)
    assert out["entries"][14]["GIVEI"] == 14


def test_dispatch_returns_raw_for_unknown_mt():
    out = decode_sbas_message(_pack(_h(15) + [(0, 220)]))
    assert out["msg_type"] == 15
    assert isinstance(out["raw"], bytes)


def test_dispatch_routes_known_mts():
    payload_mt1 = _pack(_h(1) + [(0, 210), (1, 2)])
    assert decode_sbas_message(payload_mt1)["msg_type"] == 1
    payload_mt7 = _pack(_h(7) + [(3, 4), (1, 2), (0, 2)] + [(j, 4) for j in range(51)])
    out = decode_sbas_message(payload_mt7)
    assert out["msg_type"] == 7
    assert out["IODP"] == 1
    assert out["ai_index"][3] == 3
