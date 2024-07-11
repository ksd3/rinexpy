"""Tests for the GPS LNAV subframe decoders."""

from __future__ import annotations

import pytest
from pytest import approx

from rinexpy.gps_lnav import (
    PREAMBLE,
    decode_lnav_subframe1,
    decode_lnav_subframe2,
    decode_lnav_subframe3,
    encode_lnav_words,
)

_PI = 3.1415926535898


def _build(field_specs, subframe_id, tow_count=12345):
    """Prepend the TLM + HOW words to the specs and pack."""
    # Word 1 (TLM): preamble(8) + TLM message(14) + reserved(2) = 24 data bits
    # Word 2 (HOW): tow(17) + alert(1) + A-S(1) + sf_id(3) + 2 parity-helper bits = 24
    prefix = [
        (PREAMBLE, 8),
        (0, 14),    # TLM message
        (0, 2),     # reserved
        (tow_count, 17),
        (0, 1),     # alert flag
        (0, 1),     # A-S flag
        (subframe_id, 3),
        (0, 2),     # solve-for-parity bits
    ]
    return encode_lnav_words(prefix + field_specs)


def test_subframe1_round_trips_known_values():
    """Pack synthetic clock fields, decode, expect them back."""
    week = 999             # 10-bit field, must be < 1024
    ca_p_l2 = 1
    ura = 2
    health = 0
    iodc_msb = 3            # top 2 bits
    l2_p_data_flag = 0
    word4_reserved = 0
    word56_reserved_lo = 0
    word56_reserved_hi = 0
    word7_reserved = 0
    # T_GD: pick a known signed integer that we'll multiply by 2^-31 below.
    tgd_raw = -7
    iodc_lsb = 0xAB
    toc_raw = 100           # toc = 100 * 16 = 1600 s
    af2_raw = 5
    af1_raw = -123
    af0_raw = 0x123456 & ((1 << 22) - 1)  # arbitrary 22-bit

    spec = [
        (week, 10),
        (ca_p_l2, 2),
        (ura, 4),
        (health, 6),
        (iodc_msb, 2),
        # word 4
        (l2_p_data_flag, 1),
        (word4_reserved, 23),
        # words 5, 6
        (word56_reserved_lo, 24),
        (word56_reserved_hi, 24),
        # word 7: 16 reserved + T_GD(8)
        (word7_reserved, 16),
        (tgd_raw & 0xFF, 8),
        # word 8: IODC_lsb(8) + t_oc(16)
        (iodc_lsb, 8),
        (toc_raw, 16),
        # word 9: a_f2(8) + a_f1(16)
        (af2_raw & 0xFF, 8),
        (af1_raw & 0xFFFF, 16),
        # word 10: a_f0(22) + 2 parity-helper
        (af0_raw, 22),
        (0, 2),
    ]
    words = _build(spec, subframe_id=1)
    out = decode_lnav_subframe1(words)

    assert out["subframe_id"] == 1
    assert out["week"] == week
    assert out["ca_or_p_on_l2"] == ca_p_l2
    assert out["URA"] == ura
    assert out["SV_health"] == health
    assert out["IODC"] == (iodc_msb << 8) | iodc_lsb
    assert out["T_GD_s"] == approx(tgd_raw * 2**-31)
    assert out["t_oc_s"] == toc_raw * 16
    assert out["a_f2_s_per_s2"] == approx(af2_raw * 2**-55)
    assert out["a_f1_s_per_s"] == approx(af1_raw * 2**-43)
    # a_f0 wasn't sign-bit-set; check unsigned interpretation matches.
    sign = af0_raw & (1 << 21)
    expected_af0 = af0_raw - (1 << 22) if sign else af0_raw
    assert out["a_f0_s"] == approx(expected_af0 * 2**-31)


def test_subframe2_round_trips():
    """Pack ephemeris part 1 fields, decode, recover them."""
    iode = 77
    crs_raw = -123          # signed, scale 2^-5 m
    delta_n_raw = 42        # signed, scale 2^-43 semicircles/s
    m0_raw = -100000        # signed 32-bit, scale 2^-31 semicircles
    cuc_raw = 50            # signed, scale 2^-29 rad
    e_raw = 0x10000000      # unsigned 32-bit, scale 2^-33
    cus_raw = -50           # signed, scale 2^-29 rad
    sqrt_a_raw = 0x1ABCDEF0  # unsigned 32-bit, scale 2^-19 sqrt(m)
    toe_raw = 1000          # unsigned, scale 16 s
    fit_flag = 0
    aodo = 27

    spec = [
        # word 3: IODE(8) + C_rs(16)
        (iode, 8),
        (crs_raw & 0xFFFF, 16),
        # word 4: dn(16) + M_0_msb(8)
        (delta_n_raw & 0xFFFF, 16),
        ((m0_raw >> 24) & 0xFF, 8),
        # word 5: M_0_lsb(24)
        (m0_raw & 0xFFFFFF, 24),
        # word 6: C_uc(16) + e_msb(8)
        (cuc_raw & 0xFFFF, 16),
        ((e_raw >> 24) & 0xFF, 8),
        # word 7: e_lsb(24)
        (e_raw & 0xFFFFFF, 24),
        # word 8: C_us(16) + sqrt(A)_msb(8)
        (cus_raw & 0xFFFF, 16),
        ((sqrt_a_raw >> 24) & 0xFF, 8),
        # word 9: sqrt(A)_lsb(24)
        (sqrt_a_raw & 0xFFFFFF, 24),
        # word 10: t_oe(16) + fit(1) + AODO(5) + 2 unused
        (toe_raw, 16),
        (fit_flag, 1),
        (aodo, 5),
        (0, 2),
    ]
    words = _build(spec, subframe_id=2)
    out = decode_lnav_subframe2(words)

    assert out["subframe_id"] == 2
    assert out["IODE"] == iode
    assert out["C_rs_m"] == approx(crs_raw * 2**-5)
    assert out["delta_n_rad_s"] == approx(delta_n_raw * 2**-43 * _PI)
    assert out["M_0_rad"] == approx(m0_raw * 2**-31 * _PI)
    assert out["C_uc_rad"] == approx(cuc_raw * 2**-29)
    assert out["e"] == approx(e_raw * 2**-33)
    assert out["C_us_rad"] == approx(cus_raw * 2**-29)
    assert out["sqrt_A_root_m"] == approx(sqrt_a_raw * 2**-19)
    assert out["t_oe_s"] == toe_raw * 16
    assert out["fit_interval_flag"] == fit_flag
    assert out["AODO"] == aodo


def test_subframe3_round_trips():
    """Pack ephemeris part 2 fields, decode, recover them."""
    cic_raw = 11
    omega0_raw = -123456789  # signed 32-bit
    cis_raw = -22
    i0_raw = 50000000        # signed 32-bit
    crc_raw = 333            # signed
    omega_raw = -22222222    # signed 32-bit
    omega_dot_raw = -77      # signed 24-bit
    iode = 88
    idot_raw = -55           # signed 14-bit

    spec = [
        # word 3: C_ic(16) + Omega_0_msb(8)
        (cic_raw & 0xFFFF, 16),
        ((omega0_raw >> 24) & 0xFF, 8),
        # word 4: Omega_0_lsb(24)
        (omega0_raw & 0xFFFFFF, 24),
        # word 5: C_is(16) + i_0_msb(8)
        (cis_raw & 0xFFFF, 16),
        ((i0_raw >> 24) & 0xFF, 8),
        # word 6: i_0_lsb(24)
        (i0_raw & 0xFFFFFF, 24),
        # word 7: C_rc(16) + omega_msb(8)
        (crc_raw & 0xFFFF, 16),
        ((omega_raw >> 24) & 0xFF, 8),
        # word 8: omega_lsb(24)
        (omega_raw & 0xFFFFFF, 24),
        # word 9: Omega_dot(24)
        (omega_dot_raw & 0xFFFFFF, 24),
        # word 10: IODE(8) + IDOT(14) + 2 unused
        (iode, 8),
        (idot_raw & 0x3FFF, 14),
        (0, 2),
    ]
    words = _build(spec, subframe_id=3)
    out = decode_lnav_subframe3(words)

    assert out["subframe_id"] == 3
    assert out["C_ic_rad"] == approx(cic_raw * 2**-29)
    assert out["Omega_0_rad"] == approx(omega0_raw * 2**-31 * _PI)
    assert out["C_is_rad"] == approx(cis_raw * 2**-29)
    assert out["i_0_rad"] == approx(i0_raw * 2**-31 * _PI)
    assert out["C_rc_m"] == approx(crc_raw * 2**-5)
    assert out["omega_rad"] == approx(omega_raw * 2**-31 * _PI)
    assert out["Omega_dot_rad_s"] == approx(omega_dot_raw * 2**-43 * _PI)
    assert out["IODE"] == iode
    assert out["IDOT_rad_s"] == approx(idot_raw * 2**-43 * _PI)


def test_bad_preamble_rejected():
    """Words with the wrong preamble raise."""
    # Build a subframe 1 with a wrong preamble (0x00).
    words = encode_lnav_words(
        [
            (0x00, 8),     # bad preamble
            (0, 14),
            (0, 2),
            (12345, 17),
            (0, 1),
            (0, 1),
            (1, 3),        # subframe id
            (0, 2),
        ]
    )
    with pytest.raises(ValueError, match="preamble"):
        decode_lnav_subframe1(words)


def test_subframe_id_mismatch_rejected():
    """Calling decode_lnav_subframe1 on a subframe-2 packet raises."""
    words = _build([(0, 240 - 64)], subframe_id=2)  # placeholder body
    with pytest.raises(ValueError, match="expected subframe 1"):
        decode_lnav_subframe1(words)


def test_too_few_words_rejected():
    """Fewer than 10 words is a programming error."""
    with pytest.raises(ValueError, match="needs 10 words"):
        decode_lnav_subframe1([0] * 9)


def test_tow_count_decoded():
    """The HOW TOW count comes through unchanged."""
    words = _build([(0, 240 - 64)], subframe_id=1, tow_count=99999)
    out = decode_lnav_subframe1(words)
    assert out["tow_count"] == 99999


def test_encode_pads_when_specs_short():
    """encode_lnav_words pads with zeros when the spec is < 240 bits."""
    words = encode_lnav_words([(PREAMBLE, 8)])
    assert len(words) == 10
    # First word: 8-bit preamble at the top, rest zero, then 6 parity zeros.
    assert words[0] >> 6 == PREAMBLE << 16
