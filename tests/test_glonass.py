"""Tests for GLONASS FDMA helpers."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.glonass import (
    CHANNEL_MAX,
    CHANNEL_MIN,
    C_M_PER_S,
    F_L1OF_BASE_HZ,
    F_L1OF_STEP_HZ,
    F_L2OF_BASE_HZ,
    F_L2OF_STEP_HZ,
    decode_glonass_string,
    decode_glonass_string1,
    decode_glonass_string2,
    decode_glonass_string3,
    frequencies_array,
    iono_free_phase,
    iono_free_pseudorange,
    l1_frequency_hz,
    l1_wavelength_m,
    l2_frequency_hz,
    l2_wavelength_m,
    phase_cycles_to_meters,
)


def test_channel_zero_returns_base_frequencies():
    assert l1_frequency_hz(0) == F_L1OF_BASE_HZ
    assert l2_frequency_hz(0) == F_L2OF_BASE_HZ


def test_extreme_channels_match_published_values():
    """Channel -7 and +6 produce the spec-published frequencies."""
    # L1OF: -7 -> 1598.0625 MHz, +6 -> 1605.375 MHz.
    assert l1_frequency_hz(-7) == pytest.approx(1598.0625e6)
    assert l1_frequency_hz(6) == pytest.approx(1605.375e6)
    # L2OF: -7 -> 1242.9375 MHz, +6 -> 1248.625 MHz.
    assert l2_frequency_hz(-7) == pytest.approx(1242.9375e6)
    assert l2_frequency_hz(6) == pytest.approx(1248.625e6)


def test_wavelengths_match_inverse_frequencies():
    for k in range(CHANNEL_MIN, CHANNEL_MAX + 1):
        assert l1_wavelength_m(k) == pytest.approx(C_M_PER_S / l1_frequency_hz(k))
        assert l2_wavelength_m(k) == pytest.approx(C_M_PER_S / l2_frequency_hz(k))


def test_out_of_range_channel_raises():
    with pytest.raises(ValueError, match="channel"):
        l1_frequency_hz(7)
    with pytest.raises(ValueError, match="channel"):
        l2_frequency_hz(-8)


def test_frequencies_array_vectorized():
    chans = np.array([-7, 0, 3, 6])
    f1, f2 = frequencies_array(chans)
    assert f1.shape == (4,)
    assert f2.shape == (4,)
    # Same as scalar form.
    for i, k in enumerate(chans):
        assert f1[i] == pytest.approx(l1_frequency_hz(int(k)))
        assert f2[i] == pytest.approx(l2_frequency_hz(int(k)))


def test_frequencies_array_rejects_out_of_range():
    with pytest.raises(ValueError, match="channel"):
        frequencies_array(np.array([0, 7, 1]))


def test_iono_free_pseudorange_matches_scalar_formula():
    chans = np.array([-3, 4])
    p1 = np.array([23_000_000.0, 23_500_000.0])
    p2 = np.array([23_000_010.0, 23_500_010.0])
    ifree = iono_free_pseudorange(p1, p2, chans)
    f1, f2 = frequencies_array(chans)
    expected = (f1 ** 2 * p1 - f2 ** 2 * p2) / (f1 ** 2 - f2 ** 2)
    np.testing.assert_allclose(ifree, expected)


def test_iono_free_phase_matches_scalar_formula():
    chans = np.array([0, -7, 6])
    l1 = np.array([23_000_000.0, 23_500_000.0, 24_000_000.0])
    l2 = np.array([23_000_010.0, 23_500_010.0, 24_000_010.0])
    ifree = iono_free_phase(l1, l2, chans)
    f1, f2 = frequencies_array(chans)
    expected = (f1 ** 2 * l1 - f2 ** 2 * l2) / (f1 ** 2 - f2 ** 2)
    np.testing.assert_allclose(ifree, expected)


def test_phase_cycles_to_meters_uses_per_sv_wavelength():
    chans = np.array([0, -5])
    phi = np.array([100_000.0, 100_000.0])    # same cycle count
    result = phase_cycles_to_meters(phi, chans, band="L1")
    # Different channels -> different wavelengths -> different meters.
    assert result[0] != result[1]
    # And each component matches the scalar wavelength.
    assert result[0] == pytest.approx(phi[0] * l1_wavelength_m(0))
    assert result[1] == pytest.approx(phi[1] * l1_wavelength_m(-5))


def test_phase_cycles_to_meters_l2():
    chans = np.array([3])
    phi = np.array([1000.0])
    out = phase_cycles_to_meters(phi, chans, band="L2")
    assert out[0] == pytest.approx(1000.0 * l2_wavelength_m(3))


# ---------------------------------------------------------------------------
# Raw string decoders
# ---------------------------------------------------------------------------


def _sm_encode(value: int, n: int) -> int:
    """Encode a signed integer in n-bit sign-magnitude."""
    if value < 0:
        return (1 << (n - 1)) | (-value)
    return value


def _pack_string(fields: list[tuple[int, int, int]]) -> bytes:
    """Pack a list of ``(icd_first_bit, n_bits, raw_value)`` into 11 bytes.

    icd_first_bit is the highest ICD bit number of the field; bits are
    placed at offset (85 - icd_first_bit) MSB-first, matching the
    decoder. Output is exactly 88 bits (11 bytes), where the last
    3 bits are zero pad past ICD bit 1.
    """
    bits = ["0"] * 88
    for icd_first, n, value in fields:
        offset = 85 - icd_first
        raw = value & ((1 << n) - 1)
        bs = f"{raw:0{n}b}"
        for i, b in enumerate(bs):
            bits[offset + i] = b
    s = "".join(bits)
    return bytes(int(s[i : i + 8], 2) for i in range(0, 88, 8))


def test_glonass_string1_round_trip():
    # X = -1234 km magnitude, X_dot = 0.5 km/s, X_dot_dot = -0.001 km/s^2
    x_raw = _sm_encode(-1234 * (1 << 11), 27)              # 2^-11 km
    xdot_raw = _sm_encode(int(0.5 * (1 << 20)), 24)        # 2^-20 km/s
    xddot_raw = _sm_encode(-(int(0.001 * (1 << 30))) >> 25, 5)  # not great; pick a small fittable value
    # use exact representable: 3 * 2^-30 km/s^2
    xddot_raw = _sm_encode(3, 5)
    # t_k: 12h 30min 30s -> raw = (12<<7) | (30<<1) | 1 = 1536 | 60 | 1 = 1597
    t_k_h, t_k_m, t_k_30 = 12, 30, 1
    t_k_raw = (t_k_h << 7) | (t_k_m << 1) | t_k_30
    payload = _pack_string([
        (84, 4, 1),                # m = 1
        (78, 2, 2),                # P1
        (76, 12, t_k_raw),         # t_k
        (64, 24, xdot_raw),
        (40, 5, xddot_raw),
        (35, 27, x_raw),
    ])
    out = decode_glonass_string1(payload)
    assert out["string"] == 1
    assert out["P1"] == 2
    assert out["t_k_s"] == 12 * 3600 + 30 * 60 + 30
    assert out["x_m"] == pytest.approx(-1234.0 * 1000.0)
    assert out["x_dot_m_s"] == pytest.approx(0.5 * 1000.0)
    assert out["x_dot_dot_m_s2"] == pytest.approx(3 * 2 ** -30 * 1000.0)


def test_glonass_string2_round_trip():
    y_raw = _sm_encode(5000 * (1 << 11), 27)
    ydot_raw = _sm_encode(-int(0.25 * (1 << 20)), 24)
    yddot_raw = _sm_encode(-2, 5)
    payload = _pack_string([
        (84, 4, 2),
        (80, 3, 4),                # B_n
        (77, 1, 1),                # P2
        (76, 7, 48),               # t_b = 48 * 15 min = 12h
        (64, 24, ydot_raw),
        (40, 5, yddot_raw),
        (35, 27, y_raw),
    ])
    out = decode_glonass_string2(payload)
    assert out["string"] == 2
    assert out["B_n"] == 4
    assert out["P2"] == 1
    assert out["t_b_s"] == 48 * 15 * 60
    assert out["y_m"] == pytest.approx(5000.0 * 1000.0)
    assert out["y_dot_m_s"] == pytest.approx(-0.25 * 1000.0)
    assert out["y_dot_dot_m_s2"] == pytest.approx(-2 * 2 ** -30 * 1000.0)


def test_glonass_string3_round_trip():
    z_raw = _sm_encode(-7777 * (1 << 11), 27)
    zdot_km_s_quanta = 100000
    zdot_raw = _sm_encode(zdot_km_s_quanta, 24)
    zddot_raw = _sm_encode(1, 5)
    gamma_raw = _sm_encode(-3, 11)
    payload = _pack_string([
        (84, 4, 3),
        (80, 1, 1),                # P3
        (79, 11, gamma_raw),
        (67, 2, 2),                # P (mode flag)
        (65, 1, 0),                # l_n
        (64, 24, zdot_raw),
        (40, 5, zddot_raw),
        (35, 27, z_raw),
    ])
    out = decode_glonass_string3(payload)
    assert out["string"] == 3
    assert out["P3"] == 1
    assert out["P"] == 2
    assert out["l_n"] == 0
    assert out["gamma_n"] == pytest.approx(-3 * 2 ** -40)
    assert out["z_m"] == pytest.approx(-7777.0 * 1000.0)
    assert out["z_dot_m_s"] == pytest.approx(zdot_km_s_quanta * 2 ** -20 * 1000.0)
    assert out["z_dot_dot_m_s2"] == pytest.approx(2 ** -30 * 1000.0)


def test_glonass_dispatch_unknown_string_returns_raw():
    payload = _pack_string([(84, 4, 7)])
    out = decode_glonass_string(payload)
    assert out["string"] == 7
    assert isinstance(out["raw"], bytes)


def test_glonass_wrong_string_number_raises():
    p = _pack_string([(84, 4, 2)])
    with pytest.raises(ValueError):
        decode_glonass_string1(p)
