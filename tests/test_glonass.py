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
