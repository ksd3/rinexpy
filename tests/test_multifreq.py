"""Tests for the dual-frequency LAMBDA helpers."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.multifreq import (
    LAMBDA_L1,
    LAMBDA_L2,
    LAMBDA_WL,
    lambda_dual_freq,
    melbourne_wubbena,
    narrow_lane_phase,
    resolve_wide_lane,
    split_wl_into_l1_l2,
    wide_lane_phase,
)


def _synthesize(rho_m: np.ndarray, n1: np.ndarray, n2: np.ndarray):
    """Build noise-free (phi1, phi2, p1, p2) consistent with a known geometry.

    Pseudorange = geometric range; phase (cycles) = range/lambda + N.
    """
    phi1 = rho_m / LAMBDA_L1 + n1
    phi2 = rho_m / LAMBDA_L2 + n2
    p1 = rho_m
    p2 = rho_m
    return phi1, phi2, p1, p2


def test_lane_wavelengths():
    assert approx(0.190294, abs=1e-5) == LAMBDA_L1
    assert approx(0.244210, abs=1e-5) == LAMBDA_L2
    assert approx(0.861918, abs=1e-5) == LAMBDA_WL
    # Sanity: WL >> L1, L2 -> easier to fix.
    assert LAMBDA_WL > LAMBDA_L1
    assert LAMBDA_WL > LAMBDA_L2


def test_wide_lane_phase_recovers_n1_minus_n2():
    """For zero geometric range, WL phase = N1 - N2."""
    n1 = np.array([5, -3, 0, 17])
    n2 = np.array([2, -1, 0, 13])
    phi1, phi2, _, _ = _synthesize(np.zeros(4), n1, n2)
    wl = wide_lane_phase(phi1, phi2)
    np.testing.assert_allclose(wl, n1 - n2, atol=1e-9)


def test_narrow_lane_phase_returns_finite():
    phi1 = np.array([1000.0, 2000.0])
    phi2 = np.array([800.0, 1500.0])
    nl = narrow_lane_phase(phi1, phi2)
    assert np.all(np.isfinite(nl))


def test_melbourne_wubbena_recovers_wl():
    """MW combination = lambda_WL * N_WL when noise-free."""
    rho = np.array([2.0e7, 2.1e7, 2.2e7])
    n1 = np.array([100, 200, 300])
    n2 = np.array([50, 150, 250])
    phi1, phi2, p1, p2 = _synthesize(rho, n1, n2)
    mw = melbourne_wubbena(phi1, phi2, p1, p2)
    expected = LAMBDA_WL * (n1 - n2)
    np.testing.assert_allclose(mw, expected, atol=1e-6)


def test_resolve_wide_lane_synthetic():
    rho = np.array([2.0e7] * 5)
    n1 = np.array([100, 200, 300, 400, 500])
    n2 = np.array([50, 150, 250, 350, 450])
    phi1, phi2, p1, p2 = _synthesize(rho, n1, n2)
    out = resolve_wide_lane(phi1, phi2, p1, p2)
    np.testing.assert_array_equal(out["N_WL"], n1 - n2)
    assert out["fraction_fixed"] == 1.0


def test_split_wl_into_l1_l2_round_trip():
    """split(N_WL=N1-N2, N_NL=N1+N2) recovers (N1, N2) exactly."""
    n1 = np.array([5, 7, -3, 11])
    n2 = np.array([2, 3, -1, 4])
    n_wl = n1 - n2
    n_nl = n1 + n2
    a, b = split_wl_into_l1_l2(n_wl, n_nl)
    np.testing.assert_array_equal(a, n1)
    np.testing.assert_array_equal(b, n2)


def test_split_wl_inconsistent_raises():
    """Odd N1+N2 sum is impossible by construction; raise."""
    with pytest.raises(ValueError):
        split_wl_into_l1_l2(np.array([1]), np.array([2]))


def test_lambda_dual_freq_with_float_ambiguities():
    """Synthetic data: dual-freq LAMBDA recovers both N1 and N2 from
    float ambiguities (no pseudoranges needed when geometry is already
    differenced out of the float estimates)."""
    n1_true = np.array([10, 20, 30, 40, 50, 60])
    n2_true = np.array([5, 15, 25, 35, 45, 55])
    # In a real RTK, the LSQ produces float ambiguities very close to
    # the true integers. Synthesize: integer + small noise.
    rng = np.random.default_rng(0)
    a_l1_float = n1_true + rng.normal(0, 0.05, size=n1_true.size)
    a_l2_float = n2_true + rng.normal(0, 0.05, size=n2_true.size)
    out = lambda_dual_freq(a_l1_float, a_l2_float)
    assert out["fraction_fixed"] == 1.0
    np.testing.assert_array_equal(out["N_L1"], n1_true)
    np.testing.assert_array_equal(out["N_L2"], n2_true)


def test_lambda_dual_freq_without_pseudoranges():
    """When no PR is given, falls back to rounding the L1-L2 difference."""
    a_l1 = np.array([10.05, 20.0, 30.0])
    a_l2 = np.array([5.05, 15.05, 25.0])
    out = lambda_dual_freq(a_l1, a_l2)
    np.testing.assert_array_equal(out["N_WL"], [10 - 5, 20 - 15, 30 - 25])


def test_lambda_dual_freq_unfixed_when_noisy():
    """Large rounding residuals (>= 0.25 cycles) mark an SV unfixed."""
    a_l1 = np.array([10.0, 20.4, 30.0])
    a_l2 = np.array([5.0, 15.0, 25.0])
    out = lambda_dual_freq(a_l1, a_l2, sigma_threshold=0.25)
    assert out["fixed_mask"][1] is np.False_ or not out["fixed_mask"][1]
    assert out["N_L1"][1] == 0  # zeroed when unfixed
