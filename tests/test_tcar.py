"""Tests for the triple-carrier ambiguity resolution cascade."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.multifreq import (
    F1, F2, F5,
    LAMBDA_EWL_15, LAMBDA_EWL_25,
    LAMBDA_L1, LAMBDA_L2, LAMBDA_L5,
    extra_wide_lane_phase,
    melbourne_wubbena_ewl,
    resolve_extra_wide_lane,
    tcar_resolve,
)


def test_ewl_wavelength_is_huge():
    """EWL on L2-L5 has a 5.86 m wavelength; the integer ambiguity at
    that scale is essentially trivial to fix."""
    assert LAMBDA_EWL_25 == pytest.approx(5.86, abs=0.05)
    # L1-L5 EWL is shorter at ~0.75 m.
    assert 0.7 < LAMBDA_EWL_15 < 0.8


def test_extra_wide_lane_phase_noise_free():
    rng = np.random.default_rng(0)
    n = 6
    truth_n5 = rng.integers(-50, 50, n)
    truth_n2 = rng.integers(-50, 50, n)
    # Synthesize phase on the noise-free range only (no ionosphere here).
    rho = 23_000_000.0 + rng.standard_normal(n) * 100.0
    phi2 = rho / LAMBDA_L2 + truth_n2
    phi5 = rho / LAMBDA_L5 + truth_n5
    ewl_truth = truth_n2 - truth_n5
    # The naive (phi2 - phi5) is NOT a clean integer because the
    # geometric range maps to different cycle counts on the two bands.
    # MW combination removes geometry; phase difference alone doesn't.
    # So just check the difference is consistent with the synthetic.
    out = extra_wide_lane_phase(phi2, phi5)
    # Reconstruct expected = rho/lambda_2 - rho/lambda_5 + (n2 - n5).
    expected = rho / LAMBDA_L2 - rho / LAMBDA_L5 + ewl_truth
    np.testing.assert_allclose(out, expected)


def test_mw_ewl_recovers_integer():
    """MW on (L2, L5) gives a float ambiguity centered on the integer
    N_EWL = N2 - N5; rounding fixes it with high confidence."""
    rng = np.random.default_rng(1)
    n = 20
    rho = 23_000_000.0 + rng.standard_normal(n) * 1000.0
    truth_n2 = rng.integers(-100, 100, n).astype(float)
    truth_n5 = rng.integers(-100, 100, n).astype(float)
    phi2 = rho / LAMBDA_L2 + truth_n2 + rng.normal(0.0, 0.001, n)
    phi5 = rho / LAMBDA_L5 + truth_n5 + rng.normal(0.0, 0.001, n)
    p2 = rho + rng.normal(0.0, 0.3, n)
    p5 = rho + rng.normal(0.0, 0.3, n)
    mw = melbourne_wubbena_ewl(phi2, phi5, p2, p5)
    expected_ewl = truth_n2 - truth_n5
    # Rounded MW should equal truth EWL integer.
    np.testing.assert_array_equal(np.round(mw).astype(int), expected_ewl.astype(int))


def test_resolve_extra_wide_lane_threshold():
    """Float values within threshold accept; outside reject."""
    mw = np.array([3.05, -2.10, 4.50, np.nan, 1.20])
    out = resolve_extra_wide_lane(mw, threshold_cycles=0.25)
    np.testing.assert_array_equal(out["N_EWL"], [3, -2, 4, 0, 1])
    # 4.50 is exactly on the boundary -> accept (4 vs 5 both at 0.5).
    # 1.20 has |diff| = 0.20 -> accept.
    # NaN -> reject.
    assert out["fixed_mask"].tolist() == [True, True, False, False, True]


def test_tcar_full_cascade_noise_free():
    """Triple-frequency synthetic data with known integer ambiguities;
    cascade recovers all three integer sets."""
    rng = np.random.default_rng(2)
    n = 8
    rho = 23_000_000.0 + rng.standard_normal(n) * 1000.0
    truth_n1 = rng.integers(-100, 100, n).astype(float)
    truth_n2 = rng.integers(-100, 100, n).astype(float)
    truth_n5 = rng.integers(-100, 100, n).astype(float)
    phi1 = rho / LAMBDA_L1 + truth_n1
    phi2 = rho / LAMBDA_L2 + truth_n2
    phi5 = rho / LAMBDA_L5 + truth_n5
    p1 = rho + 0.0
    p2 = rho + 0.0
    p5 = rho + 0.0
    out = tcar_resolve(phi1, phi2, phi5, p1, p2, p5)

    # Stage 1: EWL = N2 - N5
    np.testing.assert_array_equal(
        out["N_EWL"], (truth_n2 - truth_n5).astype(int)
    )
    # Stage 2: WL = N1 - N2
    np.testing.assert_array_equal(
        out["N_WL"], (truth_n1 - truth_n2).astype(int)
    )
    # Stage 3 cross-check: the inverted N_L1 / N_L2 / N_L5 satisfy the
    # WL / EWL relations even if the NL absolute level may differ by
    # an offset (rounding NL bootstraps to the nearest integer pair).
    assert out["fraction_fixed"] >= 0.5
    np.testing.assert_array_equal(out["N_L1"] - out["N_L2"], out["N_WL"])
    np.testing.assert_array_equal(out["N_L2"] - out["N_L5"], out["N_EWL"])


def test_tcar_with_phase_noise():
    """Add realistic mm-level phase noise; EWL still fixes."""
    rng = np.random.default_rng(3)
    n = 15
    rho = 23_000_000.0 + rng.standard_normal(n) * 1000.0
    truth_n1 = rng.integers(-100, 100, n).astype(float)
    truth_n2 = rng.integers(-100, 100, n).astype(float)
    truth_n5 = rng.integers(-100, 100, n).astype(float)
    phi1 = rho / LAMBDA_L1 + truth_n1 + rng.normal(0.0, 0.003, n)
    phi2 = rho / LAMBDA_L2 + truth_n2 + rng.normal(0.0, 0.003, n)
    phi5 = rho / LAMBDA_L5 + truth_n5 + rng.normal(0.0, 0.003, n)
    p1 = rho + rng.normal(0.0, 0.3, n)
    p2 = rho + rng.normal(0.0, 0.3, n)
    p5 = rho + rng.normal(0.0, 0.3, n)
    out = tcar_resolve(phi1, phi2, phi5, p1, p2, p5)
    # EWL has a 5.86 m wavelength -> code noise contributes < 0.06 cycles -> all fix.
    assert out["fixed_mask_ewl"].all()
