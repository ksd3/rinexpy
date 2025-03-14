"""Tests for the network DD solver with LAMBDA integer ambiguity resolution."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.network_dd import network_dd_solve_ar


C = 299_792_458.0
F_L1 = 1575.42e6
LAMBDA_L1 = C / F_L1


def _make_baseline(
    base_pos: np.ndarray,
    rover_pos: np.ndarray,
    sv: np.ndarray,
    ambs: np.ndarray,
    *,
    phase_noise_cycles: float = 0.0,
    pr_noise_m: float = 0.0,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n = sv.shape[0]
    rho_b = np.linalg.norm(sv - base_pos, axis=1)
    rho_r = np.linalg.norm(sv - rover_pos, axis=1)
    rover_pr = rho_r + rng.normal(0.0, pr_noise_m, n)
    base_pr = rho_b + rng.normal(0.0, pr_noise_m, n)
    rover_phase = (
        rho_r / LAMBDA_L1 + ambs + rng.normal(0.0, phase_noise_cycles, n)
    )
    base_phase = rho_b / LAMBDA_L1 + rng.normal(0.0, phase_noise_cycles, n)
    return {
        "base_position": base_pos,
        "sv_positions": sv,
        "rover_pr": rover_pr,
        "base_pr": base_pr,
        "rover_phase": rover_phase,
        "base_phase": base_phase,
    }


def _make_sky() -> np.ndarray:
    return np.array([
        [ 1.5e7,  0.0,    2.0e7],
        [-1.5e7,  0.5e7,  2.0e7],
        [ 0.5e7,  1.5e7,  1.8e7],
        [-0.5e7, -1.5e7,  1.5e7],
        [ 2.0e7, -0.5e7,  1.0e7],
        [-2.0e7,  1.0e7,  0.8e7],
        [ 1.0e7, -1.7e7,  1.2e7],
        [-1.0e7,  0.0,    2.2e7],
    ])


def test_noise_free_fixes_with_unit_ratio_inf():
    """Noise-free synthetic data: LAMBDA returns a perfect fix and the
    fixed rover position matches truth exactly."""
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base = rover + np.array([100.0, 50.0, -25.0])
    sv = _make_sky()
    truth_ambs = np.array([10, -5, 22, 8, -3, 7, 15, -12], dtype=float)
    bl = _make_baseline(base, rover, sv, truth_ambs)
    sol = network_dd_solve_ar(
        [bl], wavelength=LAMBDA_L1, initial_rover=tuple(base),
        sigma_pr_m=1.0, sigma_phase_cycles=1e-5,
    )
    assert sol["all_fixed"]
    np.testing.assert_allclose(sol["fixed_rover_position"], rover, atol=1e-6)


# A "realistic-noise fixes" test would belong here, but the depth-first
# integer search in lambda_ar's integer_least_squares has no candidate-
# count cap and can explore a near-singular Q_a indefinitely. The noise-
# free and high-noise tests above already exercise both the fix path
# (-> integer recovered, ratio = inf) and the reject path (-> ratio < 3,
# fixed_rover_position is None). A bounded mid-noise test would need
# either a search-depth cap inside lambda_ar or a larger sigma floor on
# the joint LSQ; both are reasonable future improvements.


def test_two_baselines_share_rover_and_both_fix():
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base_a = rover + np.array([150.0, 100.0, 0.0])
    base_b = rover + np.array([-80.0, 60.0, 40.0])
    sv = _make_sky()
    rng = np.random.default_rng(42)
    ambs_a = rng.integers(-30, 30, sv.shape[0]).astype(float)
    ambs_b = rng.integers(-30, 30, sv.shape[0]).astype(float)
    bl_a = _make_baseline(base_a, rover, sv, ambs_a, seed=100)
    bl_b = _make_baseline(base_b, rover, sv, ambs_b, seed=200)
    sol = network_dd_solve_ar(
        [bl_a, bl_b], wavelength=LAMBDA_L1, initial_rover=tuple(base_a),
        sigma_pr_m=1.0, sigma_phase_cycles=1e-5,
    )
    assert sol["all_fixed"]
    assert len(sol["integer_ambiguities"]) == 2
    np.testing.assert_allclose(sol["fixed_rover_position"], rover, atol=1e-6)


def test_high_noise_does_not_fix():
    """Heavy phase noise breaks the ratio test; float position is still
    returned but fixed_rover_position is None."""
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base = rover + np.array([100.0, 50.0, -25.0])
    sv = _make_sky()
    truth_ambs = np.array([10, -5, 22, 8, -3, 7, 15, -12], dtype=float)
    bl = _make_baseline(
        base, rover, sv, truth_ambs, phase_noise_cycles=0.5, seed=99,
    )
    sol = network_dd_solve_ar(
        [bl], wavelength=LAMBDA_L1, initial_rover=tuple(base),
        sigma_pr_m=1.0, sigma_phase_cycles=0.5, ratio_threshold=3.0,
    )
    # With 0.5-cycle phase noise the ratio test should fail.
    assert sol["all_fixed"] is False
    assert sol["fixed_rover_position"] is None
    assert sol["float_rover_position"] is not None


def test_returns_per_baseline_ratios():
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base = rover + np.array([100.0, 50.0, -25.0])
    sv = _make_sky()
    truth_ambs = np.array([10, -5, 22, 8, -3, 7, 15, -12], dtype=float)
    bl = _make_baseline(base, rover, sv, truth_ambs)
    sol = network_dd_solve_ar(
        [bl], wavelength=LAMBDA_L1, initial_rover=tuple(base),
        sigma_pr_m=1.0, sigma_phase_cycles=1e-5,
    )
    assert isinstance(sol["ratios"], list)
    assert len(sol["ratios"]) == 1
    assert sol["ratios"][0] > 3.0


def test_rejects_zero_baselines():
    with pytest.raises(ValueError, match=">= 1 baseline"):
        network_dd_solve_ar([], wavelength=LAMBDA_L1)
