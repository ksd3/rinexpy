"""Tests for the network double-difference solver."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.network_dd import network_dd_solve


C = 299_792_458.0
F_L1 = 1575.42e6
LAMBDA_L1 = C / F_L1


def _make_baseline(
    base_pos: np.ndarray,
    rover_pos: np.ndarray,
    sv: np.ndarray,
    ambs: np.ndarray,
    *,
    pr_noise: float = 0.0,
    phase_noise_cycles: float = 0.0,
    seed: int = 0,
) -> dict:
    """Build a synthetic per-baseline observation set from known truth."""
    rng = np.random.default_rng(seed)
    n = sv.shape[0]
    rho_b = np.linalg.norm(sv - base_pos, axis=1)
    rho_r = np.linalg.norm(sv - rover_pos, axis=1)
    # Identical receiver / SV clock biases on rover and base cancel in
    # the DD, so we don't need to add them here.
    rover_pr = rho_r + rng.normal(0.0, pr_noise, n)
    base_pr = rho_b + rng.normal(0.0, pr_noise, n)
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
    """A 6-satellite sky distribution used across the test cases."""
    return np.array([
        [ 1.5e7,  0.0,    2.0e7],
        [-1.5e7,  0.5e7,  2.0e7],
        [ 0.5e7,  1.5e7,  1.8e7],
        [-0.5e7, -1.5e7,  1.5e7],
        [ 2.0e7, -0.5e7,  1.0e7],
        [-2.0e7,  1.0e7,  0.8e7],
    ])


def test_single_baseline_matches_truth_noise_free():
    """One baseline in the network = single-baseline RTK; truth recovered."""
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base = rover + np.array([100.0, 50.0, -25.0])
    sv = _make_sky()
    ambs = np.array([10, -5, 22, 8, -3, 0], dtype=float)
    bl = _make_baseline(base, rover, sv, ambs)
    sol = network_dd_solve([bl], wavelength=LAMBDA_L1, initial_rover=tuple(base))
    np.testing.assert_allclose(sol["rover_position"], rover, atol=1e-6)


def test_two_baselines_share_rover_position():
    """Two bases. Rover position must come out the same; each baseline
    has its own ambiguity set."""
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base_a = rover + np.array([150.0, 100.0,   0.0])
    base_b = rover + np.array([-80.0,  60.0,  40.0])
    sv = _make_sky()
    ambs_a = np.array([12, -7, 25, 9, -1, 2], dtype=float)
    ambs_b = np.array([ 3, 18, -2, 11, 4, -6], dtype=float)
    bl_a = _make_baseline(base_a, rover, sv, ambs_a, seed=1)
    bl_b = _make_baseline(base_b, rover, sv, ambs_b, seed=2)
    sol = network_dd_solve(
        [bl_a, bl_b], wavelength=LAMBDA_L1, initial_rover=tuple(base_a)
    )
    np.testing.assert_allclose(sol["rover_position"], rover, atol=1e-6)
    assert len(sol["ambiguities"]) == 2


def test_three_baselines_with_phase_noise():
    """Three bases plus mm-level phase noise. Position should still be
    mm-level."""
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base_a = rover + np.array([200.0, 100.0,  20.0])
    base_b = rover + np.array([-120.0, 70.0,  30.0])
    base_c = rover + np.array([  60.0,-180.0,-10.0])
    sv = _make_sky()
    rng = np.random.default_rng(42)
    ambs_a = rng.integers(-50, 50, 6).astype(float)
    ambs_b = rng.integers(-50, 50, 6).astype(float)
    ambs_c = rng.integers(-50, 50, 6).astype(float)
    bl_a = _make_baseline(base_a, rover, sv, ambs_a, phase_noise_cycles=0.005, seed=10)
    bl_b = _make_baseline(base_b, rover, sv, ambs_b, phase_noise_cycles=0.005, seed=20)
    bl_c = _make_baseline(base_c, rover, sv, ambs_c, phase_noise_cycles=0.005, seed=30)
    sol = network_dd_solve(
        [bl_a, bl_b, bl_c], wavelength=LAMBDA_L1, initial_rover=tuple(base_a)
    )
    err = np.linalg.norm(np.array(sol["rover_position"]) - rover)
    # Three baselines and 0.5-cm carrier noise: ~mm position error.
    assert err < 0.01


def test_rejects_zero_baselines():
    with pytest.raises(ValueError, match=">= 1 baseline"):
        network_dd_solve([], wavelength=LAMBDA_L1)


def test_rejects_too_few_satellites():
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base = rover + np.array([100.0, 0.0, 0.0])
    sv = _make_sky()[:4]   # only 4 SVs
    bl = _make_baseline(base, rover, sv, np.zeros(4))
    with pytest.raises(ValueError, match=">= 5"):
        network_dd_solve([bl], wavelength=LAMBDA_L1)


def test_recovered_ambiguities_are_close_to_truth():
    """Float ambiguities should match the truth integers within < 0.5 cycles
    in the noiseless case."""
    rover = np.array([6_378_137.0, 0.0, 0.0])
    base = rover + np.array([100.0, 50.0, -25.0])
    sv = _make_sky()
    truth_ambs = np.array([10, -5, 22, 8, -3, 0], dtype=float)
    bl = _make_baseline(base, rover, sv, truth_ambs)
    sol = network_dd_solve([bl], wavelength=LAMBDA_L1, initial_rover=tuple(base))
    # Reference SV's ambiguity is differenced out; the recovered DD
    # ambiguities equal truth_others - truth_ref.
    ref = sol["reference_sv"][0]
    others = [i for i in range(6) if i != ref]
    expected_dd = truth_ambs[others] - truth_ambs[ref]
    np.testing.assert_allclose(sol["ambiguities"][0], expected_dd, atol=0.5)
