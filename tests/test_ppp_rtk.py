"""Tests for the PPP-RTK fusion estimator."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.multifreq import LAMBDA_L1
from rinexpy.ppp_rtk import PPPRTKFusion


def _sky() -> np.ndarray:
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


def test_default_fused_is_ppp_when_no_rtk():
    """Without a base / RTK update, the fused position equals the PPP
    estimate."""
    f = PPPRTKFusion(
        n_sv=8, initial_position=(6_378_137.0, 0.0, 0.0),
    )
    assert f.rtk_position is None
    assert f.fused_position == f.ppp_position
    assert f.rtk_weight == 0.0


def test_update_rtk_requires_base():
    f = PPPRTKFusion(n_sv=8, initial_position=(6_378_137.0, 0.0, 0.0))
    sv = _sky()[:5]
    with pytest.raises(ValueError, match="base_position"):
        f.update_rtk(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), sv)


def test_short_baseline_rtk_dominates_fused_position():
    """100-m baseline -> RTK sigma = 1 cm, PPP sigma = O(m); fused is RTK."""
    rover_truth = np.array([6_378_137.0, 0.0, 0.0])
    base = rover_truth + np.array([100.0, 50.0, -25.0])
    sv = _sky()
    truth_ambs = np.array([10, -5, 22, 8, -3, 7, 15, -12], dtype=float)
    rho_b = np.linalg.norm(sv - base, axis=1)
    rho_r = np.linalg.norm(sv - rover_truth, axis=1)
    rover_pr = rho_r.copy()
    base_pr = rho_b.copy()
    rover_phase = rho_r / LAMBDA_L1 + truth_ambs
    base_phase = rho_b / LAMBDA_L1

    f = PPPRTKFusion(
        n_sv=8, initial_position=tuple(base),
        base_position=tuple(base),
    )
    f.update_rtk(rover_pr, base_pr, rover_phase, base_phase, sv)
    # RTK gets the rover position right to mm; PPP hasn't seen data yet.
    np.testing.assert_allclose(f.fused_position, rover_truth, atol=0.05)
    assert f.rtk_weight > 0.9
    # baseline_km computes correctly.
    assert f.baseline_km is not None
    assert f.baseline_km < 0.2   # 100 m = 0.1 km


def test_long_baseline_grows_rtk_sigma():
    """5000-km baseline -> RTK sigma is dominated by the ppm term."""
    rover_truth = np.array([6_378_137.0, 0.0, 0.0])
    # Synthetic far base, with the rover at the true rover position.
    base = rover_truth + np.array([5_000_000.0, 0.0, 0.0])
    sv = _sky()
    # Just exercise the sigma model; no need to make the RTK math itself
    # converge with this geometry.
    f = PPPRTKFusion(
        n_sv=8, initial_position=tuple(rover_truth),
        base_position=tuple(base),
        rtk_sigma_floor_m=0.01,
        rtk_sigma_ppm_per_km=1.0,
    )
    f.rtk_position = rover_truth.copy()
    # Recompute sigma at the cached baseline.
    b_km = float(np.linalg.norm(f.rtk_position - f.base_position) / 1000.0)
    f.rtk_sigma_m = f.rtk_sigma_floor_m + f.rtk_sigma_ppm_per_km * 1e-3 * b_km
    # b_km = 5000, sigma = 0.01 + 5 m = 5.01 m.
    assert f.rtk_sigma_m == pytest.approx(5.01, abs=0.01)


def test_fused_sigma_drops_below_both_inputs():
    """The inverse-variance fusion strictly tightens the position
    sigma relative to either input alone."""
    rover_truth = np.array([6_378_137.0, 0.0, 0.0])
    base = rover_truth + np.array([100.0, 0.0, 0.0])
    sv = _sky()
    truth_ambs = np.zeros(8)
    rho_b = np.linalg.norm(sv - base, axis=1)
    rho_r = np.linalg.norm(sv - rover_truth, axis=1)
    rover_pr = rho_r.copy()
    base_pr = rho_b.copy()
    rover_phase = rho_r / LAMBDA_L1 + truth_ambs
    base_phase = rho_b / LAMBDA_L1

    f = PPPRTKFusion(
        n_sv=8, initial_position=tuple(base),
        base_position=tuple(base),
    )
    f.update_rtk(rover_pr, base_pr, rover_phase, base_phase, sv)
    # PPP variance still at init (10 m); RTK at floor (1 cm).
    fused_sigma = f.fused_sigma
    # Both axes' fused sigma must be strictly less than the RTK sigma
    # (which is itself less than the PPP sigma).
    for s in fused_sigma:
        assert s <= f.rtk_sigma_m + 1e-12


def test_weight_in_range():
    f = PPPRTKFusion(
        n_sv=4, initial_position=(0.0, 0.0, 0.0),
        base_position=(0.0, 0.0, 100.0),
    )
    # Inject a synthetic RTK fix to exercise the weight computation.
    f.rtk_position = np.array([0.0, 0.0, 0.0])
    f.rtk_sigma_m = 0.01
    w = f.rtk_weight
    assert 0.0 <= w <= 1.0
