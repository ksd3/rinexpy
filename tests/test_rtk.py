"""Tests for the RTK float-ambiguity solver."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.rtk import double_difference_solve

_LAMBDA_L1 = 0.190293672798365  # GPS L1 wavelength in m


def _build_synthetic(rover, base, sv, true_amb_cycles, wavelength):
    """Build noise-free pseudorange + phase observations for both receivers."""
    rho_r = np.linalg.norm(sv - rover, axis=1)
    rho_b = np.linalg.norm(sv - base, axis=1)
    pr_r = rho_r.copy()
    pr_b = rho_b.copy()
    # Phase: rho/wavelength + integer ambiguity (cycles)
    phase_r = rho_r / wavelength + true_amb_cycles
    phase_b = rho_b / wavelength + true_amb_cycles
    return pr_r, pr_b, phase_r, phase_b


def test_rtk_recovers_baseline():
    """Noise-free synthetic data: float-DD recovers the baseline to mm."""
    rng = np.random.default_rng(42)
    base = np.array(lla_to_ecef(40, -3, 0))
    rover = base + np.array([10.0, -5.0, 2.0])  # 10-m baseline

    sv = np.array(
        [
            base + np.array([2.0e7, 1.0e7, 1.5e7]),
            base + np.array([-2.0e7, 1.0e7, 1.5e7]),
            base + np.array([0, 2.0e7, 2.0e7]),
            base + np.array([0, -2.0e7, 1.0e7]),
            base + np.array([1.5e7, 0, 1.7e7]),
            base + np.array([-1.0e7, -1.5e7, 1.3e7]),
        ]
    )
    true_amb = rng.integers(-1000, 1000, size=sv.shape[0])
    pr_r, pr_b, ph_r, ph_b = _build_synthetic(rover, base, sv, true_amb, _LAMBDA_L1)

    sol = double_difference_solve(
        pr_r, pr_b, ph_r, ph_b, sv, tuple(base), wavelength=_LAMBDA_L1
    )
    bx, by, bz = sol["baseline"]
    assert bx == approx(10.0, abs=1e-3)
    assert by == approx(-5.0, abs=1e-3)
    assert bz == approx(2.0, abs=1e-3)


def test_rtk_too_few_sats():
    sv = np.zeros((4, 3))
    pr = np.zeros(4)
    with pytest.raises(ValueError):
        double_difference_solve(
            pr, pr, pr, pr, sv, (0, 0, 0), wavelength=_LAMBDA_L1
        )


def test_rtk_returns_rover_position():
    """rover_position = base + baseline."""
    base = np.array(lla_to_ecef(0, 0, 0))
    rover = base + np.array([3.0, 4.0, 0.0])
    sv = np.array(
        [
            base + np.array([2.0e7, 1.0e7, 1.5e7]),
            base + np.array([-2.0e7, 1.0e7, 1.5e7]),
            base + np.array([0, 2.0e7, 2.0e7]),
            base + np.array([0, -2.0e7, 1.0e7]),
            base + np.array([1.5e7, 0, 1.7e7]),
        ]
    )
    pr_r, pr_b, ph_r, ph_b = _build_synthetic(
        rover, base, sv, np.zeros(sv.shape[0]), _LAMBDA_L1
    )
    sol = double_difference_solve(
        pr_r, pr_b, ph_r, ph_b, sv, tuple(base), wavelength=_LAMBDA_L1
    )
    assert sol["rover_position"][0] == approx(rover[0], abs=1e-3)
    assert sol["rover_position"][1] == approx(rover[1], abs=1e-3)
    assert sol["rover_position"][2] == approx(rover[2], abs=1e-3)
