"""Tests for the sequential (multi-epoch) RTK solver with carry-over."""

from __future__ import annotations

import numpy as np
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.rtk import SequentialRTK

_LAMBDA_L1 = 0.190293672798365


def _obs(rover, base, sv, true_amb, wavelength):
    rho_r = np.linalg.norm(sv - rover, axis=1)
    rho_b = np.linalg.norm(sv - base, axis=1)
    return rho_r.copy(), rho_b.copy(), rho_r / wavelength + true_amb, rho_b / wavelength + true_amb


def _setup_geometry():
    base = np.array(lla_to_ecef(40, -3, 0))
    sv = np.array([
        base + np.array([2.0e7, 1.0e7, 1.5e7]),
        base + np.array([-2.0e7, 1.0e7, 1.5e7]),
        base + np.array([0, 2.0e7, 2.0e7]),
        base + np.array([0, -2.0e7, 1.0e7]),
        base + np.array([1.5e7, 0, 1.7e7]),
        base + np.array([-1.0e7, -1.5e7, 1.3e7]),
    ])
    return base, sv


def test_sequential_rtk_carries_fix_across_clean_epochs():
    base, sv = _setup_geometry()
    rover = base + np.array([2.5, 1.7, -0.4])
    rng = np.random.default_rng(11)
    true_amb = rng.integers(-100, 100, size=sv.shape[0])
    sv_ids = [f"G{i:02d}" for i in range(sv.shape[0])]

    rtk = SequentialRTK(tuple(base), wavelength=_LAMBDA_L1)

    # Epoch 1: cold start, should resolve fix.
    pr_r, pr_b, ph_r, ph_b = _obs(rover, base, sv, true_amb, _LAMBDA_L1)
    r1 = rtk.update(sv_ids, pr_r, pr_b, ph_r, ph_b, sv)
    assert r1["fixed_accepted"]
    bx, by, bz = r1["baseline"]
    assert bx == approx(2.5, abs=1e-3)
    assert r1["carry_over_count"] == 0

    # Epoch 2: rover moves slightly, same SVs, no slip.
    rover2 = rover + np.array([0.5, -0.2, 0.1])
    pr_r2, pr_b2, ph_r2, ph_b2 = _obs(rover2, base, sv, true_amb, _LAMBDA_L1)
    r2 = rtk.update(sv_ids, pr_r2, pr_b2, ph_r2, ph_b2, sv)
    assert r2["fixed_accepted"]
    bx2, by2, bz2 = r2["baseline"]
    assert bx2 == approx(3.0, abs=1e-3)
    assert r2["carry_over_count"] > 0


def test_sequential_rtk_detects_cycle_slip():
    base, sv = _setup_geometry()
    rover = base + np.array([1.0, 2.0, 0.5])
    true_amb = np.array([10, 20, 30, 40, 50, 60])
    sv_ids = [f"G{i:02d}" for i in range(sv.shape[0])]

    rtk = SequentialRTK(tuple(base), wavelength=_LAMBDA_L1)
    pr_r, pr_b, ph_r, ph_b = _obs(rover, base, sv, true_amb, _LAMBDA_L1)
    rtk.update(sv_ids, pr_r, pr_b, ph_r, ph_b, sv)

    # Inject a 5-cycle slip on SV index 2 (rover-side only).
    rover2 = rover + np.array([0.2, 0.1, 0.0])
    pr_r2, pr_b2, ph_r2, ph_b2 = _obs(rover2, base, sv, true_amb, _LAMBDA_L1)
    ph_r2[2] += 5  # rover-only carrier-phase jump = slip
    r2 = rtk.update(sv_ids, pr_r2, pr_b2, ph_r2, ph_b2, sv)
    assert "G02" in r2["slipped_svs"]


def test_sequential_rtk_replay_no_epoch_rejected():
    """Roadmap acceptance: a moving-baseline replay produces continuous
    fixed solutions without rejecting > 5 % of epochs."""
    base, sv = _setup_geometry()
    true_amb = np.array([10, 20, 30, 40, 50, 60])
    sv_ids = [f"G{i:02d}" for i in range(sv.shape[0])]
    rtk = SequentialRTK(tuple(base), wavelength=_LAMBDA_L1)

    n_epochs = 20
    rejected = 0
    for k in range(n_epochs):
        rover = base + np.array([1.0 + 0.1 * k, 2.0 - 0.05 * k, 0.5 + 0.02 * k])
        pr_r, pr_b, ph_r, ph_b = _obs(rover, base, sv, true_amb, _LAMBDA_L1)
        r = rtk.update(sv_ids, pr_r, pr_b, ph_r, ph_b, sv)
        if not r["fixed_accepted"]:
            rejected += 1
    assert rejected / n_epochs <= 0.05


def test_sequential_rtk_reset_clears_state():
    base, sv = _setup_geometry()
    true_amb = np.array([10, 20, 30, 40, 50, 60])
    rover = base + np.array([1.0, 2.0, 0.5])
    sv_ids = [f"G{i:02d}" for i in range(sv.shape[0])]
    rtk = SequentialRTK(tuple(base), wavelength=_LAMBDA_L1)
    pr_r, pr_b, ph_r, ph_b = _obs(rover, base, sv, true_amb, _LAMBDA_L1)
    rtk.update(sv_ids, pr_r, pr_b, ph_r, ph_b, sv)
    assert rtk._lock
    rtk.reset()
    assert not rtk._lock
    assert not rtk._integer_amb
