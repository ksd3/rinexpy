"""Tests for the single- and dual-frequency cycle slip repair routines."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.qc import (
    detect_slips_mw,
    detect_slips_phase_only,
    repair_slips,
    repair_slips_dual,
)


def test_repair_no_slips_leaves_input_unchanged():
    n = 50
    phi = np.linspace(1_000_000.0, 1_000_050.0, n)
    slips = np.zeros(n, dtype=bool)
    repaired = repair_slips(phi, slips)
    np.testing.assert_allclose(repaired, phi)


def test_repair_single_integer_jump():
    """A 7-cycle slip introduced mid-series should be removed cleanly."""
    n = 40
    truth = 1_000_000.0 + np.arange(n) * 0.1  # constant slope of 0.1 cyc/epoch
    phi = truth.copy()
    phi[20:] += 7   # 7-cycle slip at epoch 20
    slips = np.zeros(n, dtype=bool)
    slips[20] = True
    repaired = repair_slips(phi, slips, fit_window=5)
    np.testing.assert_allclose(repaired, truth)


def test_repair_multiple_slips_in_sequence():
    n = 60
    truth = 1_000_000.0 + np.arange(n) * 0.1
    phi = truth.copy()
    phi[20:] += 5
    phi[40:] += -3
    slips = np.zeros(n, dtype=bool)
    slips[20] = True
    slips[40] = True
    repaired = repair_slips(phi, slips, fit_window=5)
    np.testing.assert_allclose(repaired, truth)


def test_repair_handles_nans_at_slip_epoch():
    n = 40
    truth = 1_000_000.0 + np.arange(n) * 0.1
    phi = truth.copy()
    phi[20] = np.nan
    slips = np.zeros(n, dtype=bool)
    slips[20] = True
    # No jump, but slip epoch is NaN -> repair leaves the array alone
    # except for the NaN.
    repaired = repair_slips(phi, slips, fit_window=5)
    assert np.isnan(repaired[20])
    np.testing.assert_allclose(repaired[21:], truth[21:])


def test_repair_dual_resolves_n1_n2_jump():
    """Generate a synthetic dual-frequency record with a known (dN1, dN2)
    slip and confirm the dual-frequency repair restores both signals."""
    n = 50
    f1 = 1575.42e6
    f2 = 1227.60e6
    c = 299_792_458.0
    lam1 = c / f1
    lam2 = c / f2
    # Linear-in-time geometry to make MW perfectly stationary in truth.
    geom = 22_000_000.0 + np.arange(n) * 10.0
    N1 = 12_345
    N2 =  8_765
    phi1 = geom / lam1 + N1
    phi2 = geom / lam2 + N2
    p1 = geom + 0.0
    p2 = geom + 0.0
    # Inject a (dN1, dN2) = (4, 1) slip at epoch 25.
    phi1[25:] += 4
    phi2[25:] += 1
    slips = np.zeros(n, dtype=bool)
    slips[25] = True
    p1r, p2r = repair_slips_dual(phi1, phi2, p1, p2, slips, f1=f1, f2=f2)
    np.testing.assert_allclose(p1r[25:], geom[25:] / lam1 + N1)
    np.testing.assert_allclose(p2r[25:], geom[25:] / lam2 + N2)


def test_detect_then_repair_round_trip():
    """End-to-end: phase-only detector flags a synthetic slip, the
    repair removes it, and the post-repair phase matches truth."""
    n = 80
    truth = 1_000_000.0 + np.arange(n) * 0.05 - 0.0001 * np.arange(n) ** 2
    phi = truth.copy()
    phi[50:] += -11
    flags = detect_slips_phase_only(phi, threshold_cycles=2.0)
    assert flags[50]
    repaired = repair_slips(phi, flags, fit_window=8)
    # Allow tiny rounding noise where the quadratic-curvature prediction
    # is slightly off — but the integer should be right and the residual
    # should be tiny (well under 1 cycle).
    diff = repaired[50:] - truth[50:]
    assert np.max(np.abs(diff)) < 0.5
