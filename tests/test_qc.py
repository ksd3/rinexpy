"""Tests for cycle slip detection."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from rinexpy.qc import (
    detect_slips,
    detect_slips_geometry_free,
    detect_slips_mw,
    detect_slips_phase_only,
)

_C = 299_792_458.0
_F_L1 = 1575.42e6
_F_L2 = 1227.60e6
_LAMBDA_L1 = _C / _F_L1
_LAMBDA_L2 = _C / _F_L2


def _simulate_dual_freq(n, slip_at=None, slip_cycles=0.0, seed=42, code_noise_m=0.05):
    """Synthetic dual-frequency observations with shared geometry + iono.

    Default code noise is 0.05 m (geodetic-quality receiver). Raw MW
    is noise-dominated by the code term, so this level is what keeps
    MW first-diff RMS well below the default 0.5-cycle threshold.
    """
    rng = np.random.default_rng(seed)
    gamma = (_F_L1 / _F_L2) ** 2
    rho = 2.5e7 + np.arange(n) * 100.0
    iono = 5.0
    n1, n2 = 1000.0, 1500.0
    phi1 = (rho - iono) / _LAMBDA_L1 + n1 + rng.normal(0, 0.005, n)
    phi2 = (rho - gamma * iono) / _LAMBDA_L2 + n2 + rng.normal(0, 0.005, n)
    p1 = rho + iono + rng.normal(0, code_noise_m, n)
    p2 = rho + gamma * iono + rng.normal(0, code_noise_m, n)
    if slip_at is not None:
        phi1[slip_at:] += slip_cycles
    return phi1, phi2, p1, p2


def test_phase_only_clean():
    """Smooth phase, no slips detected."""
    n = 100
    phi = 1000.0 + np.arange(n) * 1.5 + 0.01 * np.sin(np.arange(n) * 0.1)
    assert not detect_slips_phase_only(phi).any()


def test_phase_only_detects_5cycle_jump():
    n = 100
    phi = 1000.0 + np.arange(n) * 1.5
    phi[50:] += 5.0
    slips = detect_slips_phase_only(phi)
    assert slips[50]


def test_phase_only_handles_short_inputs():
    assert not detect_slips_phase_only(np.array([])).any()
    assert not detect_slips_phase_only(np.array([1.0, 2.0])).any()


def test_phase_only_handles_nan():
    """A NaN epoch shouldn't trip a false slip."""
    n = 50
    phi = 1000.0 + np.arange(n) * 0.5
    phi[20] = np.nan
    slips = detect_slips_phase_only(phi)
    # NaN propagates into d2 -> isfinite filter -> no slip flagged.
    assert not slips[22]


def test_gf_clean():
    """Constant ionosphere: GF first-diff stays at zero."""
    phi1, phi2, _, _ = _simulate_dual_freq(100)
    assert not detect_slips_geometry_free(phi1, phi2).any()


def test_gf_detects_1cycle_slip():
    """ROADMAP acceptance: 1-cycle slip flagged."""
    phi1, phi2, _, _ = _simulate_dual_freq(100, slip_at=30, slip_cycles=1.0)
    slips = detect_slips_geometry_free(phi1, phi2)
    assert slips[30]


def test_mw_clean():
    """Clean dual-freq data: MW noise stays below the 0.5-cycle threshold."""
    phi1, phi2, p1, p2 = _simulate_dual_freq(100)
    slips = detect_slips_mw(phi1, phi2, p1, p2)
    # Allow at most a couple of noise-driven false positives.
    assert slips.sum() <= 2


def test_mw_detects_1cycle_slip():
    """ROADMAP acceptance: 1-cycle slip on phi1 is detected at the slip epoch."""
    phi1, phi2, p1, p2 = _simulate_dual_freq(100, slip_at=50, slip_cycles=1.0)
    slips = detect_slips_mw(phi1, phi2, p1, p2)
    assert slips[50]


def test_detect_slips_picks_mw_when_dual_freq_present():
    """Full dual-freq code + carrier: detect_slips picks MW."""
    n = 60
    phi1, phi2, p1, p2 = _simulate_dual_freq(n)
    phi1b, phi2b, p1b, p2b = _simulate_dual_freq(n, slip_at=20, slip_cycles=1.0, seed=7)
    obs = xr.Dataset(
        {
            "L1C": (("time", "sv"), np.column_stack([phi1, phi1b])),
            "L2C": (("time", "sv"), np.column_stack([phi2, phi2b])),
            "C1C": (("time", "sv"), np.column_stack([p1, p1b])),
            "C2C": (("time", "sv"), np.column_stack([p2, p2b])),
        },
        coords={"time": np.arange(n), "sv": ["G01", "G02"]},
    )
    rep = detect_slips(obs)
    assert rep["methods_by_sv"]["G01"] == "mw"
    assert rep["methods_by_sv"]["G02"] == "mw"
    assert 20 in rep["slips_by_sv"]["G02"]
    assert 20 not in rep["slips_by_sv"]["G01"]


def test_detect_slips_falls_back_to_gf():
    """No code data: detect_slips uses the GF combination."""
    phi1, phi2, _, _ = _simulate_dual_freq(60)
    obs = xr.Dataset(
        {
            "L1C": (("time", "sv"), phi1[:, None]),
            "L2C": (("time", "sv"), phi2[:, None]),
        },
        coords={"time": np.arange(60), "sv": ["G01"]},
    )
    assert detect_slips(obs)["methods_by_sv"]["G01"] == "gf"


def test_detect_slips_falls_back_to_phase_only():
    """No L2 at all: phase-only on L1."""
    n = 60
    phi1 = 1.0e8 + np.arange(n) * 1000.0
    obs = xr.Dataset(
        {"L1C": (("time", "sv"), phi1[:, None])},
        coords={"time": np.arange(n), "sv": ["G01"]},
    )
    assert detect_slips(obs)["methods_by_sv"]["G01"] == "phase"


def test_detect_slips_requires_l1():
    obs = xr.Dataset(
        {"C1C": (("time", "sv"), [[1.0]])},
        coords={"time": [0], "sv": ["G01"]},
    )
    with pytest.raises(ValueError, match="L1"):
        detect_slips(obs)


def test_detect_slips_handles_rinex2_names():
    """Falls back to RINEX 2 P1/P2/L1/L2 names too."""
    n = 60
    phi1, phi2, p1, p2 = _simulate_dual_freq(n, slip_at=15, slip_cycles=1.0)
    obs = xr.Dataset(
        {
            "L1": (("time", "sv"), phi1[:, None]),
            "L2": (("time", "sv"), phi2[:, None]),
            "P1": (("time", "sv"), p1[:, None]),
            "P2": (("time", "sv"), p2[:, None]),
        },
        coords={"time": np.arange(n), "sv": ["G01"]},
    )
    rep = detect_slips(obs)
    assert rep["methods_by_sv"]["G01"] == "mw"
    assert 15 in rep["slips_by_sv"]["G01"]
