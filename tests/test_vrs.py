"""Tests for the VRS (Virtual Reference Station) synthesizer."""

from __future__ import annotations

import numpy as np
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.rtk import double_difference_solve
from rinexpy.vrs import synthesize_vrs

_LAMBDA_L1 = 0.190293672798365


def _build(bases_lla, rover_lla, sv_positions, true_amb_cycles, wavelength,
           atmosphere_m: float = 0.0):
    """Build a synthetic network: each base sees the SAME satellite
    positions (i.e. the SVs are far overhead and don't move across the
    short network baseline). Each base experiences the same atmosphere
    bias on every SV."""
    sv = np.asarray(sv_positions, dtype=float)
    bases = []
    for lat, lon, alt in bases_lla:
        base = np.array(lla_to_ecef(lat, lon, alt))
        rho = np.linalg.norm(sv - base, axis=1)
        bases.append({
            "base_position": tuple(base),
            "sv_positions": sv.copy(),
            "pr": rho + atmosphere_m,
            "phase": rho / wavelength + true_amb_cycles,
        })
    rover = np.array(lla_to_ecef(*rover_lla))
    rover_rho = np.linalg.norm(sv - rover, axis=1)
    rover_obs = {
        "pr": rover_rho + atmosphere_m,
        "phase": rover_rho / wavelength + true_amb_cycles,
        "sv_positions": sv.copy(),
    }
    return bases, rover, rover_obs


def _sv_geometry():
    """Absolute SV ECEF positions (far enough above Earth that small
    surface baselines don't change which SV-to-base vector you get)."""
    earth = np.array(lla_to_ecef(40.0, -3.0, 0.0))
    return np.array([
        earth + np.array([2.0e7, 1.0e7, 1.5e7]),
        earth + np.array([-2.0e7, 1.0e7, 1.5e7]),
        earth + np.array([0, 2.0e7, 2.0e7]),
        earth + np.array([0, -2.0e7, 1.0e7]),
        earth + np.array([1.5e7, 0, 1.7e7]),
        earth + np.array([-1.0e7, -1.5e7, 1.3e7]),
    ])


def test_vrs_master_only_returns_rover_geometry():
    """With one base, VRS should produce observations whose pseudorange
    equals |sv - rover| + the master's residual (which is the
    atmosphere if any)."""
    sv = _sv_geometry()
    bases, rover, _ = _build(
        bases_lla=[(40.0, -3.0, 100.0)],
        rover_lla=(40.05, -3.05, 100.0),
        sv_positions=sv,
        true_amb_cycles=np.zeros(sv.shape[0]),
        wavelength=_LAMBDA_L1,
        atmosphere_m=5.0,
    )
    vrs = synthesize_vrs(bases, tuple(rover), wavelength=_LAMBDA_L1)
    rho_rover = np.linalg.norm(np.asarray(vrs["sv_positions"]) - rover, axis=1)
    # Master residual is +5 m (atmosphere); VRS pseudorange = rho_rover + 5.
    assert np.allclose(vrs["pr"], rho_rover + 5.0, atol=1e-6)


def test_vrs_three_bases_solve_recovers_rover():
    """With 3+ bases and a synthesized VRS, an ordinary single-baseline
    DD solve against the VRS should return a baseline close to zero
    (rover - VRS position = 0)."""
    sv = _sv_geometry()
    bases, rover, rover_obs = _build(
        bases_lla=[
            (40.00, -3.00, 100.0),
            (40.20, -3.05, 100.0),
            (40.00, -2.80, 100.0),
        ],
        rover_lla=(40.10, -2.95, 100.0),
        sv_positions=sv,
        true_amb_cycles=np.zeros(sv.shape[0]),
        wavelength=_LAMBDA_L1,
        atmosphere_m=3.0,
    )
    vrs = synthesize_vrs(bases, tuple(rover), wavelength=_LAMBDA_L1)
    sol = double_difference_solve(
        rover_obs["pr"], vrs["pr"],
        rover_obs["phase"], vrs["phase"],
        rover_obs["sv_positions"],
        vrs["base_position"],
        wavelength=_LAMBDA_L1,
    )
    bx, by, bz = sol["baseline"]
    assert np.sqrt(bx * bx + by * by + bz * bz) < 0.5


def test_vrs_requires_bases():
    import pytest
    with pytest.raises(ValueError):
        synthesize_vrs([], (0, 0, 0), wavelength=_LAMBDA_L1)


def test_vrs_sv_ordering_mismatch_raises():
    import pytest
    sv = _sv_geometry()
    bases, rover, _ = _build(
        bases_lla=[(40.0, -3.0, 100.0), (40.05, -3.0, 100.0)],
        rover_lla=(40.02, -3.0, 100.0),
        sv_positions=sv,
        true_amb_cycles=np.zeros(sv.shape[0]),
        wavelength=_LAMBDA_L1,
    )
    # Truncate one base's SV table.
    bases[1]["sv_positions"] = bases[1]["sv_positions"][:-1]
    with pytest.raises(ValueError):
        synthesize_vrs(bases, tuple(rover), wavelength=_LAMBDA_L1)
