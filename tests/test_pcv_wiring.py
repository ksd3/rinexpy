"""Tests for the antex.pcv_corrections_for_observations helper.

The underlying apply_antex_pcv is covered elsewhere; this test verifies
the per-epoch wrapper that turns satellite ECEF positions and an
antenna entry into a vector of per-SV PCV corrections ready to plug
into spp_solve / ppp_solve_code_only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rinexpy.antex import pcv_corrections_for_observations


def _make_entry_noazi() -> dict:
    """Synthetic ANTEX entry: NOAZI PCV table for G01, no azimuth grid."""
    # 19 zenith points 0..90 deg every 5 deg, all values in mm. Linear
    # ramp so PCV at zenith=0 is 0 mm and at zenith=90 is 9.0 mm.
    zen = np.linspace(0.0, 90.0, 19)
    noazi = zen * 0.1   # mm; 0 at zenith, 9 mm at horizon
    return {
        "type": "GENERIC",
        "frequencies": {
            "G01": {
                "noazi": noazi,
                "zenith_deg": zen,
            }
        },
    }


def test_pcv_shape_matches_input():
    entry = _make_entry_noazi()
    station = np.array([0.0, 0.0, 6_378_137.0])   # North pole
    # 3 satellites at high / mid / low elevation. Place them above the
    # pole so we know their elevations exactly.
    sv = np.array([
        [0.0, 0.0, 6_378_137.0 + 20_200_000.0],     # zenith (el=90, zen=0)
        [20_200_000.0, 0.0, 6_378_137.0 + 5_000_000.0],   # low elevation
        [0.0, 0.0, 6_378_137.0 - 20_200_000.0],     # below the station
    ])
    pcv = pcv_corrections_for_observations(entry, "G01", sv, station)
    assert pcv.shape == (3,)


def test_pcv_zenith_is_zero():
    entry = _make_entry_noazi()
    station = np.array([0.0, 0.0, 6_378_137.0])
    sv = np.array([[0.0, 0.0, 6_378_137.0 + 20_200_000.0]])
    pcv = pcv_corrections_for_observations(entry, "G01", sv, station)
    # NOAZI value at zenith=0 is 0 mm.
    assert pcv[0] == pytest.approx(0.0, abs=1e-9)


def test_pcv_low_elevation_is_largest():
    """At low elevation (zenith near 90), the linear-ramp NOAZI returns
    the largest correction; at zenith=0 it's zero."""
    entry = _make_entry_noazi()
    station = np.array([0.0, 0.0, 6_378_137.0])
    # SV at near-horizon elevation
    near_horizon = np.array([[20_200_000.0, 0.0, 6_378_137.0 + 500_000.0]])
    zenith = np.array([[0.0, 0.0, 6_378_137.0 + 20_200_000.0]])
    pcv_h = pcv_corrections_for_observations(entry, "G01", near_horizon, station)
    pcv_z = pcv_corrections_for_observations(entry, "G01", zenith, station)
    assert pcv_h[0] > pcv_z[0]


def test_pcv_missing_frequency_returns_zero():
    entry = _make_entry_noazi()
    station = np.array([0.0, 0.0, 6_378_137.0])
    sv = np.array([[20_200_000.0, 0.0, 6_378_137.0 + 5_000_000.0]])
    # Frequency G02 isn't in the entry -> 0.0 fallback.
    pcv = pcv_corrections_for_observations(entry, "G02", sv, station)
    assert pcv[0] == pytest.approx(0.0)


def test_pcv_applies_to_spp_pseudoranges():
    """End-to-end: pre-correct pseudoranges for PCV then run SPP.

    The corrected solution moves by ~the PCV magnitude (mm). This test
    confirms the API matches what a PPP caller would do.
    """
    from rinexpy.positioning import spp_solve
    entry = _make_entry_noazi()
    # Place receiver near surface, 4 satellites at varying elevations.
    rng = np.random.default_rng(42)
    station_true = np.array([6_378_137.0, 0.0, 0.0])
    sv = np.array([
        [26_578_000.0,   1_000_000.0,    500_000.0],
        [25_000_000.0,  -2_000_000.0,  10_000_000.0],
        [20_000_000.0,  10_000_000.0, -10_000_000.0],
        [22_000_000.0,   5_000_000.0,  -5_000_000.0],
    ])
    rho = np.linalg.norm(sv - station_true, axis=1)
    # Add the PCV signal as if it were embedded in the raw obs.
    pcv = pcv_corrections_for_observations(entry, "G01", sv, station_true)
    pr_raw = rho + pcv
    # Solve with raw obs first.
    sol_raw = spp_solve(sv, pr_raw, initial_guess=(0.0, 0.0, 0.0))
    # Now correct.
    sol_corr = spp_solve(sv, pr_raw - pcv, initial_guess=(0.0, 0.0, 0.0))
    # Corrected solution should be much closer to truth.
    err_raw = np.linalg.norm(np.array(sol_raw["position"]) - station_true)
    err_corr = np.linalg.norm(np.array(sol_corr["position"]) - station_true)
    assert err_corr < err_raw or err_corr < 1e-3
