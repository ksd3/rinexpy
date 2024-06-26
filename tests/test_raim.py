"""Tests for the RAIM fault-detection-and-exclusion wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.positioning import _chi2_quantile, _norm_quantile, spp_solve_raim

_C = 299_792_458.0


def _synthetic_pseudoranges(rx_ecef, sv_ecef, clock_bias_s=0.0):
    rx = np.asarray(rx_ecef)
    diff = np.asarray(sv_ecef) - rx
    return np.linalg.norm(diff, axis=1) + _C * clock_bias_s


SV_GEOMETRY = np.array(
    [
        [2.0e7, 1.0e7, 1.5e7],
        [-2.0e7, 1.0e7, 1.5e7],
        [0.0, 2.0e7, 2.0e7],
        [0.0, -2.0e7, 1.0e7],
        [1.5e7, 0.0, 1.7e7],
        [-1.5e7, -1.0e7, 1.8e7],
        [1.0e7, -1.5e7, 1.6e7],
    ]
)


def test_raim_passes_on_clean_data():
    """No injected fault: RAIM accepts the solution on the first try."""
    truth = np.array(lla_to_ecef(40, -3, 100))
    pr = _synthetic_pseudoranges(truth, SV_GEOMETRY)
    sol = spp_solve_raim(SV_GEOMETRY, pr, sigma_pr=5.0, p_fa=1e-4)
    assert not sol["fault_detected"]
    assert not sol["raim_failed"]
    assert sol["excluded_svs"] == []
    assert sol["raim_test"] < sol["raim_threshold"]
    for i in range(3):
        assert sol["position"][i] == approx(truth[i], abs=1e-3)


def test_raim_flags_50m_bias():
    """ROADMAP acceptance: a 50 m bias on one SV is detected and excluded."""
    truth = np.array(lla_to_ecef(40, -3, 100))
    pr = _synthetic_pseudoranges(truth, SV_GEOMETRY)
    bad_sv = 2
    pr[bad_sv] += 50.0
    sol = spp_solve_raim(SV_GEOMETRY, pr, sigma_pr=5.0, p_fa=1e-4)
    assert sol["fault_detected"]
    assert not sol["raim_failed"]
    assert bad_sv in sol["excluded_svs"]
    # After excluding the bad SV, the recovered position should match truth.
    for i in range(3):
        assert sol["position"][i] == approx(truth[i], abs=1e-3)


def test_raim_gives_up_after_too_many_exclusions():
    """Two bad SVs but max_exclusions=1: RAIM reports failure."""
    truth = np.array(lla_to_ecef(40, -3, 100))
    pr = _synthetic_pseudoranges(truth, SV_GEOMETRY)
    pr[2] += 200.0
    pr[5] += 200.0
    sol = spp_solve_raim(
        SV_GEOMETRY, pr, sigma_pr=5.0, p_fa=1e-4, max_exclusions=1
    )
    assert sol["fault_detected"]
    assert sol["raim_failed"]


def test_raim_rejects_too_few_sats():
    sv = SV_GEOMETRY[:4]
    pr = np.array([2.5e7, 2.6e7, 2.7e7, 2.8e7])
    with pytest.raises(ValueError):
        spp_solve_raim(sv, pr)


def test_norm_quantile_endpoints():
    """Acklam's approximation matches the standard normal table at canonical points."""
    assert _norm_quantile(0.5) == approx(0.0, abs=1e-9)
    assert _norm_quantile(0.975) == approx(1.959964, abs=1e-4)
    assert _norm_quantile(0.99) == approx(2.326348, abs=1e-4)
    # Wide-tail
    assert _norm_quantile(1 - 1e-4) == approx(3.71902, abs=1e-3)


def test_chi2_quantile_matches_table():
    """Wilson-Hilferty within ~1% of scipy.stats.chi2.ppf at typical RAIM operating points."""
    # Reference values from scipy.stats.chi2.ppf(1 - 1e-4, df).
    cases = {
        5: 25.745,
        7: 29.878,
        10: 35.564,
    }
    for df, expected in cases.items():
        got = _chi2_quantile(1 - 1e-4, df)
        # Wilson-Hilferty is ~3% off at df=5, tightens to ~1% by df=10.
        assert got == approx(expected, rel=0.03), f"df={df}: got {got}, expected {expected}"


def test_norm_quantile_rejects_out_of_range():
    with pytest.raises(ValueError):
        _norm_quantile(0.0)
    with pytest.raises(ValueError):
        _norm_quantile(1.0)


def test_chi2_quantile_rejects_zero_df():
    with pytest.raises(ValueError):
        _chi2_quantile(0.5, 0)
