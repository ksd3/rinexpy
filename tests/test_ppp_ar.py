"""Tests for fix_iono_free_ambiguity (PPP ambiguity resolution).

Constructs synthetic dual-frequency observations from known integer
N1, N2 plus controllable code/phase noise, runs the WL/NL fix, and
verifies recovery.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.multifreq import (
    F1,
    F2,
    LAMBDA_L1,
    LAMBDA_L2,
    LAMBDA_NL,
    LAMBDA_WL,
    fix_iono_free_ambiguity,
)

_C = 299_792_458.0


def _synthetic_arc(
    n1: int, n2: int, n_epoch: int = 50,
    sigma_code_m: float = 0.5, sigma_phase_m: float = 0.005,
    seed: int = 0,
):
    """Build noiseless / noisy code+phase observations for known N1, N2.

    Returns: p1, p2 (m), phi1, phi2 (cycles), true B_IF (m).
    """
    rng = np.random.default_rng(seed)
    # Range varies across epochs (rough orbit motion).
    rho = 25.0e6 + np.arange(n_epoch) * 100.0
    # No iono / tropo / clock for the test; just range + ambiguity + noise.
    p1 = rho + rng.normal(0, sigma_code_m, n_epoch)
    p2 = rho + rng.normal(0, sigma_code_m, n_epoch)
    phi1 = (rho + n1 * LAMBDA_L1) / LAMBDA_L1 + rng.normal(
        0, sigma_phase_m / LAMBDA_L1, n_epoch
    )
    phi2 = (rho + n2 * LAMBDA_L2) / LAMBDA_L2 + rng.normal(
        0, sigma_phase_m / LAMBDA_L2, n_epoch
    )
    # True iono-free ambiguity in meters.
    true_b_if = 0.5 * LAMBDA_WL * (n1 - n2) + 0.5 * LAMBDA_NL * (n1 + n2)
    return p1, p2, phi1, phi2, true_b_if


def test_fixes_known_integers_noiselessly():
    """Noiseless data with N1=1003, N2=997 → recover N1, N2 exactly."""
    n1, n2 = 1003, 997
    p1, p2, phi1, phi2, true_b_if = _synthetic_arc(
        n1, n2, n_epoch=20, sigma_code_m=0, sigma_phase_m=0
    )
    out = fix_iono_free_ambiguity(p1, p2, phi1, phi2, true_b_if)
    assert out["wl_fixed"] and out["nl_fixed"] and out["parity_ok"]
    assert out["n_wl"] == n1 - n2
    assert out["n_nl"] == n1 + n2
    assert out["n1"] == n1
    assert out["n2"] == n2
    assert out["fixed_iono_free_ambig_m"] == approx(true_b_if, abs=1e-9)


def test_fix_survives_realistic_code_noise():
    """0.5 m code RMS over 50 epochs averages down enough to fix WL."""
    n1, n2 = 5000, 4995
    p1, p2, phi1, phi2, true_b_if = _synthetic_arc(
        n1, n2, n_epoch=50, sigma_code_m=0.5, sigma_phase_m=0.005, seed=42,
    )
    out = fix_iono_free_ambiguity(p1, p2, phi1, phi2, true_b_if)
    assert out["wl_fixed"]
    assert out["n_wl"] == n1 - n2
    if out["nl_fixed"]:
        assert out["n1"] == n1
        assert out["n2"] == n2


def test_rejects_when_float_b_if_far_from_integer_grid():
    """An iono-free ambiguity that doesn't land near any integer-pair
    combination should fail the NL fix."""
    n1, n2 = 1000, 998
    p1, p2, phi1, phi2, true_b_if = _synthetic_arc(
        n1, n2, n_epoch=20, sigma_code_m=0, sigma_phase_m=0
    )
    # Perturb B_IF off the integer grid by half a narrow-lane wavelength.
    out = fix_iono_free_ambiguity(
        p1, p2, phi1, phi2, true_b_if + 0.5 * LAMBDA_NL / 2,
        sigma_nl_cycles=0.1,
    )
    assert out["wl_fixed"]
    assert not out["nl_fixed"]
    assert out["n1"] is None and out["n2"] is None


def test_rejects_on_parity_mismatch():
    """When the float B_IF lands on an N_NL of the wrong parity, the fix
    is rejected even if N_NL itself rounds cleanly."""
    n1, n2 = 1000, 999    # N_WL = 1 (odd), N_NL = 1999 (odd) -- same parity
    p1, p2, phi1, phi2, true_b_if = _synthetic_arc(
        n1, n2, n_epoch=20, sigma_code_m=0, sigma_phase_m=0
    )
    # Bump B_IF by one half-wavelength so N_NL rounds to 2000 (even),
    # mismatching the WL parity.
    out = fix_iono_free_ambiguity(
        p1, p2, phi1, phi2, true_b_if + LAMBDA_NL / 2,
        sigma_nl_cycles=0.6,
    )
    assert out["wl_fixed"]
    # Parity check should fire even though both individual rounds succeeded.
    assert not out["parity_ok"] or out["n1"] is None


def test_handles_empty_input():
    """All-NaN MW is reported as unfixed without crashing."""
    n_epoch = 10
    p1 = np.full(n_epoch, np.nan)
    p2 = np.full(n_epoch, np.nan)
    phi1 = np.full(n_epoch, np.nan)
    phi2 = np.full(n_epoch, np.nan)
    out = fix_iono_free_ambiguity(p1, p2, phi1, phi2, 0.0)
    assert not out["wl_fixed"]
    assert out["n1"] is None


def test_lambda_constants_match_derivation():
    """Sanity: lambda_WL * (f1 - f2) == c and lambda_NL * (f1 + f2) == c."""
    assert LAMBDA_WL * (F1 - F2) == approx(_C, rel=1e-9)
    assert LAMBDA_NL * (F1 + F2) == approx(_C, rel=1e-9)


def test_iono_free_ambiguity_decomposition_identity():
    """B_IF = lambda_WL/2 * N_WL + lambda_NL/2 * N_NL is the formula used."""
    n1, n2 = 1234, 1230
    n_wl = n1 - n2
    n_nl = n1 + n2
    b_if = 0.5 * LAMBDA_WL * n_wl + 0.5 * LAMBDA_NL * n_nl
    # Also verify via the IF combo definition: B_IF = (alpha*N1*lam1 - N2*lam2)/(alpha-1)
    alpha = (F1 / F2) ** 2
    direct = (alpha * n1 * LAMBDA_L1 - n2 * LAMBDA_L2) / (alpha - 1)
    assert b_if == approx(direct, rel=1e-9)
