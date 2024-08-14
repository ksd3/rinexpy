"""Integration test for PPP using a bundled SP3 fixture.

We use the SP3 satellite positions and clocks as the precise products,
pick a synthetic receiver at a known ECEF, build the noise-free
iono-free pseudorange from those, and check that PPP recovers the
truth to sub-meter. This exercises the load_sp3 -> ppp_solve_code_only
wiring end-to-end (km<->m conversion, microsecond<->second conversion,
SV alignment).
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import rinexpy as rp
from rinexpy.geodesy import lla_to_ecef
from rinexpy.positioning import ppp_solve_code_only

_C = 299_792_458.0


@pytest.fixture(scope="module")
def sp3_fixture():
    return rp.load_sp3("tests/data/igs19362.sp3c")


def test_ppp_with_sp3_recovers_synthetic_position(sp3_fixture):
    """Use real SP3 sat positions and clocks; reconstruct the synthetic PRs
    and verify PPP recovers the receiver position to sub-meter."""
    sp3 = sp3_fixture

    # Pick a mid-file epoch with many valid SVs.
    sp3_epoch = sp3.time.values[10]
    # SP3 position is in km; clock in microseconds. Bad/missing clocks are
    # the SP3 sentinel 999999.999... (= 1.0 s scaled by 1e6 in microseconds).
    pos_km = sp3.position.sel(time=sp3_epoch).values   # (sv, 3)
    clk_us = sp3.clock.sel(time=sp3_epoch).values      # (sv,) in microseconds
    sv_labels = sp3.sv.values

    # Convert and drop SVs that don't have a clean position or clock.
    pos_m = pos_km * 1000.0
    clk_s = clk_us * 1e-6
    finite = (
        np.all(np.isfinite(pos_m), axis=1)
        & np.isfinite(clk_s)
        & (np.abs(clk_s) < 1.0)  # SP3 missing-clock sentinel is +/-1.0 s
    )
    sv_ecef = pos_m[finite]
    sat_clock = clk_s[finite]
    kept = [str(s) for s, ok in zip(sv_labels, finite) if ok]

    # Drop SVs below the horizon for the chosen receiver to keep the
    # geometry sane.
    truth_rx = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    los = sv_ecef - truth_rx
    rho = np.linalg.norm(los, axis=1)
    up = truth_rx / np.linalg.norm(truth_rx)
    elev_rad = np.arcsin((los @ up) / rho)
    visible = elev_rad > np.deg2rad(5.0)
    sv_ecef = sv_ecef[visible]
    sat_clock = sat_clock[visible]
    rho = rho[visible]
    assert sv_ecef.shape[0] >= 6, "need at least 6 visible SVs for a clean PPP"

    # Synthetic noise-free iono-free PR: PR = rho - c * dt_sv.
    pr_if = rho - _C * sat_clock

    sol = ppp_solve_code_only(pr_if, sv_ecef, sat_clock)
    err = np.linalg.norm(np.array(sol["position"]) - truth_rx)
    assert err < 0.01, f"PPP recovered with {err:.3f} m error"
    assert sol["clock_bias"] == approx(0.0, abs=1e-9)


def test_ppp_with_sp3_handles_misaligned_inputs(sp3_fixture):
    """If we pass mismatched shapes, ppp_solve_code_only raises cleanly."""
    sp3 = sp3_fixture
    pos = sp3.position.sel(time=sp3.time.values[10]).values * 1000.0
    clk = sp3.clock.sel(time=sp3.time.values[10]).values * 1e-6
    finite = np.all(np.isfinite(pos), axis=1) & np.isfinite(clk) & (np.abs(clk) < 1.0)
    sv = pos[finite][:5]
    clk = clk[finite][:5]
    pr = np.ones(4)  # wrong length on purpose
    with pytest.raises(ValueError, match="match"):
        ppp_solve_code_only(pr, sv, clk)
