"""Tests for the float code-only PPP solver."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.positioning import (
    apply_light_time_and_earth_rotation,
    iono_free_pseudorange,
    ppp_solve_code_only,
    spp_solve,
)

_C = 299_792_458.0
_F_L1 = 1575.42e6
_F_L2 = 1227.60e6

SV_GEOM = np.array(
    [
        [2.0e7, 1.0e7, 1.5e7],
        [-2.0e7, 1.0e7, 1.5e7],
        [0.0, 2.0e7, 2.0e7],
        [0.0, -2.0e7, 1.0e7],
        [1.5e7, 0.0, 1.7e7],
    ]
)


def _geometric_ranges(rx_ecef, sv_ecef):
    return np.linalg.norm(np.asarray(sv_ecef) - np.asarray(rx_ecef), axis=1)


def test_iono_free_cancels_dispersive_term():
    """A 5 m iono on P1 maps to 5*alpha on P2; IF combo zeroes it out."""
    rho = 2.5e7
    iono = 5.0
    alpha = (_F_L1 / _F_L2) ** 2
    p1 = np.array([rho + iono])
    p2 = np.array([rho + alpha * iono])
    out = iono_free_pseudorange(p1, p2)
    # Floating-point round-off at 25e6 m sits near machine epsilon; allow
    # ~10 nm tolerance.
    assert out[0] == approx(rho, abs=1e-7)


def test_iono_free_handles_array_shape():
    """Vectorised inputs work as expected."""
    p1 = np.linspace(1e7, 2e7, 5)
    p2 = p1 + 0.5  # arbitrary frequency-dependent bias
    out = iono_free_pseudorange(p1, p2)
    assert out.shape == p1.shape


def test_ppp_recovers_truth_with_exact_products():
    """No iono, no tropo, sat clocks = 0: PPP matches the SPP solution."""
    truth = np.array(lla_to_ecef(40, -3, 100))
    rho = _geometric_ranges(truth, SV_GEOM)
    pr_if = rho  # iono-free, no clocks, no tropo
    sat_clock = np.zeros(SV_GEOM.shape[0])
    sol = ppp_solve_code_only(pr_if, SV_GEOM, sat_clock)
    for i in range(3):
        assert sol["position"][i] == approx(truth[i], abs=1e-3)
    assert sol["clock_bias"] == approx(0.0, abs=1e-9)


def test_ppp_corrects_for_satellite_clocks():
    """Apply a known sat clock offset: PPP recovers truth, SPP would not."""
    truth = np.array(lla_to_ecef(40, -3, 100))
    rho = _geometric_ranges(truth, SV_GEOM)
    # Pretend each SV has a different clock offset.
    sat_clock = np.array([1e-6, -2e-6, 5e-7, -1e-6, 3e-7])  # microsecond scale
    # PR = rho - c * dt_sat (sat is ahead means signal travels less time
    # for the same observed range).
    pr = rho - _C * sat_clock
    pr_if = pr  # treat as already iono-free
    sol = ppp_solve_code_only(pr_if, SV_GEOM, sat_clock)
    for i in range(3):
        assert sol["position"][i] == approx(truth[i], abs=1e-3)
    # Without the sat clock correction, SPP biases would be ~hundreds of meters.
    sol_spp = spp_solve(SV_GEOM, pr)
    naive_pos = np.array(sol_spp["position"])
    err_naive = np.linalg.norm(naive_pos - truth)
    err_ppp = np.linalg.norm(np.array(sol["position"]) - truth)
    assert err_ppp < 0.01
    assert err_naive > 100.0  # sat clocks unmodelled => big position bias


def test_ppp_applies_tropospheric_correction():
    """A per-SV tropo correction is subtracted from the pseudorange."""
    truth = np.array(lla_to_ecef(40, -3, 100))
    rho = _geometric_ranges(truth, SV_GEOM)
    # Inject a per-SV tropo delay (3-10 m, elevation-dependent in practice).
    tropo = np.array([2.5, 4.0, 3.2, 6.0, 2.8])
    pr = rho + tropo  # observed = range + tropo
    sat_clock = np.zeros(SV_GEOM.shape[0])
    sol = ppp_solve_code_only(
        pr, SV_GEOM, sat_clock, tropospheric_delay_m=tropo
    )
    for i in range(3):
        assert sol["position"][i] == approx(truth[i], abs=1e-3)


def test_ppp_full_correction_stack():
    """Sat clocks + tropo + iono all modelled: PPP recovers truth."""
    truth = np.array(lla_to_ecef(40, -3, 100))
    rho = _geometric_ranges(truth, SV_GEOM)
    sat_clock = np.array([1e-6, -2e-6, 5e-7, -1e-6, 3e-7])
    tropo = np.array([2.5, 4.0, 3.2, 6.0, 2.8])
    iono = np.array([5.0, 6.0, 4.0, 7.0, 4.5])
    alpha = (_F_L1 / _F_L2) ** 2
    # Build dual-freq pseudoranges with iono, tropo, sat clock.
    p1 = rho + iono + tropo - _C * sat_clock
    p2 = rho + alpha * iono + tropo - _C * sat_clock
    pr_if = iono_free_pseudorange(p1, p2)
    sol = ppp_solve_code_only(
        pr_if, SV_GEOM, sat_clock, tropospheric_delay_m=tropo
    )
    err = np.linalg.norm(np.array(sol["position"]) - truth)
    assert err < 0.01


def test_ppp_rejects_too_few_satellites():
    pr = np.array([2.5e7, 2.6e7, 2.7e7])
    sv = SV_GEOM[:3]
    sat_clock = np.zeros(3)
    with pytest.raises(ValueError, match="needs >= 4"):
        ppp_solve_code_only(pr, sv, sat_clock)


def test_ppp_rejects_mismatched_shapes():
    pr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sv = SV_GEOM
    sat_clock = np.zeros(4)  # wrong length
    with pytest.raises(ValueError, match="match"):
        ppp_solve_code_only(pr, sv, sat_clock)


def test_ppp_rejects_bad_sv_shape():
    pr = np.array([1.0, 2.0, 3.0, 4.0])
    sv = np.zeros((4, 2))  # not 3 columns
    sat_clock = np.zeros(4)
    with pytest.raises(ValueError, match="shape"):
        ppp_solve_code_only(pr, sv, sat_clock)


def test_light_time_correction_on_synthetic_sp3():
    """A stationary satellite (zero velocity in our synthetic SP3) sits at
    the same position whether or not we apply the light-time correction;
    we just verify the helper runs and returns the right shape."""
    import xarray as xr

    times = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:15:00", "2020-01-01T00:30:00"],
        dtype="datetime64[ns]",
    )
    # One SV, sitting at a fixed location in km. Make all three epochs the
    # same so Lagrange interpolation returns that fixed location for any
    # query inside the window.
    fixed_km = np.array([20000.0, 0.0, 15000.0])
    pos = np.broadcast_to(fixed_km, (3, 1, 3)).copy()
    sp3 = xr.Dataset(
        {"position": (("time", "sv", "ECEF"), pos)},
        coords={"time": times, "sv": ["G99"], "ECEF": ["x", "y", "z"]},
    )
    rx = np.array([0.0, 0.0, 0.0])
    pos_m = apply_light_time_and_earth_rotation(
        sp3, times[1], rx, "G99"
    )
    assert pos_m.shape == (3,)
    # Stationary SV in inertial frame: after the iteration, the ECEF
    # position has rotated by Omega * light_time. The magnitude is
    # preserved (rotation is rigid). We check that.
    truth_m = fixed_km * 1000.0
    assert np.linalg.norm(pos_m) == approx(np.linalg.norm(truth_m), rel=1e-9)


def test_light_time_iteration_converges():
    """The fixed-point iteration converges to a stable position in 2-3 steps."""
    import xarray as xr

    times = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:15:00", "2020-01-01T00:30:00"],
        dtype="datetime64[ns]",
    )
    fixed_km = np.array([20000.0, 5000.0, 15000.0])
    pos = np.broadcast_to(fixed_km, (3, 1, 3)).copy()
    sp3 = xr.Dataset(
        {"position": (("time", "sv", "ECEF"), pos)},
        coords={"time": times, "sv": ["G99"], "ECEF": ["x", "y", "z"]},
    )
    rx = np.array([1e6, 2e6, 5e6])
    # max_iter=1 vs max_iter=3 should converge to the same answer.
    pos_one = apply_light_time_and_earth_rotation(
        sp3, times[1], rx, "G99", max_iter=1
    )
    pos_three = apply_light_time_and_earth_rotation(
        sp3, times[1], rx, "G99", max_iter=3
    )
    assert np.allclose(pos_one, pos_three, atol=1e-6)
