"""Tests for the static PPP Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.kalman import StaticPPPFilter
from rinexpy.multifreq import F1, F2, LAMBDA_L1, LAMBDA_L2

_C = 299_792_458.0


def _make_kinematic_session(
    n_epoch=200, n_sv=8, sigma_code_m=0.5, sigma_phase_m=0.005,
    seed=0,
):
    """Synthetic dual-frequency static session with varied SV geometry."""
    rng = np.random.default_rng(seed)
    rng_geom = np.random.default_rng(123)
    truth_rx = np.array(lla_to_ecef(40.0, -3.0, 100.0))

    base_az = rng_geom.uniform(0, 2 * np.pi, n_sv)
    base_el = rng_geom.uniform(np.deg2rad(15), np.deg2rad(85), n_sv)
    az_rate = rng_geom.uniform(1e-3, 5e-3, n_sv)
    el_rate = rng_geom.uniform(-2e-4, 2e-4, n_sv)
    sat_alt = 2.0e7
    lat = np.deg2rad(40.0)
    lon = np.deg2rad(-3.0)
    sl, cl = np.sin(lat), np.cos(lat)
    sg, cg = np.sin(lon), np.cos(lon)
    sv_ecef = np.empty((n_epoch, n_sv, 3))
    for k in range(n_epoch):
        az = base_az + az_rate * k
        el = np.clip(base_el + el_rate * k, np.deg2rad(5), np.deg2rad(90))
        e_e = sat_alt * np.cos(el) * np.sin(az)
        e_n = sat_alt * np.cos(el) * np.cos(az)
        e_u = sat_alt * np.sin(el)
        dx = -sg * e_e - sl * cg * e_n + cl * cg * e_u
        dy = cg * e_e - sl * sg * e_n + cl * sg * e_u
        dz = cl * e_n + sl * e_u
        sv_ecef[k, :, 0] = truth_rx[0] + dx
        sv_ecef[k, :, 1] = truth_rx[1] + dy
        sv_ecef[k, :, 2] = truth_rx[2] + dz

    sat_clock = np.zeros((n_epoch, n_sv))
    tropo = np.zeros((n_epoch, n_sv))

    # Iono-free ambiguities in meters: arbitrary float per SV.
    true_b_if = rng.uniform(-100.0, 100.0, n_sv)

    pr_if = np.empty((n_epoch, n_sv))
    phase_if = np.empty((n_epoch, n_sv))
    for k in range(n_epoch):
        rho = np.linalg.norm(sv_ecef[k] - truth_rx, axis=1)
        pr_if[k] = rho + rng.normal(0, sigma_code_m, n_sv)
        phase_if[k] = rho + true_b_if + rng.normal(0, sigma_phase_m, n_sv)
    return {
        "truth_rx": truth_rx,
        "sv_ecef": sv_ecef,
        "sat_clock": sat_clock,
        "tropo": tropo,
        "pr_if": pr_if,
        "phase_if": phase_if,
        "true_b_if": true_b_if,
    }


def test_filter_recovers_position_under_realistic_noise():
    """200 epochs of synthetic phase observations converge the static
    position to a few centimetres."""
    s = _make_kinematic_session(n_epoch=200, n_sv=8,
                                 sigma_code_m=0.5, sigma_phase_m=0.005)
    flt = StaticPPPFilter(
        n_sv=8,
        initial_position=tuple(s["truth_rx"] + np.array([10.0, 10.0, 10.0])),
        sigma_code=0.5, sigma_phase=0.005,
    )
    for k in range(200):
        flt.predict(dt=1.0)
        flt.update(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["pr_if"][k], s["phase_if"][k],
            tropo_m=s["tropo"][k],
        )
    err = np.linalg.norm(np.array(flt.position) - s["truth_rx"])
    assert err < 0.10, f"static PPP filter recovered {err:.4f} m"


def test_filter_converges_toward_truth_over_time():
    """Position error should monotonically (roughly) drop across epochs."""
    s = _make_kinematic_session(n_epoch=100, n_sv=8,
                                 sigma_code_m=0.3, sigma_phase_m=0.003)
    flt = StaticPPPFilter(
        n_sv=8,
        initial_position=tuple(s["truth_rx"] + np.array([5.0, 5.0, 5.0])),
        sigma_code=0.3, sigma_phase=0.003,
    )
    errs = []
    for k in range(100):
        flt.predict(dt=1.0)
        flt.update(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["pr_if"][k], s["phase_if"][k],
            tropo_m=s["tropo"][k],
        )
        errs.append(np.linalg.norm(np.array(flt.position) - s["truth_rx"]))
    # Final 10-epoch median should be far below the first 10's.
    assert np.median(errs[-10:]) < np.median(errs[:10]) / 3


def test_filter_position_sigma_decreases_with_data():
    """Position uncertainty (1-sigma) shrinks as observations accumulate."""
    s = _make_kinematic_session(n_epoch=80, n_sv=8)
    flt = StaticPPPFilter(
        n_sv=8, initial_position=tuple(s["truth_rx"]),
    )
    sigma_initial = max(flt.position_sigma)
    for k in range(80):
        flt.predict(dt=1.0)
        flt.update(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["pr_if"][k], s["phase_if"][k],
            tropo_m=s["tropo"][k],
        )
    sigma_final = max(flt.position_sigma)
    assert sigma_final < sigma_initial / 5, (
        f"position sigma went {sigma_initial:.3f} -> {sigma_final:.3f} m"
    )


def test_filter_handles_missing_observations():
    """NaN entries in pr_if / phase_if just skip those SVs."""
    s = _make_kinematic_session(n_epoch=30, n_sv=8)
    flt = StaticPPPFilter(n_sv=8, initial_position=tuple(s["truth_rx"]))
    # Knock out half the obs at random.
    rng = np.random.default_rng(99)
    for k in range(30):
        pr = s["pr_if"][k].copy()
        ph = s["phase_if"][k].copy()
        mask = rng.random(8) < 0.5
        pr[mask] = np.nan
        ph[mask] = np.nan
        flt.predict(dt=1.0)
        flt.update(
            s["sv_ecef"][k], s["sat_clock"][k], pr, ph,
            tropo_m=s["tropo"][k],
        )
    err = np.linalg.norm(np.array(flt.position) - s["truth_rx"])
    assert err < 5.0  # loose; data dropout limits convergence


def test_reset_ambiguity_clears_one_sv():
    """reset_ambiguity wipes the slot to zero with high variance."""
    s = _make_kinematic_session(n_epoch=20, n_sv=4)
    flt = StaticPPPFilter(n_sv=4, initial_position=tuple(s["truth_rx"]))
    for k in range(20):
        flt.predict(dt=1.0)
        flt.update(s["sv_ecef"][k], s["sat_clock"][k],
                   s["pr_if"][k], s["phase_if"][k], tropo_m=s["tropo"][k])
    # Ambig 0 should be near its float estimate.
    a_before = flt.ambiguities_m[0]
    sigma_before = float(np.sqrt(flt.P[4, 4]))
    flt.reset_ambiguity(0)
    assert flt.ambiguities_m[0] == 0.0
    sigma_after = float(np.sqrt(flt.P[4, 4]))
    assert sigma_after > sigma_before * 10
    # Other slots are untouched.
    assert flt.ambiguities_m[1] != 0.0 or abs(a_before) < 1e-9


def test_filter_rejects_shape_mismatch():
    flt = StaticPPPFilter(n_sv=4, initial_position=(1.0, 2.0, 3.0))
    bad_sv = np.zeros((3, 3))    # only 3 SVs
    with pytest.raises(ValueError, match="shape"):
        flt.update(bad_sv, np.zeros(3), np.zeros(3), np.zeros(3))


def test_filter_rejects_negative_dt():
    flt = StaticPPPFilter(n_sv=4, initial_position=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="dt"):
        flt.predict(dt=-1.0)


def test_clock_variance_grows_with_predict():
    """Time updates should inflate the clock-state variance per the
    random-walk model."""
    flt = StaticPPPFilter(n_sv=4, initial_position=(0.0, 0.0, 0.0),
                          sigma_clock_init=10.0, sigma_clock_rate_m=5.0)
    v0 = flt.P[3, 3]
    flt.predict(dt=4.0)
    v1 = flt.P[3, 3]
    assert v1 == approx(v0 + 5.0 ** 2 * 4.0)


def test_filter_returns_sane_clock_bias():
    """clock_bias_s converts the state's c*dt to seconds."""
    s = _make_kinematic_session(n_epoch=50, n_sv=8)
    flt = StaticPPPFilter(n_sv=8, initial_position=tuple(s["truth_rx"]))
    for k in range(50):
        flt.predict(dt=1.0)
        flt.update(s["sv_ecef"][k], s["sat_clock"][k],
                   s["pr_if"][k], s["phase_if"][k], tropo_m=s["tropo"][k])
    # True clock is zero in our synthetic; filter should be within a
    # microsecond.
    assert abs(flt.clock_bias_s) < 1e-6


def test_gnssfilter_alias_is_staticpppfilter():
    """ROADMAP acceptance API: rinexpy.kalman.GNSSFilter is the named
    EKF entry point."""
    from rinexpy.kalman import GNSSFilter, StaticPPPFilter

    assert GNSSFilter is StaticPPPFilter
    flt = GNSSFilter(n_sv=4, initial_position=(0.0, 0.0, 0.0))
    assert isinstance(flt, StaticPPPFilter)
