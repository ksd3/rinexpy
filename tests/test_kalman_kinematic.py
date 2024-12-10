"""Tests for the position-random-walk (kinematic) mode of StaticPPPFilter."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.kalman import StaticPPPFilter

_C = 299_792_458.0


def _make_session(
    velocity_m_s,
    n_epoch=200, n_sv=8, sigma_code_m=0.5, sigma_phase_m=0.005, seed=0,
):
    """Synthetic dual-frequency session with optional linear receiver motion.

    Returns a dict with per-epoch ground-truth positions and the
    matching iono-free code + phase observations.
    """
    rng = np.random.default_rng(seed)
    rng_geom = np.random.default_rng(123)
    base_rx = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    v = np.asarray(velocity_m_s, dtype=float)

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
    truth_rx_t = np.empty((n_epoch, 3))
    for k in range(n_epoch):
        truth_rx_t[k] = base_rx + v * k    # dt=1 s, so k seconds elapsed
        az = base_az + az_rate * k
        el = np.clip(base_el + el_rate * k, np.deg2rad(5), np.deg2rad(90))
        e_e = sat_alt * np.cos(el) * np.sin(az)
        e_n = sat_alt * np.cos(el) * np.cos(az)
        e_u = sat_alt * np.sin(el)
        dx = -sg * e_e - sl * cg * e_n + cl * cg * e_u
        dy = cg * e_e - sl * sg * e_n + cl * sg * e_u
        dz = cl * e_n + sl * e_u
        sv_ecef[k, :, 0] = base_rx[0] + dx
        sv_ecef[k, :, 1] = base_rx[1] + dy
        sv_ecef[k, :, 2] = base_rx[2] + dz

    sat_clock = np.zeros((n_epoch, n_sv))
    tropo = np.zeros((n_epoch, n_sv))
    true_b_if = rng.uniform(-100.0, 100.0, n_sv)
    pr_if = np.empty((n_epoch, n_sv))
    phase_if = np.empty((n_epoch, n_sv))
    for k in range(n_epoch):
        rho = np.linalg.norm(sv_ecef[k] - truth_rx_t[k], axis=1)
        pr_if[k] = rho + rng.normal(0, sigma_code_m, n_sv)
        phase_if[k] = rho + true_b_if + rng.normal(0, sigma_phase_m, n_sv)
    return {
        "truth_rx_t": truth_rx_t,
        "sv_ecef": sv_ecef,
        "sat_clock": sat_clock,
        "tropo": tropo,
        "pr_if": pr_if,
        "phase_if": phase_if,
    }


def test_kinematic_filter_tracks_constant_velocity_receiver():
    """A receiver moving at 1 m/s should be tracked by the filter to ~few m
    when sigma_position_rate is tuned to that motion scale."""
    s = _make_session(velocity_m_s=(1.0, 0.0, 0.0),
                      n_epoch=200, n_sv=8)
    # process noise > expected velocity * sqrt(dt) so the filter doesn't
    # damp out real motion.
    flt = StaticPPPFilter(
        n_sv=8,
        initial_position=tuple(s["truth_rx_t"][0]),
        sigma_position_rate_m=1.5,
    )
    final_errs = []
    for k in range(200):
        flt.predict(dt=1.0)
        flt.update(s["sv_ecef"][k], s["sat_clock"][k],
                   s["pr_if"][k], s["phase_if"][k], tropo_m=s["tropo"][k])
        err = np.linalg.norm(np.array(flt.position) - s["truth_rx_t"][k])
        final_errs.append(err)
    # Steady-state error (after burn-in) should be small.
    assert np.median(final_errs[50:]) < 5.0, (
        f"median tracking error {np.median(final_errs[50:]):.3f} m"
    )


def test_static_filter_lags_a_moving_receiver():
    """A truly-static filter (sigma_position_rate_m=0) on moving data
    builds up a tracking error roughly equal to the cumulative motion."""
    s = _make_session(velocity_m_s=(1.0, 0.0, 0.0),
                      n_epoch=200, n_sv=8)
    flt = StaticPPPFilter(
        n_sv=8,
        initial_position=tuple(s["truth_rx_t"][0]),
        sigma_position_rate_m=0.0,
    )
    for k in range(200):
        flt.predict(dt=1.0)
        flt.update(s["sv_ecef"][k], s["sat_clock"][k],
                   s["pr_if"][k], s["phase_if"][k], tropo_m=s["tropo"][k])
    final_err = np.linalg.norm(np.array(flt.position) - s["truth_rx_t"][-1])
    # Truth moved 200 m; filter held near the start, so error >> 100 m.
    assert final_err > 50.0, (
        f"static filter should lag a 200 m motion; got err = {final_err:.3f} m"
    )


def test_position_variance_grows_when_kinematic():
    """predict() with sigma_position_rate_m > 0 grows position variance."""
    flt = StaticPPPFilter(
        n_sv=4,
        initial_position=(0.0, 0.0, 0.0),
        sigma_position_rate_m=2.0,
    )
    v0 = flt.P[0, 0]
    flt.predict(dt=5.0)
    v1 = flt.P[0, 0]
    assert v1 == approx(v0 + 2.0 ** 2 * 5.0)


def test_kinematic_filter_rejects_negative_rate():
    with pytest.raises(ValueError, match="sigma_position_rate_m"):
        StaticPPPFilter(
            n_sv=4, initial_position=(0.0, 0.0, 0.0),
            sigma_position_rate_m=-1.0,
        )


def test_static_default_does_not_grow_position_variance():
    """Default sigma_position_rate_m=0 keeps the static-filter behaviour."""
    flt = StaticPPPFilter(n_sv=4, initial_position=(0.0, 0.0, 0.0))
    v0 = flt.P[0, 0]
    flt.predict(dt=10.0)
    assert flt.P[0, 0] == v0
