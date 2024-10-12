"""Tests for the static-batch float-ambiguity PPP solver."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.positioning import (
    iono_free_phase,
    iono_free_pseudorange,
    ppp_solve_static_batch,
)

_C = 299_792_458.0
_F_L1 = 1575.42e6
_F_L2 = 1227.60e6


def _synthetic_batch(n_epoch=20, n_sv=8, seed=0,
                     sigma_code=1.0, sigma_phase=0.005):
    """Build a synthetic static-PPP batch with known truth.

    Returns: pr_if, phase_if, sv_ecef, sat_clock, tropo, truth_rx, true_N.
    Satellites move along a circle around the receiver to give varied
    geometry across epochs.
    """
    rng = np.random.default_rng(seed)
    truth_rx = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    # Distribute SVs roughly uniformly on the celestial sphere above the
    # receiver, then let each one sweep across the sky during the session.
    rng_geom = np.random.default_rng(123)
    base_az = rng_geom.uniform(0, 2 * np.pi, n_sv)
    base_el = rng_geom.uniform(np.deg2rad(15), np.deg2rad(85), n_sv)
    # Orbital motion: rate (rad/s in az, elevation drift) varies per SV
    az_rate = rng_geom.uniform(1e-3, 5e-3, n_sv)
    el_rate = rng_geom.uniform(-2e-4, 2e-4, n_sv)
    sat_alt = 2.0e7  # meters above receiver (rough GPS orbit altitude)
    sv_ecef = np.empty((n_epoch, n_sv, 3))
    for k in range(n_epoch):
        az = base_az + az_rate * k
        el = np.clip(base_el + el_rate * k, np.deg2rad(5), np.deg2rad(90))
        # Local ENU offset, distance equal to sat_alt (toy line-of-sight model).
        e_e = sat_alt * np.cos(el) * np.sin(az)
        e_n = sat_alt * np.cos(el) * np.cos(az)
        e_u = sat_alt * np.sin(el)
        # Convert ENU at truth_rx (lat=40, lon=-3) into ECEF offsets.
        lat = np.deg2rad(40.0)
        lon = np.deg2rad(-3.0)
        sl, cl = np.sin(lat), np.cos(lat)
        sg, cg = np.sin(lon), np.cos(lon)
        dx = -sg * e_e - sl * cg * e_n + cl * cg * e_u
        dy =  cg * e_e - sl * sg * e_n + cl * sg * e_u
        dz =                  cl * e_n + sl * e_u
        sv_ecef[k, :, 0] = truth_rx[0] + dx
        sv_ecef[k, :, 1] = truth_rx[1] + dy
        sv_ecef[k, :, 2] = truth_rx[2] + dz
    sat_clock = np.zeros((n_epoch, n_sv))
    tropo = np.zeros((n_epoch, n_sv))
    true_N = rng.uniform(-100, 100, n_sv)  # per-SV iono-free ambiguity in meters

    pr_if = np.empty((n_epoch, n_sv))
    phase_if = np.empty((n_epoch, n_sv))
    for k in range(n_epoch):
        diff = sv_ecef[k] - truth_rx
        geom = np.linalg.norm(diff, axis=1)
        pr_if[k] = geom + rng.normal(0, sigma_code, n_sv)
        phase_if[k] = geom + true_N + rng.normal(0, sigma_phase, n_sv)
    return pr_if, phase_if, sv_ecef, sat_clock, tropo, truth_rx, true_N


def test_static_batch_recovers_position_below_centimeter():
    """Noiseless synthetic: position recovered to mm."""
    pr, ph, sv, sc, tr, truth, _ = _synthetic_batch(
        n_epoch=20, n_sv=8, sigma_code=0.0, sigma_phase=0.0
    )
    sol = ppp_solve_static_batch(
        pr, ph, sv, sc, tropo=tr,
        initial_position=tuple(truth + np.array([5.0, 5.0, 5.0])),
    )
    err = np.linalg.norm(np.array(sol["position"]) - truth)
    assert err < 1e-3, f"noiseless recovery {err:.4f} m"


def test_static_batch_recovers_position_at_realistic_noise():
    """With realistic code/phase noise, position recovers to a few dm.

    The expected sigma is dominated by the code noise (since the float
    ambiguities absorb the phase-vs-code mean offset and the code drives
    the absolute scale). With 1 m code RMS on 100 epochs / 8 SVs the
    formal position sigma sits at ~10 cm; a 50 cm threshold is loose
    enough to be a stable wiring test.
    """
    pr, ph, sv, sc, tr, truth, _ = _synthetic_batch(
        n_epoch=100, n_sv=8, sigma_code=1.0, sigma_phase=0.005,
    )
    sol = ppp_solve_static_batch(
        pr, ph, sv, sc, tropo=tr,
        initial_position=tuple(truth + np.array([10.0, 10.0, 10.0])),
        max_iter=20,
    )
    err = np.linalg.norm(np.array(sol["position"]) - truth)
    assert err < 0.5, f"noisy recovery {err:.4f} m"


def test_static_batch_returns_per_epoch_clock():
    """Clock estimate has length n_epoch."""
    pr, ph, sv, sc, tr, truth, _ = _synthetic_batch(n_epoch=10, n_sv=6)
    sol = ppp_solve_static_batch(
        pr, ph, sv, sc, tropo=tr, initial_position=tuple(truth),
    )
    assert sol["clock_per_epoch"].shape == (10,)
    assert sol["ambiguities_m"].shape == (6,)


def test_static_batch_recovers_ambiguities_qualitatively():
    """Float ambiguities should be within a few meters of the truth even
    when the float solver isn't constrained to integers."""
    pr, ph, sv, sc, tr, truth, true_N = _synthetic_batch(
        n_epoch=30, n_sv=6, sigma_code=0.5, sigma_phase=0.005,
    )
    sol = ppp_solve_static_batch(
        pr, ph, sv, sc, tropo=tr, initial_position=tuple(truth),
    )
    # Float ambiguities should be within ~1 m of truth at this noise level.
    err = np.max(np.abs(sol["ambiguities_m"] - true_N))
    assert err < 2.0, f"max float ambiguity error {err:.3f} m"


def test_static_batch_rejects_shape_mismatch():
    pr, ph, sv, sc, _, truth, _ = _synthetic_batch(n_epoch=5, n_sv=5)
    with pytest.raises(ValueError):
        ppp_solve_static_batch(
            pr, ph[:4], sv, sc, initial_position=tuple(truth),
        )


def test_iono_free_phase_cancels_iono():
    """A 5 m iono advance on L1 (and 5*alpha on L2) cancels in the combo."""
    alpha = (_F_L1 / _F_L2) ** 2
    base = 2.5e7
    l1 = np.array([base - 5.0])
    l2 = np.array([base - alpha * 5.0])
    out = iono_free_phase(l1, l2)
    assert out[0] == approx(base, abs=1e-7)


def test_iono_free_phase_uses_same_alpha_as_code():
    """Iono-free phase and code coefficients line up exactly."""
    alpha = (_F_L1 / _F_L2) ** 2
    l1 = np.array([1.0])
    l2 = np.array([0.5])
    p1 = np.array([1.0])
    p2 = np.array([0.5])
    assert iono_free_phase(l1, l2)[0] == approx(
        iono_free_pseudorange(p1, p2)[0], abs=1e-12
    )
    # And manual check: (alpha*1 - 0.5)/(alpha - 1)
    expected = (alpha * 1.0 - 0.5) / (alpha - 1.0)
    assert iono_free_phase(l1, l2)[0] == approx(expected, abs=1e-12)
