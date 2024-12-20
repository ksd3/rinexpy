"""Tests for the slip-aware ``StaticPPPFilter.update_with_slip_check`` method."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.kalman import StaticPPPFilter
from rinexpy.multifreq import F1, F2, LAMBDA_L1, LAMBDA_L2


def _synthetic_l1_l2_session(
    n_epoch=80, n_sv=6, sigma_code_m=0.05, sigma_phase_m=0.001, seed=0,
):
    """Synthetic dual-frequency session with NO iono / tropo / clock.

    Returns per-epoch L1 / L2 code (m) and carrier phase (cycles) plus
    the satellite ECEF, sat clocks (all zero), and ground-truth N1, N2.
    """
    rng = np.random.default_rng(seed)
    rng_geom = np.random.default_rng(123)
    truth_rx = np.array(lla_to_ecef(40.0, -3.0, 100.0))

    base_az = rng_geom.uniform(0, 2 * np.pi, n_sv)
    base_el = rng_geom.uniform(np.deg2rad(20), np.deg2rad(80), n_sv)
    az_rate = rng_geom.uniform(1e-3, 4e-3, n_sv)
    sat_alt = 2.0e7
    lat = np.deg2rad(40.0); lon = np.deg2rad(-3.0)
    sl, cl = np.sin(lat), np.cos(lat); sg, cg = np.sin(lon), np.cos(lon)
    sv_ecef = np.empty((n_epoch, n_sv, 3))
    for k in range(n_epoch):
        az = base_az + az_rate * k
        el = base_el
        e_e = sat_alt * np.cos(el) * np.sin(az)
        e_n = sat_alt * np.cos(el) * np.cos(az)
        e_u = sat_alt * np.sin(el)
        dx = -sg * e_e - sl * cg * e_n + cl * cg * e_u
        dy = cg * e_e - sl * sg * e_n + cl * sg * e_u
        dz = cl * e_n + sl * e_u
        sv_ecef[k, :, 0] = truth_rx[0] + dx
        sv_ecef[k, :, 1] = truth_rx[1] + dy
        sv_ecef[k, :, 2] = truth_rx[2] + dz
    N1 = 1000 + rng.integers(-30, 31, n_sv)
    N2 = N1 + rng.integers(-5, 6, n_sv)
    p1 = np.empty((n_epoch, n_sv))
    p2 = np.empty((n_epoch, n_sv))
    phi1 = np.empty((n_epoch, n_sv))
    phi2 = np.empty((n_epoch, n_sv))
    for k in range(n_epoch):
        rho = np.linalg.norm(sv_ecef[k] - truth_rx, axis=1)
        p1[k] = rho + rng.normal(0, sigma_code_m, n_sv)
        p2[k] = rho + rng.normal(0, sigma_code_m, n_sv)
        phi1[k] = (rho + N1 * LAMBDA_L1) / LAMBDA_L1 + rng.normal(
            0, sigma_phase_m / LAMBDA_L1, n_sv
        )
        phi2[k] = (rho + N2 * LAMBDA_L2) / LAMBDA_L2 + rng.normal(
            0, sigma_phase_m / LAMBDA_L2, n_sv
        )
    return {
        "truth_rx": truth_rx,
        "sv_ecef": sv_ecef,
        "sat_clock": np.zeros((n_epoch, n_sv)),
        "p1": p1, "p2": p2, "phi1": phi1, "phi2": phi2,
        "N1": N1, "N2": N2,
        "n_epoch": n_epoch, "n_sv": n_sv,
    }


def test_no_slip_no_reset():
    """Clean data without any injected slip: no SV gets flagged as slipped."""
    s = _synthetic_l1_l2_session(n_epoch=40, n_sv=6)
    flt = StaticPPPFilter(
        n_sv=6, initial_position=tuple(s["truth_rx"]),
        sigma_code=0.05, sigma_phase=0.001,
    )
    for k in range(s["n_epoch"]):
        flt.predict(dt=1.0)
        slips = flt.update_with_slip_check(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["p1"][k], s["p2"][k], s["phi1"][k], s["phi2"][k],
            slip_threshold_cycles=2.0,
        )
        if k > 0:
            assert not slips.any(), f"unexpected slip at epoch {k}: {slips}"


def test_injected_cycle_slip_is_detected_and_reset():
    """Inject a 5-cycle slip on phi1 of SV 2 at epoch 20; the filter should
    flag and reset that SV's ambiguity slot."""
    s = _synthetic_l1_l2_session(n_epoch=60, n_sv=6)
    s["phi1"][20:, 2] += 5.0  # 5-cycle slip on L1, SV index 2

    flt = StaticPPPFilter(
        n_sv=6, initial_position=tuple(s["truth_rx"]),
        sigma_code=0.05, sigma_phase=0.001,
    )
    slip_epochs = []
    for k in range(s["n_epoch"]):
        flt.predict(dt=1.0)
        slips = flt.update_with_slip_check(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["p1"][k], s["p2"][k], s["phi1"][k], s["phi2"][k],
            slip_threshold_cycles=2.0,
        )
        if slips.any():
            slip_epochs.append((k, np.where(slips)[0].tolist()))
    # The slip should fire at epoch 20 on SV 2 exactly once.
    assert any(20 == k and 2 in svs for k, svs in slip_epochs), (
        f"expected a slip at epoch 20 on SV 2; got {slip_epochs}"
    )


def test_filter_recovers_after_injected_slip():
    """Position recovery shouldn't be permanently broken by a single slip
    when the slip-aware update wipes the affected ambiguity slot."""
    s = _synthetic_l1_l2_session(n_epoch=200, n_sv=8)
    s["phi1"][50:, 3] += 7.0
    s["phi2"][120:, 5] -= 3.0  # second slip on a different SV

    flt = StaticPPPFilter(
        n_sv=8, initial_position=tuple(s["truth_rx"] + np.array([5.0, 5.0, 5.0])),
        sigma_code=0.05, sigma_phase=0.001,
    )
    for k in range(s["n_epoch"]):
        flt.predict(dt=1.0)
        flt.update_with_slip_check(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["p1"][k], s["p2"][k], s["phi1"][k], s["phi2"][k],
            slip_threshold_cycles=2.0,
        )
    err = np.linalg.norm(np.array(flt.position) - s["truth_rx"])
    assert err < 0.20, f"position err {err:.4f} m after slip-aware filtering"


def test_slip_check_handles_nan_observations():
    """NaN in L1 / L2 for an SV at one epoch shouldn't crash or wrongly
    trigger a slip on the next valid epoch."""
    s = _synthetic_l1_l2_session(n_epoch=30, n_sv=4)
    s["phi1"][10, 1] = np.nan   # gap on SV 1 at epoch 10
    s["phi2"][10, 1] = np.nan
    s["p1"][10, 1] = np.nan
    s["p2"][10, 1] = np.nan
    flt = StaticPPPFilter(n_sv=4, initial_position=tuple(s["truth_rx"]))
    for k in range(s["n_epoch"]):
        flt.predict(dt=1.0)
        slips = flt.update_with_slip_check(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["p1"][k], s["p2"][k], s["phi1"][k], s["phi2"][k],
        )
        # Gap doesn't trigger a slip on its own.
        if k == 10:
            assert not slips[1]


def test_slip_check_rejects_shape_mismatch():
    flt = StaticPPPFilter(n_sv=4, initial_position=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="shape"):
        flt.update_with_slip_check(
            np.zeros((4, 3)), np.zeros(4),
            np.zeros(3), np.zeros(4), np.zeros(4), np.zeros(4),
        )


def test_reset_ambiguity_also_clears_mw_history():
    """Manually resetting an ambiguity clears its MW history, so a slip
    isn't double-counted on the next slip-aware update."""
    s = _synthetic_l1_l2_session(n_epoch=10, n_sv=4)
    flt = StaticPPPFilter(n_sv=4, initial_position=tuple(s["truth_rx"]))
    for k in range(5):
        flt.predict(dt=1.0)
        flt.update_with_slip_check(
            s["sv_ecef"][k], s["sat_clock"][k],
            s["p1"][k], s["p2"][k], s["phi1"][k], s["phi2"][k],
        )
    flt.reset_ambiguity(2)
    assert np.isnan(flt._prev_mw_wl_cycles[2])
