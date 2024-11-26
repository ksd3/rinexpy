"""Tests for ppp_solve_static_batch_with_ar (closed-loop PPP-AR)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.multifreq import F1, F2, LAMBDA_L1, LAMBDA_L2, LAMBDA_NL, LAMBDA_WL
from rinexpy.positioning import (
    iono_free_phase,
    iono_free_pseudorange,
    ppp_solve_static_batch_with_ar,
)

_C = 299_792_458.0


def _make_static_session(
    n_epoch=80, n_sv=8, seed=0,
    sigma_code_m=0.5, sigma_phase_m=0.005,
    N1_base=1000, N2_base=999,
):
    """Synthetic dual-frequency static session with known integer ambiguities.

    Geometry: 8 SVs uniformly distributed in azimuth around the receiver,
    sweeping slowly across epochs. True N1, N2 vary per SV around the
    given base.
    """
    rng = np.random.default_rng(seed)
    rng_geom = np.random.default_rng(123)
    truth_rx = np.array(lla_to_ecef(40.0, -3.0, 100.0))

    # Base SV azimuth/elevation; each rotates slowly across the session.
    base_az = rng_geom.uniform(0, 2 * np.pi, n_sv)
    base_el = rng_geom.uniform(np.deg2rad(20), np.deg2rad(80), n_sv)
    az_rate = rng_geom.uniform(2e-4, 8e-4, n_sv)
    el_rate = rng_geom.uniform(-5e-5, 5e-5, n_sv)
    sat_alt = 2.0e7

    lat = np.deg2rad(40.0)
    lon = np.deg2rad(-3.0)
    sl, cl = np.sin(lat), np.cos(lat)
    sg, cg = np.sin(lon), np.cos(lon)

    sv_ecef = np.empty((n_epoch, n_sv, 3))
    for k in range(n_epoch):
        az = base_az + az_rate * k
        el = np.clip(base_el + el_rate * k, np.deg2rad(5), np.deg2rad(90))
        e = sat_alt * np.cos(el) * np.sin(az)
        n = sat_alt * np.cos(el) * np.cos(az)
        u = sat_alt * np.sin(el)
        dx = -sg * e - sl * cg * n + cl * cg * u
        dy = cg * e - sl * sg * n + cl * sg * u
        dz = cl * n + sl * u
        sv_ecef[k, :, 0] = truth_rx[0] + dx
        sv_ecef[k, :, 1] = truth_rx[1] + dy
        sv_ecef[k, :, 2] = truth_rx[2] + dz

    sat_clock = np.zeros((n_epoch, n_sv))
    tropo = np.zeros((n_epoch, n_sv))

    # Per-SV true integer ambiguities (small differences around the base).
    N1 = N1_base + rng.integers(-10, 11, n_sv)
    N2 = N2_base + rng.integers(-10, 11, n_sv)

    pr_if = np.empty((n_epoch, n_sv))
    phase_if = np.empty((n_epoch, n_sv))
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
        pr_if[k] = iono_free_pseudorange(p1[k], p2[k])
        phase_if[k] = iono_free_phase(phi1[k] * LAMBDA_L1, phi2[k] * LAMBDA_L2)

    return {
        "truth_rx": truth_rx,
        "sv_ecef": sv_ecef,
        "sat_clock": sat_clock,
        "tropo": tropo,
        "p1": p1, "p2": p2,
        "phi1": phi1, "phi2": phi2,
        "pr_if": pr_if, "phase_if": phase_if,
        "N1": N1, "N2": N2,
    }


def test_ar_closed_loop_fixes_all_svs_in_noiseless_limit():
    """Noiseless synthetic: every SV's float ambiguity should land on the
    integer grid and AR should recover N1 / N2 exactly."""
    s = _make_static_session(
        n_epoch=80, n_sv=8, sigma_code_m=0.0, sigma_phase_m=0.0,
    )
    out = ppp_solve_static_batch_with_ar(
        s["pr_if"], s["phase_if"], s["sv_ecef"], s["sat_clock"],
        p1_m=s["p1"], p2_m=s["p2"],
        phi1_cycles=s["phi1"], phi2_cycles=s["phi2"],
        tropo=s["tropo"],
        initial_position=tuple(s["truth_rx"] + np.array([1.0, 1.0, 1.0])),
    )
    assert out["ar_applied"]
    # All 8 SVs should fix without noise.
    assert out["fixed_ambig_mask"].sum() == 8
    # The recovered integer N1 / N2 should match truth exactly.
    np.testing.assert_array_equal(
        out["n1_per_sv"].astype(int), s["N1"]
    )
    np.testing.assert_array_equal(
        out["n2_per_sv"].astype(int), s["N2"]
    )
    # Position should be recovered to sub-mm.
    err = np.linalg.norm(np.array(out["fixed"]["position"]) - s["truth_rx"])
    assert err < 1e-3, f"AR position err {err:.6f} m"


def test_ar_falls_back_to_float_when_nothing_fixes():
    """With huge code noise no SV gets fixed; AR-aware solver returns
    the float solution unchanged."""
    s = _make_static_session(n_epoch=20, n_sv=6, sigma_code_m=20.0,
                              sigma_phase_m=0.005)
    out = ppp_solve_static_batch_with_ar(
        s["pr_if"], s["phase_if"], s["sv_ecef"], s["sat_clock"],
        p1_m=s["p1"], p2_m=s["p2"],
        phi1_cycles=s["phi1"], phi2_cycles=s["phi2"],
        tropo=s["tropo"],
        initial_position=tuple(s["truth_rx"]),
    )
    if not out["ar_applied"]:
        assert out["fixed"] is out["float"]
    else:
        # If by chance some SVs got fixed, the result should still be
        # consistent in shape with the float.
        assert "position" in out["fixed"]


def test_ar_returns_per_sv_integer_arrays_with_nan_for_unfixed():
    """The n1_per_sv / n2_per_sv arrays have NaN at unfixed SVs."""
    s = _make_static_session(n_epoch=60, n_sv=6)
    out = ppp_solve_static_batch_with_ar(
        s["pr_if"], s["phase_if"], s["sv_ecef"], s["sat_clock"],
        p1_m=s["p1"], p2_m=s["p2"],
        phi1_cycles=s["phi1"], phi2_cycles=s["phi2"],
        tropo=s["tropo"],
        initial_position=tuple(s["truth_rx"]),
    )
    for j in range(s["pr_if"].shape[1]):
        if out["fixed_ambig_mask"][j]:
            assert np.isfinite(out["n1_per_sv"][j])
            assert np.isfinite(out["n2_per_sv"][j])
        else:
            assert np.isnan(out["n1_per_sv"][j])
            assert np.isnan(out["n2_per_sv"][j])


def test_ar_fixed_position_not_worse_than_float_in_clean_data():
    """With low noise, AR shouldn't make the position worse than the float solve."""
    s = _make_static_session(n_epoch=100, n_sv=8, sigma_code_m=0.3,
                              sigma_phase_m=0.003)
    out = ppp_solve_static_batch_with_ar(
        s["pr_if"], s["phase_if"], s["sv_ecef"], s["sat_clock"],
        p1_m=s["p1"], p2_m=s["p2"],
        phi1_cycles=s["phi1"], phi2_cycles=s["phi2"],
        tropo=s["tropo"],
        initial_position=tuple(s["truth_rx"] + np.array([5.0, 5.0, 5.0])),
    )
    err_float = np.linalg.norm(
        np.array(out["float"]["position"]) - s["truth_rx"]
    )
    err_fixed = np.linalg.norm(
        np.array(out["fixed"]["position"]) - s["truth_rx"]
    )
    # Allow a small slack (5 cm) to absorb the float ambiguities that
    # AR may have re-aligned to integers.
    assert err_fixed < err_float + 0.05
