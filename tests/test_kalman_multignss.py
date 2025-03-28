"""Tests for the multi-constellation PPP filter with inter-system biases."""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy.kalman_multignss import StaticPPPFilterMultiGNSS


def _sky() -> np.ndarray:
    return np.array([
        [ 1.5e7,  0.0,    2.0e7],   # GPS
        [-1.5e7,  0.5e7,  2.0e7],   # GPS
        [ 0.5e7,  1.5e7,  1.8e7],   # GPS
        [-0.5e7, -1.5e7,  1.5e7],   # Galileo
        [ 2.0e7, -0.5e7,  1.0e7],   # Galileo
        [-2.0e7,  1.0e7,  0.8e7],   # BeiDou
        [ 1.0e7, -1.7e7,  1.2e7],   # BeiDou
        [-1.0e7,  0.0,    2.2e7],   # BeiDou
    ])


CONSTS = ["G", "G", "G", "E", "E", "C", "C", "C"]


def _wet_mapping(sv, rx):
    diff = sv - rx
    rho = np.linalg.norm(diff, axis=1)
    cos_zen = -np.dot(diff, rx / np.linalg.norm(rx)) / rho
    return np.clip(1.0 / np.sqrt(np.maximum(1.0 - cos_zen ** 2, 1e-6)), 1.0, 10.0)


def test_gps_only_has_no_isb_states():
    f = StaticPPPFilterMultiGNSS(
        n_sv=4, constellations=["G"] * 4,
        initial_position=(0.0, 0.0, 0.0),
    )
    assert f.n_isb == 0
    assert f.isb_m("G") == 0.0


def test_mixed_constellations_get_one_isb_each():
    f = StaticPPPFilterMultiGNSS(
        n_sv=8, constellations=CONSTS,
        initial_position=(0.0, 0.0, 0.0),
    )
    assert f.n_isb == 2   # E and C
    assert f.isb_m("G") == 0.0
    assert f.isb_m("E") == 0.0       # initial state zero
    assert f.isb_m("C") == 0.0
    with pytest.raises(KeyError):
        f.isb_m("R")    # GLONASS isn't in this filter


def test_constellations_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length"):
        StaticPPPFilterMultiGNSS(
            n_sv=4, constellations=["G", "E"],
            initial_position=(0.0, 0.0, 0.0),
        )


def test_isb_recovers_known_offsets():
    """Synthesize code observations with known per-constellation ISBs
    and confirm the filter recovers them within a few iterations."""
    truth_pos = np.array([6_378_137.0, 0.0, 0.0])
    sv = _sky()
    mw = _wet_mapping(sv, truth_pos)
    rho_truth = np.linalg.norm(sv - truth_pos, axis=1)
    # Truth ISBs: E = 5 m, C = -3 m (relative to GPS).
    isb_truth = {"G": 0.0, "E": 5.0, "C": -3.0}
    pr = np.array([
        rho_truth[j] + isb_truth[CONSTS[j]] for j in range(len(sv))
    ])

    f = StaticPPPFilterMultiGNSS(
        n_sv=8, constellations=CONSTS,
        initial_position=tuple(truth_pos),
        initial_zwd_m=0.0,
        sigma_position_init=1e-3,
        sigma_clock_init=1e-3,
        sigma_zwd_init=1e-3,
        sigma_isb_init=100.0,
        sigma_code=0.1,
    )
    for _ in range(20):
        ph = np.full(8, np.nan)
        f.update(sv, np.zeros(8), pr, ph, wet_mapping=mw)

    assert abs(f.isb_m("E") - 5.0) < 0.1
    assert abs(f.isb_m("C") - (-3.0)) < 0.1


def test_full_obs_with_zwd_and_isb():
    """Code + phase observations + non-trivial ZWD + per-system ISBs."""
    truth_pos = np.array([6_378_137.0, 0.0, 0.0])
    truth_zwd = 0.15
    truth_isb = {"G": 0.0, "E": 4.0, "C": -2.5}
    sv = _sky()
    mw = _wet_mapping(sv, truth_pos)
    rho = np.linalg.norm(sv - truth_pos, axis=1)
    rng = np.random.default_rng(0)
    true_ambs = rng.integers(-100, 100, 8).astype(float) * 0.1  # in meters
    pr = np.array([
        rho[j] + mw[j] * truth_zwd + truth_isb[CONSTS[j]]
        for j in range(8)
    ])
    ph = pr + true_ambs

    f = StaticPPPFilterMultiGNSS(
        n_sv=8, constellations=CONSTS,
        initial_position=tuple(truth_pos),
        initial_zwd_m=0.0,
        sigma_position_init=0.1,
        sigma_clock_init=1.0,
        sigma_zwd_init=1.0,
        sigma_isb_init=10.0,
    )
    for _ in range(30):
        f.update(sv, np.zeros(8), pr, ph, wet_mapping=mw)

    assert abs(f.zwd_m - truth_zwd) < 0.03
    assert abs(f.isb_m("E") - 4.0) < 0.3
    assert abs(f.isb_m("C") - (-2.5)) < 0.3
    err = np.linalg.norm(np.array(f.position) - truth_pos)
    assert err < 0.5


def test_predict_grows_isb_variance_only_when_rate_set():
    f1 = StaticPPPFilterMultiGNSS(
        n_sv=4, constellations=["G", "E", "C", "G"],
        initial_position=(0.0, 0.0, 0.0),
        sigma_isb_rate_m_per_sqrt_hr=0.0,
    )
    v_e_before = f1.P[f1._idx_isb_start + f1._isb_map["E"],
                     f1._idx_isb_start + f1._isb_map["E"]]
    f1.predict(dt=3600.0)
    v_e_after = f1.P[f1._idx_isb_start + f1._isb_map["E"],
                    f1._idx_isb_start + f1._isb_map["E"]]
    assert v_e_after == v_e_before     # rate = 0 -> no growth

    f2 = StaticPPPFilterMultiGNSS(
        n_sv=4, constellations=["G", "E", "C", "G"],
        initial_position=(0.0, 0.0, 0.0),
        sigma_isb_rate_m_per_sqrt_hr=0.1,
    )
    v_e_before = f2.P[f2._idx_isb_start + f2._isb_map["E"],
                     f2._idx_isb_start + f2._isb_map["E"]]
    f2.predict(dt=3600.0)
    v_e_after = f2.P[f2._idx_isb_start + f2._isb_map["E"],
                    f2._idx_isb_start + f2._isb_map["E"]]
    assert v_e_after - v_e_before == pytest.approx(0.1 ** 2, rel=1e-9)
