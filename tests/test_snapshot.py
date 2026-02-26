"""Tests for the code-phase-only snapshot positioning solver."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.snapshot import snapshot_positioning

_C = 299_792_458.0
_CHIP_LEN_M = _C / 1.023e6
_PERIOD_LEN_M = _C * 1e-3


def _gen_arrangement():
    truth = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    sv = np.array([
        truth + np.array([2.0e7, 1.0e7, 1.5e7]),
        truth + np.array([-2.0e7, 1.0e7, 1.5e7]),
        truth + np.array([0, 2.0e7, 2.0e7]),
        truth + np.array([0, -2.0e7, 1.0e7]),
        truth + np.array([1.5e7, 0, 1.7e7]),
        truth + np.array([-1.0e7, -1.5e7, 1.3e7]),
    ])
    return truth, sv


def test_snapshot_recovers_truth_from_close_prior():
    truth, sv = _gen_arrangement()
    true_bias_s = 1.234e-7
    true_pr = np.linalg.norm(sv - truth, axis=1) + true_bias_s * _C
    # Code phase = fractional portion within one C/A code period.
    frac_m = np.mod(true_pr, _PERIOD_LEN_M)
    code_chips = frac_m / _CHIP_LEN_M

    # Coarse prior 30 km away (typical cell-tower accuracy).
    prior = truth + np.array([20_000.0, -10_000.0, 5000.0])
    out = snapshot_positioning(code_chips, sv, tuple(prior))
    px, py, pz = out["position_ecef"]
    assert px == approx(truth[0], abs=1.0)
    assert py == approx(truth[1], abs=1.0)
    assert pz == approx(truth[2], abs=1.0)
    assert out["time_bias_s"] == approx(true_bias_s, abs=1e-9)
    # All ms-integer counts agree with the truth pseudoranges.
    assert np.allclose(out["pseudoranges_m"], true_pr, atol=1.0)


def test_snapshot_lla_within_a_few_meters():
    truth, sv = _gen_arrangement()
    true_pr = np.linalg.norm(sv - truth, axis=1)
    code_chips = (np.mod(true_pr, _PERIOD_LEN_M)) / _CHIP_LEN_M
    prior = truth + np.array([10_000.0, 10_000.0, 0.0])
    out = snapshot_positioning(code_chips, sv, tuple(prior))
    lat, lon, alt = out["lla"]
    # 1 deg latitude ~ 111 km, so 5 m tolerance maps to ~4.5e-5 deg.
    assert lat == approx(40.0, abs=1e-4)
    assert lon == approx(-3.0, abs=1e-4)
    assert alt == approx(100.0, abs=5.0)


def test_snapshot_requires_min_four_svs():
    truth, sv = _gen_arrangement()
    code_chips = np.zeros(3)
    with pytest.raises(ValueError):
        snapshot_positioning(code_chips, sv[:3], tuple(truth))


def test_snapshot_converges_within_iter_limit():
    truth, sv = _gen_arrangement()
    true_pr = np.linalg.norm(sv - truth, axis=1)
    code_chips = (np.mod(true_pr, _PERIOD_LEN_M)) / _CHIP_LEN_M
    out = snapshot_positioning(
        code_chips, sv, tuple(truth + np.array([5000.0, 0, 0])), max_iter=20
    )
    assert out["n_iter"] <= 20
