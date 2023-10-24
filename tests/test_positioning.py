"""Tests for the SPP solver."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import lla_to_ecef
from rinexpy.positioning import spp_solve

_C = 299_792_458.0


def _synthetic_pseudoranges(rx_ecef, sv_ecef, clock_bias_s=0.0):
    """Build noise-free pseudoranges consistent with a known receiver position."""
    rx = np.asarray(rx_ecef)
    diff = np.asarray(sv_ecef) - rx
    return np.linalg.norm(diff, axis=1) + _C * clock_bias_s


def test_spp_recovers_known_position():
    """Noise-free synthetic data: SPP recovers the truth to mm precision."""
    truth_rx = lla_to_ecef(40, -3, 100)
    sv = np.array(
        [
            [2.0e7, 1.0e7, 1.5e7],
            [-2.0e7, 1.0e7, 1.5e7],
            [0.0, 2.0e7, 2.0e7],
            [0.0, -2.0e7, 1.0e7],
            [1.5e7, 0.0, 1.7e7],
        ]
    )
    pr = _synthetic_pseudoranges(truth_rx, sv)
    sol = spp_solve(sv, pr)
    assert sol["position"][0] == approx(truth_rx[0], abs=1e-3)
    assert sol["position"][1] == approx(truth_rx[1], abs=1e-3)
    assert sol["position"][2] == approx(truth_rx[2], abs=1e-3)
    assert sol["clock_bias"] == approx(0.0, abs=1e-9)


def test_spp_recovers_clock_bias():
    """Synthetic data with a known clock bias is recovered."""
    truth_rx = lla_to_ecef(0, 0, 0)
    sv = np.array(
        [
            [2.6e7, 0, 0],
            [0, 2.6e7, 0],
            [0, 0, 2.6e7],
            [-2.6e7 / 2, -2.6e7 / 2, 2.6e7 / 2],
        ]
    )
    bias_s = 1e-4  # 100 µs
    pr = _synthetic_pseudoranges(truth_rx, sv, clock_bias_s=bias_s)
    sol = spp_solve(sv, pr)
    assert sol["clock_bias"] == approx(bias_s, abs=1e-9)


def test_spp_too_few_sats():
    sv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    pr = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        spp_solve(sv, pr)


def test_spp_returns_lla():
    truth_rx = lla_to_ecef(40, -3, 100)
    sv = np.array(
        [
            [2.0e7, 1.0e7, 1.5e7],
            [-2.0e7, 1.0e7, 1.5e7],
            [0.0, 2.0e7, 2.0e7],
            [0.0, -2.0e7, 1.0e7],
            [1.5e7, 0.0, 1.7e7],
        ]
    )
    pr = _synthetic_pseudoranges(truth_rx, sv)
    sol = spp_solve(sv, pr)
    lat, lon, alt = sol["lla"]
    assert lat == approx(40, abs=1e-4)
    assert lon == approx(-3, abs=1e-4)
    assert alt == approx(100, abs=1.0)
