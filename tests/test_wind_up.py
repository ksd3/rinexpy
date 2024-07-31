"""Tests for the carrier-phase wind-up correction."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from rinexpy.geodesy import phase_wind_up_correction


def _rotate(axis, angle, vec):
    """Rotate ``vec`` around the unit ``axis`` by ``angle`` radians (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    v = np.asarray(vec, dtype=float)
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)


def test_aligned_axes_zero_correction():
    """Identical receiver and satellite axes, LOS along z: correction ~0."""
    los = np.array([0.0, 0.0, 1.0])
    xs = np.array([1.0, 0.0, 0.0])
    ys = np.array([0.0, 1.0, 0.0])
    xr = np.array([1.0, 0.0, 0.0])
    yr = np.array([0.0, 1.0, 0.0])
    cyc = phase_wind_up_correction(xs, ys, xr, yr, los)
    assert cyc == approx(0.0, abs=1e-9)


def test_quarter_rotation_about_los_is_quarter_cycle():
    """Rotating the receiver antenna by 90 degrees about LOS gives ~0.25 cycles."""
    los = np.array([0.0, 0.0, 1.0])
    xs = np.array([1.0, 0.0, 0.0])
    ys = np.array([0.0, 1.0, 0.0])
    angle = np.pi / 2
    xr = _rotate(los, angle, xs)
    yr = _rotate(los, angle, ys)
    cyc = phase_wind_up_correction(xs, ys, xr, yr, los)
    assert abs(cyc) == approx(0.25, abs=1e-6)


def test_unwrapping_keeps_continuity():
    """Sweeping the receiver azimuth past pi yields a continuous (unwrapped) series."""
    los = np.array([0.0, 0.0, 1.0])
    xs = np.array([1.0, 0.0, 0.0])
    ys = np.array([0.0, 1.0, 0.0])
    angles = np.linspace(0, 4 * np.pi, 80)  # two full rotations
    previous = 0.0
    series = []
    for a in angles:
        xr = _rotate(los, a, xs)
        yr = _rotate(los, a, ys)
        previous = phase_wind_up_correction(xs, ys, xr, yr, los, previous_cycles=previous)
        series.append(previous)
    # The series should look like a (monotonic) ramp of ~2 full cycles.
    series = np.array(series)
    assert abs(series[-1] - series[0]) == approx(2.0, abs=0.05)
    # No discontinuities larger than ~0.1 cycle (each step is ~0.05 cycle).
    assert np.max(np.abs(np.diff(series))) < 0.1


def test_los_zero_returns_previous():
    """Degenerate LOS: function falls back to ``previous_cycles``."""
    zero = np.zeros(3)
    cyc = phase_wind_up_correction(
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        zero,
        previous_cycles=3.14,
    )
    assert cyc == 3.14


def test_arccos_clamping_handles_numerical_overshoot():
    """Identical axes, but LOS not orthogonal: cos_theta can land just outside [-1, 1] in float."""
    los = np.array([1.0, 1.0, 1.0])
    xs = np.array([1.0, 0.0, 0.0])
    ys = np.array([0.0, 1.0, 0.0])
    xr = np.array([1.0, 0.0, 0.0])
    yr = np.array([0.0, 1.0, 0.0])
    cyc = phase_wind_up_correction(xs, ys, xr, yr, los)
    assert np.isfinite(cyc)
