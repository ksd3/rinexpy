"""Parity tests for the C++ SSR orbit + clock correction kernels.

Skipped when rinexpy_native isn't importable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from rinexpy import _native
from rinexpy.realtime import RealtimeOrbitClock, _SSRClockEntry, _SSREntry

pytest.importorskip("rinexpy_native")


def _force(on: bool) -> None:
    if on:
        _native.have_apply_ssr = (
            lambda: _native._apply_ssr_orbit_correction is not None
        )
    else:
        _native.have_apply_ssr = lambda: False


def test_native_apply_ssr_available():
    assert _native.have_apply_ssr() is True


def _build_cache():
    cache = RealtimeOrbitClock()
    now = datetime.now(timezone.utc)
    cache.ssr_orbit[5] = _SSREntry(
        received=now, iod=0,
        radial_m=0.123, along_m=-0.456, cross_m=0.789,
        dot_radial_m_per_s=1e-4, dot_along_m_per_s=2e-4,
        dot_cross_m_per_s=-3e-4,
    )
    cache.ssr_clock[5] = _SSRClockEntry(
        received=now, c0_m=1.5, c1_m_per_s=0.001, c2_m_per_s2=0.0,
    )
    return cache


def test_orbit_correction_matches_python():
    cache = _build_cache()
    r = np.array([20e6, 5e6, 18e6])
    v = np.array([1000.0, -1500.0, 2000.0])

    _force(False)
    py = cache.apply_orbit_correction(5, r, v, elapsed_s=2.0)
    _force(True)
    cpp = cache.apply_orbit_correction(5, r, v, elapsed_s=2.0)
    np.testing.assert_allclose(cpp, py, rtol=0, atol=1e-12)


def test_clock_correction_matches_python():
    cache = _build_cache()
    _force(False)
    py = cache.apply_clock_correction(5, 1e-6, elapsed_s=2.0)
    _force(True)
    cpp = cache.apply_clock_correction(5, 1e-6, elapsed_s=2.0)
    assert abs(cpp - py) < 1e-18


def test_orbit_falls_back_when_no_correction():
    """Missing SSR entry must still leave the broadcast position alone."""
    cache = RealtimeOrbitClock()
    r = np.array([20e6, 5e6, 18e6])
    v = np.array([1000.0, -1500.0, 2000.0])
    _force(True)
    out = cache.apply_orbit_correction(99, r, v, elapsed_s=0.0)
    np.testing.assert_array_equal(out, r)


def teardown_module():
    _force(True)
