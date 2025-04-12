"""Tests for the real-time orbit/clock cache."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rinexpy.realtime import RealtimeOrbitClock


def test_ingest_ssr_orbit_message():
    cache = RealtimeOrbitClock()
    msg = {
        "msg_id": 1057,
        "satellites": [
            {
                "prn": 5,
                "iode": 42,
                "delta_radial_m": 0.123,
                "delta_along_track_m": 0.456,
                "delta_cross_track_m": -0.789,
                "dot_delta_radial_m_per_s": 0.001,
                "dot_delta_along_track_m_per_s": 0.002,
                "dot_delta_cross_track_m_per_s": -0.003,
            },
        ],
    }
    cache.ingest(msg)
    e = cache.ssr_orbit_for(5)
    assert e is not None
    assert e.iod == 42
    assert e.radial_m == pytest.approx(0.123)
    assert cache.ssr_orbit_for(99) is None


def test_ingest_ssr_clock_message():
    cache = RealtimeOrbitClock()
    msg = {
        "msg_id": 1058,
        "satellites": [
            {"prn": 7, "c0_m": -0.55, "c1_m_per_s": 0.01, "c2_m_per_s2": 1e-5},
        ],
    }
    cache.ingest(msg)
    e = cache.ssr_clock_for(7)
    assert e is not None
    assert e.c0_m == pytest.approx(-0.55)


def test_stale_ssr_returns_none():
    cache = RealtimeOrbitClock(ssr_validity_s=0.0)   # immediately stale
    cache.ingest({
        "msg_id": 1057,
        "satellites": [{
            "prn": 5, "iode": 1,
            "delta_radial_m": 0, "delta_along_track_m": 0, "delta_cross_track_m": 0,
            "dot_delta_radial_m_per_s": 0, "dot_delta_along_track_m_per_s": 0,
            "dot_delta_cross_track_m_per_s": 0,
        }],
    })
    # Validity = 0 -> immediately stale.
    assert cache.ssr_orbit_for(5) is None


def test_apply_orbit_correction_no_velocity_returns_unchanged():
    cache = RealtimeOrbitClock()
    cache.ingest({
        "msg_id": 1057,
        "satellites": [{
            "prn": 5, "iode": 1,
            "delta_radial_m": 10.0, "delta_along_track_m": 5.0, "delta_cross_track_m": -3.0,
            "dot_delta_radial_m_per_s": 0, "dot_delta_along_track_m_per_s": 0,
            "dot_delta_cross_track_m_per_s": 0,
        }],
    })
    pos = np.array([2.6e7, 0.0, 0.0])
    corrected = cache.apply_orbit_correction(5, pos)
    np.testing.assert_allclose(corrected, pos)   # no velocity -> falls back


def test_apply_orbit_correction_with_velocity():
    cache = RealtimeOrbitClock()
    cache.ingest({
        "msg_id": 1057,
        "satellites": [{
            "prn": 5, "iode": 1,
            "delta_radial_m": 10.0, "delta_along_track_m": 0.0, "delta_cross_track_m": 0.0,
            "dot_delta_radial_m_per_s": 0, "dot_delta_along_track_m_per_s": 0,
            "dot_delta_cross_track_m_per_s": 0,
        }],
    })
    pos = np.array([2.6e7, 0.0, 0.0])
    vel = np.array([0.0, 3_870.0, 0.0])   # circular orbit speed
    corrected = cache.apply_orbit_correction(5, pos, sv_velocity_ecef=vel)
    # 10-m radial correction (toward Earth) -> position shrinks in x by 10 m.
    np.testing.assert_allclose(corrected, pos - np.array([10.0, 0.0, 0.0]), atol=1e-9)


def test_apply_clock_correction_subtracts_in_seconds():
    cache = RealtimeOrbitClock()
    cache.ingest({
        "msg_id": 1058,
        "satellites": [{"prn": 5, "c0_m": 299.792458, "c1_m_per_s": 0, "c2_m_per_s2": 0}],
    })
    # c0 = 1 us in meters (= 299.792458 m); should subtract 1 us from broadcast.
    corrected = cache.apply_clock_correction(5, broadcast_clock_s=1e-3)
    assert corrected == pytest.approx(1e-3 - 1e-6, abs=1e-15)


def test_ingest_routes_broadcast_ephemerides():
    cache = RealtimeOrbitClock()
    cache.ingest({"msg_id": 1019, "prn": 11})
    cache.ingest({"msg_id": 1042, "prn": 30})
    cache.ingest({"msg_id": 1045, "prn": 8})
    cache.ingest({"msg_id": 1020, "prn": 17})
    assert ("G", 11) in cache.broadcast
    assert ("C", 30) in cache.broadcast
    assert ("E", 8) in cache.broadcast
    assert ("R", 17) in cache.broadcast


def test_ingest_has_message():
    cache = RealtimeOrbitClock()
    # MT 2 orbit message
    cache.ingest({
        "header": {"message_type": 2},
        "msg_id": None,
        "payload": {
            "satellites": [
                {"gnss_id": 2, "prn": 5, "delta_radial_m": 1.0,
                 "delta_along_track_m": 0.5, "delta_cross_track_m": -0.3, "iod": 17},
            ],
        },
    })
    assert (2, 5) in cache.has_orbit
    assert cache.has_orbit[(2, 5)]["delta_radial_m"] == 1.0
