"""Tests for the GPS time utilities."""

from __future__ import annotations

from datetime import datetime

import pytest

from rinexpy.gpstime import (
    GPS_EPOCH,
    datetime_to_gps,
    datetime_to_gps_seconds,
    gps_seconds_to_datetime,
    gps_to_datetime,
    gps_week_rollover,
    leap_seconds_at,
)


def test_gps_epoch_is_week_zero():
    assert datetime_to_gps(GPS_EPOCH) == (0, 0.0)


def test_known_known_week_number():
    # GPS_EPOCH (1980-01-06) is week 0. 1983 weeks * 7 days = 13881 days,
    # so GPS week 1983 begins 2018-01-07 00:00:18 GPS == 2018-01-07 UTC
    # (after subtracting the 18-leap-second offset).
    week, sow = datetime_to_gps(datetime(2018, 1, 7))
    assert week == 1983
    # Subtract the 18-second leap offset and we're 18s into the week.
    assert sow == pytest.approx(18.0, abs=1e-6)


def test_round_trip():
    t = datetime(2020, 6, 15, 12, 34, 56)
    week, sow = datetime_to_gps(t)
    back = gps_to_datetime(week, sow)
    assert abs((back - t).total_seconds()) < 1e-6


def test_continuous_seconds_round_trip():
    t = datetime(2024, 3, 14, 1, 59, 26, 535_897)
    s = datetime_to_gps_seconds(t)
    back = gps_seconds_to_datetime(s)
    assert abs((back - t).total_seconds()) < 1e-6


def test_leap_seconds_growing():
    assert leap_seconds_at(datetime(1980, 1, 1)) == 19
    assert leap_seconds_at(datetime(1985, 1, 1)) == 22
    assert leap_seconds_at(datetime(2017, 6, 1)) == 37
    assert leap_seconds_at(datetime(2026, 1, 1)) == 37  # no leap since 2017


def test_rollover_resolves_close_to_reference():
    week_full = 2356
    ten_bit = week_full % 1024
    resolved = gps_week_rollover(ten_bit, datetime(2025, 4, 1))
    assert resolved == week_full
