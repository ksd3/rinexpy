"""Tests for the time-parsing helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rinexpy._time import (
    normalize_interval,
    normalize_tlim,
    parse_header_epoch,
    parse_nav2_epoch,
    parse_nav3_epoch,
    parse_obs2_epoch,
    parse_obs3_epoch,
)


def test_parse_obs2_epoch_simple():
    line = " 18  6 22  6 17 30.0000000  0  3"
    assert parse_obs2_epoch(line) == datetime(2018, 6, 22, 6, 17, 30)


def test_parse_obs2_epoch_two_digit_year_pivot():
    # 79 -> 2079, 80 -> 1980
    line = " 79  1  1  0  0  0.0000000  0  1"
    assert parse_obs2_epoch(line).year == 2079
    line = " 80  1  1  0  0  0.0000000  0  1"
    assert parse_obs2_epoch(line).year == 1980


def test_parse_obs2_epoch_bad_flag():
    line = " 10  3  5  0  0  0.0000000  9  1"
    with pytest.raises(ValueError):
        parse_obs2_epoch(line)


def test_parse_obs3_epoch():
    line = "> 2018 05 13 01 30 00.0000000  0 25"
    assert parse_obs3_epoch(line) == datetime(2018, 5, 13, 1, 30, 0)


def test_parse_obs3_epoch_ns_matches_python_datetime():
    """The int-ns parser must agree with the datetime parser to nanosecond
    precision on the same line."""
    import numpy as np

    from rinexpy._time import parse_obs3_epoch_ns

    line = "> 2018 05 13 01 30 00.0500000  0 25"
    ns = parse_obs3_epoch_ns(line)
    dt = parse_obs3_epoch(line)
    assert np.datetime64(ns, "ns") == np.datetime64(dt, "ns")


def test_parse_obs3_epoch_ns_rejects_bad_line():
    from rinexpy._time import parse_obs3_epoch_ns

    with pytest.raises(ValueError):
        parse_obs3_epoch_ns("X 2018 05 13 01 30 00.0000000  0 25")


def test_datetime_to_ns_round_trip():
    """``datetime_to_ns`` and ``parse_obs3_epoch_ns`` use the same algorithm."""
    import numpy as np

    from rinexpy._time import datetime_to_ns

    dt = datetime(2018, 5, 13, 1, 30, 0)
    assert np.datetime64(datetime_to_ns(dt), "ns") == np.datetime64(dt, "ns")


def test_parse_obs3_epoch_microseconds():
    line = "> 2018 05 13 01 30 00.5000000  0 25"
    assert parse_obs3_epoch(line) == datetime(2018, 5, 13, 1, 30, 0, 500_000)


def test_parse_obs3_epoch_wrong_marker():
    with pytest.raises(ValueError):
        parse_obs3_epoch("X 2018 05 13 01 30 00.0000000  0 25")


def test_parse_nav2_epoch():
    # Format: 'SV YY MM DD HH MM SS.S' - SV in cols 0-1, then space-separated.
    line = " 6 99  9  2 19  0  0.0 -.839701388031D-03"
    assert parse_nav2_epoch(line) == datetime(1999, 9, 2, 19, 0, 0)


def test_parse_nav3_epoch():
    line = "G07 2018 06 22 08 00 00 -.123D-04"
    assert parse_nav3_epoch(line) == datetime(2018, 6, 22, 8, 0, 0)


def test_parse_header_epoch_robust():
    # Note: spaces in the seconds field that some receivers emit.
    field = "  2018     6    22     6    17   30.0000000     GPS"
    t = parse_header_epoch(field)
    assert t == datetime(2018, 6, 22, 6, 17, 30)


def test_normalize_tlim_none():
    assert normalize_tlim(None) is None


def test_normalize_tlim_strings():
    a, b = normalize_tlim(("2018-01-01", "2018-01-02"))
    assert a == datetime(2018, 1, 1)
    assert b == datetime(2018, 1, 2)


def test_normalize_tlim_datetimes():
    a, b = (datetime(2018, 1, 1), datetime(2018, 1, 2))
    assert normalize_tlim((a, b)) == (a, b)


def test_normalize_tlim_swapped():
    with pytest.raises(ValueError):
        normalize_tlim((datetime(2020, 1, 1), datetime(2019, 1, 1)))


def test_normalize_tlim_wrong_length():
    with pytest.raises(ValueError):
        normalize_tlim((datetime(2020, 1, 1),))  # type: ignore[arg-type]


def test_normalize_interval_seconds():
    assert normalize_interval(30) == timedelta(seconds=30)
    assert normalize_interval(15.5) == timedelta(seconds=15.5)


def test_normalize_interval_timedelta():
    td = timedelta(minutes=2)
    assert normalize_interval(td) is td


def test_normalize_interval_none():
    assert normalize_interval(None) is None


def test_normalize_interval_negative():
    with pytest.raises(ValueError):
        normalize_interval(-1)


def test_normalize_interval_wrong_type():
    with pytest.raises(TypeError):
        normalize_interval("30")  # type: ignore[arg-type]
