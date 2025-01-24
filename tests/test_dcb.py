"""Tests for the SINEX-BIAS DCB reader and application."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from rinexpy.dcb import C_M_PER_S, correct_pseudorange, get_bias, read_bsx


SAMPLE_BSX = """\
%=BIA 1.00 IGS 2024:047:00000 IGS 2024:040:00000 2024:046:86399 R 00000003
*-------------------------------------------------------------------------------
+FILE/REFERENCE
 DESCRIPTION       TEST SINEX-BIAS FILE
-FILE/REFERENCE
*-------------------------------------------------------------------------------
+BIAS/SOLUTION
*BIAS  SVN_   PRN_ STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___
 OSB   G063   G05 ----      C1W       2024:040:00000 2024:046:86400 ns                 -7.1234       0.0123
 OSB   G063   G05 ----      C2W       2024:040:00000 2024:046:86400 ns                 -3.5678       0.0089
 DSB   G063   G05 ----      C1W  C2W  2024:040:00000 2024:046:86400 ns                 -3.5556       0.0150
 OSB   ----   ---  ALGO      C1W       2024:040:00000 2024:046:86400 ns                  1.2345       0.0200
-BIAS/SOLUTION
%ENDBIA
"""


@pytest.fixture
def bsx_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.bsx"
    p.write_text(SAMPLE_BSX)
    return p


def test_read_bsx_returns_four_records(bsx_file: Path):
    records = read_bsx(bsx_file)
    assert len(records) == 4
    types = [r["bias_type"] for r in records]
    assert types == ["OSB", "OSB", "DSB", "OSB"]


def test_read_bsx_parses_fields(bsx_file: Path):
    records = read_bsx(bsx_file)
    r = records[0]
    assert r["prn"] == "G05"
    assert r["svn"] == "G063"
    assert r["obs1"] == "C1W"
    assert r["obs2"] == ""
    assert r["station"] == ""
    assert r["unit"] == "ns"
    assert r["value"] == pytest.approx(-7.1234)
    assert r["stddev"] == pytest.approx(0.0123)
    assert r["start"] == datetime(2024, 2, 9, tzinfo=timezone.utc)


def test_read_bsx_station_bias_parses_station_code(bsx_file: Path):
    records = read_bsx(bsx_file)
    rx = next(r for r in records if r["station"] == "ALGO")
    assert rx["prn"] == "---"   # leading dashes preserved
    assert rx["obs1"] == "C1W"
    assert rx["value"] == pytest.approx(1.2345)


def test_get_bias_returns_value_in_meters(bsx_file: Path):
    records = read_bsx(bsx_file)
    b = get_bias(records, prn="G05", obs1="C1W")
    # -7.1234 ns * c -> -2.135 m roughly.
    assert b == pytest.approx(-7.1234 * C_M_PER_S * 1e-9)


def test_get_bias_returns_none_outside_validity_window(bsx_file: Path):
    records = read_bsx(bsx_file)
    epoch = datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert get_bias(records, prn="G05", obs1="C1W", epoch=epoch) is None


def test_get_bias_returns_value_inside_window(bsx_file: Path):
    records = read_bsx(bsx_file)
    epoch = datetime(2024, 2, 12, 12, 0, 0, tzinfo=timezone.utc)
    b = get_bias(records, prn="G05", obs1="C1W", epoch=epoch)
    assert b is not None
    assert b == pytest.approx(-7.1234 * C_M_PER_S * 1e-9)


def test_correct_pseudorange_applies_satellite_bias_only(bsx_file: Path):
    records = read_bsx(bsx_file)
    raw = 23_456_789.0
    corrected = correct_pseudorange(raw, prn="G05", obs_code="C1W", records=records)
    expected = raw + (-7.1234 * C_M_PER_S * 1e-9)
    assert corrected == pytest.approx(expected)


def test_correct_pseudorange_applies_satellite_and_receiver_bias(bsx_file: Path):
    records = read_bsx(bsx_file)
    raw = 23_456_789.0
    corrected = correct_pseudorange(
        raw, prn="G05", obs_code="C1W", records=records, station="ALGO"
    )
    sat = -7.1234 * C_M_PER_S * 1e-9
    rx = 1.2345 * C_M_PER_S * 1e-9
    assert corrected == pytest.approx(raw + sat + rx)


def test_correct_pseudorange_no_match_returns_raw(bsx_file: Path):
    records = read_bsx(bsx_file)
    raw = 23_456_789.0
    # Unknown PRN -> no bias applied, no error.
    corrected = correct_pseudorange(raw, prn="G99", obs_code="C1W", records=records)
    assert corrected == raw


def test_get_bias_dsb_lookup(bsx_file: Path):
    records = read_bsx(bsx_file)
    b = get_bias(records, prn="G05", obs1="C1W", obs2="C2W")
    assert b == pytest.approx(-3.5556 * C_M_PER_S * 1e-9)
