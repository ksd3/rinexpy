"""Tests for the SINEX-BIAS DCB reader and application."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from rinexpy.dcb import (
    C_M_PER_S, correct_pseudorange, get_bias, read_bsx, read_code_dcb,
)


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


def test_spp_solve_applies_dcb_records(bsx_file: Path):
    """spp_solve(dcb_records=..., dcb_obs_code=..., sv_labels=...)
    should produce a clock-bias offset equal to the common DCB bias
    when every SV has the same OSB."""
    import numpy as np
    from rinexpy.geodesy import lla_to_ecef
    from rinexpy.positioning import spp_solve

    truth = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    sv = np.array([
        truth + np.array([2.0e7, 1.0e7, 1.5e7]),
        truth + np.array([-2.0e7, 1.0e7, 1.5e7]),
        truth + np.array([0.0, 2.0e7, 2.0e7]),
        truth + np.array([0.0, -2.0e7, 1.0e7]),
        truth + np.array([1.5e7, 0.0, 1.7e7]),
    ])
    sv_labels = ["G05", "G05", "G05", "G05", "G05"]
    pr = np.linalg.norm(sv - truth, axis=1)
    records = read_bsx(bsx_file)

    sol_off = spp_solve(sv, pr)
    sol_on = spp_solve(
        sv, pr.copy(),
        sv_labels=sv_labels,
        dcb_records=records,
        dcb_obs_code="C1W",
    )
    # Sat-side OSB for G05 / C1W is -7.1234 ns -> ~-2.14 m. Applied to
    # every SV identically, so position stays put and only clock moves.
    expected_m = -7.1234 * C_M_PER_S * 1e-9
    diff_pos = np.linalg.norm(
        np.array(sol_on["position"]) - np.array(sol_off["position"])
    )
    diff_clk_m = C_M_PER_S * (sol_on["clock_bias"] - sol_off["clock_bias"])
    assert diff_pos < 1e-6
    assert diff_clk_m == pytest.approx(expected_m, abs=1e-6)


def test_spp_solve_applies_tgd_map(bsx_file: Path):
    """spp_solve(tgd_map=..., sv_labels=...) subtracts c*gamma*TGD per
    SV, shifting the receiver-clock estimate."""
    import numpy as np
    from rinexpy.geodesy import lla_to_ecef
    from rinexpy.positioning import spp_solve

    truth = np.array(lla_to_ecef(40.0, -3.0, 100.0))
    sv = np.array([
        truth + np.array([2.0e7, 1.0e7, 1.5e7]),
        truth + np.array([-2.0e7, 1.0e7, 1.5e7]),
        truth + np.array([0.0, 2.0e7, 2.0e7]),
        truth + np.array([0.0, -2.0e7, 1.0e7]),
        truth + np.array([1.5e7, 0.0, 1.7e7]),
    ])
    sv_labels = ["G01", "G02", "G03", "G04", "G05"]
    pr = np.linalg.norm(sv - truth, axis=1)
    tgd_map = {s: 5e-9 for s in sv_labels}  # 5 ns common TGD

    sol_off = spp_solve(sv, pr)
    sol_on = spp_solve(sv, pr.copy(), sv_labels=sv_labels, tgd_map=tgd_map)
    expected_m = -C_M_PER_S * 5e-9
    diff_clk_m = C_M_PER_S * (sol_on["clock_bias"] - sol_off["clock_bias"])
    assert diff_clk_m == pytest.approx(expected_m, abs=1e-6)


def test_spp_solve_dcb_requires_sv_labels(bsx_file: Path):
    import numpy as np
    from rinexpy.positioning import spp_solve
    sv = np.zeros((5, 3))
    sv[:, 0] = np.linspace(2.0e7, 2.5e7, 5)
    pr = np.full(5, 2.3e7)
    records = read_bsx(bsx_file)
    with pytest.raises(ValueError):
        spp_solve(sv, pr, dcb_records=records, dcb_obs_code="C1W")


# ---------------------------------------------------------------------------
# Legacy AIUB CODE monthly DCB reader (pre-2017 format)
# ---------------------------------------------------------------------------


_SAMPLE_CODE_P1P2 = """\
                P1-P2 DIFFERENTIAL CODE BIASES (DCB) FOR SATELLITES AND STATIONS
                CODE'S MONTHLY DCB SOLUTION
                Reference: AIUB Astronomical Institute, University of Bern

PRN / STATION NAME        VALUE (NS)  RMS (NS)
***  ****************     *****.****  ****.****
G01                       -2.5470     0.0245
G02                        0.6730     0.0312
G05                       -1.1234     0.0210
ALGO                       2.4500     0.0500
"""

_SAMPLE_CODE_P1C1 = """\
                P1-C1 DIFFERENTIAL CODE BIASES (DCB) FOR SATELLITES AND STATIONS
                Reference: AIUB

PRN / STATION NAME        VALUE (NS)  RMS (NS)
G01                       -0.4900     0.0520
G05                        0.3210     0.0480
"""


@pytest.fixture
def code_p1p2_file(tmp_path: Path) -> Path:
    p = tmp_path / "P1P21001.DCB"
    p.write_text(_SAMPLE_CODE_P1P2)
    return p


def test_read_code_dcb_parses_satellite_records(code_p1p2_file: Path):
    records = read_code_dcb(code_p1p2_file)
    sats = [r for r in records if r["prn"]]
    assert len(sats) == 3
    g05 = next(r for r in sats if r["prn"] == "G05")
    assert g05["obs1"] == "C1W"  # P1
    assert g05["obs2"] == "C2W"  # P2
    assert g05["bias_type"] == "DSB"
    assert g05["unit"] == "ns"
    assert g05["value"] == pytest.approx(-1.1234)


def test_read_code_dcb_parses_station_records(code_p1p2_file: Path):
    records = read_code_dcb(code_p1p2_file)
    algo = next(r for r in records if r["station"] == "ALGO")
    assert algo["prn"] == ""
    assert algo["value"] == pytest.approx(2.4500)


def test_read_code_dcb_infers_validity_window_from_filename(code_p1p2_file: Path):
    """P1P21001.DCB -> January 2010 validity window."""
    records = read_code_dcb(code_p1p2_file)
    rec = records[0]
    assert rec["start"].year == 2010
    assert rec["start"].month == 1
    assert rec["end"].year == 2010
    assert rec["end"].month == 1
    assert rec["end"].day == 31


def test_read_code_dcb_explicit_year_month(tmp_path: Path):
    p = tmp_path / "anything.dcb"
    p.write_text(_SAMPLE_CODE_P1P2)
    records = read_code_dcb(p, year=2015, month=6)
    assert records[0]["start"].year == 2015
    assert records[0]["start"].month == 6


def test_read_code_dcb_p1c1_section(tmp_path: Path):
    p = tmp_path / "P1C11005.DCB"
    p.write_text(_SAMPLE_CODE_P1C1)
    records = read_code_dcb(p)
    g01 = next(r for r in records if r["prn"] == "G01")
    assert g01["obs1"] == "C1W"  # P1
    assert g01["obs2"] == "C1C"  # C1
    assert g01["value"] == pytest.approx(-0.4900)


def test_read_code_dcb_records_plug_into_get_bias(code_p1p2_file: Path):
    """CODE records should be queryable by get_bias just like SINEX
    records - that's the whole point of unifying the schema."""
    records = read_code_dcb(code_p1p2_file)
    b = get_bias(records, prn="G05", obs1="C1W", obs2="C2W")
    assert b == pytest.approx(-1.1234 * C_M_PER_S * 1e-9)
