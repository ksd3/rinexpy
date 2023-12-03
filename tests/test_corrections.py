"""Tests for ANTEX/IONEX/CLK *application* helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pytest import approx

from rinexpy.antex import apply_antex_pcv, find_antenna, load_antex
from rinexpy.clk import interpolate_clk, load_clk
from rinexpy.ionex import interp_tec, load_ionex, slant_tec

# ---------------------------------------------------------------------------
# ANTEX application
# ---------------------------------------------------------------------------

_ATX_SAMPLE = """\
     1.4            M                                       ANTEX VERSION / SYST
A                                                           PCV TYPE / REFANT
                                                            END OF HEADER
                                                            START OF ANTENNA
TRM41249.00     NONE                                        TYPE / SERIAL NO
     0.0  90.0   5.0                                        ZEN1 / ZEN2 / DZEN
     1                                                      # OF FREQUENCIES
   G01                                                      START OF FREQUENCY
      0.50      0.50     60.00                              NORTH / EAST / UP
   NOAZI    0.00    1.00    2.00    3.00    4.00    5.00    6.00    7.00    8.00    9.00   10.00   11.00   12.00   13.00   14.00   15.00   16.00   17.00   18.00
   G01                                                      END OF FREQUENCY
                                                            END OF ANTENNA
"""


@pytest.fixture
def atx(tmp_path):
    p = tmp_path / "x.atx"
    p.write_text(_ATX_SAMPLE)
    return load_antex(p)


def test_find_antenna_by_type(atx):
    e = find_antenna(atx, "TRM41249.00     NONE")
    assert e is not None


def test_apply_antex_pcv_zenith(atx):
    e = atx[0]
    # NOAZI[0] = 0 mm at zen=0 -> 0 m
    assert apply_antex_pcv(e, "G01", 90.0) == approx(0.0)


def test_apply_antex_pcv_horizon(atx):
    e = atx[0]
    # NOAZI[18] = 18 mm at zen=90 -> 0.018 m
    assert apply_antex_pcv(e, "G01", 0.0) == approx(0.018)


def test_apply_antex_pcv_unknown_freq(atx):
    e = atx[0]
    assert apply_antex_pcv(e, "G02", 45.0) == 0.0


# ---------------------------------------------------------------------------
# IONEX application
# ---------------------------------------------------------------------------

_INX_SAMPLE = """\
     1.0            IONOSPHERE MAPS    GPS                 IONEX VERSION / TYPE
  2024     3    14     0     0     0                        EPOCH OF FIRST MAP
  2024     3    14     0    10     0                        EPOCH OF LAST MAP
   600                                                      INTERVAL
     2                                                      # OF MAPS IN FILE
    -1                                                      EXPONENT
   2.5  -2.5  -2.5                                          LAT1 / LAT2 / DLAT
   0.0   5.0   2.5                                          LON1 / LON2 / DLON
                                                            END OF HEADER
     1                                                      START OF TEC MAP
  2024     3    14     0     0     0                        EPOCH OF CURRENT MAP
   2.5   0.0   5.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   10   20   30
   0.0   0.0   5.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   40   50   60
  -2.5   0.0   5.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   70   80   90
                                                            END OF TEC MAP
     2                                                      START OF TEC MAP
  2024     3    14     0    10     0                        EPOCH OF CURRENT MAP
   2.5   0.0   5.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   20   30   40
   0.0   0.0   5.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   50   60   70
  -2.5   0.0   5.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   80   90  100
                                                            END OF TEC MAP
                                                            END OF FILE
"""


@pytest.fixture
def inx(tmp_path):
    p = tmp_path / "x.inx"
    p.write_text(_INX_SAMPLE)
    return load_ionex(p)


def test_interp_tec_at_grid_point(inx):
    # At the (lat=0, lon=0, t=2024-03-14T00:00) corner the value is 4.0 TECU.
    v = interp_tec(inx, 0.0, 0.0, datetime(2024, 3, 14, 0, 0))
    assert v == approx(4.0, abs=1e-6)


def test_interp_tec_temporal_midpoint(inx):
    """Midway between the two epochs the values should average."""
    v0 = interp_tec(inx, 0.0, 0.0, datetime(2024, 3, 14, 0, 0))
    v1 = interp_tec(inx, 0.0, 0.0, datetime(2024, 3, 14, 0, 10))
    v_mid = interp_tec(inx, 0.0, 0.0, datetime(2024, 3, 14, 0, 5))
    assert v_mid == approx((v0 + v1) / 2, abs=1e-6)


def test_interp_tec_outside_returns_nan(inx):
    import math

    out = interp_tec(inx, 0.0, 0.0, datetime(2024, 3, 14, 1, 0))
    assert math.isnan(out)


def test_slant_tec_zenith_equals_vertical():
    assert slant_tec(10.0, 90.0) == approx(10.0, abs=1e-6)


def test_slant_tec_low_elevation_amplifies():
    assert slant_tec(10.0, 10.0) > 25  # sec(z')/cos(z') ~3 at el=10


# ---------------------------------------------------------------------------
# CLK interpolation
# ---------------------------------------------------------------------------

_CLK_SAMPLE = """\
     3.00           CLOCK              GPS                 RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01         2024 03 14 00 00  0.000000  1   -1.000D-04
AS G01         2024 03 14 00 05  0.000000  1   -2.000D-04
"""


@pytest.fixture
def clk(tmp_path):
    p = tmp_path / "x.clk"
    p.write_text(_CLK_SAMPLE)
    return load_clk(p)


def test_clk_interpolate_at_endpoint(clk):
    assert interpolate_clk(clk, "G01", datetime(2024, 3, 14, 0, 0)) == approx(-1e-4)


def test_clk_interpolate_midpoint(clk):
    v = interpolate_clk(clk, "G01", datetime(2024, 3, 14, 0, 2, 30))
    assert v == approx(-1.5e-4, abs=1e-9)


def test_clk_interpolate_outside_nan(clk):
    import math

    assert math.isnan(interpolate_clk(clk, "G01", datetime(2024, 3, 14, 1, 0)))
    assert math.isnan(interpolate_clk(clk, "G99", datetime(2024, 3, 14, 0, 0)))


def test_clk_interpolate_at_t_outside_range(clk):
    import math

    earlier = datetime(2024, 3, 14) - timedelta(hours=1)
    assert math.isnan(interpolate_clk(clk, "G01", earlier))
