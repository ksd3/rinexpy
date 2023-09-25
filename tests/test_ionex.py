"""Tests for the IONEX (.inx) reader."""

from __future__ import annotations

import pytest

from rinexpy.ionex import load_ionex

# Minimal IONEX: 3 lat * 5 lon grid, 1 epoch.
_SAMPLE = """\
     1.0            IONOSPHERE MAPS    GPS                 IONEX VERSION / TYPE
  2024     3    14     0     0     0                        EPOCH OF FIRST MAP
  2024     3    14     0     0     0                        EPOCH OF LAST MAP
   300                                                      INTERVAL
     1                                                      # OF MAPS IN FILE
    -1                                                      EXPONENT
   2.5  -2.5  -2.5                                          LAT1 / LAT2 / DLAT
   0.0  10.0   2.5                                          LON1 / LON2 / DLON
                                                            END OF HEADER
     1                                                      START OF TEC MAP
  2024     3    14     0     0     0                        EPOCH OF CURRENT MAP
   2.5   0.0  10.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   10   20   30   40   50
   0.0   0.0  10.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
   60   70   80   90  100
  -2.5   0.0  10.0   2.5 100.0                              LAT/LON1/LON2/DLON/H
  110  120  130  140  150
                                                            END OF TEC MAP
                                                            END OF FILE
"""


@pytest.fixture
def sample_inx(tmp_path):
    p = tmp_path / "sample.inx"
    p.write_text(_SAMPLE)
    return p


def test_ionex_dims(sample_inx):
    ds = load_ionex(sample_inx)
    assert ds.time.size == 1
    assert ds.lat.size == 3
    assert ds.lon.size == 5


def test_ionex_values(sample_inx):
    ds = load_ionex(sample_inx)
    # values are integers * 10^exponent (=-1) -> divide by 10
    assert ds.tec.values[0, 0, 0] == pytest.approx(1.0)
    assert ds.tec.values[0, 0, 4] == pytest.approx(5.0)
    assert ds.tec.values[0, 2, 4] == pytest.approx(15.0)


def test_ionex_axis_order(sample_inx):
    ds = load_ionex(sample_inx)
    assert ds.lat.values[0] == 2.5
    assert ds.lat.values[-1] == -2.5
