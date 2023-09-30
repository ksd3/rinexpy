"""Tests for the ANTEX (.atx) reader."""

from __future__ import annotations

import pytest

from rinexpy.antex import load_antex

_SAMPLE = """\
     1.4            M                                       ANTEX VERSION / SYST
A                                                           PCV TYPE / REFANT
                                                            END OF HEADER
                                                            START OF ANTENNA
TRM41249.00     NONE                                        TYPE / SERIAL NO
     0.0  90.0   5.0                                        ZEN1 / ZEN2 / DZEN
     2                                                      # OF FREQUENCIES
   G01                                                      START OF FREQUENCY
      0.50      0.50     60.00                              NORTH / EAST / UP
   NOAZI    0.00    0.10    0.20    0.30    0.40    0.50    0.60    0.70    0.80    0.90    1.00    1.10    1.20    1.30    1.40    1.50    1.60    1.70    1.80    2.00
   G01                                                      END OF FREQUENCY
   G02                                                      START OF FREQUENCY
      0.40      0.40     58.00                              NORTH / EAST / UP
   NOAZI    0.00    0.05    0.10    0.15    0.20    0.25    0.30    0.35    0.40    0.45    0.50    0.55    0.60    0.65    0.70    0.75    0.80    0.85    0.90    1.00
   G02                                                      END OF FREQUENCY
                                                            END OF ANTENNA
"""


@pytest.fixture
def sample_atx(tmp_path):
    p = tmp_path / "sample.atx"
    p.write_text(_SAMPLE)
    return p


def test_antex_one_antenna(sample_atx):
    entries = load_antex(sample_atx)
    assert len(entries) == 1
    e = entries[0]
    assert e["type"] == "TRM41249.00     NONE"
    assert set(e["frequencies"]) == {"G01", "G02"}


def test_antex_offsets(sample_atx):
    entries = load_antex(sample_atx)
    g01 = entries[0]["frequencies"]["G01"]
    assert g01["north"] == 0.5
    assert g01["east"] == 0.5
    assert g01["up"] == 60.0


def test_antex_noazi_length(sample_atx):
    entries = load_antex(sample_atx)
    g01 = entries[0]["frequencies"]["G01"]
    # ZEN1=0, ZEN2=90, DZEN=5 -> 19 values; the parser ignores extras.
    assert g01["noazi"].size == 19
    assert g01["noazi"][0] == 0.0
    # The 19th NOAZI value in the sample is 1.80 (we stop one short of
    # the trailing 2.00 because the formula gives exactly 19 cells).
    assert g01["noazi"][-1] == 1.8
