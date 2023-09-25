"""Tests for the RINEX clock (.clk) reader."""

from __future__ import annotations

import pytest
from pytest import approx

from rinexpy.clk import load_clk

# Minimal valid .clk file: header + one AR + two AS records for two SVs.
_SAMPLE = """\
     3.00           CLOCK              GPS                 RINEX VERSION / TYPE
COD                                                         ANALYSIS CENTER
                                                            END OF HEADER
AR ALGO        2024 03 14 00 00  0.000000  2    1.234D-09  3.0D-12
AS G01         2024 03 14 00 00  0.000000  2   -1.000D-04  1.0D-12
AS G02         2024 03 14 00 00  0.000000  2   -2.500D-04  2.0D-12
AS G01         2024 03 14 00 05  0.000000  2   -1.001D-04  1.0D-12
AS G02         2024 03 14 00 05  0.000000  2   -2.501D-04  2.0D-12
"""


@pytest.fixture
def sample_clk(tmp_path):
    p = tmp_path / "sample.clk"
    p.write_text(_SAMPLE)
    return p


def test_clk_basic(sample_clk):
    ds = load_clk(sample_clk)
    assert sorted(ds.sv.values) == ["G01", "G02"]
    assert ds.time.size == 2


def test_clk_values(sample_clk):
    ds = load_clk(sample_clk)
    assert ds.bias.sel(sv="G01").values == approx([-1e-4, -1.001e-4])
    assert ds.bias.sel(sv="G02").values == approx([-2.5e-4, -2.501e-4])


def test_clk_times(sample_clk):
    import numpy as np

    ds = load_clk(sample_clk)
    assert ds.time.values[0] == np.datetime64("2024-03-14T00:00:00")
    assert ds.time.values[1] == np.datetime64("2024-03-14T00:05:00")


def test_clk_stations_attr(sample_clk):
    ds = load_clk(sample_clk)
    assert ds.attrs["stations"] == ["ALGO"]
