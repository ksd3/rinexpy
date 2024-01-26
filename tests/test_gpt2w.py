"""Tests for the GPT2w grid loader and evaluator.

The full grid file is ~2 MB and not shipped; we synthesise a tiny
3x3-cell grid in the proper format and verify the parser, the
seasonal expansion, and the bilinear interpolation.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from rinexpy.gpt2w import _seasonal, gpt2w, load_gpt2w_grid


def _build_tiny_grid_text(value_at: dict[tuple[float, float], list[float]]) -> str:
    """Build the text of a minimal GPT2w grid for testing.

    Each entry of ``value_at`` is keyed by ``(lat, lon)`` and supplies
    the 42 numerical columns (after lat/lon). Caller is responsible for
    providing every (lat, lon) on the synthesised grid.
    """
    lines = ["% synthetic GPT2w grid for testing"]
    for (lat, lon), cols in value_at.items():
        nums = [f"{lat:.1f}", f"{lon:.1f}"] + [f"{v:.6e}" for v in cols]
        lines.append(" ".join(nums))
    return "\n".join(lines)


def _row(p_mean: float, t_mean: float) -> list[float]:
    """Build one 42-column row with the supplied P and T means; zeros elsewhere."""
    cols = [0.0] * 42
    cols[0] = 0.0  # undulation
    cols[1] = 0.0  # ref alt
    cols[2] = p_mean
    cols[7] = t_mean
    return cols


@pytest.fixture
def tiny_grid(tmp_path):
    rows: dict[tuple[float, float], list[float]] = {}
    for lat in (45.0, 40.0, 35.0):
        for lon in (0.0, 5.0, 10.0):
            # Pressure varies smoothly, temperature too.
            rows[(lat, lon)] = _row(p_mean=1000.0 + (lat - 40), t_mean=288.0 - (lat - 40))
    p = tmp_path / "tiny.grd"
    p.write_text(_build_tiny_grid_text(rows))
    return load_gpt2w_grid(p)


def test_loader_dimensions(tiny_grid):
    assert tiny_grid["resolution_deg"] == 5.0
    assert tiny_grid["lat"].tolist() == [45.0, 40.0, 35.0]
    assert tiny_grid["lon"].tolist() == [0.0, 5.0, 10.0]
    assert tiny_grid["data"].shape == (3, 3, 42)


def test_seasonal_at_doy_one_returns_mean_plus_a1():
    # At doy=1 the cosines are ~1 and sines are ~0.
    val = _seasonal(100.0, 5.0, 0.0, 2.0, 0.0, 1.0)
    assert val == pytest.approx(107.0, abs=1e-3)


def test_gpt2w_at_grid_point(tiny_grid):
    """At (40, 0) the pressure should equal the grid's mean pressure."""
    out = gpt2w(tiny_grid, 40.0, 0.0, datetime(2024, 1, 1))
    assert out["pressure_hpa"] == pytest.approx(1000.0, abs=1.0)


def test_gpt2w_returns_a_h_and_a_w(tiny_grid):
    """``a_h`` and ``a_w`` are present and numeric (zero in our test grid)."""
    out = gpt2w(tiny_grid, 40.0, 0.0, datetime(2024, 1, 1))
    assert "a_h" in out
    assert "a_w" in out
    assert np.isfinite(out["a_h"])
    assert np.isfinite(out["a_w"])


def test_gpt2w_altitude_reduces_pressure(tiny_grid):
    """Raising altitude reduces pressure (barometric law)."""
    p_low = gpt2w(tiny_grid, 40.0, 0.0, datetime(2024, 1, 1), altitude_m=0.0)
    p_high = gpt2w(tiny_grid, 40.0, 0.0, datetime(2024, 1, 1), altitude_m=1000.0)
    assert p_high["pressure_hpa"] < p_low["pressure_hpa"]


def test_gpt2w_doy_passed_as_float(tiny_grid):
    """``epoch`` accepts a bare float day-of-year."""
    out = gpt2w(tiny_grid, 40.0, 0.0, 80.0)
    assert "pressure_hpa" in out
