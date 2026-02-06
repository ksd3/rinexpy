"""Tests for the RINEX 4 NAV STO / EOP / ION record reader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from rinexpy.nav4 import load_nav4


def _fmt(values: list[float]) -> str:
    """Format a row of 4 values as the RINEX 4 continuation 19-char/field
    layout: 4 leading spaces + 4 fields of 19 chars each (positive
    numbers carry a leading space; negative numbers fill it with '-')."""
    body = ""
    for v in values:
        s = f"{v: .12E}".replace("E", "D")
        # Pad to exactly 19 chars (sign + 1 digit + . + 12 digits + Dee = 19).
        if not s.startswith(("-", " ")):
            s = " " + s
        body += s
    return "    " + body + "\n"


_STO_RECORD = (
    "> STO G UTC(USNO)\n"
    "G01 2022 06 25 00 00 00 GPUT                                          \n"
    + _fmt([1.862645149230e-09, 4.440892098501e-15, 529200.0, 2215.0])
)

_EOP_RECORD = (
    "> EOP G LNAV\n"
    "G01 2022 06 25 00 00 00                                                                \n"
    + _fmt([0.1234, 1.234e-4, 0.0, 0.0])
    + _fmt([-0.1234, 1.234e-4, 0.0, 0.0])
    + _fmt([518400.0, 1.234e-4, 1.234e-8, 0.0])
)

_ION_RECORD = (
    "> ION G LNAV\n"
    "G01 2022 06 25 00 00 00 KLOB                                          \n"
    + _fmt([1e-8, 2e-8, 3e-8, 4e-8])
    + _fmt([1e5, 2e5, 3e5, 4e5])
    + _fmt([1.0, 0.0, 0.0, 0.0])
)

_HEADER = (
    "     4.00           N: GNSS NAV DATA    M (MIXED)            RINEX VERSION / TYPE\n"
    "                                                            END OF HEADER\n"
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "synthetic.rnx"
    p.write_text(_HEADER + body)
    return p


def test_load_nav4_parses_sto(tmp_path):
    out = load_nav4(_write(tmp_path, _STO_RECORD))
    assert len(out["STO"]) == 1
    rec = out["STO"][0]
    assert rec["sv"] == "G01"
    assert rec["system"] == "G"
    assert rec["message_type"] == "UTC(USNO)"
    assert rec["epoch"] == datetime(2022, 6, 25)
    assert rec["utc_id"] == "GPUT"
    assert rec["A0_s"] == pytest.approx(1.862645149230e-09)
    assert rec["A1_s_per_s"] == pytest.approx(4.440892098501e-15)
    assert rec["T_t_s"] == pytest.approx(529200.0)
    assert rec["W_t_weeks"] == 2215


def test_load_nav4_parses_eop(tmp_path):
    out = load_nav4(_write(tmp_path, _EOP_RECORD))
    assert len(out["EOP"]) == 1
    rec = out["EOP"][0]
    assert rec["sv"] == "G01"
    assert rec["PM_X_arcsec"] == pytest.approx(0.1234)
    assert rec["PM_Y_arcsec"] == pytest.approx(-0.1234)
    assert rec["T_EOP_s"] == pytest.approx(518400.0)
    assert rec["dUT1_s"] == pytest.approx(1.234e-4)
    assert rec["dUT1_dot_s_per_day"] == pytest.approx(1.234e-8)


def test_load_nav4_parses_ion_klobuchar(tmp_path):
    out = load_nav4(_write(tmp_path, _ION_RECORD))
    assert len(out["ION"]) == 1
    rec = out["ION"][0]
    assert rec["model"] == "KLOB"
    assert rec["alpha"] == [
        pytest.approx(1e-8), pytest.approx(2e-8),
        pytest.approx(3e-8), pytest.approx(4e-8),
    ]
    assert rec["beta"] == [
        pytest.approx(1e5), pytest.approx(2e5),
        pytest.approx(3e5), pytest.approx(4e5),
    ]
    assert rec["region_code"] == 1


def test_load_nav4_handles_mixed_records(tmp_path):
    body = _STO_RECORD + _EOP_RECORD + _ION_RECORD
    out = load_nav4(_write(tmp_path, body))
    assert len(out["STO"]) == 1
    assert len(out["EOP"]) == 1
    assert len(out["ION"]) == 1
    assert len(out["EPH"]) == 0
