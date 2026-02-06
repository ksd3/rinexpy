"""RINEX 4 NAV reader for STO / EOP / ION records.

RINEX 4 introduces a structured per-record header so a single NAV file
can carry several message classes alongside the broadcast ephemeris:

    > EPH G LNAV
    G01 2022 06 25 00 00 00 ...  (7-line ephemeris record)
    > STO G UTC(USNO)
    G01 2022 06 25 00 00 00 GPUT
        A0      A1      T_t     W_t
    > EOP G LNAV
    G01 2022 06 25 00 00 00
        PM_X    PM_X_dot   PM_X_acc   <reserved>
        PM_Y    PM_Y_dot   PM_Y_acc   <reserved>
        T_EOP   dUT1       dUT1_dot   dUT1_acc
    > ION G LNAV
    G01 2022 06 25 00 00 00 KLOB
        alpha0  alpha1  alpha2  alpha3
        beta0   beta1   beta2   beta3
        region

This module reads STO, EOP, and ION records into typed Python dicts.
EPH records can already be consumed by :func:`rinexpy.nav3.rinexnav3`
after stripping the ``>`` header line, so they are returned as the raw
continuation block here (callers who want a typed ephemeris view should
use :func:`rinexnav4` which separates the EPH stream out for nav3).

Reference: RINEX 4.01 §5 + Annex A.2.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from ._io import opener
from ._time import parse_nav3_epoch
from ._types import FileLike

_FIELD_WIDTH = 19


def _fortran_float(s: str) -> float:
    """Convert a Fortran ``D``/``E`` numeric field to float; empty -> 0."""
    s = s.strip().replace("D", "E").replace("d", "E")
    return float(s) if s else 0.0


def _split_fields(line: str, n: int, *, start: int = 4) -> list[float]:
    """Split a RINEX numeric continuation line into ``n`` 19-char fields."""
    out: list[float] = []
    line = line.rstrip("\n")
    for i in range(n):
        a = start + i * _FIELD_WIDTH
        b = a + _FIELD_WIDTH
        out.append(_fortran_float(line[a:b]))
    return out


def _parse_header_line(line: str) -> tuple[str, str, str]:
    """Parse a ``> RECORD_TYPE SYS MSG_TYPE`` header line.

    Returns ``(record_type, system_letter, message_type)``.
    """
    parts = line[1:].split()
    if len(parts) < 3:
        raise ValueError(f"malformed RINEX 4 record header: {line!r}")
    return parts[0], parts[1], " ".join(parts[2:])


def _parse_epoch_prefix(line: str) -> tuple[str, datetime, str]:
    """Read the standard 'PRN YYYY MM DD HH MM SS  <suffix>' line and
    return ``(sv, epoch, suffix)``.

    ``suffix`` is whatever follows the time stamp (e.g. ``GPUT`` for
    STO, ``KLOB`` for ION). Empty if absent.
    """
    sv = line[:3].replace(" ", "0")
    epoch = parse_nav3_epoch(line)
    # Epoch occupies columns 4..23 (3 + 1 + 4 + 5*3 = ...); the suffix
    # starts after that, often with leading spaces.
    suffix = line[23:].strip() if len(line) > 23 else ""
    # Strip any numeric tail (some readers include leading values on the
    # same line - keep only alphanumeric+parens leading token).
    if suffix:
        suffix = suffix.split()[0]
    return sv, epoch, suffix


def _parse_sto_record(stream, header: tuple[str, str, str]) -> dict[str, Any]:
    """Read one STO record (header + epoch line + one 4-field data line)."""
    record_type, sys_letter, msg_type = header
    epoch_line = stream.readline()
    sv, epoch, ut_id = _parse_epoch_prefix(epoch_line)
    data = _split_fields(stream.readline(), 4)
    return {
        "type": "STO",
        "sv": sv,
        "system": sys_letter,
        "message_type": msg_type,
        "epoch": epoch,
        "utc_id": ut_id,
        "A0_s": data[0],
        "A1_s_per_s": data[1],
        "T_t_s": data[2],
        "W_t_weeks": int(data[3]),
    }


def _parse_eop_record(stream, header: tuple[str, str, str]) -> dict[str, Any]:
    """Read one EOP record (header + epoch line + three 4-field data lines)."""
    record_type, sys_letter, msg_type = header
    sv, epoch, _ = _parse_epoch_prefix(stream.readline())
    row_x = _split_fields(stream.readline(), 4)
    row_y = _split_fields(stream.readline(), 4)
    row_ut1 = _split_fields(stream.readline(), 4)
    return {
        "type": "EOP",
        "sv": sv,
        "system": sys_letter,
        "message_type": msg_type,
        "epoch": epoch,
        "PM_X_arcsec": row_x[0],
        "PM_X_dot_arcsec_per_day": row_x[1],
        "PM_X_acc_arcsec_per_day2": row_x[2],
        "PM_Y_arcsec": row_y[0],
        "PM_Y_dot_arcsec_per_day": row_y[1],
        "PM_Y_acc_arcsec_per_day2": row_y[2],
        "T_EOP_s": row_ut1[0],
        "dUT1_s": row_ut1[1],
        "dUT1_dot_s_per_day": row_ut1[2],
        "dUT1_acc_s_per_day2": row_ut1[3],
    }


def _parse_ion_record(stream, header: tuple[str, str, str]) -> dict[str, Any]:
    """Read one ION record. Supports Klobuchar (8 + 1 fields), NeQuick-G
    (3 + region, single data line), and BDGIM (9 alpha-i, two data lines).
    The model type comes from either the message-type or the epoch-line
    suffix."""
    record_type, sys_letter, msg_type = header
    sv, epoch, suffix = _parse_epoch_prefix(stream.readline())
    model = (suffix or msg_type).upper()

    common = {
        "type": "ION",
        "sv": sv,
        "system": sys_letter,
        "message_type": msg_type,
        "epoch": epoch,
        "model": model,
    }

    if model.startswith("KLOB"):
        alpha = _split_fields(stream.readline(), 4)
        beta = _split_fields(stream.readline(), 4)
        region_line = stream.readline()
        region = _split_fields(region_line, 4)[0] if region_line.strip() else 0.0
        return {**common, "alpha": alpha, "beta": beta, "region_code": int(region)}

    if model.startswith("NEQG") or model.startswith("NEQ"):
        row = _split_fields(stream.readline(), 4)
        return {
            **common,
            "a_i0": row[0],
            "a_i1": row[1],
            "a_i2": row[2],
            "region_code": int(row[3]),
        }

    if model.startswith("BDGIM"):
        a1 = _split_fields(stream.readline(), 4)
        a2 = _split_fields(stream.readline(), 4)
        a3 = _split_fields(stream.readline(), 4)
        return {**common, "alpha": a1 + a2 + a3[:1]}

    # Unknown model — return raw 4-field rows for the caller to interpret.
    rows = [_split_fields(stream.readline(), 4) for _ in range(2)]
    return {**common, "raw_rows": rows}


def _parse_eph_record(
    stream, header: tuple[str, str, str], n_data_lines: int
) -> dict[str, Any]:
    """Read one EPH record. Returns the message type plus the raw
    continuation text so it can be fed through :mod:`rinexpy.nav3`'s
    ephemeris parser unchanged."""
    record_type, sys_letter, msg_type = header
    epoch_line = stream.readline()
    sv, epoch, _ = _parse_epoch_prefix(epoch_line)
    raw = epoch_line[23:].rstrip("\n")
    for _ in range(n_data_lines):
        raw += stream.readline()[4:].rstrip("\n")
    return {
        "type": "EPH",
        "sv": sv,
        "system": sys_letter,
        "message_type": msg_type,
        "epoch": epoch,
        "raw_data": raw.replace("D", "E"),
    }


_EPH_DATA_LINES_BY_SYS = {"G": 7, "E": 7, "C": 7, "J": 7, "I": 7, "R": 3, "S": 3}


def load_nav4(fn: FileLike) -> dict[str, list[dict[str, Any]]]:
    """Read a RINEX 4 NAV file and return its records grouped by class.

    Returns a dict with keys ``"EPH"``, ``"STO"``, ``"EOP"``, ``"ION"``
    (any of which may be missing if the file has no such records). Each
    value is a list of typed Python dicts; see the per-record helpers
    in this module for the field set.

    The file header is skipped silently. Unknown record types or
    malformed records are skipped rather than raising.
    """
    if isinstance(fn, (str, Path)):
        fn = Path(fn).expanduser()
    out: dict[str, list[dict[str, Any]]] = {
        "EPH": [], "STO": [], "EOP": [], "ION": [],
    }
    with opener(fn) as f:
        # Skip header to END OF HEADER using readline() throughout, to
        # avoid mixing iterator readahead with the per-record readline()
        # calls in the helpers below.
        while True:
            line = f.readline()
            if not line or "END OF HEADER" in line:
                break
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(">"):
                continue
            try:
                header = _parse_header_line(line)
            except ValueError:
                continue
            rt, sys_letter, _ = header
            try:
                if rt == "STO":
                    out["STO"].append(_parse_sto_record(f, header))
                elif rt == "EOP":
                    out["EOP"].append(_parse_eop_record(f, header))
                elif rt == "ION":
                    out["ION"].append(_parse_ion_record(f, header))
                elif rt == "EPH":
                    n = _EPH_DATA_LINES_BY_SYS.get(sys_letter, 7)
                    out["EPH"].append(_parse_eph_record(f, header, n))
                else:
                    continue
            except (StopIteration, ValueError):
                continue
    return out


__all__ = ["load_nav4"]
