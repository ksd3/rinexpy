"""Differential code bias (DCB) reader and application.

Code observations from different signals see slightly different hardware
delays inside the satellite and the receiver. The ionosphere-free
combination cancels first-order ionospheric delay only if the two
input pseudoranges are bias-corrected; absolute TEC mapping likewise
requires both satellite and receiver biases removed.

Operational products are published as SINEX-BIAS (``.BSX``) files by
CODE, CAS, DLR and IGS. Each ``BIAS_SOLUTION`` line carries one bias
value (in nanoseconds) for a (satellite or receiver) site, a pair of
observable codes, an estimation epoch range, and an uncertainty.

This module ships a tolerant SINEX-BIAS reader and a small helper to
look up biases by (PRN, obs_code, obs_code) at an epoch, plus a
``correct_pseudorange`` convenience that applies the bias as a range
correction (bias_ns * c / 1e9 meters).

The reader doesn't validate the SINEX-BIAS header end-of-block markers
strictly -- it just iterates ``BIAS_SOLUTION`` blocks and parses whatever
lines start with a ``+`` / ``OSB`` / ``DSB`` keyword. That matches the
real-world product variability across agencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

C_M_PER_S = 299_792_458.0


def _parse_sinex_date(token: str) -> datetime:
    """Parse a SINEX YY:DOY:SOD timestamp into a UTC datetime.

    SINEX-BIAS dates look like ``2024:047:00000`` (year, day-of-year,
    seconds-of-day). The year is 4-digit in modern files; older ones use
    2-digit (a leading 0 is added; 0-49 -> 2000+, 50-99 -> 1900+).
    """
    parts = token.split(":")
    if len(parts) != 3:
        raise ValueError(f"bad SINEX date token: {token!r}")
    yr = int(parts[0])
    doy = int(parts[1])
    sod = int(parts[2])
    if yr < 100:
        yr = 2000 + yr if yr < 50 else 1900 + yr
    return datetime(yr, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, seconds=sod
    )


def read_bsx(path) -> list[dict[str, Any]]:
    """Parse a SINEX-BIAS (.BSX) file into a flat list of bias records.

    Each returned record has:

    - ``bias_type``  -- ``"OSB"`` (observable-specific bias) or
      ``"DSB"`` (differential signal bias).
    - ``svn``  / ``prn`` -- satellite vehicle number / PRN (one or both
      may be empty depending on whether this is a satellite or receiver
      bias).
    - ``station`` -- 4-character station code, or empty for satellite
      biases.
    - ``obs1``, ``obs2`` -- RINEX-3 observation codes (``obs2`` empty for
      OSB, populated for DSB).
    - ``start``, ``end`` -- estimation interval as UTC datetimes.
    - ``unit`` -- usually ``"ns"``; we don't convert.
    - ``value`` -- bias value in the given unit.
    - ``stddev`` -- formal uncertainty in the given unit (NaN if absent).

    Parameters
    ----------
    path:
        Path to a SINEX-BIAS file.

    Returns
    -------
    list of dict
        Bias records, in file order. Empty list if the file has no
        ``BIAS_SOLUTION`` block.

    Raises
    ------
    ValueError
        If the file cannot be opened or a header line is malformed in a
        way that prevents parsing the block boundaries.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    in_solution = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("+BIAS/SOLUTION"):
                in_solution = True
                continue
            if line.startswith("-BIAS/SOLUTION"):
                in_solution = False
                continue
            if not in_solution:
                continue
            # Skip column-header / comment lines starting with ``*``.
            if line.startswith("*") or not line.strip():
                continue
            tok = line.split()
            if len(tok) < 9:
                continue
            bias_type = tok[0]
            if bias_type not in ("OSB", "DSB", "ISB"):
                continue
            # Column layout per the IGS SINEX-BIAS 1.00 spec:
            #   BIAS_TYPE SVN PRN STATION OBS1 OBS2 START END UNIT VALUE
            #   STDDEV [SLOPE SLOPE_STDDEV]
            svn = tok[1]
            prn = tok[2]
            station = tok[3] if tok[3] != "----" else ""
            obs1 = tok[4]
            # Some records (typically OSB) omit obs2; tok[5] is then the
            # start-date token. Dates contain ":" so they distinguish
            # cleanly from observation codes like "C2W".
            if ":" in tok[5]:
                obs2 = ""
                idx = 5
            else:
                obs2 = tok[5] if tok[5] != "----" else ""
                idx = 6
            try:
                start = _parse_sinex_date(tok[idx])
                end = _parse_sinex_date(tok[idx + 1])
            except ValueError:
                continue
            unit = tok[idx + 2]
            try:
                value = float(tok[idx + 3])
            except (IndexError, ValueError):
                continue
            try:
                stddev = float(tok[idx + 4])
            except (IndexError, ValueError):
                stddev = float("nan")
            records.append({
                "bias_type": bias_type,
                "svn": svn,
                "prn": prn,
                "station": station,
                "obs1": obs1,
                "obs2": obs2,
                "start": start,
                "end": end,
                "unit": unit,
                "value": value,
                "stddev": stddev,
            })
    return records


def get_bias(
    records: list[dict[str, Any]],
    *,
    prn: str = "",
    station: str = "",
    obs1: str,
    obs2: str = "",
    epoch: datetime | None = None,
) -> float | None:
    """Look up a bias value for one (PRN/station, obs1, obs2, epoch).

    Returns the bias in **meters** (converted from nanoseconds via the
    speed of light). For OSB records ``obs2`` is left empty; for DSB
    records both are needed. If no matching record covers the epoch,
    returns ``None``.

    Parameters
    ----------
    records:
        Output of :func:`read_bsx`.
    prn:
        Satellite PRN, e.g. ``"G05"``. Leave empty to match receiver
        records only.
    station:
        4-character station code. Leave empty to match satellite
        records only.
    obs1, obs2:
        RINEX-3 observation codes (e.g. ``"C1W"``, ``"C2W"``).
    epoch:
        Observation epoch. If ``None``, the first record matching the
        identifiers is returned regardless of time.

    Returns
    -------
    float or None
        Bias as a range correction in meters, or ``None`` if no record
        matches.
    """
    if epoch is not None and epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    for r in records:
        if prn and r["prn"] != prn:
            continue
        if station and r["station"] != station:
            continue
        if r["obs1"] != obs1 or r["obs2"] != obs2:
            continue
        if epoch is not None and not (r["start"] <= epoch <= r["end"]):
            continue
        val_ns = r["value"]
        if r["unit"] == "ns":
            return val_ns * C_M_PER_S * 1e-9
        elif r["unit"] in ("m", "metres", "meters"):
            return val_ns
        else:
            # Unknown unit: best-effort treat as ns.
            return val_ns * C_M_PER_S * 1e-9
    return None


def correct_pseudorange(
    pseudorange_m: float,
    *,
    prn: str,
    obs_code: str,
    records: list[dict[str, Any]],
    epoch: datetime | None = None,
    station: str = "",
) -> float:
    """Subtract satellite (and optional receiver) OSB bias from a pseudorange.

    OSB sign convention (IGS): the published bias value is **added** to
    the raw observation to obtain the bias-corrected observation. This
    function therefore returns ``pseudorange_m + b_sv + b_rx`` where
    ``b_sv`` is the satellite OSB and ``b_rx`` is the receiver OSB
    (the latter only included when ``station`` is given and a matching
    record exists).
    """
    b_sv = get_bias(records, prn=prn, obs1=obs_code, epoch=epoch) or 0.0
    b_rx = 0.0
    if station:
        b_rx = get_bias(records, station=station, obs1=obs_code, epoch=epoch) or 0.0
    return pseudorange_m + b_sv + b_rx


# ---------------------------------------------------------------------------
# Legacy CODE DCB (AIUB) reader
# ---------------------------------------------------------------------------
#
# Before the SINEX-BIAS format (~2017), AIUB / CODE published *monthly*
# DCB files in a fixed-column ASCII format:
#
#     P1P2YYMM.DCB  -- P1 - P2 differential code biases (the main one)
#     P1C1YYMM.DCB  -- P1 - C1
#     P2C2YYMM.DCB  -- P2 - C2
#
# A single file usually carries multiple sections (satellite + station
# blocks per bias type). Each data line is:
#
#     PRN_or_STATION(cols 1-20)  VALUE_NS(cols 21-35)  RMS_NS(cols 36-50)
#
# The header above the data identifies which bias-pair the section is
# (e.g. ``P1-P2 DIFFERENTIAL CODE BIASES``). This reader returns records
# in the same shape as :func:`read_bsx`, so callers can pass them
# straight to :func:`get_bias` / :func:`correct_pseudorange` and to
# ``ppp_solve(dcb_records=...)`` / ``spp_solve(dcb_records=...)``.

#: Translation table: CODE's "P1/P2/C1/C2" -> RINEX-3 observation codes.
_CODE_OBS_TO_RNX3: dict[str, str] = {
    "P1": "C1W",
    "P2": "C2W",
    "C1": "C1C",
    "C2": "C2C",
}


def _detect_dcb_pair(line: str) -> tuple[str, str] | None:
    """Return the (obs1, obs2) RINEX-3 codes implied by a section header,
    or None if the line is not a recognised header."""
    upper = line.upper()
    for tag in ("P1-P2", "P1-C1", "P2-C2", "P1P2", "P1C1", "P2C2"):
        if tag in upper:
            tag = tag.replace("-", "")
            a, b = tag[:2], tag[2:]
            return _CODE_OBS_TO_RNX3.get(a, ""), _CODE_OBS_TO_RNX3.get(b, "")
    return None


def _month_window(year: int, month: int) -> tuple[datetime, datetime]:
    """Return the (start, end) validity window for a monthly DCB file
    as UTC datetimes covering the whole calendar month."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    return start, end


def read_code_dcb(
    path,
    *,
    year: int | None = None,
    month: int | None = None,
) -> list[dict[str, Any]]:
    """Parse an AIUB CODE monthly DCB file into bias records.

    Each record matches the shape returned by :func:`read_bsx`, with
    ``bias_type="DSB"`` (every CODE DCB is differential), and the
    ``obs1`` / ``obs2`` translated to RINEX-3 codes via
    :data:`_CODE_OBS_TO_RNX3`.

    Parameters
    ----------
    path:
        Path to the (already-decompressed) ``.DCB`` text file.
    year, month:
        Optional year + month for the record's validity window. If
        omitted, an attempt is made to infer them from the filename
        suffix (``P1P2YYMM.DCB``); failing that the window is left at
        an open ``(1900, 2099)`` range.

    Returns
    -------
    list[dict]
        Bias records compatible with :func:`get_bias` /
        :func:`correct_pseudorange`.
    """
    path = Path(path)
    text = path.read_text(errors="ignore")

    # Validity window: explicit args > filename suffix > open range.
    if year is None or month is None:
        stem = path.stem.upper()
        if len(stem) >= 8 and stem[:4] in ("P1P2", "P1C1", "P2C2"):
            try:
                yy = int(stem[4:6])
                mm = int(stem[6:8])
                yr = 2000 + yy if yy < 90 else 1900 + yy
                if 1 <= mm <= 12:
                    year, month = yr, mm
            except ValueError:
                pass
    if year is not None and month is not None:
        start, end = _month_window(int(year), int(month))
    else:
        start = datetime(1900, 1, 1, tzinfo=timezone.utc)
        end = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    out: list[dict[str, Any]] = []
    current_pair: tuple[str, str] | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.startswith("*"):
            continue
        pair = _detect_dcb_pair(raw)
        if pair is not None:
            current_pair = pair
            continue
        if current_pair is None:
            continue
        # Data line: first 4 chars hold a satellite PRN like "G01"
        # (possibly followed by 16 chars of station/site description),
        # or a 4-char station code followed by a longer name. The
        # value (ns) and RMS (ns) follow after a wide space block.
        tokens = raw.split()
        if len(tokens) < 2:
            continue
        ident = tokens[0]
        try:
            value_ns = float(tokens[-2])
        except (ValueError, IndexError):
            try:
                value_ns = float(tokens[-1])
            except ValueError:
                continue
        is_prn = len(ident) == 3 and ident[0] in "GREJCIS" and ident[1:].isdigit()
        record = {
            "bias_type": "DSB",
            "prn": ident if is_prn else "",
            "station": "" if is_prn else ident,
            "obs1": current_pair[0],
            "obs2": current_pair[1],
            "start": start,
            "end": end,
            "unit": "ns",
            "value": value_ns,
        }
        out.append(record)
    return out


__all__ = ["correct_pseudorange", "get_bias", "read_bsx", "read_code_dcb"]
