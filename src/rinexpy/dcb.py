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


__all__ = ["correct_pseudorange", "get_bias", "read_bsx"]
