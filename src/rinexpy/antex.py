"""ANTEX (.atx) antenna phase center variation reader.

Reference: https://files.igs.org/pub/data/format/antex14.txt

Each ANTEX entry is bracketed by ``START OF ANTENNA`` / ``END OF ANTENNA``.
Within an antenna entry, one or more frequencies (``START OF FREQUENCY`` /
``END OF FREQUENCY``) carry phase-center offsets and a NOAZI / azimuth-
dependent PCV grid.

The output is a list of dicts (one per antenna) — ANTEX is too irregular
to make a single ``xarray.Dataset`` worthwhile.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ._common import fortran_float
from ._io import opener
from ._types import FileLike

log = logging.getLogger(__name__)


def load_antex(fn: FileLike) -> list[dict[str, Any]]:
    """Read an ANTEX file into a list of antenna entries.

    Parameters
    ----------
    fn:
        Path or open text stream of an ``.atx`` file.

    Returns
    -------
    list[dict]
        One entry per antenna. Each dict has keys: ``type`` (model name),
        ``serial``, ``valid_from``, ``valid_until`` (or None), and
        ``frequencies``: a dict ``{freq_id: {north, east, up, noazi, pcv}}``
        where ``noazi`` is a 1-D ndarray of zenith-angle PCV values and
        ``pcv`` (when present) is a 2-D ``(azi, zen)`` ndarray.
    """
    import numpy as np

    entries: list[dict[str, Any]] = []
    with opener(fn) as f:
        # Read header for DAZI (azimuth step) and skip to END OF HEADER.
        dazi = 0.0
        for line in f:
            label = line[60:].strip()
            if label == "DAZI":
                try:
                    dazi = float(line[0:8])
                except ValueError:
                    dazi = 0.0
            if "END OF HEADER" in line:
                break

        cur: dict[str, Any] | None = None
        cur_freq: str | None = None
        zen1 = zen2 = dzen = None

        for line in f:
            label = line[60:].strip()
            if label == "START OF ANTENNA":
                cur = {"frequencies": {}}
            elif label == "END OF ANTENNA":
                if cur is not None:
                    entries.append(cur)
                cur = None
            elif cur is None:
                continue
            elif label == "TYPE / SERIAL NO":
                cur["type"] = line[:20].strip()
                cur["serial"] = line[20:40].strip()
            elif label == "VALID FROM":
                cur["valid_from"] = _parse_atx_epoch(line)
            elif label == "VALID UNTIL":
                cur["valid_until"] = _parse_atx_epoch(line)
            elif label == "ZEN1 / ZEN2 / DZEN":
                zen1 = float(line[2:8])
                zen2 = float(line[8:14])
                dzen = float(line[14:20])
            elif label == "# OF FREQUENCIES":
                pass  # ignored; we count by walking blocks
            elif label == "START OF FREQUENCY":
                cur_freq = line[3:6].strip()
                cur["frequencies"][cur_freq] = {"pcv_rows": []}
            elif label == "END OF FREQUENCY":
                f_entry = cur["frequencies"][cur_freq]
                if f_entry["pcv_rows"]:
                    f_entry["pcv"] = np.array(f_entry["pcv_rows"])
                    # Build the azimuth axis from DAZI: rows are at
                    # 0, DAZI, 2*DAZI, ... 360 (the spec wraps; we keep
                    # both endpoints so callers can interp without modulo).
                    if dazi > 0:
                        n_az = int(round(360.0 / dazi)) + 1
                        f_entry["azimuth_deg"] = np.linspace(0.0, 360.0, n_az)
                    if zen1 is not None and zen2 is not None and dzen is not None:
                        n_zen = int(round((zen2 - zen1) / dzen)) + 1
                        f_entry["zenith_deg"] = np.linspace(zen1, zen2, n_zen)
                f_entry.pop("pcv_rows", None)
                cur_freq = None
            elif cur_freq is not None and label == "NORTH / EAST / UP":
                f_entry = cur["frequencies"][cur_freq]
                f_entry["north"] = fortran_float(line[0:10])
                f_entry["east"] = fortran_float(line[10:20])
                f_entry["up"] = fortran_float(line[20:30])
            elif cur_freq is not None:
                # Data line: detect NOAZI or numeric azimuth in cols 0-8.
                # ANTEX value rows can extend past col 60 (which holds
                # data, not a label), so we must NOT skip on label != ''.
                head = line[:8]
                if head.strip() == "NOAZI":
                    if zen1 is None or zen2 is None or dzen is None:
                        continue
                    n = int((zen2 - zen1) / dzen) + 1
                    vals = [fortran_float(line[8 + i * 8 : 16 + i * 8]) for i in range(n)]
                    cur["frequencies"][cur_freq]["noazi"] = np.array(vals)
                else:
                    try:
                        float(head)
                    except ValueError:
                        continue
                    if zen1 is None or zen2 is None or dzen is None:
                        continue
                    n = int((zen2 - zen1) / dzen) + 1
                    vals = [fortran_float(line[8 + i * 8 : 16 + i * 8]) for i in range(n)]
                    cur["frequencies"][cur_freq]["pcv_rows"].append(vals)

    return entries


def _parse_atx_epoch(line: str) -> datetime | None:
    """Parse an ANTEX VALID FROM/UNTIL date line."""
    try:
        return datetime(
            int(line[0:6]),
            int(line[6:12]),
            int(line[12:18]),
            int(line[18:24]) if line[18:24].strip() else 0,
            int(line[24:30]) if line[24:30].strip() else 0,
            int(float(line[30:43])) if line[30:43].strip() else 0,
        )
    except ValueError:
        return None


def find_antenna(
    entries: list[dict[str, Any]],
    type_code: str,
    *,
    serial: str | None = None,
    epoch: datetime | None = None,
) -> dict[str, Any] | None:
    """Find an ANTEX entry by type code (and optionally serial / epoch).

    Parameters
    ----------
    entries:
        List returned by :func:`load_antex`.
    type_code:
        Antenna model name (left-justified to 20 chars in ANTEX).
    serial:
        Optional serial number; ``""`` matches the generic (non-IGS-cal)
        entry. ``None`` (the default) returns the first match by type.
    epoch:
        If given, prefer an entry whose ``valid_from`` <= epoch <=
        ``valid_until``; ignore unbounded entries when a bounded match
        exists.

    Returns
    -------
    dict | None
        The first matching entry, or ``None`` if no match.
    """
    candidates = [e for e in entries if e.get("type", "").rstrip() == type_code.rstrip()]
    if serial is not None:
        candidates = [e for e in candidates if e.get("serial", "").rstrip() == serial.rstrip()]
    if not candidates:
        return None
    if epoch is None:
        return candidates[0]
    bounded = [
        e
        for e in candidates
        if e.get("valid_from") and e["valid_from"] <= epoch
        and (e.get("valid_until") is None or e["valid_until"] >= epoch)
    ]
    return (bounded or candidates)[0]


def apply_antex_pcv(
    entry: dict[str, Any],
    freq_id: str,
    el_deg: float,
    *,
    az_deg: float | None = None,
) -> float:
    """Return the antenna PCV correction (m) for a (frequency, az, el).

    Parameters
    ----------
    entry:
        ANTEX antenna entry from :func:`load_antex` (or :func:`find_antenna`).
    freq_id:
        Frequency label (e.g. ``"G01"``, ``"G02"``).
    el_deg:
        Satellite elevation angle in degrees.
    az_deg:
        Optional satellite azimuth in degrees. If given AND the entry
        has an azimuth-dependent ``pcv`` grid, the 2-D grid is
        bilinearly interpolated. Otherwise (or for entries without a
        ``DAZI`` step) the azimuth-independent ``noazi`` row is used.

    Returns
    -------
    float
        PCV correction in meters. Returns ``0.0`` when the requested
        frequency is absent. Subtract this from the observed pseudorange
        / carrier-phase to remove the antenna effect:

        ::

            corrected = observation - apply_antex_pcv(ant, "G01", el, az_deg=az)
    """
    import numpy as np

    f_entry = entry.get("frequencies", {}).get(freq_id)
    if f_entry is None:
        return 0.0
    zen = 90.0 - el_deg
    if zen < 0 or zen > 90:
        return 0.0

    # Azimuth-dependent path.
    if (
        az_deg is not None
        and "pcv" in f_entry
        and "azimuth_deg" in f_entry
        and "zenith_deg" in f_entry
    ):
        az_axis = f_entry["azimuth_deg"]
        zen_axis = f_entry["zenith_deg"]
        grid = f_entry["pcv"]  # (n_az, n_zen)
        if grid.shape == (az_axis.size, zen_axis.size):
            val_mm = float(_bilinear(az_axis, zen_axis, grid, az_deg % 360.0, zen))
            return val_mm * 1e-3

    # NOAZI fallback.
    if "noazi" not in f_entry:
        return 0.0
    noazi = f_entry["noazi"]
    grid = (
        f_entry["zenith_deg"]
        if "zenith_deg" in f_entry
        else np.linspace(0.0, 90.0, noazi.size)
    )
    val_mm = float(np.interp(zen, grid, noazi))
    return val_mm * 1e-3


def _bilinear(x_axis, y_axis, grid, x: float, y: float) -> float:
    """Bilinear interpolation on a regular ``(x_axis, y_axis)`` grid.

    Used by :func:`apply_antex_pcv` for the 2-D PCV table; kept as a
    private helper because the only consumer is right above.
    """
    import numpy as np

    x_idx = int(np.searchsorted(x_axis, x))
    x_idx = max(1, min(x_axis.size - 1, x_idx))
    x0, x1 = x_axis[x_idx - 1], x_axis[x_idx]
    wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)

    y_idx = int(np.searchsorted(y_axis, y))
    y_idx = max(1, min(y_axis.size - 1, y_idx))
    y0, y1 = y_axis[y_idx - 1], y_axis[y_idx]
    wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    v00 = grid[x_idx - 1, y_idx - 1]
    v01 = grid[x_idx - 1, y_idx]
    v10 = grid[x_idx, y_idx - 1]
    v11 = grid[x_idx, y_idx]
    v0 = v00 * (1 - wy) + v01 * wy
    v1 = v10 * (1 - wy) + v11 * wy
    return float(v0 * (1 - wx) + v1 * wx)


__all__ = ["apply_antex_pcv", "find_antenna", "load_antex"]
