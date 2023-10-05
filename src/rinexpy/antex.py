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
        # Skip header up to END OF HEADER.
        for line in f:
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
                if "noazi" in f_entry and f_entry["pcv_rows"]:
                    f_entry["pcv"] = np.array(f_entry["pcv_rows"])
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


__all__ = ["load_antex"]
