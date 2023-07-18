"""RINEX-3 navigation message reader.

The original ``georinex.nav3`` builds an ``xarray.Dataset`` per satellite and
``xarray.merge``s them together — quadratic in the number of unique
satellites. This rewrite instead:

1. Collects all parsed records as a flat list of ``(sv, time, values)`` tuples.
2. Sorts records into per-SV time-major buffers.
3. Builds a single ``xarray.Dataset`` from a 3-D NumPy array.

The duplicate-SV-record handling (when a receiver re-broadcasts the same
SV at the same epoch with different ephemerides) is preserved — duplicates
become ``"E04_1"``, ``"E04_2"`` SV labels, matching georinex's behavior.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import numpy as np
import xarray as xr

from ._io import opener
from ._time import parse_nav3_epoch
from ._types import FileLike
from .headers import navheader3

log = logging.getLogger(__name__)

# Lines per SV record (after the epoch line) per system letter.
_NL_BY_SYS: dict[str, int] = {"C": 7, "E": 7, "G": 7, "J": 7, "R": 3, "S": 3, "I": 7}

# Column where numerical data starts on continuation lines.
_STARTCOL = 4
# Width of one Fortran ``D``/``E`` numeric field.
_FIELD_WIDTH = 19

#: Fields whose units differ between km (RINEX 3 R/S systems) and m. The
#: rinexpy convention follows georinex: emit meters everywhere.
_KM_TO_M_FIELDS: frozenset[str] = frozenset(
    {"X", "dX", "dX2", "Y", "dY", "dY2", "Z", "dZ", "dZ2"}
)

#: Per-system field tables. Lengths matter (31 for full Keplerian-style; 15
#: for GLONASS/SBAS), the last few entries are RINEX 3.04 "spare" slots.
_FIELDS_BY_SYS: dict[str, list[str]] = {
    "G": [
        "SVclockBias", "SVclockDrift", "SVclockDriftRate",
        "IODE", "Crs", "DeltaN", "M0",
        "Cuc", "Eccentricity", "Cus", "sqrtA",
        "Toe", "Cic", "Omega0", "Cis",
        "Io", "Crc", "omega", "OmegaDot",
        "IDOT", "CodesL2", "GPSWeek", "L2Pflag",
        "SVacc", "health", "TGD", "IODC",
        "TransTime", "FitIntvl", "spare0", "spare1",
    ],
    "C": [
        "SVclockBias", "SVclockDrift", "SVclockDriftRate",
        "AODE", "Crs", "DeltaN", "M0",
        "Cuc", "Eccentricity", "Cus", "sqrtA",
        "Toe", "Cic", "Omega0", "Cis",
        "Io", "Crc", "omega", "OmegaDot",
        "IDOT", "spare0", "BDTWeek", "spare1",
        "SVacc", "SatH1", "TGD1", "TGD2",
        "TransTime", "AODC", "spare2", "spare3",
    ],
    "R": [
        "SVclockBias", "SVrelFreqBias", "MessageFrameTime",
        "X", "dX", "dX2", "health",
        "Y", "dY", "dY2", "FreqNum",
        "Z", "dZ", "dZ2", "AgeOpInfo",
    ],
    "S": [
        "SVclockBias", "SVrelFreqBias", "MessageFrameTime",
        "X", "dX", "dX2", "health",
        "Y", "dY", "dY2", "URA",
        "Z", "dZ", "dZ2", "IODN",
    ],
    "J": [
        "SVclockBias", "SVclockDrift", "SVclockDriftRate",
        "IODE", "Crs", "DeltaN", "M0",
        "Cuc", "Eccentricity", "Cus", "sqrtA",
        "Toe", "Cic", "Omega0", "Cis",
        "Io", "Crc", "omega", "OmegaDot",
        "IDOT", "CodesL2", "GPSWeek", "L2Pflag",
        "SVacc", "health", "TGD", "IODC",
        "TransTime", "FitIntvl", "spare0", "spare1",
    ],
    "E": [
        "SVclockBias", "SVclockDrift", "SVclockDriftRate",
        "IODnav", "Crs", "DeltaN", "M0",
        "Cuc", "Eccentricity", "Cus", "sqrtA",
        "Toe", "Cic", "Omega0", "Cis",
        "Io", "Crc", "omega", "OmegaDot",
        "IDOT", "DataSrc", "GALWeek", "spare0",
        "SISA", "health", "BGDe5a", "BGDe5b",
        "TransTime", "spare1", "spare2", "spare3",
    ],
    "I": [
        "SVclockBias", "SVclockDrift", "SVclockDriftRate",
        "IODEC", "Crs", "DeltaN", "M0",
        "Cuc", "Eccentricity", "Cus", "sqrtA",
        "Toe", "Cic", "Omega0", "Cis",
        "Io", "Crc", "omega", "OmegaDot",
        "spare0", "BDTWeek", "spare1", "URA",
        "health", "TGD", "spare2", "TransTime",
        "spare3", "spare4", "spare5",
    ],
}


def _skip(stream: IO[str], n: int) -> None:
    for _ in range(n):
        stream.readline()


def _num_fields(raw: str) -> int:
    """Return the number of 19-char fields present in a raw concatenated record."""
    return math.ceil(len(raw.rstrip()) / _FIELD_WIDTH)


def _select_fields(full: list[str], sv_sys: str, n_present: int) -> list[str]:
    """Map an incomplete RINEX-3 NAV record to the right subset of fields.

    Receivers vary in whether they emit the optional "spare" fields. This
    helper deduces which fields are present from the count.
    """
    cf = full
    if sv_sys == "G":
        match n_present:
            case 30:
                cf = cf[:-1]
            case 29:
                cf = cf[:-2]
            case 28:
                cf = cf[:-3]
    elif sv_sys == "C":
        match n_present:
            case 27:
                cf = cf[:20] + [cf[21]] + cf[23:29]
            case 28:
                cf = cf[:22] + cf[23:29]
            case 29:
                cf = cf[:29]
            case 30:
                cf = cf[:30]
    elif sv_sys == "J":
        match n_present:
            case 28 | 29:
                n_present = 29
                cf = cf[:29]
            case 30:
                cf = cf[:30]
    elif sv_sys == "E":
        match n_present:
            case 29:
                cf = cf[:-2]
            case 28:
                cf = cf[:-3]
            case 27:
                cf = cf[:22] + cf[23:-3]
    elif sv_sys == "I" and n_present == 28:
        cf = cf[:28]

    if n_present != len(cf):
        raise ValueError(
            f"system {sv_sys} NAV record has {n_present} fields, "
            f"no matching field-set with len {len(cf)}"
        )
    return cf


def rinexnav3(
    fn: FileLike,
    *,
    use: set[str] | None = None,
    tlim: tuple[datetime, datetime] | None = None,
) -> xr.Dataset:
    """Read a RINEX-3 navigation file into an ``xarray.Dataset``.

    Parameters
    ----------
    fn:
        Path or open text stream of a RINEX-3 NAV file.
    use:
        Optional set of single-letter system codes to keep. Records for
        other systems are skipped without parsing.
    tlim:
        Optional ``(start, stop)`` datetime bounds.

    Returns
    -------
    xarray.Dataset
        Dataset with ``time`` and ``sv`` coords. Duplicate SV records at the
        same epoch are split into ``Sxx``, ``Sxx_1``, ``Sxx_2`` etc. (matching
        georinex's behavior).

    Notes
    -----
    Missing trailing fields (e.g. a missing GPS ``FitIntvl``) are interpreted
    as zero, per the RINEX 3.04 spec.
    """
    if isinstance(fn, (str, Path)):
        fn = Path(fn).expanduser()

    if isinstance(use, str):
        use = {use}

    raws_per_sv: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    sv_order: list[str] = []
    svtypes: list[str] = []

    with opener(fn) as f:
        header = navheader3(f)

        for line in f:
            if line.startswith("\n") or not line.strip():
                continue
            try:
                t = parse_nav3_epoch(line)
            except ValueError:
                continue

            sys_letter = line[0]
            n_lines = _NL_BY_SYS.get(sys_letter)
            if n_lines is None:
                # Unknown system — skip cautiously (one line; may corrupt
                # subsequent reads, but unknown systems are rare).
                continue

            if use is not None and sys_letter not in use:
                _skip(f, n_lines)
                continue
            if tlim is not None and (t < tlim[0] or t > tlim[1]):
                _skip(f, n_lines)
                continue

            sv = line[:3].replace(" ", "0")
            if sv not in raws_per_sv:
                sv_order.append(sv)
            if not svtypes or svtypes[-1] != sys_letter:
                svtypes.append(sys_letter)

            raw = line[23:80]
            for _ in range(n_lines):
                raw += f.readline()[_STARTCOL:80]
            raws_per_sv[sv].append((t, raw.replace("D", "E").replace("\n", "")))

    return _assemble_nav3(header, raws_per_sv, sv_order, svtypes, fn)


def _assemble_nav3(
    header: dict[Hashable, Any],
    raws_per_sv: dict[str, list[tuple[datetime, str]]],
    sv_order: list[str],
    svtypes: list[str],
    fn: FileLike,
) -> xr.Dataset:
    """Build the final ``xarray.Dataset`` from per-SV record buffers."""
    if not raws_per_sv:
        nav = xr.Dataset(coords={"time": [], "sv": []})
        nav.attrs["version"] = header.get("version", 0)
        nav.attrs["svtype"] = svtypes
        nav.attrs["rinextype"] = "nav"
        if isinstance(fn, Path):
            nav.attrs["filename"] = fn.name
        return nav

    # Step 1: Determine all unique (sv, time) pairs and split duplicates into
    # SV-suffix variants. We also collect each pair's parsed values into a
    # flat dict so the final allocation is dense.
    times_set: set[datetime] = set()
    expanded: dict[str, dict[datetime, np.ndarray]] = {}
    var_fields: dict[str, list[str]] = {}

    for sv in sv_order:
        records = raws_per_sv[sv]
        sys_letter = sv[0]
        full_fields = _FIELDS_BY_SYS[sys_letter]
        # Group by time so we can detect duplicates per epoch.
        by_time: dict[datetime, list[str]] = defaultdict(list)
        for t, raw in records:
            by_time[t].append(raw)

        max_dups = max(len(v) for v in by_time.values())
        # Pre-create SV variants up to the highest duplication count.
        sv_variants: list[str] = []
        for i in range(max_dups):
            label = sv if i == 0 else f"{sv}_{i}"
            sv_variants.append(label)
            expanded[label] = {}
            var_fields[label] = full_fields

        for t, raws_at_t in by_time.items():
            for i, raw in enumerate(raws_at_t):
                label = sv_variants[i]
                n_present = _num_fields(raw)
                try:
                    cf = _select_fields(full_fields, sys_letter, n_present)
                except ValueError as e:
                    log.warning("%s: %s", sv, e)
                    continue
                values = np.full(len(full_fields), np.nan, dtype=float)
                # Map present fields back to their canonical positions.
                positions = [full_fields.index(name) for name in cf]
                for k, pos in enumerate(positions):
                    chunk = raw[k * _FIELD_WIDTH : (k + 1) * _FIELD_WIDTH].strip()
                    # Empty chunks within a present record are spec'd as 0,
                    # not NaN (RINEX 3.04 §6, missing field interpretation).
                    if not chunk:
                        values[pos] = 0.0
                        continue
                    try:
                        values[pos] = float(chunk)
                    except ValueError:
                        values[pos] = 0.0
                # Fields trimmed by _select_fields (typically trailing
                # FitIntvl / spare slots that the receiver omitted) also
                # default to zero per the RINEX 3.04 spec, except for the
                # purely-decorative "spare*" slots which we leave as NaN
                # because they have no defined unit.
                for k, name in enumerate(full_fields):
                    if name not in cf and not name.startswith("spare"):
                        values[k] = 0.0
                expanded[label][t] = values
                times_set.add(t)

    times_arr = np.array(sorted(times_set), dtype="datetime64[ns]")
    time_index = {t: i for i, t in enumerate(sorted(times_set))}

    # Sort SVs (with variants) consistently.
    sv_labels = sorted(expanded)
    n_t = times_arr.size
    n_sv = len(sv_labels)

    # All systems share the same {field: column} index space — but field sets
    # differ per system. We need to know the union of all field names so we
    # can allocate a single array.
    all_fields: list[str] = []
    seen_fields: set[str] = set()
    for label in sv_labels:
        for fld in var_fields[label]:
            if fld not in seen_fields:
                all_fields.append(fld)
                seen_fields.add(fld)

    field_index = {fld: i for i, fld in enumerate(all_fields)}
    data = np.full((len(all_fields), n_t, n_sv), np.nan, dtype=float)

    for j, label in enumerate(sv_labels):
        sys_letter = label[0]
        for t, vals in expanded[label].items():
            it = time_index[t]
            for k, fname in enumerate(var_fields[label]):
                data[field_index[fname], it, j] = vals[k]

    # GLONASS / SBAS km-to-m fix (applied once, vectorized).
    for fname in _KM_TO_M_FIELDS:
        if fname not in field_index:
            continue
        idx = field_index[fname]
        # Only apply for SV columns belonging to R or S systems.
        sv_mask = np.array([label[0] in {"R", "S"} for label in sv_labels])
        if sv_mask.any():
            data[idx, :, sv_mask] *= 1000.0

    nav = xr.Dataset(
        {fname: (("time", "sv"), data[i]) for i, fname in enumerate(all_fields)},
        coords={"time": times_arr, "sv": sv_labels},
    )

    if "IONOSPHERIC CORR" in header:
        corr = header["IONOSPHERIC CORR"]
        if "GPSA" in corr and "GPSB" in corr:
            nav.attrs["ionospheric_corr_GPS"] = np.hstack((corr["GPSA"], corr["GPSB"]))
        if "GAL" in corr:
            nav.attrs["ionospheric_corr_GAL"] = corr["GAL"]
        if "QZSA" in corr and "QZSB" in corr:
            nav.attrs["ionospheric_corr_QZS"] = np.hstack((corr["QZSA"], corr["QZSB"]))
        if "BDSA" in corr and "BDSB" in corr:
            nav.attrs["ionospheric_corr_BDS"] = np.hstack((corr["BDSA"], corr["BDSB"]))
        if "IRNA" in corr and "IRNB" in corr:
            nav.attrs["ionospheric_corr_IRN"] = np.hstack((corr["IRNA"], corr["IRNB"]))

    nav.attrs["version"] = header["version"]
    nav.attrs["svtype"] = svtypes
    nav.attrs["rinextype"] = "nav"
    if isinstance(fn, Path):
        nav.attrs["filename"] = fn.name

    return nav


def navtime3(fn: FileLike) -> np.ndarray:
    """Return all unique epoch timestamps in a RINEX-3 NAV file."""
    times: list[datetime] = []
    with opener(fn) as f:
        navheader3(f)
        for line in f:
            try:
                times.append(parse_nav3_epoch(line))
            except ValueError:
                continue
            n_lines = _NL_BY_SYS.get(line[0])
            if n_lines is not None:
                _skip(f, n_lines)
    return np.unique(np.asarray(times, dtype="datetime64[ms]"))


__all__ = ["navtime3", "rinexnav3"]
