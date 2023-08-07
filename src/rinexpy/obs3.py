"""RINEX-3 observation file reader.

This is the headline optimization in rinexpy. ``georinex.obs3._epoch`` builds
an ``xarray.Dataset`` for each (epoch, system) tuple and ``xarray.merge``s it
into a running aggregate; the merge re-allocates a coordinate index every
time and the cost is O(N^2) in the number of epochs. The upstream README
explicitly attributes "over 60% of time" to that pattern.

This module instead:

1. Walks the file once, collecting per-epoch SV labels and the decoded raw
   string for each SV into per-system buffers.
2. After the walk, allocates dense ``(n_times, n_svs)`` arrays per
   measurement name and back-fills them from the buffers.
3. Builds a single :class:`xarray.Dataset` from those arrays.

Functional behavior matches georinex (same coords, same data variables, same
attrs). Test fixtures r3all.nc / r3G.nc / r3GR.nc are reproduced bit-exactly
to within float tolerance.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Any

import numpy as np
import xarray as xr

from ._common import determine_time_system
from ._io import opener
from ._time import normalize_interval, parse_obs3_epoch
from ._types import FileLike, MeasSelection, SystemSelection
from .headers import obsheader3

log = logging.getLogger(__name__)

# Width of one OBS3 numeric field (14 cols + 1 LLI + 1 SSI = 16).
_FIELD_WIDTH = 14
_CELL_WIDTH = 16  # value + LLI + SSI

# Measurement-label prefixes that get a paired ``Llli`` (loss-of-lock indicator)
# column in the output when ``useindicators=True``. SSI is emitted for every
# measurement; LLI only for L1/L2 carrier-phase observables.
_LLI_PREFIXES: frozenset[str] = frozenset({"L1", "L2"})


def rinexobs3(
    fn: FileLike,
    use: SystemSelection = None,
    *,
    tlim: tuple[datetime, datetime] | None = None,
    useindicators: bool = False,
    meas: MeasSelection = None,
    verbose: bool = False,
    fast: bool = False,
    interval: float | int | timedelta | None = None,
) -> xr.Dataset:
    """Read a RINEX-3 OBS file into an ``xarray.Dataset``.

    Parameters
    ----------
    fn:
        Path or open text stream of a RINEX-3 OBS file.
    use:
        Optional set of single-letter system codes to keep.
    tlim:
        Optional ``(start, stop)`` datetime bounds.
    useindicators:
        If True, emit ``Lxxxlli`` and ``Xxxxssi`` indicator columns alongside
        the measurement values.
    meas:
        Optional list of measurement-label prefixes to keep.
    verbose:
        If True, prints the current epoch to stderr while parsing.
    fast:
        Reserved for parity with georinex; currently unused (the rewritten
        reader is always single-pass and no longer needs the speculative-
        preallocation hack).
    interval:
        Optional decimation: skip epochs closer than ``interval`` to the
        previous kept epoch.

    Returns
    -------
    xarray.Dataset
        With coords ``time`` and ``sv``, one data variable per measurement
        label, plus measurement-specific ``ssi``/``lli`` columns when
        ``useindicators`` is set.

    Raises
    ------
    TypeError
        If ``tlim`` is provided with non-``datetime`` bounds.
    """
    interval = normalize_interval(interval)

    if isinstance(use, str):
        use = {use}
    if isinstance(meas, str):
        meas = [meas]
    if not meas or not meas[0].strip():
        meas = None

    if tlim is not None and not isinstance(tlim[0], datetime):
        raise TypeError("time bounds must be datetime.datetime")

    with opener(fn) as f:
        hdr = obsheader3(f, use=use, meas=meas)
        epochs, time_offsets = _walk_epochs(f, hdr, tlim=tlim, interval=interval, verbose=verbose)

    data = _assemble_obs3(epochs, hdr, useindicators=useindicators)
    _attach_obs3_attrs(data, hdr, fn=fn, time_offsets=time_offsets)
    return data


def _walk_epochs(
    f: IO[str],
    hdr: Mapping[Hashable, Any],
    *,
    tlim: tuple[datetime, datetime] | None,
    interval: timedelta | None,
    verbose: bool,
) -> tuple[list[tuple[datetime, list[str], list[str]]], list[float]]:
    """Single-pass walk of the OBS3 data section.

    Returns a list of ``(time, sv_labels, raw_lines)`` tuples — one per epoch.
    The ``raw_lines`` element is a list of fixed-width data strings, one per
    SV in this epoch (they are not yet merged or decoded).
    """
    epochs: list[tuple[datetime, list[str], list[str]]] = []
    time_offsets: list[float] = []
    last_epoch: datetime | None = None

    for line in f:
        if not line.startswith(">"):
            break

        try:
            t = parse_obs3_epoch(line)
        except ValueError:
            log.debug("garbage line in OBS3 file: %r", line[:80])
            continue

        try:
            time_offsets.append(float(line[41:56]))
        except ValueError:
            pass

        n_sv = int(line[33:35])
        sv_list: list[str] = []
        raw_lines: list[str] = []
        for _ in range(n_sv):
            sv_line = f.readline()
            sv_list.append(sv_line[:3])
            raw_lines.append(sv_line[3:])

        if tlim is not None:
            if t < tlim[0]:
                continue
            if t > tlim[1]:
                break

        if interval is not None:
            if last_epoch is None:
                last_epoch = t
            elif t - last_epoch < interval:
                continue
            else:
                last_epoch += interval

        if verbose:
            print(t, end="\r")

        epochs.append((t, sv_list, raw_lines))

    return epochs, time_offsets


def _decode_sv_line(raw: str, n_obs: int) -> np.ndarray:
    """Decode one SV's measurement line into ``(n_obs, 3)`` float array.

    Each measurement occupies a 16-byte cell: 14 chars of value, 1 char LLI,
    1 char SSI. Empty cells (whitespace) become ``NaN``.
    """
    out = np.full((n_obs, 3), np.nan, dtype=float)
    # Pad the line to the full width so we don't have to bounds-check below.
    padded = raw.ljust(n_obs * _CELL_WIDTH)
    for k in range(n_obs):
        cell = padded[k * _CELL_WIDTH : (k + 1) * _CELL_WIDTH]
        value_part = cell[:_FIELD_WIDTH]
        if value_part.strip():
            try:
                out[k, 0] = float(value_part)
            except ValueError:
                pass
        lli = cell[_FIELD_WIDTH : _FIELD_WIDTH + 1]
        if lli.strip():
            try:
                out[k, 1] = float(lli)
            except ValueError:
                pass
        ssi = cell[_FIELD_WIDTH + 1 : _CELL_WIDTH]
        if ssi.strip():
            try:
                out[k, 2] = float(ssi)
            except ValueError:
                pass
    return out


def _assemble_obs3(
    epochs: list[tuple[datetime, list[str], list[str]]],
    hdr: Mapping[Hashable, Any],
    *,
    useindicators: bool,
) -> xr.Dataset:
    """Build the final ``xarray.Dataset`` from the per-epoch buffers.

    The hot path: instead of ``xarray.merge`` per epoch, allocate dense
    arrays sized once and write into them with vectorised assignments.
    """
    fields: dict[str, list[str]] = hdr["fields"]
    fields_ind: dict[str, Any] = hdr["fields_ind"]
    f_max: int = hdr["Fmax"]

    if not epochs or not fields:
        return xr.Dataset(coords={"time": [], "sv": []})

    times = [e[0] for e in epochs]
    times_arr = np.array(times, dtype="datetime64[ns]")
    n_t = len(times)

    # Determine global SV set per system, sorted alphabetically within each
    # system. Systems themselves are then concatenated in alphabetical order
    # to match georinex's xarray.merge-driven SV ordering. We clean
    # 'G 7' -> 'G07' here so labels collate correctly.
    sys_keys = sorted(fields)
    per_sys_svs: dict[str, list[str]] = {sk: [] for sk in sys_keys}
    seen_sv: dict[str, set[str]] = {sk: set() for sk in sys_keys}
    for _, svs, _ in epochs:
        for label in svs:
            sk = label[0]
            if sk not in fields:
                continue
            clean = label.replace(" ", "0")
            if clean not in seen_sv[sk]:
                seen_sv[sk].add(clean)
                per_sys_svs[sk].append(clean)
    for sk in per_sys_svs:
        per_sys_svs[sk].sort()

    # Per-system index lookup tables and per-system measurement allocations.
    sv_index_by_sys: dict[str, dict[str, int]] = {
        sk: {sv: i for i, sv in enumerate(per_sys_svs[sk])} for sk in fields
    }

    # Per-system data buffers: (n_meas_for_sys, n_t, n_sv_for_sys)
    sys_data: dict[str, np.ndarray] = {}
    sys_lli: dict[str, np.ndarray] = {}
    sys_ssi: dict[str, np.ndarray] = {}
    for sk in sys_keys:
        n_sv = len(per_sys_svs[sk])
        n_meas = len(fields[sk])
        sys_data[sk] = np.full((n_meas, n_t, n_sv), np.nan, dtype=float)
        if useindicators:
            sys_lli[sk] = np.full((n_meas, n_t, n_sv), np.nan, dtype=float)
            sys_ssi[sk] = np.full((n_meas, n_t, n_sv), np.nan, dtype=float)

    # Single pass: decode each epoch's SVs into the dense buffers.
    for i, (_t, svs, raws) in enumerate(epochs):
        for sv_label, raw in zip(svs, raws):
            sk = sv_label[0]
            if sk not in fields:
                continue
            decoded = _decode_sv_line(raw, f_max)
            # decoded has shape (Fmax, 3): (value, LLI, SSI).
            ind = fields_ind[sk]
            if isinstance(ind, np.ndarray):
                # boolean mask over Fmax*3 cells - reshape to (Fmax, 3) view.
                # We restored the original cell shape so per-meas indexing works.
                # ind is laid out as [val0, lli0, ssi0, val1, lli1, ssi1, ...]
                mask_3 = ind.reshape(-1, 3)[:f_max, 0]
                decoded = decoded[mask_3]
            clean_sv = sv_label.replace(" ", "0")
            j = sv_index_by_sys[sk][clean_sv]
            n_meas = len(fields[sk])
            sys_data[sk][:, i, j] = decoded[:n_meas, 0]
            if useindicators:
                sys_lli[sk][:, i, j] = decoded[:n_meas, 1]
                sys_ssi[sk][:, i, j] = decoded[:n_meas, 2]

    # Build the union of all SV labels in stable order (sorted within each
    # system; systems themselves concatenated in alphabetical order).
    all_svs: list[str] = []
    sv_offset: dict[str, int] = {}
    for sk in sys_keys:
        sv_offset[sk] = len(all_svs)
        all_svs.extend(per_sys_svs[sk])

    n_sv_total = len(all_svs)
    if n_sv_total == 0:
        return xr.Dataset(coords={"time": times_arr, "sv": []})

    # Project per-system buffers into the union SV axis. Different systems
    # never share a measurement label (e.g. C1C is per-system in the spec),
    # but if they do, later systems overwrite earlier ones.
    var_buffers: dict[str, np.ndarray] = {}
    var_lli: dict[str, np.ndarray] = {}
    var_ssi: dict[str, np.ndarray] = {}
    for sk in sys_keys:
        offset = sv_offset[sk]
        n_sv = len(per_sys_svs[sk])
        for k, label in enumerate(fields[sk]):
            buf = var_buffers.setdefault(label, np.full((n_t, n_sv_total), np.nan))
            buf[:, offset : offset + n_sv] = sys_data[sk][k]
            if useindicators:
                if label[:2] in _LLI_PREFIXES:
                    lbuf = var_lli.setdefault(f"{label}lli", np.full((n_t, n_sv_total), np.nan))
                    lbuf[:, offset : offset + n_sv] = sys_lli[sk][k]
                sbuf = var_ssi.setdefault(f"{label}ssi", np.full((n_t, n_sv_total), np.nan))
                sbuf[:, offset : offset + n_sv] = sys_ssi[sk][k]

    data_vars: dict[str, tuple] = {name: (("time", "sv"), arr) for name, arr in var_buffers.items()}
    if useindicators:
        for name, arr in var_lli.items():
            data_vars[name] = (("time", "sv"), arr)
        for name, arr in var_ssi.items():
            data_vars[name] = (("time", "sv"), arr)

    return xr.Dataset(data_vars, coords={"time": times_arr, "sv": all_svs})


def _attach_obs3_attrs(
    data: xr.Dataset,
    hdr: Mapping[Hashable, Any],
    *,
    fn: FileLike,
    time_offsets: list[float],
) -> None:
    """Populate the OBS3 attribute keys (interval, position, time_system, ...)."""
    data.attrs["version"] = hdr.get("version", 0)
    if "interval" in hdr:
        data.attrs["interval"] = hdr["interval"]
    elif "time" in data.coords and data.time.size > 0:
        try:
            data.attrs["interval"] = float(np.median(np.diff(data.time) / np.timedelta64(1, "s")))
        except (TypeError, ValueError):
            data.attrs["interval"] = float("nan")
    else:
        data.attrs["interval"] = float("nan")

    data.attrs["rinextype"] = "obs"
    data.attrs["fast_processing"] = 0
    data.attrs["time_system"] = determine_time_system(hdr)
    if isinstance(fn, Path):
        data.attrs["filename"] = fn.name
    if "position" in hdr:
        data.attrs["position"] = hdr["position"]
        if "position_geodetic" in hdr:
            data.attrs["position_geodetic"] = hdr["position_geodetic"]
    if time_offsets:
        data.attrs["time_offset"] = time_offsets
    if "RCV CLOCK OFFS APPL" in hdr:
        try:
            data.attrs["receiver_clock_offset_applied"] = int(hdr["RCV CLOCK OFFS APPL"])
        except (KeyError, ValueError):
            pass


def obstime3(fn: FileLike, *, verbose: bool = False) -> np.ndarray:
    """Return all unique epoch timestamps in a RINEX-3 OBS file."""
    times: list[datetime] = []
    with opener(fn) as f:
        for line in f:
            if line.startswith("> "):
                try:
                    times.append(parse_obs3_epoch(line))
                except (ValueError, IndexError):
                    log.debug("not a time line: %r", line[:80])
    arr = np.asarray(times, dtype="datetime64[us]")
    return arr


__all__ = ["obstime3", "rinexobs3"]
