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

from . import _jit, _native
from ._common import determine_time_system
from ._io import opener
from ._time import datetime_to_ns, normalize_interval, parse_obs3_epoch_ns
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
    use_jit: bool | None = None,
    use_native: bool | None = None,
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
    use_jit:
        If True, use the optional numba-jitted decoder from
        :mod:`rinexpy._jit` (requires the ``jit`` extra). If ``None``
        (default), follow the ``RINEXPY_USE_JIT`` environment variable.
        Numba carries a ~1.5 s one-time JIT cost; the speedup is only
        worth it on files with many epochs.
    use_native:
        If True, use the optional C++ decoder from
        :mod:`rinexpy._native` (requires the ``native`` extra, which
        installs ``rinexpy-native``). Strictly fastest path on real
        files (~3x over JIT, ~18x over pure Python). When ``True``
        and ``rinexpy_native`` is missing, raises ``ImportError``.
        ``None`` (default) follows the ``RINEXPY_USE_NATIVE`` env var.
        Takes precedence over ``use_jit`` when both are enabled.

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

    # Backend selection: native > jit > pure-Python.
    if use_native is None:
        native = _native.is_enabled()
    else:
        native = bool(use_native) and _native.is_available()
    if use_native and not _native.is_available():
        raise ImportError(
            "use_native=True requires the rinexpy-native package; "
            "install with `uv add 'rinexpy[native]'`."
        )
    if native:
        jit = False
    elif use_jit is None:
        jit = _jit.is_enabled()
    else:
        jit = bool(use_jit) and _jit.is_available()
    data = _assemble_obs3(
        epochs, hdr, useindicators=useindicators, use_jit=jit, use_native=native
    )
    _attach_obs3_attrs(data, hdr, fn=fn, time_offsets=time_offsets)
    return data


def _walk_epochs(
    f: IO[str],
    hdr: Mapping[Hashable, Any],
    *,
    tlim: tuple[datetime, datetime] | None,
    interval: timedelta | None,
    verbose: bool,
) -> tuple[list[tuple[int, list[str], list[str]]], list[float]]:
    """Single-pass walk of the OBS3 data section.

    Returns a list of ``(epoch_ns, sv_labels, raw_lines)`` tuples - one per
    epoch. ``epoch_ns`` is an integer number of nanoseconds since the Unix
    epoch (``parse_obs3_epoch_ns``), which lets the assemble step build the
    final ``datetime64[ns]`` coord via ``np.asarray(...).view('datetime64[ns]')``
    in ~40x less time than ``np.array(list_of_datetime, dtype='datetime64[ns]')``.
    """
    epochs: list[tuple[int, list[str], list[str]]] = []
    time_offsets: list[float] = []

    # tlim and interval get coerced to integer-ns once, so the inner-loop
    # comparisons are pure int arithmetic instead of datetime ordering.
    tlim_ns = (datetime_to_ns(tlim[0]), datetime_to_ns(tlim[1])) if tlim is not None else None
    interval_ns = int(interval.total_seconds() * 1_000_000_000) if interval is not None else None
    last_epoch_ns: int | None = None

    for line in f:
        if not line.startswith(">"):
            break

        try:
            t_ns = parse_obs3_epoch_ns(line)
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

        if tlim_ns is not None:
            if t_ns < tlim_ns[0]:
                continue
            if t_ns > tlim_ns[1]:
                break

        if interval_ns is not None:
            if last_epoch_ns is None:
                last_epoch_ns = t_ns
            elif t_ns - last_epoch_ns < interval_ns:
                continue
            else:
                last_epoch_ns += interval_ns

        if verbose:
            print(t_ns, end="\r")

        epochs.append((t_ns, sv_list, raw_lines))

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


def _build_flat_buffer(
    epochs: list[tuple[int, list[str], list[str]]], f_max: int
) -> tuple[np.ndarray, int]:
    """Concatenate every SV line in ``epochs`` into one flat byte buffer.

    Each line is right-padded to ``f_max * _CELL_WIDTH`` ASCII spaces.
    Returns ``(buffer, n_lines)``. Used by both the JIT and native
    OBS3 decoder paths.
    """
    n_lines = sum(len(svs) for _, svs, _ in epochs)
    line_bytes = f_max * _CELL_WIDTH
    if n_lines == 0:
        return np.empty(0, dtype=np.uint8), 0
    flat = np.empty(n_lines * line_bytes, dtype=np.uint8)
    flat.fill(32)  # ASCII space — emulates ljust to the per-line width
    pos = 0
    for _, _svs, raws in epochs:
        for raw in raws:
            n = min(len(raw), line_bytes)
            flat[pos : pos + n] = np.frombuffer(
                raw[:n].encode("ascii", errors="replace"), dtype=np.uint8
            )
            pos += line_bytes
    return flat, n_lines


def _batch_decode_jit(epochs: list[tuple[int, list[str], list[str]]], f_max: int) -> np.ndarray:
    """Bulk-decode every SV line via the numba JIT kernel.

    Pre-flattens the per-epoch raw lines into one ``np.uint8`` buffer
    (so the per-call dispatch into numba happens exactly once per file)
    and calls :func:`rinexpy._jit.decode_obs_batch`.
    """
    flat, n_lines = _build_flat_buffer(epochs, f_max)
    if n_lines == 0:
        return np.empty((0, f_max * 3), dtype=np.float64)
    return _jit.decode_obs_batch(flat, n_lines, f_max)


def _batch_decode_native(
    epochs: list[tuple[int, list[str], list[str]]], f_max: int
) -> np.ndarray:
    """Bulk-decode every SV line via the C++ extension.

    Same shape as :func:`_batch_decode_jit` but dispatches into the
    optional ``rinexpy_native`` package. Caller is responsible for
    checking ``_native.is_available()`` first; this function will
    raise :class:`ImportError` otherwise via the wrapper.
    """
    flat, n_lines = _build_flat_buffer(epochs, f_max)
    if n_lines == 0:
        return np.empty((0, f_max * 3), dtype=np.float64)
    return _native.decode_obs_batch(flat, n_lines, f_max)


def _assemble_obs3(
    epochs: list[tuple[int, list[str], list[str]]],
    hdr: Mapping[Hashable, Any],
    *,
    useindicators: bool,
    use_jit: bool = False,
    use_native: bool = False,
) -> xr.Dataset:
    """Build the final ``xarray.Dataset`` from the per-epoch buffers.

    The hot path: instead of ``xarray.merge`` per epoch, allocate dense
    arrays sized once and write into them with vectorised assignments.

    Three decoder back-ends are dispatched, in priority order:

    - ``use_native=True``: the C++ kernel from
      :mod:`rinexpy._native` (requires ``rinexpy-native``);
    - ``use_jit=True``: the numba kernel from :mod:`rinexpy._jit`;
    - otherwise: the pure-Python :func:`_decode_sv_line`.

    All three are numerically equivalent for well-formed input.
    """
    fields: dict[str, list[str]] = hdr["fields"]
    fields_ind: dict[str, Any] = hdr["fields_ind"]
    f_max: int = hdr["Fmax"]

    if not epochs or not fields:
        return xr.Dataset(coords={"time": [], "sv": []})

    # Epochs already carry int-ns; the .view here is the cheap part of the
    # whole assembly (~1 ms for 100k epochs vs ~60 ms for the old
    # np.array(list_of_datetime, dtype='datetime64[ns]') path).
    times_arr = np.asarray([e[0] for e in epochs], dtype="int64").view("datetime64[ns]")
    n_t = times_arr.size

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
    # Native > JIT > pure-Python.
    if use_native:
        decoded_per_line = _batch_decode_native(epochs, f_max)
    elif use_jit:
        decoded_per_line = _batch_decode_jit(epochs, f_max)
    else:
        decoded_per_line = None
    line_idx = 0
    for i, (_t, svs, raws) in enumerate(epochs):
        for sv_label, raw in zip(svs, raws):
            sk = sv_label[0]
            if sk not in fields:
                line_idx += 1
                continue
            if decoded_per_line is not None:
                decoded = decoded_per_line[line_idx].reshape(f_max, 3)
            else:
                decoded = _decode_sv_line(raw, f_max)
            line_idx += 1
            ind = fields_ind[sk]
            if isinstance(ind, np.ndarray):
                # boolean mask over Fmax*3 cells - reshape to (Fmax, 3) view.
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
    """Return all unique epoch timestamps in a RINEX-3 OBS file.

    Returns
    -------
    numpy.ndarray
        ``datetime64[us]`` array of epoch times, in file order.
    """
    times_ns: list[int] = []
    with opener(fn) as f:
        for line in f:
            if line.startswith("> "):
                try:
                    times_ns.append(parse_obs3_epoch_ns(line))
                except (ValueError, IndexError):
                    log.debug("not a time line: %r", line[:80])
    arr_ns = np.asarray(times_ns, dtype="int64").view("datetime64[ns]")
    return arr_ns.astype("datetime64[us]")


__all__ = ["obstime3", "rinexobs3"]
