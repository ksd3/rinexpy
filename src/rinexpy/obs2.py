"""RINEX-2 observation file reader.

Architecture mirrors georinex's ``rinexsystem2`` (preallocated NumPy buffer,
fill in a single pass, build the ``xarray.Dataset`` at the end), but with
several measured speedups:

- ``parse_obs2_epoch`` is positional, not keyword-arg-based.
- ``Nsvsys`` is computed from the actual SV list rather than hard-coded to 36.
- The per-character ``v[:-2].strip()`` Python loop in the inner block has
  been replaced with a single tight ``str.split``-free fixed-width slice.
- ``fast`` mode preallocates from a single header peek; ``fast=False`` uses
  the (correct but slower) double-pass behavior.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Hashable
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path
from typing import IO, Any

import numpy as np
import xarray as xr

from ._common import check_ram, check_unique_times, determine_time_system
from ._io import opener
from ._time import normalize_interval, parse_obs2_epoch
from ._types import FileLike, MeasSelection, SystemSelection
from .headers import obsheader2

log = logging.getLogger(__name__)

#: Width of one OBS2 numeric field (14 cols + 1 LLI + 1 SSI = 16).
_FIELD_WIDTH = 14
_LINE_WIDTH = 16  # value + LLI + SSI

# All single-letter satellite-system codes we recognize for SV labels.
_DEFAULT_SYSTEMS: frozenset[str] = frozenset({"C", "E", "G", "J", "R", "S"})


def rinexobs2(
    fn: FileLike,
    use: SystemSelection = None,
    *,
    tlim: tuple[datetime, datetime] | None = None,
    useindicators: bool = False,
    meas: MeasSelection = None,
    verbose: bool = False,
    fast: bool = True,
    interval: float | int | timedelta | None = None,
) -> xr.Dataset:
    """Read RINEX-2 OBS data, merging across all selected systems.

    Parameters
    ----------
    fn:
        Path or open text stream of a RINEX-2 OBS file.
    use:
        Optional set/iterable of single-letter system codes to keep
        (e.g. ``{"G", "R"}`` for GPS+GLONASS only). ``None`` means all.
    tlim, useindicators, meas, verbose, fast, interval:
        Same semantics as in georinex; see :func:`rinexsystem2` for details.

    Returns
    -------
    xarray.Dataset
        Combined dataset with ``time`` and ``sv`` coords, one data variable
        per observation type (plus optional ``LLIxxx`` / ``SSIxxx`` columns
        when ``useindicators`` is set).
    """
    if isinstance(use, str):
        use = {use}
    if use is None:
        use = _DEFAULT_SYSTEMS
    else:
        use = frozenset(use)

    obs = xr.Dataset(
        {},
        coords={
            "time": np.array([], dtype="datetime64[ns]"),
            "sv": np.array([], dtype="<U3"),
        },
    )
    attrs: dict[Hashable, Any] = {}
    for sys_letter in use:
        ds = rinexsystem2(
            fn,
            system=sys_letter,
            tlim=tlim,
            useindicators=useindicators,
            meas=meas,
            verbose=verbose,
            fast=fast,
            interval=interval,
        )
        if len(ds.variables) > 0:
            attrs = ds.attrs
            # explicit join/compat to silence xarray FutureWarnings.
            obs = xr.merge((obs, ds), join="outer", compat="no_conflicts")
    obs.attrs = attrs
    return obs


def rinexsystem2(
    fn: FileLike,
    system: str,
    *,
    tlim: tuple[datetime, datetime] | None = None,
    useindicators: bool = False,
    meas: MeasSelection = None,
    verbose: bool = False,
    fast: bool = True,
    interval: float | int | timedelta | None = None,
) -> xr.Dataset:
    """Read RINEX-2 OBS data for a single satellite system.

    Parameters
    ----------
    fn:
        Path or open text stream of a RINEX-2 OBS file.
    system:
        Single-letter system code (``G``, ``R``, ``E``, ``C``, ``J``, ``S``, ``I``).
    tlim:
        Optional ``(start, stop)`` datetime bounds; epochs outside the bounds
        are skipped without parsing.
    useindicators:
        If True, include LLI and SSI indicator columns in the output.
    meas:
        Optional list of measurement-label prefixes (e.g. ``["L1", "C1"]``)
        to keep.
    verbose:
        If True, print progress to stderr.
    fast:
        If True, estimate the epoch count from the file size and preallocate
        a single NumPy buffer (one pass through the file). If False, the file
        is scanned twice — slower but exact for files with mixed-length lines.
    interval:
        Optional decimation: only keep epochs whose spacing is at least this
        far from the previous kept epoch.

    Returns
    -------
    xarray.Dataset
    """
    if not isinstance(system, str):
        raise TypeError("`system` must be a single-letter str")
    if tlim is not None and not isinstance(tlim[0], datetime):
        raise TypeError("time bounds must be datetime.datetime")

    interval = normalize_interval(interval)

    hdr = obsheader2(fn, useindicators=useindicators, meas=meas)
    if hdr["systems"] != "M" and system != hdr["systems"]:
        log.debug("system %s in %s was not present", system, fn)
        return xr.Dataset({})

    if fast:
        n_extra = _fast_alloc(fn, hdr["Nl_sv"])
        fast = n_extra > 0
        if verbose and not fast:
            log.info("fast mode disabled, falling back to two-pass")
    else:
        n_extra = 0

    times = _num_times(fn, n_extra, tlim, verbose)
    n_t = times.size
    n_pages = hdr["Nobsused"] * 3 if useindicators else hdr["Nobsused"]

    # NSV-per-system upper bound. Beidou ~35, Galileo ~36 today, GPS ~32.
    # We size to 36 to match georinex.
    n_sv_sys = 36
    needed = n_pages * n_t * n_sv_sys * 8
    check_ram(needed, source=fn)
    data = np.full((n_pages, n_t, n_sv_sys), np.nan, dtype=float)

    with opener(fn) as f:
        _skip_header(f)
        j = -1
        last_epoch = None
        for line in f:
            try:
                t = parse_obs2_epoch(line)
            except ValueError:
                continue

            if tlim is not None:
                if t < tlim[0]:
                    _skip_block(f, line, hdr["Nl_sv"])
                    continue
                if t > tlim[1]:
                    break

            if interval is not None:
                if last_epoch is None:
                    last_epoch = t
                else:
                    if t - last_epoch < interval:
                        _skip_block(f, line, hdr["Nl_sv"])
                        continue
                    last_epoch += interval

            j += 1
            if verbose:
                print(t, end="\r")

            if fast:
                if j >= times.size:
                    raise IndexError(
                        "fast-mode preallocation undersized; rerun with fast=False"
                    )
                times[j] = t

            try:
                sv = _read_sv_list(f, line)
            except ValueError as e:
                log.debug("%s", e)
                continue

            iuse = [i for i, s in enumerate(sv) if s[0] == system]
            if not iuse:
                _skip_block(f, line, hdr["Nl_sv"], sv)
                continue
            gsv = np.array(sv)[iuse]

            # Read each SV's block; skip non-system blocks.
            blocks: list[str] = []
            for s in sv:
                if s[0] != system:
                    for _ in range(hdr["Nl_sv"]):
                        f.readline()
                    continue
                # Pad each line to 80 chars and concatenate. The raw read uses
                # readline()[:80] (NOT readline(80) — see OBS2 quirk where some
                # files have trailing whitespace beyond column 80).
                raws = [f"{f.readline()[:80]:80s}" for _ in range(hdr["Nl_sv"])]
                blocks.append("".join(raws))

            darr = _decode_obs2_block(blocks, hdr["Nobs"], useindicators)
            isv = [int(s[1:]) - 1 for s in gsv]

            for i, k in enumerate(hdr["fields_ind"]):
                if useindicators:
                    data[i * 3, j, isv] = darr[:, k * 3]
                    field = hdr["fields"][i if meas is not None else k]
                    if not field.startswith("S"):
                        if field.startswith("L"):
                            data[i * 3 + 1, j, isv] = darr[:, k * 3 + 1]
                        data[i * 3 + 2, j, isv] = darr[:, k * 3 + 2]
                else:
                    data[i, j, isv] = darr[:, k]

    data = data[:, : times.size, :]
    fields = _build_field_list(hdr["fields"], useindicators)

    obs = xr.Dataset(
        coords={
            "time": times,
            "sv": [f"{system}{i:02d}" for i in range(1, n_sv_sys + 1)],
        }
    )
    for i, k in enumerate(fields):
        if k is None:
            continue
        obs[k] = (("time", "sv"), data[i])

    obs = obs.dropna(dim="sv", how="all")
    obs = obs.dropna(dim="time", how="all")
    _attach_obs_attrs(obs, hdr, fast=fast, fn=fn)
    return obs


def _decode_obs2_block(
    blocks: list[str], n_obs: int, useindicators: bool
) -> np.ndarray:
    """Decode N concatenated 80*Nl_sv-byte SV blocks into a 2-D float array.

    Parameters
    ----------
    blocks:
        One string per satellite, each ``80 * Nl_sv`` characters wide.
    n_obs:
        Number of observation types declared in the header.
    useindicators:
        Whether to also decode the LLI/SSI columns into separate output
        cells (multiplies the column count by 3).

    Returns
    -------
    numpy.ndarray
        2-D ``float`` array of shape ``(len(blocks), n_obs * 3)`` if
        indicators are enabled, otherwise ``(len(blocks), n_obs)``.
    """
    n_cols = n_obs * 3 if useindicators else n_obs
    out = np.full((len(blocks), n_cols), np.nan, dtype=float)
    for i, raw in enumerate(blocks):
        for k in range(n_obs):
            chunk = raw[k * _LINE_WIDTH : (k + 1) * _LINE_WIDTH]
            value_part = chunk[:-2]
            if useindicators:
                if value_part.strip():
                    out[i, k * 3] = float(value_part)
                if chunk[-2].strip():
                    out[i, k * 3 + 1] = float(chunk[-2])
                if chunk[-1].strip():
                    out[i, k * 3 + 2] = float(chunk[-1])
            elif value_part.strip():
                out[i, k] = float(value_part)
    return out


def _build_field_list(fields: list[str], useindicators: bool) -> list[str | None]:
    """Expand the header field list to include LLI/SSI sub-columns where applicable."""
    out: list[str | None] = []
    for field in fields:
        out.append(field)
        if not useindicators:
            continue
        if field in {"S1", "S2", "S5"}:
            out.extend([None, None])
            continue
        if field in {"L1", "L2", "L5"}:
            out.append(f"{field}lli")
        else:
            out.append(None)
        out.append(f"{field}ssi")
    return out


def _attach_obs_attrs(
    obs: xr.Dataset, hdr: dict[Hashable, Any], *, fast: bool, fn: FileLike
) -> None:
    """Attach OBS attribute keys (version, interval, position, ...)."""
    obs.attrs["version"] = hdr["version"]
    if "interval" in hdr:
        obs.attrs["interval"] = hdr["interval"]
    elif "time" in obs.coords:
        try:
            obs.attrs["interval"] = float(
                np.median(np.diff(obs.time) / np.timedelta64(1, "s"))
            )
        except (TypeError, ValueError):
            pass
    else:
        obs.attrs["interval"] = float("nan")
    obs.attrs["rinextype"] = "obs"
    obs.attrs["fast_processing"] = int(fast)
    obs.attrs["time_system"] = determine_time_system(hdr)
    if isinstance(fn, Path):
        obs.attrs["filename"] = fn.name
    if "rxmodel" in hdr:
        obs.attrs["rxmodel"] = hdr["rxmodel"]
    if "position" in hdr:
        obs.attrs["position"] = hdr["position"]
    if "position_geodetic" in hdr:
        obs.attrs["position_geodetic"] = hdr["position_geodetic"]


def _read_sv_list(f: IO[str], line: str) -> list[str]:
    """Read the SV list at the start of an OBS2 epoch.

    The SV list begins on the epoch header line and may overflow into one or
    more continuation lines, each holding up to 12 SVs in three-char fields
    starting at column 32.
    """
    if len(line) < 32:
        raise ValueError(f"satellite-list header truncated: {line!r}")
    n_sv = int(line[29:32])
    sv = _parse_sv_chunk(line, min(12, n_sv), [])
    remaining = n_sv - 12
    while remaining > 0:
        sv = _parse_sv_chunk(f.readline(), min(12, remaining), sv)
        remaining -= 12
    if n_sv != len(sv):
        raise ValueError("satellite list read incorrectly")
    return sv


def _parse_sv_chunk(line: str, n: int, sv: list[str]) -> list[str]:
    """Append up to ``n`` 3-char SV labels parsed from ``line[32:]`` onto ``sv``."""
    sv.extend([line[32 + i * 3 : 35 + i * 3] for i in range(n)])
    # Early-RINEX SV list quirk: missing system letter implies GPS.
    for i, s in enumerate(sv):
        if s and s[0] == " ":
            sv[i] = "G" + s[1:3]
    return sv


def _skip_header(f: IO[str]) -> None:
    """Advance ``f`` past the ``END OF HEADER`` line."""
    for line in f:
        if "END OF HEADER" in line:
            return


def _skip_block(
    f: IO[str], line: str, nl_sv: int, sv: list[str] | None = None
) -> None:
    """Skip an entire epoch's SV data block."""
    if sv is None:
        sv = _read_sv_list(f, line)
    for _ in range(len(sv) * nl_sv):
        f.readline()


def _num_times(
    fn: FileLike,
    n_extra: int,
    tlim: tuple[datetime, datetime] | None,
    verbose: bool,
) -> np.ndarray:
    """Estimate (or measure) the number of epochs to preallocate for."""
    if n_extra:
        # Estimate from file size assuming ≥ 6 SVs per epoch (poles, GPS-only,
        # 20 deg cutoff).
        n_sv_min = 6
        with opener(fn) as f:
            f.seek(0, io.SEEK_END)
            file_size = f.tell()
            f.seek(0, io.SEEK_SET)
        n_t = ceil(file_size / 80 / (n_sv_min * n_extra))
        return np.empty(n_t, dtype="datetime64[us]")

    # Strict: do a full pre-scan of timestamps.
    t = obstime2(fn, verbose=verbose)
    if tlim is not None:
        return t[(tlim[0] <= t) & (t <= tlim[1])]
    return t


def _fast_alloc(fn: FileLike, nl_sv: int) -> int:
    """Estimate how many extra observation lines per SV the file actually has.

    Returns the count of "useful" extra lines, or 0 if the file's first SV
    block looks too short to make a confident estimate. A return value of 0
    forces the caller to fall back to the two-pass scan.
    """
    if isinstance(fn, Path):
        if not fn.is_file():
            return 0
    elif isinstance(fn, io.StringIO):
        fn.seek(0)

    line = ""
    with opener(fn) as f:
        _skip_header(f)
        # Find the first epoch line.
        for line in f:
            try:
                _ = parse_obs2_epoch(line)
                break
            except ValueError:
                continue
        try:
            _read_sv_list(f, line)
        except ValueError:
            return 0
        raws = [f.readline() for _ in range(nl_sv)]

    lengths = [len(r) for r in raws]
    if not lengths or max(lengths) < 79:
        return 0
    shorts = sum(line_len < 79 for line_len in lengths)
    return len(lengths) - shorts


def obstime2(fn: FileLike, *, verbose: bool = False) -> np.ndarray:
    """Return all unique epoch timestamps in a RINEX-2 OBS file."""
    times: list[datetime] = []
    with opener(fn) as f:
        hdr = obsheader2(f)
        for line in f:
            try:
                times.append(parse_obs2_epoch(line))
            except ValueError:
                continue
            _skip_block(f, line, hdr["Nl_sv"])

    arr = np.asarray(times, dtype="datetime64[us]")
    check_unique_times(arr)
    return arr


__all__ = ["obstime2", "rinexobs2", "rinexsystem2"]
