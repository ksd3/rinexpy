"""Iterator-style readers for files larger than RAM.

The default :func:`rinexpy.load` materializes the entire file as one
``xarray.Dataset``. For multi-day RINEX-3 OBS files (1 Hz, mixed
constellation, full observation set) that can be many GB and exceed
local memory.

This module provides per-epoch generators:

- :func:`iter_obs3_epochs`: yields ``(time, xarray.Dataset)`` per OBS-3 epoch
- :func:`iter_nav3_records`: yields ``(time, sv, dict)`` per NAV-3 record

The caller can stream-process or stream-write without ever holding the
full file in memory. Typical usage:

.. code-block:: python

    import rinexpy
    for t, ds in rinexpy.iter_obs3_epochs("huge.rnx.gz"):
        ds.to_netcdf(f"epoch_{t.isoformat()}.nc")

These iterators do **not** support the ``fast`` preallocation path,
``meas`` masking by per-system bool array, or ``useindicators=True``
(the indicator columns add complexity that's not worth it for the
streaming use case). They do support ``use=`` (system selection) and
``tlim=``/``interval=`` decimation since those skip whole epochs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import IO

import numpy as np
import xarray as xr

from ._io import opener
from ._time import datetime_to_ns, normalize_interval, parse_obs3_epoch_ns
from ._types import FileLike, SystemSelection
from .headers import obsheader3
from .obs3 import _decode_sv_line

log = logging.getLogger(__name__)


def iter_obs3_epochs(
    fn: FileLike,
    *,
    use: SystemSelection = None,
    tlim: tuple[datetime, datetime] | None = None,
    interval: float | int | timedelta | None = None,
) -> Iterator[tuple[datetime, xr.Dataset]]:
    """Yield ``(time, xarray.Dataset)`` for each epoch in a RINEX-3 OBS file.

    Parameters
    ----------
    fn:
        Path or open text stream of a RINEX-3 OBS file.
    use:
        Optional set of single-letter system codes to keep.
    tlim:
        Optional ``(start, stop)`` datetime bounds; epochs outside are skipped.
    interval:
        Optional decimation: skip epochs closer than this to the previous
        kept epoch.

    Yields
    ------
    tuple
        ``(datetime, xarray.Dataset)``. The dataset has ``time`` of length 1
        and ``sv`` sized to the SVs present at that epoch only.

    Notes
    -----
    Memory footprint is one epoch at a time, so this is appropriate for
    files larger than RAM. Compared to :func:`rinexpy.load`, the per-epoch
    overhead is higher (you pay one ``xarray.Dataset`` construction per
    epoch instead of one for the whole file), so on small files
    :func:`load` is faster.
    """
    if isinstance(use, str):
        use_set: set[str] | None = {use}
    elif use is None:
        use_set = None
    else:
        use_set = set(use)

    interval_td = normalize_interval(interval)
    tlim_ns = (datetime_to_ns(tlim[0]), datetime_to_ns(tlim[1])) if tlim is not None else None
    interval_ns = (
        int(interval_td.total_seconds() * 1_000_000_000) if interval_td is not None else None
    )

    with opener(fn) as f:
        hdr = obsheader3(f, use=use_set, meas=None)
        last_epoch_ns: int | None = None
        yield from _stream_obs3_body(f, hdr, tlim_ns, interval_ns, last_epoch_ns)


def _stream_obs3_body(
    f: IO[str],
    hdr,
    tlim_ns: tuple[int, int] | None,
    interval_ns: int | None,
    last_epoch_ns: int | None,
) -> Iterator[tuple[datetime, xr.Dataset]]:
    """Inner generator that walks the data section one epoch at a time.

    Split out so :func:`iter_obs3_epochs` can keep its argument-normalization
    code in one place. Each yield builds a fresh single-time ``Dataset``
    sized only to the SVs present at that epoch (no global SV axis,
    unlike the bulk reader).
    """
    fields = hdr["fields"]
    f_max = hdr["Fmax"]

    for line in f:
        if not line.startswith(">"):
            return
        try:
            t_ns = parse_obs3_epoch_ns(line)
        except ValueError:
            log.debug("garbage line in OBS3 stream: %r", line[:80])
            continue

        n_sv = int(line[33:35])
        sv_labels: list[str] = []
        raw_lines: list[str] = []
        for _ in range(n_sv):
            sv_line = f.readline()
            sv_labels.append(sv_line[:3])
            raw_lines.append(sv_line[3:])

        if tlim_ns is not None:
            if t_ns < tlim_ns[0]:
                continue
            if t_ns > tlim_ns[1]:
                return

        if interval_ns is not None:
            if last_epoch_ns is None:
                last_epoch_ns = t_ns
            elif t_ns - last_epoch_ns < interval_ns:
                continue
            else:
                last_epoch_ns += interval_ns

        ds = _build_epoch_dataset(t_ns, sv_labels, raw_lines, fields, f_max)
        if ds is not None:
            yield ds.time.values[0].astype("datetime64[us]").astype(datetime), ds


def _build_epoch_dataset(
    t_ns: int,
    sv_labels: list[str],
    raw_lines: list[str],
    fields,
    f_max: int,
) -> xr.Dataset | None:
    """Decode one epoch into a single-time, narrow-sv ``xarray.Dataset``.

    Returns ``None`` if no SVs from the requested ``fields`` were present in
    this epoch (so the caller can skip empty yields).
    """
    keep_idx = [i for i, lab in enumerate(sv_labels) if lab[0] in fields]
    if not keep_idx:
        return None
    clean_svs = [sv_labels[i].replace(" ", "0") for i in keep_idx]

    decoded = np.stack([_decode_sv_line(raw_lines[i], f_max) for i in keep_idx])

    data_vars: dict[str, tuple] = {}
    for sk, labels in fields.items():
        sv_in_sys = [(j, sv_labels[i]) for j, i in enumerate(keep_idx) if sv_labels[i][0] == sk]
        if not sv_in_sys:
            continue
        rows = [j for j, _ in sv_in_sys]
        for k, label in enumerate(labels):
            col = decoded[rows, k, 0]
            buf = data_vars.setdefault(
                label, (("time", "sv"), np.full((1, len(clean_svs)), np.nan))
            )
            arr = buf[1]
            for r, j in enumerate(rows):
                arr[0, j] = col[r]

    times_arr = np.asarray([t_ns], dtype="int64").view("datetime64[ns]")
    return xr.Dataset(data_vars, coords={"time": times_arr, "sv": clean_svs})


__all__ = ["iter_obs3_epochs"]
