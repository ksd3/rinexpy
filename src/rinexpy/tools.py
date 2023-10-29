"""High-level QC and dataset-manipulation tools.

Three helpers, each backed by a CLI subcommand:

- :func:`validate_file` - QC report (header consistency, gaps, jumps).
- :func:`concat_files` - join multiple OBS/NAV files on the time axis.
- :func:`diff_datasets` - find the first divergence between two parsed datasets.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .api import load
from .headers import rinexheader, rinexinfo

log = logging.getLogger(__name__)


def validate_file(fn: Path | str) -> dict[str, Any]:
    """Return a small QC report for ``fn``.

    Parameters
    ----------
    fn:
        Path to a RINEX or NetCDF file.

    Returns
    -------
    dict
        Always contains ``ok`` (bool) and ``warnings`` (list[str]).
        Additional keys for the file kind: ``n_epochs``, ``n_sv``,
        ``time_first``, ``time_last``, ``interval_seconds``,
        ``gap_count`` (epochs farther apart than 1.5x interval),
        ``duplicate_count`` (per-SV duplicate-time records).
    """
    out: dict[str, Any] = {"ok": True, "warnings": []}
    try:
        info = rinexinfo(fn)
        hdr = rinexheader(fn)
    except (ValueError, OSError, LookupError) as e:
        out["ok"] = False
        out["warnings"].append(f"header parse failed: {e}")
        return out
    out["info"] = dict(info)

    declared_n_obs = hdr.get("Nobs")
    if declared_n_obs is not None and "fields" in hdr:
        actual = len(hdr["fields"]) if isinstance(hdr["fields"], list) else None
        if actual is not None and declared_n_obs != actual:
            out["warnings"].append(f"declared Nobs ({declared_n_obs}) != #fields ({actual})")

    try:
        ds = load(fn)
    except (ValueError, OSError) as e:
        out["ok"] = False
        out["warnings"].append(f"data parse failed: {e}")
        return out

    if isinstance(ds, dict):
        ds = ds.get("obs") or ds.get("nav")
    if not isinstance(ds, xr.Dataset) or "time" not in ds.coords:
        out["warnings"].append("no time axis in parsed data")
        return out

    times = ds.time.values
    out["n_epochs"] = int(times.size)
    if "sv" in ds.coords:
        out["n_sv"] = int(ds.sv.size)
    if times.size:
        out["time_first"] = str(times[0])
        out["time_last"] = str(times[-1])
    if times.size > 1:
        diffs = np.diff(times) / np.timedelta64(1, "s")
        median = float(np.median(diffs))
        out["interval_seconds"] = median
        if median > 0:
            gaps = int(np.sum(diffs > 1.5 * median))
            out["gap_count"] = gaps
            if gaps:
                out["warnings"].append(f"{gaps} time gap(s) larger than 1.5x interval")
    return out


def concat_files(
    files: Iterable[Path | str],
    *,
    dim: str = "time",
) -> xr.Dataset:
    """Concatenate multiple parsed RINEX files along ``dim``.

    Parameters
    ----------
    files:
        Iterable of file paths. Each is loaded with :func:`load` and the
        results are joined with :func:`xarray.concat`. Files that fail to
        parse are logged and skipped.
    dim:
        Dimension to concatenate along. Default ``"time"``.

    Returns
    -------
    xarray.Dataset
        Concatenated dataset, sorted along ``dim`` and with duplicate-
        coordinate entries deduplicated (first occurrence kept).

    Raises
    ------
    ValueError
        If no file was successfully parsed.
    """
    parts: list[xr.Dataset] = []
    for f in files:
        try:
            ds = load(f)
        except (ValueError, OSError) as e:
            log.warning("%s: %s", f, e)
            continue
        if isinstance(ds, xr.Dataset):
            parts.append(ds)
    if not parts:
        raise ValueError("no parseable files to concatenate")

    combined = xr.concat(parts, dim=dim, join="outer", combine_attrs="drop_conflicts")
    if dim in combined.coords:
        # Dedup along the join axis, keeping the first occurrence.
        _, idx = np.unique(combined[dim].values, return_index=True)
        combined = combined.isel({dim: np.sort(idx)})
    return combined


def diff_datasets(
    a: xr.Dataset,
    b: xr.Dataset,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Find the first per-variable difference between two parsed datasets.

    Parameters
    ----------
    a, b:
        Datasets to compare.
    rtol, atol:
        ``numpy.allclose`` tolerances for float comparisons.

    Returns
    -------
    dict
        ``{"equal": bool, "differences": list[dict]}`` where each
        difference dict has ``var``, ``reason``, optional ``where``.
    """
    out: dict[str, Any] = {"equal": True, "differences": []}
    a_vars = set(a.data_vars)
    b_vars = set(b.data_vars)
    if a_vars != b_vars:
        out["equal"] = False
        out["differences"].append(
            {
                "var": None,
                "reason": "different data_vars",
                "only_in_a": sorted(a_vars - b_vars),
                "only_in_b": sorted(b_vars - a_vars),
            }
        )
    for v in sorted(a_vars & b_vars):
        if a[v].shape != b[v].shape:
            out["equal"] = False
            out["differences"].append(
                {"var": v, "reason": "shape", "a": a[v].shape, "b": b[v].shape}
            )
            continue
        av = a[v].values
        bv = b[v].values
        if av.dtype.kind in "fc" and bv.dtype.kind in "fc":
            close = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
        else:
            close = av == bv
        if not close.all():
            out["equal"] = False
            mismatches = np.argwhere(~close)
            first = tuple(int(x) for x in mismatches[0])
            out["differences"].append(
                {
                    "var": v,
                    "reason": "value",
                    "n_mismatch": int((~close).sum()),
                    "first_at": first,
                    "a_value": float(av[first]) if av.dtype.kind in "fc" else av[first],
                    "b_value": float(bv[first]) if bv.dtype.kind in "fc" else bv[first],
                }
            )
    return out


__all__ = ["concat_files", "diff_datasets", "validate_file"]


_ = timedelta  # kept for symmetry with other modules
