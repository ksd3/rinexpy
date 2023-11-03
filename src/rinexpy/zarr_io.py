"""Zarr write helper.

Zarr is a chunked, compressed N-dimensional array format that's
better-suited to cloud workflows than NetCDF (object-store native, no
HDF5 dependency). This is a thin wrapper around :meth:`xarray.Dataset.to_zarr`
that picks sane defaults for RINEX data.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def to_zarr(
    ds: xr.Dataset,
    store: Path | str,
    *,
    mode: str = "w",
    consolidated: bool = True,
) -> Path:
    """Write a parsed RINEX dataset to a Zarr store.

    Parameters
    ----------
    ds:
        The dataset to write.
    store:
        Filesystem path (or any URL Zarr understands) for the store.
    mode:
        Open mode: ``"w"`` (overwrite), ``"w-"`` (fail if exists),
        ``"a"`` (append). Default ``"w"``.
    consolidated:
        Write a consolidated metadata key. Default ``True`` (faster
        first-open for remote stores).

    Returns
    -------
    pathlib.Path
        The (resolved) path of the store.
    """
    if "time" in ds.coords and ds.time.dtype != "datetime64[ns]":
        ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
    p = Path(store).expanduser()
    ds.to_zarr(p, mode=mode, consolidated=consolidated)
    return p


__all__ = ["to_zarr"]
