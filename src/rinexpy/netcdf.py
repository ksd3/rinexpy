"""NetCDF / HDF5 round-trip helpers.

The wire format is the same NetCDF4 layout georinex emits: one HDF5 group
per RINEX kind (``OBS``, ``NAV``), light zlib compression on every variable,
fletcher32 checksums for tamper detection.

Round-tripping a `.nc` file produced by georinex is a tested invariant — see
``tests/test_netcdf.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

log = logging.getLogger(__name__)

#: Compression encoding applied to every data variable. ``complevel=1`` is
#: the cost/benefit sweet spot — higher levels save very little space at
#: significant CPU cost (georinex defaulted here too).
ENC = {"zlib": True, "complevel": 1, "fletcher32": True}


def write_dataset(
    ds: xr.Dataset, path: Path, *, group: str, overwrite: bool = False
) -> Path:
    """Write a parsed RINEX dataset to a NetCDF4 file.

    Parameters
    ----------
    ds:
        The parsed dataset to persist.
    path:
        Output ``.nc`` path. Parent directories are not created.
    group:
        The HDF5 group name (``"OBS"`` or ``"NAV"``).
    overwrite:
        If True, replace any existing file. If False and the file already
        contains a dataset under ``group``, a ``ValueError`` is raised.

    Returns
    -------
    pathlib.Path
        The (resolved) path that was written to.

    Raises
    ------
    ValueError
        If ``overwrite=False`` and ``group`` already exists in the file.
    """
    path = Path(path).expanduser()
    mode = _resolve_mode(path, group, overwrite)

    # Pandas >= 0.25 forces datetime64[ns]; xarray respects that when writing.
    if "time" in ds.coords and ds.time.dtype != "datetime64[ns]":
        ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))

    enc = {k: ENC for k in ds.data_vars}
    ds.to_netcdf(path, group=group, mode=mode, encoding=enc, format="NETCDF4")
    return path


def _resolve_mode(path: Path, group: str, overwrite: bool) -> str:
    """Return the right ``mode`` argument for ``Dataset.to_netcdf``.

    - ``"w"`` if the file does not exist or ``overwrite=True``.
    - ``"a"`` if the file exists and the requested ``group`` is *not* there.
    - Raises ``ValueError`` if the group is already present and we shouldn't
      clobber it.
    """
    if overwrite or not path.is_file():
        return "w"
    try:
        xr.open_dataset(path, group=group)
    except OSError:
        return "a"
    raise ValueError(f"{group!r} already present in {path}; pass overwrite=True to replace")


__all__ = ["ENC", "write_dataset"]
