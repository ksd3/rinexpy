"""Dask-backed lazy multi-file reader.

For multi-day RINEX archives, ``dask.array``-backed datasets let you
slice and aggregate without loading everything into RAM. This module
loads each file into a regular ``xarray.Dataset`` and then chunks it
along the time axis.

If ``dask`` is not installed, ``load_lazy`` falls back to the eager
:func:`rinexpy.tools.concat_files` and emits a warning.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import xarray as xr

from .tools import concat_files

log = logging.getLogger(__name__)


def load_lazy(
    files: Iterable[Path | str],
    *,
    chunk_size: int | dict | None = None,
) -> xr.Dataset:
    """Load multiple files into a chunked, dask-backed ``xarray.Dataset``.

    Parameters
    ----------
    files:
        Iterable of file paths.
    chunk_size:
        Either an int (chunk size along the time axis) or a dict
        (passed straight to :meth:`xarray.Dataset.chunk`). Default
        ``{"time": 1000}``.

    Returns
    -------
    xarray.Dataset
        Either a dask-backed dataset (if dask is installed) or a regular
        in-memory one.
    """
    try:
        import dask  # noqa: F401
    except ImportError:
        log.warning("dask not installed; load_lazy returns eager dataset")
        return concat_files(files)

    eager = concat_files(files)
    if chunk_size is None:
        chunks: dict[str, int] = {"time": 1000}
    elif isinstance(chunk_size, int):
        chunks = {"time": chunk_size}
    else:
        chunks = chunk_size
    return eager.chunk(chunks)


__all__ = ["load_lazy"]
