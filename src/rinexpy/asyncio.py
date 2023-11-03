"""asyncio-friendly variants of the synchronous loaders.

Each ``aload``/``aload_many`` runs the synchronous reader in a thread
pool, so it composes with other ``async`` workflows without blocking the
event loop. There's no actual concurrency speedup over a thread per file
because the readers are CPU-bound, but the API is convenient when the
calling app is already async (web servers, agents, etc.).

For CPU-parallel batch reads, prefer
:func:`rinexpy.batch_convert(..., workers=N)` which uses
``multiprocessing``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .api import load
from .gpstime import GPS_EPOCH

_ = GPS_EPOCH  # silence unused-import warnings if anyone reformats


async def aload(fn: Path | str, **kwargs: Any):
    """Async wrapper around :func:`rinexpy.load`.

    Runs in the default executor (a ``concurrent.futures.ThreadPoolExecutor``).
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: load(fn, **kwargs))


async def aload_many(files: Iterable[Path | str], **kwargs: Any) -> list:
    """Concurrently load multiple files (one task per file).

    Returns the parsed datasets in input order, with exceptions
    propagated as ``Exception`` instances inside the result list (so a
    bad file doesn't kill the whole batch).
    """
    tasks = [asyncio.create_task(aload(f, **kwargs)) for f in files]
    return await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["aload", "aload_many"]
