# Async loading

For `asyncio`-based applications, `rinexpy` includes thread-pool wrappers
around the synchronous loaders. They live in `rinexpy.asyncio`.

The wrappers do not parallelise the parsing itself; the GIL is released
inside the NumPy / xarray inner loops, so concurrent loads on a single
process see a small speedup, but the bulk of the cost is still Python.
For genuine CPU parallelism, use `batch_convert(workers=N)` which spawns
a `multiprocessing.Pool`.

## One-file load

```python
import asyncio
from rinexpy.asyncio import aload

async def main():
    ds = await aload("tests/data/demo.10o")
    print(ds.sv.size, "SVs")

asyncio.run(main())
```

`aload` accepts every keyword argument that `load` does. The work runs in
the default thread-pool executor; the caller's coroutine sees the result
when the parse finishes.

## Concurrent multi-file load

```python
import asyncio
from rinexpy.asyncio import aload_many

async def main():
    results = await aload_many([
        "tests/data/demo.10o",
        "tests/data/obs3.01gage.10o",
        "tests/data/brdc2800.15n",
    ])
    for r in results:
        if isinstance(r, Exception):
            print("error:", r)
        else:
            print(r.sv.size, "SVs")

asyncio.run(main())
```

`aload_many` runs one parse per file concurrently. The result is a list
of `xarray.Dataset` objects in the input order, with `Exception`
instances in place of any file that errored.

For most workflows the synchronous `load` + `multiprocessing.Pool` is
cleaner. The `asyncio` wrappers are for cases that already live in an
asyncio event loop (e.g. an aiohttp web server that needs to parse user
uploads on the fly).

## NTRIP astream

The NTRIP module exports an async byte stream alongside the synchronous
`stream`. Same arguments, same byte chunks; the difference is that
`astream` yields awaitably.

```python
import asyncio
from rinexpy.ntrip import astream

async def main():
    n = 0
    async for chunk in astream(
        "rtk2go.com", "MOUNT01",
        user="me", password="x", port=2101,
    ):
        n += len(chunk)
        if n > 8192:
            break
    print(f"received {n} bytes")

asyncio.run(main())
```

`afetch_sourcetable` is the async equivalent of `fetch_sourcetable`.

## Performance notes

The thread-pool wrappers do not give CPU parallelism. They give two
benefits.

**Concurrency with I/O.** A coroutine awaiting `aload` can do other work
(an HTTP request, a database write) while the parse runs.

**Backpressure.** With a bounded thread pool, you cannot saturate the
machine with parses; `asyncio` queues the extras.

For genuine CPU parallelism, the multiprocessing path is the right
answer:

```python
import rinexpy as rp

written = rp.batch_convert("data/", "*.rnx.gz", "out/", workers=0)
```

This forks a process per CPU and parses the inputs in parallel without
the GIL.

## Related pages

- [Multi-file tools](multi-file.md): the synchronous batch path.
- [RTCM and NTRIP](../formats/rtcm.md): the synchronous `stream`.
- [Streaming over RAM-sized files](streaming.md): the per-epoch iterator.
