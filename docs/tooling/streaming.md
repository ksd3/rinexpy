# Streaming over RAM-sized files

A 24-hour, 1 Hz, multi-constellation RINEX 3 OBS file can be several
hundred megabytes uncompressed. A full week of these does not fit in
memory on most machines. The streaming iterator yields one epoch at a
time, with constant memory usage in the file size.

The iterator is in `rinexpy.streaming.iter_obs3_epochs` and is also
re-exported as `rinexpy.iter_obs3_epochs`.

## The signature

```python
def iter_obs3_epochs(
    fn,                                  # path, stream, or str
    *,
    use=None,                            # set[str] of system letters
    tlim=None,                           # (start, end) datetime tuple
    interval=None,                       # seconds or timedelta
) -> Iterator[tuple[datetime, xr.Dataset]]:
```

It yields `(datetime, xarray.Dataset)` for each epoch in the file. Each
yielded Dataset holds only the SVs present at that epoch.

## Basic use

```python
import rinexpy as rp

for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(time, ds)
```

The body of the loop sees one epoch's worth of data at a time. For a 24-h
file at 1 Hz, that is 86 400 iterations. Memory usage stays roughly
constant at a few megabytes (the size of one epoch's NumPy buffers plus
the xarray overhead).

## Filtering the stream

The same filters that `load` honours work here.

```python
for time, ds in rp.iter_obs3_epochs(
    "huge.rnx.gz",
    use={"G", "E"},                            # only GPS and Galileo
    tlim=("2024-03-14T00:00", "2024-03-14T01:00"),
    interval=30,
):
    print(time, ds.sv.size)
```

`use=` reduces the SV set per epoch. `tlim=` causes the iterator to
short-circuit when the file's epochs move past `end`. `interval=` decimates
the stream by an integer factor of the file's nominal sampling rate.

The iterator does the filtering inside the parser, so records outside
the filter are skipped before they ever turn into Python floats. The
extra cost of a filter is negligible.

## Use cases

### Multi-day file conversion

Convert a multi-day file to NetCDF without ever holding the whole thing
in memory. Build the dataset epoch by epoch.

```python
import xarray as xr

streams = []
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz", use={"G"}):
    streams.append(ds.expand_dims(time=[time]))
full = xr.concat(streams, dim="time")
full.to_netcdf("huge.nc")
```

In practice, prefer `batch_convert` to convert per-file. The streaming
pattern is for cases where you need per-epoch processing logic that
cannot be combined into a single load call.

### Per-epoch QC

For a long observation session, you might want to scan for cycle slips
and multipath spikes incrementally rather than loading the whole session.

```python
from rinexpy.qc import detect_slips_phase_only
import numpy as np

# Track per-SV phase history with a 5-epoch window.
phase_history = {}

for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    for sv in ds.sv.values:
        l1 = float(ds.L1C.sel(sv=sv).values)
        if np.isnan(l1):
            continue
        history = phase_history.setdefault(sv, [])
        history.append(l1)
        if len(history) > 5:
            history.pop(0)
        if len(history) == 5:
            slips = detect_slips_phase_only(np.array(history))
            if slips[-1]:
                print(f"slip at {time} on {sv}")
```

### Real-time observation feed

When the OBS file is being written by a logger and you want to consume
it as it grows, point the iterator at the file. The iterator does not
follow the tail automatically; you would typically combine
`iter_obs3_epochs` with `time.sleep` and a re-open on EOF.

```python
import time as time_module
last_epoch = None

while True:
    last_seen = None
    for t, ds in rp.iter_obs3_epochs("growing.rnx"):
        if last_epoch is not None and t <= last_epoch:
            continue
        last_seen = t
        process_one_epoch(t, ds)
    if last_seen:
        last_epoch = last_seen
    time_module.sleep(10)
```

For genuine real-time observation, an NTRIP or serial-port reader is the
right answer; the file-tail pattern is for cases where you control the
producer.

## memory usage

The per-epoch Dataset has:

- One coord (`sv`, length = SVs at this epoch, typically 8-20).
- Per measurement label, one float64 array of length `len(sv)`.

For a typical RINEX 3 file with 8 measurement labels and 12 SVs per
epoch, one epoch is roughly 1.5 KB of data plus the xarray bookkeeping
(another 1-2 KB). With Python's GC the actual resident set grows a bit
above that, but the iterator never carries more than a few epochs at
once.

## Performance

The streaming reader is a few percent slower per epoch than the
materialising reader, because it pays the per-epoch Dataset construction
cost on every yield instead of batching. For 86 400 epochs that adds up
to roughly 200 ms of extra cost.

The trade-off is worth it any time the file exceeds available memory.

## RINEX 2 OBS

The streaming iterator only supports RINEX 3 OBS. RINEX 2 files load
into memory through `load` because the RINEX 2 grammar is harder to
stream (the per-epoch SV count is read first, then a fixed number of
lines per SV, with no length prefix).

## Related pages

- [RINEX observation files](../formats/rinex-obs.md): the materialising reader.
- [NetCDF and Zarr output](io.md): persisted forms.
- [Multi-file tools](multi-file.md): `concat_files`, `validate_file`, `diff_datasets`.
- [Async loading](async.md): the asyncio wrappers.
