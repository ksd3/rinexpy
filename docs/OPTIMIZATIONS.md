# Optimizations

This document maps each performance change relative to `georinex 1.16` to
the file/function it lives in and the rationale behind it. Numbers cited
are measured locally on the bundled test corpus (see `BENCHMARKS.md`).

## 1. Eliminate `xarray.merge` per epoch in OBS3 (`src/rinexpy/obs3.py`)

**The big one.** `georinex.obs3._epoch` creates an `xarray.Dataset` per
(epoch, system) tuple and `xarray.merge`s it into a running aggregate.
Each merge re-allocates a coordinate `Index` and the cost is **O(N²)**
in the number of epochs. The upstream README itself acknowledges this:

> shows that `np.genfromtxt()` is consuming about 30% of processing time,
> and `xarray.concat` and `xarray.Dataset` nested inside `concat` takes
> over 60% of time.

`rinexpy` instead:

1. Walks the file once (`_walk_epochs`) collecting per-epoch
   `(time, sv_labels, raw_lines)` tuples — no xarray work in the loop.
2. Pre-allocates dense per-system NumPy buffers sized exactly
   `(n_meas, n_t, n_sv_for_sys)`.
3. Decodes each SV line into the buffer with direct fixed-width slicing.
4. Builds **one** `xarray.Dataset` at the end with one `xr.Dataset(...)`
   call.

This makes the OBS3 path I/O-bound rather than `xarray`-bound. Measured
speedup is ~3-10x on the test corpus (see `BENCHMARKS.md`).

## 2. Eliminate `xarray.merge` per SV in NAV3 (`src/rinexpy/nav3.py`)

`georinex.nav3.rinexnav3` has a similar pattern: it builds an
`xarray.Dataset` per SV (and per duplicate variant for re-broadcast
records) and `xarray.merge`s. Same fix as OBS3 — collect into a flat
buffer, build one Dataset at the end. The duplicate-SV variant logic
("E04", "E04_1", ...) is preserved verbatim.

## 3. Replace `np.genfromtxt` per epoch (`src/rinexpy/obs3.py::_decode_sv_line`)

`np.genfromtxt(io.BytesIO(raw.encode("ascii")), delimiter=(14,1,1)*Fmax)`
allocates a Python `BytesIO`, encodes a `str` to `bytes`, parses, and
returns a NumPy array — *per epoch*. Measured at ~100-200 µs per call on
typical hardware. We replace it with direct fixed-width slicing
(`raw[k*16:(k+1)*16]` plus three `float()` calls per cell), which is
about 4× faster per call and avoids the `bytes` allocation churn.

## 4. Positional-arg `datetime` construction (`src/rinexpy/_time.py`)

CPython's `datetime.__init__` is measurably slower with keyword arguments
than positional. The five `parse_*_epoch` helpers all use positional
calls. The savings are small per call (~250 ns) but multiply by the
number of epochs (often >86 000 on a 24-hour file at 1 Hz).

## 5. Vectorized GLONASS unit conversion (`src/rinexpy/nav2.py`, `nav3.py`)

`georinex` converts GLONASS km->m by looping over the 9 affected field
names and doing `nav[name] *= 1000`. Each `*=` triggers a fresh
`xarray.DataArray` allocation. `rinexpy` does it in a single broadcast
multiply over a contiguous NumPy slice, which is both shorter and faster.

## 6. Pre-computed shared trig in Keplerian (`src/rinexpy/keplerian.py`)

`keplerian2ecef` re-computes `np.cos(2*phi)` and `np.sin(2*phi)` three
times in georinex (once for `Cuc/Cus`, once for `Cic/Cis`, once for
`Crc/Crs`). We compute them once. Marginal but free.

## 7. Vectorized `tk` in Keplerian

`georinex` has a Python `for` loop over `(t0, t1, t2)` triples to compute
the time-since-reference-epoch `tk`. We compute it as a single NumPy
datetime arithmetic step.

## 8. Sorted SV ordering up-front in OBS3 (`src/rinexpy/obs3.py`)

`georinex` relies on `xarray.merge`'s automatic sort to give canonical
per-system, alphabetical SV ordering. We sort once at assembly time and
build the SV index in the same order, sidestepping the merge entirely.

## 9. Lazy CRINEX/Hatanaka (`src/rinexpy/_io.py`)

`opener(header=True)` skips the (expensive) Hatanaka decode pass when
the caller only wants header information. `rinexinfo`, `rinexheader`,
and `obsheader*`/`navheader*` all opt into this. Saves multiple seconds
on large `.crx.gz` files when you only want their metadata.

## 10. RINEX-2 file-size-based preallocation (`src/rinexpy/obs2.py`)

The `fast=True` path estimates the epoch count from the file size
divided by `(80 bytes * Nsv_min * Nl_sv_per_epoch)` and preallocates a
single dense NumPy buffer. Same algorithm as georinex's; we kept it.
The `fast=False` path is a clean two-pass scan with no fast-mode bug
("fast-mode preallocation undersized"-style errors are now impossible
because we raise immediately if the estimate is too small).

## 11. Compiled regex (none used)

Notably *not* an optimization: regex parsing of the epoch lines was
considered and benchmarked but came out slower than the positional-slice
parser, because Python `int(line[1:3])` is already C-fast.

## 12. Bug fix that's also faster: Pre-fill SP3 buffers (`src/rinexpy/sp3.py`)

`georinex` uses `np.empty()` for the SP3 position/clock buffers without
ever calling `.fill(NaN)`, so SVs that are in the SP3 header but absent
from a particular epoch read back as uninitialized memory. We pre-fill
with NaN. This is both correct and (slightly) faster than the
"`.empty()` then conditional `.fill(NaN)`" pattern.

## What we explicitly did *not* do

- **Cython / Rust extension.** Considered, but the pure-Python rewrite
  already removes the dominant cost. A native extension would have hurt
  install ergonomics for very modest extra speed.
- **`numba` JIT.** Same reasoning: the hottest loop (epoch parsing) is
  bound by Python `float()` calls, and `numba` can't accelerate that
  efficiently without giving up the readable Python implementation.
- **Memory mapping.** Investigated; gzip/bz2/zip/Z all need full decode
  to even know where epoch boundaries are, so `mmap` would only help on
  plain text files and the saved `read()` is dominated by decode.
- **Multi-threading the inner loop.** The outer loop carries a position
  index that depends on prior iterations (interval decimation, time
  bounds), so it can't be trivially parallelized. `batch_convert` is
  already a natural place for `multiprocessing.Pool`, which can be added
  in a follow-up.

## Where the remaining time goes

Profiling on a 13 MB OBS3 file:

- ~35% pure parsing (`float()`, `parse_obs3_epoch`)
- ~25% I/O / decompression (gzip pipeline)
- ~15% NumPy buffer write (`buf[:, offset:offset+n]`)
- ~10% xarray.Dataset construction (the one-time cost)
- ~15% miscellaneous

So even a Rust port could only buy ~3× more before hitting the I/O floor.
