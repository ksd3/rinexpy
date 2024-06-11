# Optimizations

Each performance change relative to `georinex 1.16`, mapped to the
file and function it lives in and why it's there. Numbers are
measured locally on the bundled test corpus (see `BENCHMARKS.md`).

## 1. Drop `xarray.merge` per epoch in OBS3 (`src/rinexpy/obs3.py`)

The big win. `georinex.obs3._epoch` builds an `xarray.Dataset` per
`(epoch, system)` and `xarray.merge`s it into a running aggregate.
Each merge re-allocates the coordinate index, so the cost is O(N²)
in epoch count. The upstream README acknowledges it:

> shows that `np.genfromtxt()` is consuming about 30% of processing
> time, and `xarray.concat` and `xarray.Dataset` nested inside
> `concat` takes over 60% of time.

rinexpy does it differently:

1. Walk the file once (`_walk_epochs`) collecting per-epoch
   `(time, sv_labels, raw_lines)` tuples. No xarray work in the loop.
2. Pre-allocate dense per-system NumPy buffers sized exactly
   `(n_meas, n_t, n_sv_for_sys)`.
3. Decode each SV line into the buffer with direct fixed-width
   slicing.
4. Build one `xarray.Dataset` at the end.

Result: OBS3 is now I/O-bound, not xarray-bound. ~3-10× on the test
corpus (`BENCHMARKS.md`).

## 2. Drop `xarray.merge` per SV in NAV3 (`src/rinexpy/nav3.py`)

Same pattern as OBS3 in upstream: an `xarray.Dataset` per SV (and
per duplicate variant for re-broadcast records) merged into a
running aggregate. Same fix: collect into a flat buffer, build one
Dataset at the end. The duplicate-SV variant naming ("E04", "E04_1",
...) is preserved verbatim.

## 3. Replace `np.genfromtxt` per epoch (`src/rinexpy/obs3.py::_decode_sv_line`)

`np.genfromtxt(io.BytesIO(raw.encode("ascii")), delimiter=(14,1,1)*Fmax)`
allocates a Python `BytesIO`, encodes a `str` to `bytes`, parses,
and returns a NumPy array, once per epoch. Measured at 100-200 µs
per call on typical hardware. Replaced with direct fixed-width
slicing (`raw[k*16:(k+1)*16]` plus three `float()` calls per cell).
About 4× faster per call and no `bytes` allocation churn.

## 4. Positional `datetime` construction (`src/rinexpy/_time.py`)

CPython's `datetime.__init__` is measurably slower with keyword
arguments than positional. The five `parse_*_epoch` helpers all use
positional calls. ~250 ns per call, which adds up over 86,000+
epochs on a 24-hour 1 Hz file.

## 5. Vectorized GLONASS unit conversion (`src/rinexpy/nav2.py`, `nav3.py`)

`georinex` converts GLONASS km→m by looping over the 9 affected
field names and doing `nav[name] *= 1000`. Each `*=` allocates a
fresh `xarray.DataArray`. rinexpy does it in a single broadcast
multiply over a contiguous NumPy slice.

## 6. Pre-computed shared trig in Keplerian (`src/rinexpy/keplerian.py`)

`keplerian2ecef` re-computes `np.cos(2*phi)` and `np.sin(2*phi)`
three times in georinex (once each for `Cuc/Cus`, `Cic/Cis`,
`Crc/Crs`). rinexpy computes them once. Marginal but free.

## 7. Vectorized `tk` in Keplerian

`georinex` has a Python `for` loop over `(t0, t1, t2)` triples to
compute the time-since-reference-epoch `tk`. rinexpy does it as a
single NumPy datetime arithmetic step.

## 8. Sorted SV ordering up front (`src/rinexpy/obs3.py`)

`georinex` relies on `xarray.merge`'s automatic sort for canonical
per-system, alphabetical SV ordering. rinexpy sorts once at assembly
time and builds the SV index in that order, sidestepping the merge.

## 9. Lazy CRINEX/Hatanaka (`src/rinexpy/_io.py`)

`opener(header=True)` skips the (expensive) Hatanaka decode pass
when the caller only wants header information. `rinexinfo`,
`rinexheader`, and the `obsheader*`/`navheader*` family all opt
into this. Saves several seconds on large `.crx.gz` files when
you only need metadata.

## 10. RINEX-2 file-size-based preallocation (`src/rinexpy/obs2.py`)

The `fast=True` path estimates the epoch count from the file size
divided by `(80 bytes * Nsv_min * Nl_sv_per_epoch)` and preallocates
a single dense NumPy buffer. Same algorithm as georinex's, kept as
is. The `fast=False` path is a clean two-pass scan: if the estimate
is too small, rinexpy raises immediately instead of the
"fast-mode preallocation undersized" silent overrun.

## 11. Compiled regex (none used)

Not actually an optimization: regex parsing of the epoch lines was
benchmarked but came out slower than the positional-slice parser,
because Python's `int(line[1:3])` is already C-fast.

## 12. Bug fix that's also faster: pre-fill SP3 buffers (`src/rinexpy/sp3.py`)

`georinex` uses `np.empty()` for the SP3 position/clock buffers
without ever calling `.fill(NaN)`, so SVs that are in the SP3 header
but absent from a particular epoch read back as uninitialized
memory. rinexpy pre-fills with NaN. Correct, and (slightly) faster
than `.empty()` plus a conditional `.fill(NaN)`.

## What we didn't do

- **Cython or Rust extension.** Considered. The pure-Python rewrite
  already drops the dominant cost. A native extension would hurt
  install ergonomics for modest extra speed.
- **`numba` JIT.** Same reasoning. The hot loop (epoch parsing) is
  bound by Python `float()` calls, which numba can't accelerate
  without giving up the readable Python implementation.
- **Memory mapping.** Investigated. gzip, bz2, zip, and Z all need
  full decode to find epoch boundaries, so `mmap` would only help
  on plain text files where the saved `read()` is dominated by
  decode anyway.
- **Multi-threading the inner loop.** The outer loop carries a
  position index that depends on prior iterations (interval
  decimation, time bounds), so it can't be trivially parallelized.
  `batch_convert` is a natural place for `multiprocessing.Pool`,
  which can land as a follow-up.

## Where the remaining time goes

Profiling on a 13 MB OBS3 file:

- ~35% parsing (`float()`, `parse_obs3_epoch`)
- ~25% I/O and decompression (gzip pipeline)
- ~15% NumPy buffer writes (`buf[:, offset:offset+n]`)
- ~10% xarray.Dataset construction (one-time)
- ~15% miscellaneous

A Rust port could only buy maybe 3× more before hitting the I/O
floor.
