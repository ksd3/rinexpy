# Optimizations

The main reason `rinexpy` exists is that upstream `georinex 1.16` was slow
on RINEX 3 NAV and OBS files because of an O(N²) merge pattern. Several
other optimisations followed once that one was unblocked. This page maps
every change to the file it is in and explains why it is there.
Numbers are from the local benchmark run; see
[Benchmarks](benchmarks.md) for the full corpus.

## 1. Drop `xarray.merge` per epoch in OBS3

**File:** `src/rinexpy/obs3.py`. **main speedup:** 13-18x on RINEX 3 OBS.

`georinex.obs3._epoch` builds an `xarray.Dataset` per `(epoch, system)`
and merges it into a running aggregate with `xarray.merge`. Each merge
re-allocates the coordinate index, so the cost is O(N²) in epoch
count. The upstream README acknowledges this:

> shows that `np.genfromtxt()` is consuming about 30% of processing
> time, and `xarray.concat` and `xarray.Dataset` nested inside `concat`
> takes over 60% of time.

`rinexpy` does it differently:

1. Walk the file once (`_walk_epochs`) collecting per-epoch
 `(time, sv_labels, raw_lines)` tuples. No xarray work in the loop.
2. Pre-allocate dense per-system NumPy buffers sized exactly
 `(n_meas, n_t, n_sv_for_sys)`.
3. Decode each SV line into the buffer with direct fixed-width slicing.
4. Build one `xarray.Dataset` at the end.

Result: OBS3 is now I/O-bound, not xarray-bound.

## 2. Drop `xarray.merge` per SV in NAV3

**File:** `src/rinexpy/nav3.py`. **main speedup:** 30-33x on RINEX 3 NAV.

Same pattern as upstream OBS3: an `xarray.Dataset` per SV (and per
duplicate variant for re-broadcast records) merged into a running
aggregate. Same fix: collect into a flat buffer, build one Dataset at
the end. The duplicate-SV variant naming ("E04", "E04_1", ...) is
preserved verbatim so that downstream code that depends on the
upstream convention keeps working.

## 3. Replace `np.genfromtxt` per epoch

**File:** `src/rinexpy/obs3.py::_decode_sv_line`.

`np.genfromtxt(io.BytesIO(raw.encode("ascii")), delimiter=(14,1,1)*Fmax)`
allocates a Python `BytesIO`, encodes a `str` to `bytes`, parses, and
returns a NumPy array, once per epoch per SV. Measured at 100-200 µs
per call on typical hardware. Replaced with direct fixed-width slicing
(`raw[k*16:(k+1)*16]` plus three `float()` calls per cell). About 4x
faster per call and no `bytes` allocation churn.

## 4. Positional `datetime` construction

**File:** `src/rinexpy/_time.py`.

CPython's `datetime.__init__` is measurably slower with keyword
arguments than positional. The five `parse_*_epoch` helpers all use
positional calls. About 250 ns per call, which adds up over 86 000+
epochs on a 24-hour 1 Hz file.

## 5. Vectorised GLONASS unit conversion

**Files:** `src/rinexpy/nav2.py`, `nav3.py`.

`georinex` converts GLONASS km → m by looping over the 9 affected field
names and doing `nav[name] *= 1000`. Each `*=` allocates a fresh
`xarray.DataArray`. `rinexpy` does it in a single broadcast multiply over a
contiguous NumPy slice.

## 6. Pre-computed shared trig in Keplerian

**File:** `src/rinexpy/keplerian.py`.

`keplerian2ecef` re-computes `np.cos(2*phi)` and `np.sin(2*phi)` three
times in georinex (once each for `Cuc`/`Cus`, `Cic`/`Cis`, `Crc`/`Crs`).
`rinexpy` computes them once. Marginal speedup, free correctness.

## 7. Vectorised `tk` in Keplerian

**File:** `src/rinexpy/keplerian.py`.

`georinex` has a Python `for` loop over `(t0, t1, t2)` triples to
compute the time-since-reference-epoch `tk`. `rinexpy` does it as a
single NumPy datetime arithmetic step.

## 8. Sorted SV ordering up front

**File:** `src/rinexpy/obs3.py`.

`georinex` relies on `xarray.merge`'s automatic sort for canonical
per-system, alphabetical SV ordering. `rinexpy` sorts once at assembly
time and builds the SV index in that order, sidestepping the merge.

## 9. Lazy CRINEX / Hatanaka

**File:** `src/rinexpy/_io.py`.

`opener(header=True)` skips the (expensive) Hatanaka decode pass when
the caller only wants header information. `rinexinfo`, `rinexheader`,
and the `obsheader*` / `navheader*` family all opt into this. Saves
several seconds on large `.crx.gz` files when you only need metadata.

## 10. RINEX 2 file-size-based preallocation

**File:** `src/rinexpy/obs2.py`.

The `fast=True` path estimates the epoch count from the file size
divided by `(80 bytes * Nsv_min * Nl_sv_per_epoch)` and preallocates a
single dense NumPy buffer. Same algorithm as georinex's, kept as is.
The `fast=False` path is a clean two-pass scan: if the estimate is too
small, `rinexpy` raises immediately instead of the "fast-mode
preallocation undersized" silent overrun.

## 11. Compiled regex (not used)

Not actually an optimisation: regex parsing of the epoch lines was
benchmarked but came out slower than the positional-slice parser,
because Python's `int(line[1:3])` is already C-fast.

## 12. Pre-fill SP3 buffers (bug fix that is also faster)

**File:** `src/rinexpy/sp3.py`.

`georinex` uses `np.empty()` for the SP3 position / clock buffers
without ever calling `.fill(NaN)`, so SVs that are in the SP3 header
but absent from a particular epoch read back as uninitialised memory.
`rinexpy` pre-fills with NaN. Correct, and slightly faster than
`.empty()` plus a conditional `.fill(NaN)`.

## 13. Optional numba JIT for OBS3

**File:** `src/rinexpy/_jit.py`. **Opt-in.**

When `RINEXPY_USE_JIT=1` is set or `use_jit=True` is passed to
`rinexobs3`, the inner observation-decoding loop runs through a
numba-jitted kernel. Pre-decoded NumPy buffers avoid the Python
`float()` overhead.

About 1.9x end-to-end speedup on a 23-hour 15-second OBS3 file. The
trade-off is a one-shot JIT compile cost (~1 second on first call) and
the extra `numba` + `llvmlite` dependencies, which is why this is
opt-in rather than default.

## 14. Optional C++17 extension

**Package:** `rinexpy-native`. **Files:** `native/src/*.cpp`.

A C++17 extension that is under `native/` in the repository. The
extension provides two things:

1. A CRINEX 1 / CRINEX 3 decoder that matches the upstream `hatanaka`
 package byte-for-byte and is several times faster.
2. An in-place RINEX 3 OBS decoder that drops the parse time of a
 24-hour 30-second file from about 70 ms to about 40 ms.

When `[native]` is installed (via `uv sync --extra native`), `rinexpy`
uses it automatically. The C++ path matches the JIT path in
end-to-end wall-clock and removes the `numba` + `llvmlite` dependency
tree, so it is the recommended high-performance path.

## 15. Int-ns datetime path

**File:** `src/rinexpy/_time.py`.

Bulk `np.asarray(int_list).view('datetime64[ns]')` is about 40x faster
than `np.array(list_of_datetime, dtype='datetime64[ns]')` on the
bulk-conversion step. The parser builds a list of integer nanoseconds
during the walk and converts in one shot at assemble time.

## 16. Memory-mapped reader for plain-text files

**File:** `src/rinexpy/_io.py`.

For uncompressed RINEX files larger than 50 MB, the opener uses `mmap`
instead of `open()` + `read()`. The OS page cache then handles the
working set, which means the second `load` on the same file is nearly
free.

## What was left out

**Cython.** Considered. The pure-Python rewrite already drops the
dominant cost. A Cython extension would hurt install usability for
modest extra speed.

**Memory-mapping compressed files.** Investigated. gzip, bz2, zip, and
Z all need full decode to find epoch boundaries, so `mmap` would only
help on plain-text files where the saved `read()` is dominated by
decode anyway.

**Multi-threading the inner loop.** The outer loop carries a position
index that depends on prior iterations (interval decimation, time
bounds), so it cannot be trivially parallelised. `batch_convert` is the
natural place for `multiprocessing.Pool`, which was added.

**Per-SV thread pool inside OBS3.** Tried. The Python-side bookkeeping
overhead dominates the per-SV decode cost; even with the GIL released
in the inner loop, the threading overhead beats the parallelism.

## Where the remaining time goes

Profiling on a 13 MB OBS3 file:

- About 35% parsing (`float()`, `parse_obs3_epoch`).
- About 25% I/O and decompression (gzip pipeline).
- About 15% NumPy buffer writes (`buf[:, offset:offset+n]`).
- About 10% xarray.Dataset construction (one-time).
- About 15% miscellaneous.

A Rust port could buy maybe 3x more before hitting the I/O floor on
gzip-compressed files. For uncompressed files the I/O floor is even
closer; the marginal value of a native rewrite is small.

## Related pages

- [Benchmarks](benchmarks.md): measured numbers.
- [Architecture](architecture.md): how the layers compose.
