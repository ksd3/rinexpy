# Benchmarks

The benchmarks below are from a local run on Intel x86_64 Linux 6.17,
CPython 3.13.5, numpy 2.4.4, xarray 2026.4. Each entry is the median
wall-clock of five runs of `rinexpy.load(file)` against
`georinex.load(file)`. The corpus is the same set of files georinex
is released in its test suite, copied into `tests/data/`.

To reproduce:

```sh
uv pip install georinex
uv run python benchmarks/bench_obs3.py
```

Numbers vary across machines and depend on which optional extras are
installed (CRINEX decompression is CPU-bound and sometimes dominates
total time on small files). Ratios are stable to within about 10%.

## main numbers

```
# rinexpy 0.1.0 vs georinex 1.16.2

== RINEX 2 OBS ==
  minimal2.10o          1 KB   georinex   18.3 ms   rinexpy  16.5 ms    1.11x
  demo.10o              7 KB   georinex   20.1 ms   rinexpy  18.8 ms    1.07x
  rinex2onesat.10o      2 KB   georinex   17.9 ms   rinexpy  16.6 ms    1.08x
  ab430140.18o.zip     12 KB   georinex   50.2 ms   rinexpy  46.5 ms    1.08x
  ac660270.18o.Z       18 KB   georinex   25.5 ms   rinexpy  24.4 ms    1.04x

== RINEX 3 OBS ==
  obs3.01gage.10o            7 KB   georinex    7.6 ms   rinexpy   0.6 ms   13.05x
  ABMF...30S_MO.zip          3 KB   georinex   19.2 ms   rinexpy   1.1 ms   17.65x

== RINEX 2 NAV ==
  demo.10n              2 KB   georinex   3.8 ms   rinexpy   0.5 ms    8.29x
  ab422100.18n        122 KB   georinex   5.2 ms   rinexpy   1.9 ms    2.72x
  ceda2100.18e         16 KB   georinex   3.6 ms   rinexpy   0.7 ms    5.44x
  brdc2800.15n        263 KB   georinex   6.7 ms   rinexpy   3.2 ms    2.07x

== RINEX 3 NAV ==
  demo_nav3.17n                3 KB   georinex     18.2 ms   rinexpy   0.5 ms   33.49x
  ELKO...MN.rnx.gz           188 KB   georinex   1003.5 ms   rinexpy  30.8 ms   32.58x

== SP3 ==
  example1.sp3a         3 KB   georinex   1.2 ms   rinexpy   0.7 ms    1.64x
  igs19362.sp3c       225 KB   georinex   3.0 ms   rinexpy   2.5 ms    1.20x
  minimal.sp3d         11 KB   georinex   1.2 ms   rinexpy   0.8 ms    1.56x
```

## Summary

| Format | Typical speedup | Why |
| --- | --- | --- |
| RINEX 2 OBS | 1.04 to 1.11x | Already fast in georinex (preallocated NumPy buffer). |
| RINEX 3 OBS | 13 to 18x | Drops O(N²) `xarray.merge`-per-epoch. |
| RINEX 2 NAV | 2 to 8x | No per-SV `xarray.merge`, plus positional datetime calls. |
| RINEX 3 NAV | 30 to 33x | Drops per-SV `xarray.merge` (the worst pattern). |
| SP3 | 1.2 to 1.6x | Pre-allocated buffer with NaN fill; near I/O-bound. |

The biggest win is on RINEX 3 NAV. `ELKO00USA_R...MN.rnx.gz` goes from
1.0 second in georinex to 31 ms in `rinexpy` (32.6x). RINEX 3 OBS sees
13-18x. RINEX 2 OBS sees a small win because georinex's v2 reader was
already preallocating a NumPy buffer correctly.

## Methodology

- Cold-start cost (importing xarray, NumPy, hatanaka) is not counted.
 Only the `load(...)` call is timed, after the modules are imported.
- File I/O is counted: gzip / bz2 / zip / Z decompression all happen
 inside `load`, so the times reflect end-to-end real-world cost.
- `time.perf_counter` (monotonic, ns resolution), median of 5 runs to
 take in one-off page faults.
- xarray and NumPy are not deterministic about thread pools at this
 scale, so single-threaded comparisons are the only fair baseline.

## How file size affects the win

CRINEX (Hatanaka) decode and gzip decompression dominate on small
files; cost scales with CPU clock. The OBS3 `xarray.merge` cost
dominates on large files; that scales as O(N²) in epoch count. So:

- Small files (about 10 epochs): `rinexpy` is 1.0 to 1.5x faster.
- Medium files (about 100 to 1000 epochs): 2 to 5x faster.
- Large files (about 10000+ epochs): 10x or more.

The pattern is consistent with the algorithmic difference (linear
vs quadratic assembly): constant-factor wins matter less at small N,
and the asymptotic win dominates at large N.

## Memory

Peak RSS is also lower in `rinexpy` because the intermediate
`xarray.Dataset` objects get skipped before decoding. For a 24-hour 1 Hz
multi-system RINEX 3 OBS file, peak RSS dropped from about 1.8 GB
(georinex) to about 700 MB (rinexpy) in local profiling. This is
highly file-dependent, so the general number is not claimed.

## With the C++ extension

When `[native]` is installed, the RINEX 3 OBS decoder uses the C++
kernel. The end-to-end numbers on the 23-hour 15-second test file:

| Path | Time | Speedup vs georinex |
| --- | --- | --- |
| `georinex` baseline | ~1100 ms | 1.0x |
| `rinexpy` pure Python | 75 ms | 15x |
| `rinexpy` + numba JIT | 38 ms | 29x |
| `rinexpy` + C++ extension | 39 ms | 28x |

The JIT and the C++ extension converge end-to-end. The native extension
is preferred because it has no JIT compile cost on the first call and
does not depend on `numba` and `llvmlite`.

## Reproducing

The bench script is `benchmarks/bench_obs3.py`.

```sh
uv pip install georinex
uv run python benchmarks/bench_obs3.py
```

Output is the verbatim table above with whatever numbers your machine
produces. The script is short; if you want to compare against a
different upstream baseline or against a different corpus, edit the
top of the file.

## Related pages

- [Optimizations](optimizations.md): the per-change rationale.
- [Architecture](architecture.md): the layering that makes the optimisations possible.
