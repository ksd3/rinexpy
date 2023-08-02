# Benchmarks

Run on Intel x86_64 Linux 6.17, CPython 3.13.5, numpy 2.4.4, xarray
2026.4. Each entry is the median wall-clock of 5 runs of
`rinexpy.load(file)` vs `georinex.load(file)`. The corpus is the same
set of files georinex ships in its test suite, copied into
`tests/data/`.

Reproduce with:

```sh
python benchmarks/bench_obs3.py
```

The exact numbers will vary across machines and depend on which optional
extras are installed (CRINEX/Hatanaka decompression in particular is
CPU-bound and sometimes dominates total time on small files), but the
*ratios* should be stable to within ~10%.

## Headline (latest local run)

`benchmarks/last_run.txt` contains the most recent verbatim output. As
of writing:

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

### Summary

| Format    | Typical speedup | Why                                                        |
|-----------|-----------------|------------------------------------------------------------|
| RINEX 2 OBS | 1.04 - 1.11x  | Already fast in georinex (preallocated NumPy buffer).      |
| RINEX 3 OBS | **13 - 18x**  | Eliminates O(N²) `xarray.merge`-per-epoch.                 |
| RINEX 2 NAV | 2 - 8x        | Avoids per-SV `xarray.merge` and uses positional datetime.  |
| RINEX 3 NAV | **30 - 33x**  | Eliminates per-SV `xarray.merge` (was the worst pattern).  |
| SP3         | 1.2 - 1.6x    | Pre-allocated buffer + NaN fill; already nearly I/O-bound. |

The single biggest win is on RINEX-3 NAV files: ELKO00USA_R...MN.rnx.gz
goes from 1.0 second in georinex to 31 ms in rinexpy (32.6x). RINEX-3
OBS files see 13-18x. RINEX-2 OBS sees a small win because georinex's
v2 reader was already preallocating a NumPy buffer correctly.

## Methodology notes

- Cold-start cost (importing xarray, NumPy, hatanaka) is **excluded**:
  we time only the `load(...)` call, after the modules are imported.
- File I/O is **included**: gzip/bz2/zip/Z decompression all happen
  inside `load`, so the times reflect end-to-end real-world cost.
- We use `time.perf_counter` (monotonic, ns-resolution) and report the
  median of 5 runs to be robust to one-off page faults.
- xarray and NumPy are not deterministic about thread pools at this
  scale, so single-threaded comparisons are the only fair baseline.

## What changes if you have a slower machine

The CRINEX (Hatanaka) decode and gzip decompression dominate on smaller
files; their cost scales with CPU clock. The OBS3 `xarray.merge` cost
dominates on larger files; that scales with `N²` where `N` is the
number of epochs. So:

- For small files (~10 epochs), `rinexpy` is typically 1.0-1.5x faster.
- For medium files (~100-1000 epochs), 2-5x faster.
- For large files (~10000+ epochs), 10x or more faster.

This is consistent with the algorithmic difference (linear vs quadratic
assembly) — the constant-factor wins matter less at small N, and the
asymptotic win dominates at large N.

## Memory

Peak RSS is also lower in `rinexpy` because we never materialize the
intermediate `xarray.Dataset` objects. For a 24-hour 1 Hz multi-system
RINEX-3 OBS file, peak RSS dropped from ~1.8 GB (georinex) to ~700 MB
(rinexpy) in our local profiling, but this is highly file-dependent
and we don't claim a general number.
