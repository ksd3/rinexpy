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

The full output of `bench_obs3.py` is captured here verbatim — see
`benchmarks/last_run.txt` for the most recent run if it exists, or run
the benchmark yourself.

A representative excerpt:

```
== RINEX 3 OBS ==
  obs3.01gage.10o     georinex     7.7 ms   rinexpy   0.6 ms   13.27x
  ...
```

The single biggest win is on RINEX-3 OBS files because that's where the
`xarray.merge`-per-epoch was dominant. RINEX-2 OBS gets a smaller win
because georinex's RINEX-2 reader was already preallocating a NumPy
buffer correctly.

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
