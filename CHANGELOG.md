# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

Tracked here until the first tagged release. Once `v0.1.0` lands,
the contents below move into the `[0.1.0]` block.

## [0.1.0] - TBD

First public release.

### Added - readers

- RINEX 2 OBS and NAV (GPS, GLONASS, Galileo). Ported from
  `georinex` with measured speedups of 1.04-8x. NAV2 handles
  GLONASS km→m unit conversion and the ionospheric correction
  attribute. OBS2 has both `fast=True` (single-pass with
  speculative preallocation) and `fast=False` (exact two-pass).
- RINEX 3 OBS and NAV. Rewritten to drop the O(N²)
  `xarray.merge`-per-epoch pattern from upstream. Headline
  speedup on the bundled corpus: 13-33x over `georinex 1.16`.
- SP3-a, SP3-c, SP3-d ephemeris reader.
- RINEX clock products (`.clk`) with per-SV linear interpolation.
- IONEX (`.inx`) global TEC maps with bilinear-in-space and
  linear-in-time interpolation, plus thin-shell slant-TEC mapping.
- ANTEX (`.atx`) antenna phase-center variation reader and
  applicator. Both NOAZI and azimuth-dependent (DAZI) bilinear
  paths.
- GPT2w empirical surface-meteorology and VMF1-coefficient grid
  reader and evaluator. Grid file is user-supplied, not shipped.
- NetCDF4/HDF5 read and write with safe group-merge.
- Zarr write helper for cloud workflows.
- `iter_obs3_epochs` streaming reader for files larger than RAM.

### Added - streaming and receiver formats

- RTCM 3.x framing plus decoders for 1004 (extended L1/L2 RTK
  obs), 1005, 1006, 1019 (GPS ephemeris), 1020 (GLONASS), 1033
  (antenna descriptors), MSM4 (1074-1134), MSM7 (1077-1137, full
  per-cell signal-block decode).
- RTCM 2.x (SC-104) decoder with 6-of-8 wire stripping. Decoders
  for Type 1 (DGPS PRC), Type 3 (reference station ECEF), Type 9
  (high-rate corrections).
- NTRIP v1/v2 client (`fetch_sourcetable`, `stream`). Pure stdlib.
- NMEA-0183 ASCII sentence decoder (GGA, RMC, GSA, GSV, VTG).
- u-blox UBX binary decoder (NAV-PVT, NAV-SAT, RXM-RAWX, RXM-SFRBX).
- Septentrio SBF binary decoder (PVTGeodetic, MeasEpoch, GPSNav).
- NovAtel OEM binary decoder (BESTPOS, BESTXYZ, RAWEPHEM).
- UNAVCO BINEX framing decoder (forward byte order).
- BeiDou D1 subframe-1 and D2 page-1 raw-subframe decoder. Clock
  parameters and ionospheric coefficients per ICD-BDS-OS-200.

### Added - math and positioning

- `keplerian2ecef` vectorized Keplerian to ECEF conversion.
- `interpolate_sp3` Lagrange interpolation (default order 10).
- `interpolate_clk` linear clock-product interpolation.
- WGS-84 `ecef_to_lla` and `lla_to_ecef`, receiver-to-satellite
  `azimuth_elevation`, and the DOP set (GDOP, PDOP, HDOP, VDOP,
  TDOP).
- Tropospheric models: Saastamoinen, Niell (NMF), VMF1 (using
  GPT2w-supplied or grid-supplied `a_h`/`a_w`).
- Ionospheric models: Klobuchar broadcast L1, IONEX TEC application.
- `spp_solve` iterative single-point positioning.
- `rtk.double_difference_solve` float-ambiguity RTK baseline.
- `lambda_resolve` single-frequency LAMBDA integer ambiguity
  resolution (LDL, bootstrap, ILS search, ratio test).
- `multifreq` Wide-Lane, Narrow-Lane, Melbourne-Wübbena
  combinations and `lambda_dual_freq` for L1+L2 ambiguity fix.
- `rtk.rtk_fix` full LAMBDA-RTK loop: joint LSQ, LAMBDA, re-solve.

### Added - tooling

- `validate_file` QC report (header consistency, gaps, intervals).
- `concat_files` multi-file concatenation with dedup.
- `diff_datasets` first-divergence finder.
- `batch_convert` parallel directory-of-files NetCDF conversion
  via `multiprocessing.Pool`. `workers=` keyword; 0 means all CPUs.
- `gpstime` GPS week and leap-second utilities with 10-bit
  rollover resolution.
- `to_rinex_obs` writer for RINEX 2 and RINEX 3 OBS round-trip.

### Added - performance

- Optional `numba` JIT path (`use_jit=True` or `RINEXPY_USE_JIT=1`).
  About 1.9x end-to-end on a 23-hour 15-s OBS3 file.
- Optional C++17 extension (`rinexpy-native`, separate package;
  `use_native=True` or `RINEXPY_USE_NATIVE=1`). 18x faster on the
  parse kernel, ~3x over the JIT path, matches JIT end-to-end, and
  drops the `numba`/`llvmlite` dependency.
- Memory-mapped reader for plain-text files ≥ 50 MB.
- Int-ns datetime path. Bulk
  `np.asarray(int_list).view('datetime64[ns]')` is ~40x faster
  than `np.array(list_of_datetime, dtype='datetime64[ns]')` on the
  bulk-conversion step.

### Added - ergonomics and plotting

- `asyncio.aload` and `aload_many`. Thread-pool wrappers for
  concurrent multi-file loading from `asyncio` apps.
- `lazy.load_lazy`. Dask-backed multi-file reader.
- `plots.timeseries`, `obstimeseries`, `navtimeseries`,
  `receiver_locations`, `skyplot` (matplotlib, optional).
- Argparse `rinexpy` CLI with `read`, `times`, `info`, `convert`
  subcommands.

### Added - packaging

- `pyproject.toml` with extras for `hatanaka`, `lzw`, `netcdf`,
  `geo`, `plot`, `jit`, `zarr`, `native`, `all`.
- `uv`-managed project skeleton with a pinned `uv.lock`. Local
  install only (no PyPI publish). The optional `rinexpy-native`
  extension resolves from `./native/` as a workspace dependency.

### Added - documentation

- `README.md`, `CHANGELOG.md` (this file), `CONTRIBUTING.md`.
- `docs/`: `TUTORIAL.md` (install to RTK in 12 sections),
  `COOKBOOK.md` (short recipes), `API.md` (per-symbol reference,
  43 entries), `ARCHITECTURE.md` (module map and dataflow),
  `OPTIMIZATIONS.md` (what changed vs georinex with rationale),
  `BENCHMARKS.md` (measured perf numbers).
- `examples/`: 8 runnable scripts covering the headline workflows.
- `mkdocs.yml` for the docs site, built locally with
  `uv run mkdocs serve` and `uv run mkdocs build`.

### Tests

- 373 pytest tests covering every public function, plus parity
  tests against installed `georinex 1.16`. Full suite finishes in
  under 3 s.

### Known divergences from georinex

These are deliberate, documented inline in the test that asserts
each:

1. NAV3 trailing FitIntvl and required fields default to 0 per
   RINEX 3.04 §6 (georinex left them NaN).
2. NAV3 spare\* slots stay NaN. No defined unit; zeroing them
   would lie about the data.
3. SP3 buffers are NaN-pre-filled. `georinex` used `np.empty()`
   without `.fill(NaN)`, so SVs absent from a particular epoch
   read back as uninitialized memory, producing nondeterministic
   output.

[Unreleased]: https://github.com/ksd3/rinexpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ksd3/rinexpy/releases/tag/v0.1.0
