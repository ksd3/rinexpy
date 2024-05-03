# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Tracked here until the first tagged release. Once we tag `v0.1.0`,
the contents of this section move into the `[0.1.0]` block below.

## [0.1.0] - TBD

The first public release.

### Added — readers

- RINEX 2 OBS, RINEX 2 NAV (GPS, GLONASS, Galileo) — ported from
  `georinex` with measured speedups of 1.04–8x. NAV2 supports
  GLONASS km->m unit conversion and the ionospheric correction
  attribute. OBS2 supports both `fast=True` (single-pass with
  speculative preallocation) and `fast=False` (exact two-pass).
- RINEX 3 OBS, RINEX 3 NAV — rewritten to drop the O(N²)
  `xarray.merge`-per-epoch pattern from upstream. Headline speedup
  on the bundled corpus: **13–33× faster** than `georinex 1.16`.
- SP3-a / SP3-c / SP3-d ephemeris reader.
- RINEX clock product (`.clk`) reader and per-SV linear interpolator.
- IONEX (`.inx`) global TEC map reader with bilinear-in-space and
  linear-in-time interpolation, plus a thin-shell slant-TEC mapping.
- ANTEX (`.atx`) antenna phase-center variation reader and applicator
  with both NOAZI and azimuth-dependent (DAZI) bilinear paths.
- GPT2w empirical surface-meteorology and VMF1-coefficient grid
  reader and evaluator (grid file is user-supplied; not shipped).
- NetCDF4 / HDF5 read and write with safe group-merge.
- Zarr write helper for cloud workflows.
- `iter_obs3_epochs` per-epoch streaming reader for files larger
  than RAM.

### Added — streaming + receiver formats

- RTCM 3.x framing + decoders for messages 1004 (extended L1/L2
  RTK obs), 1005, 1006, 1019 (GPS ephemeris), 1020 (GLONASS), 1033
  (antenna descriptors), MSM4 (1074–1134), and MSM7 (1077–1137,
  with full per-cell signal-block decode).
- RTCM 2.x (SC-104) decoder with 6-of-8 wire stripping and
  decoders for Type 1 (DGPS PRC), Type 3 (reference station ECEF),
  Type 9 (high-rate corrections).
- NTRIP v1/v2 client (`fetch_sourcetable`, `stream`) — pure stdlib.
- NMEA-0183 ASCII sentence decoder (GGA, RMC, GSA, GSV, VTG).
- u-blox UBX binary decoder (NAV-PVT, NAV-SAT, RXM-RAWX, RXM-SFRBX).
- Septentrio SBF binary decoder (PVTGeodetic, MeasEpoch, GPSNav).
- NovAtel OEM binary decoder (BESTPOS, BESTXYZ, RAWEPHEM).
- UNAVCO BINEX framing decoder (forward byte order).
- BeiDou D1 subframe-1 and D2 page-1 raw-subframe decoder
  (clock parameters and ionospheric coefficients per ICD-BDS-OS-200).

### Added — math / positioning

- `keplerian2ecef` vectorised Keplerian -> ECEF conversion.
- `interpolate_sp3` Lagrange interpolation (default order 10).
- `interpolate_clk` linear clock-product interpolation.
- WGS-84 `ecef_to_lla` / `lla_to_ecef`, receiver-to-satellite
  `azimuth_elevation`, and the standard DOP set
  (GDOP/PDOP/HDOP/VDOP/TDOP).
- Tropospheric models: Saastamoinen, Niell (NMF), VMF1 (using
  GPT2w-supplied or grid-supplied `a_h`/`a_w`).
- Ionospheric models: Klobuchar broadcast L1, IONEX TEC application.
- `spp_solve` iterative single-point positioning.
- `rtk.double_difference_solve` float-ambiguity RTK baseline.
- `lambda_resolve` single-frequency LAMBDA integer ambiguity
  resolution (LDL + bootstrap + ILS search + ratio test).
- `multifreq` Wide-Lane / Narrow-Lane / Melbourne-Wübbena
  combinations and `lambda_dual_freq` for L1+L2 ambiguity fix.
- `rtk.rtk_fix` full LAMBDA-RTK loop: joint LSQ -> LAMBDA -> re-solve.

### Added — tooling

- `validate_file` quality-control report (header consistency,
  gaps, intervals).
- `concat_files` multi-file concatenation with dedup.
- `diff_datasets` first-divergence finder.
- `batch_convert` parallel directory-of-files NetCDF conversion
  via `multiprocessing.Pool` (`workers=` keyword; 0 = all CPUs).
- `gpstime` GPS week / leap-second utilities including 10-bit
  rollover resolution.
- `to_rinex_obs` writer for RINEX 2 / RINEX 3 OBS round-trip.

### Added — performance

- Optional `numba` JIT path (`use_jit=True` / `RINEXPY_USE_JIT=1`):
  ~1.9× end-to-end on a 23-h 15-s OBS3 file.
- Optional C++17 extension (`rinexpy-native` separate package;
  `use_native=True` / `RINEXPY_USE_NATIVE=1`): 18× faster on the
  parse kernel, ~3× over the JIT path; matches JIT end-to-end and
  removes the `numba`/`llvmlite` dependency.
- Memory-mapped reader for plain-text files ≥ 50 MB.
- Int-ns datetime path: bulk
  `np.asarray(int_list).view('datetime64[ns]')` is ~40× faster
  than `np.array(list_of_datetime, dtype='datetime64[ns]')` on the
  bulk-conversion step.

### Added — ergonomics + plotting

- `asyncio.aload` / `aload_many` — thread-pool wrappers for
  concurrent multi-file loading from `asyncio` apps.
- `lazy.load_lazy` — dask-backed multi-file reader.
- `plots.timeseries`, `obstimeseries`, `navtimeseries`,
  `receiver_locations`, `skyplot` (matplotlib, optional).
- Argparse `rinexpy` CLI with `read` / `times` / `info` /
  `convert` subcommands.

### Added — packaging + CI

- `pyproject.toml` with extras for `hatanaka`, `lzw`, `netcdf`,
  `geo`, `plot`, `jit`, `zarr`, `native`, and `all`.
- GitHub Actions matrix CI: lint + format check, full test matrix
  on {Linux, macOS, Windows} × {Python 3.11, 3.12, 3.13}, parity
  cross-checks against `georinex`, benchmark publishing.
- `cibuildwheel` job for the `rinexpy-native` C++ wheels with a
  separate integration job that installs the freshly-built wheel
  and runs the native test suite.

### Added — documentation

- `README.md`, `SCRATCHPAD.md` (engineering log), `CHANGELOG.md`
  (this file), `CONTRIBUTING.md`.
- `docs/`: `TUTORIAL.md` (install → RTK in 12 sections),
  `COOKBOOK.md` (copy-pasteable recipes), `API.md` (per-symbol
  reference of all 43 public entries), `ARCHITECTURE.md`
  (six-layer module map + dataflow), `OPTIMIZATIONS.md` (every
  change vs `georinex` with rationale), `BENCHMARKS.md` (measured
  perf numbers).
- `examples/`: 8 runnable scripts covering the headline workflows.
- `mkdocs.yml` for the documentation site, deployed via GitHub
  Pages on every push to `main`.

### Tests

- 373 pytest tests covering every public function, plus parity
  tests against an installed `georinex 1.16`. Full suite finishes
  in <3 s.

### Known divergences from georinex

These are intentional; documented inline in the test that asserts
each:

1. **NAV3 trailing FitIntvl / required fields** default to 0 per
   RINEX 3.04 §6 (georinex left them NaN).
2. **NAV3 spare\* slots** stay NaN (no defined unit; making them
   0 would lie about the data).
3. **SP3 buffers** are NaN-pre-filled (georinex's `np.empty()`
   leaked uninitialised heap memory for SVs absent from a
   particular epoch, producing nondeterministic output).

[Unreleased]: https://github.com/kshitijduraphe/rinexpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kshitijduraphe/rinexpy/releases/tag/v0.1.0
