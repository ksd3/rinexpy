# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

Tracked here until the next tagged release.

## [0.2.x] — current series

The v0.2 series rounds out the v0.1 surface with several new features.

### Added

- **PPP driver.** `rinexpy.ppp.ppp_solve` combines SP3 interpolation,
 CLK interpolation, optional ANTEX PCV, optional GPT2w + VMF1, optional
 DCB, and carrier-phase wind-up into one call. Static or kinematic.
- **SSR composer.** `rinexpy.ssr.SSRCorrections` takes in decoded RTCM3
 SSR messages and exposes per-(SV, epoch) orbit and clock corrections
 plus per-(SV, obs code) code biases. The PPP driver accepts `ssr=` in
 place of `clk=`.
- **Sequential RTK.** `rinexpy.rtk.SequentialRTK` carries the integer
 ambiguity fix across epochs, detects per-SV cycle slips, and runs
 partial AR on the SVs whose lock held.
- **RTCM3 SSR family.** Decoders for 1057-1068 (GPS, GLONASS) and
 1240-1263 (Galileo, QZSS, SBAS, BeiDou), plus IGS-SSR MT 4076.
- **Galileo HAS decoder.** Full HAS message-type coverage (MT 1-7).
- **Real-time orbit/clock cache.** `RealtimeOrbitClock` in
 `rinexpy.realtime` plus the NTRIP loop.
- **PPP-RTK fusion.** `PPPRTKFusion` blends a parallel PPP filter and
 RTK solver by inverse variance.
- **Multi-constellation PPP.** `StaticPPPFilterMultiGNSS` adds
 per-constellation ISBs.
- **ZWD-augmented filter.** `StaticPPPFilterZTD` carries ZWD as a state.
- **Tight GNSS/IMU EKF.** `TightINS16` accepts raw pseudoranges and
 fuses with an IMU strapdown.
- **Loose GNSS/IMU EKF.** `LooseINS15` accepts pre-computed GNSS fixes.
- **Snapshot positioning.** Van Diggelen A-GPS code-phase-only short-
 data SPP in `rinexpy.snapshot`.
- **VRS synthesis.** `rinexpy.vrs.synthesize_vrs` builds a virtual base
 from a network of physical bases.
- **Network DD.** `network_dd_solve` and `network_dd_solve_ar` for joint
 multi-baseline RTK.
- **GNSS reflectometry.** Larson 2008/2013 reflector-height retrieval.
- **Antenna calibration.** `calibrate_pcv` and `write_antex` build new
 ANTEX entries from post-fit residuals.
- **Time transfer.** P3 combination plus common-view.
- **Tides.** Solid Earth (step 1 + step 2), pole, ocean pole, ocean tide
 loading via Scherneck BLQ files.
- **DCB autodownload.** Daily SINEX-BIAS from IGS BKG (post-2017) +
 monthly CODE from AIUB (pre-2017). CDDIS path requires NASA Earthdata
 Login.
- **EOP reader.** IERS Bulletin A / C04 plus per-epoch interpolation.
- **GW-10 framer.** Furuno GW-10 SBAS L1 extractor.
- **GLONASS frequency helpers.** Per-channel L1 / L2 frequencies and
 per-SV iono-free combinations.
- **TCAR.** Three-Carrier Ambiguity Resolution for L1 + L2 + L5
 receivers.
- **Modernised-signal nav decoders.** GPS LNAV / CNAV / CNAV-2,
 Galileo F-NAV / I-NAV, GLONASS strings 1-3, BeiDou D1 / D2, NavIC
 subframes 1-2, SBAS L1 (MT 1, 2-5, 6, 7, 9, 17, 18, 24, 25, 26).
- **RINEX 4 NAV.** STO / EOP / ION record reader (Klobuchar, NeQuick-G,
 BDGIM) per RINEX 4.01 §5.
- **Plugin system.** `rinexpy.plugins` discovers external readers via
 entry points.
- **`rinexpy.cors`.** IGS daily RINEX fetcher (CDDIS / SOPAC / BKG
 mirrors).

### Performance

- Native C++17 extension as `rinexpy-native`. CRINEX 1 / 3 decoder plus
 faster RINEX 3 OBS parse kernel. Installed via `uv sync --extra native`.

## [0.1.0] — initial release

First public release.

### Added — readers

- RINEX 2 OBS and NAV (GPS, GLONASS, Galileo). Ported from `georinex`
 with measured speedups of 1.04 to 8x. NAV2 handles GLONASS km → m
 unit conversion and the ionospheric correction attribute. OBS2 has
 both `fast=True` (single-pass with speculative preallocation) and
 `fast=False` (exact two-pass).
- RINEX 3 OBS and NAV. Rewritten to drop the O(N²)
 `xarray.merge`-per-epoch pattern from upstream. main speedup on
 the bundled corpus: 13 to 33x over `georinex 1.16`.
- SP3-a, SP3-c, SP3-d ephemeris reader.
- RINEX clock products (`.clk`) with per-SV linear interpolation.
- IONEX (`.inx`) global TEC maps with bilinear-in-space and
 linear-in-time interpolation, plus thin-shell slant-TEC mapping.
- ANTEX (`.atx`) antenna phase-centre variation reader and applicator.
 Both NOAZI and azimuth-dependent (DAZI) bilinear paths.
- GPT2w empirical surface-meteorology and VMF1-coefficient grid reader
 and evaluator. Grid file is user-supplied, not released.
- NetCDF4 / HDF5 read and write with safe group merge.
- Zarr write helper for cloud workflows.
- `iter_obs3_epochs` streaming reader for files larger than RAM.

### Added — streaming and receiver formats

- RTCM 3.x framing plus decoders for 1004 (extended L1 / L2 RTK obs),
 1005, 1006, 1019 (GPS ephemeris), 1020 (GLONASS), 1033 (antenna
 descriptors), MSM4 (1074-1134), MSM7 (1077-1137, full per-cell
 signal-block decode).
- RTCM 2.x (SC-104) decoder with 6-of-8 wire stripping. Decoders for
 Type 1 (DGPS PRC), Type 3 (reference station ECEF), Type 9 (high-rate
 corrections).
- NTRIP v1 / v2 client (`fetch_sourcetable`, `stream`). Pure stdlib.
- NMEA-0183 ASCII sentence decoder (GGA, RMC, GSA, GSV, VTG).
- u-blox UBX binary decoder (NAV-PVT, NAV-SAT, RXM-RAWX, RXM-SFRBX).
- Septentrio SBF binary decoder (PVTGeodetic, MeasEpoch, GPSNav).
- NovAtel OEM binary decoder (BESTPOS, BESTXYZ, RAWEPHEM).
- UNAVCO BINEX framing decoder (forward byte order).
- BeiDou D1 subframe-1 and D2 page-1 raw-subframe decoder. Clock
 parameters and ionospheric coefficients per ICD-BDS-OS-200.

### Added — math and positioning

- `keplerian2ecef` vectorised Keplerian to ECEF conversion.
- `interpolate_sp3` Lagrange interpolation (default order 10).
- `interpolate_clk` linear clock-product interpolation.
- WGS-84 `ecef_to_lla` and `lla_to_ecef`, receiver-to-satellite
 `azimuth_elevation`, and the DOP set (GDOP, PDOP, HDOP, VDOP, TDOP).
- Tropospheric models: Saastamoinen, Niell (NMF), VMF1 (using
 GPT2w-supplied or grid-supplied `a_h` / `a_w`).
- Ionospheric models: Klobuchar broadcast L1, IONEX TEC application.
- `spp_solve` iterative single-point positioning.
- `rtk.double_difference_solve` float-ambiguity RTK baseline.
- `lambda_resolve` single-frequency LAMBDA integer ambiguity resolution
 (LDL, bootstrap, ILS search, ratio test).
- `multifreq` Wide-Lane, Narrow-Lane, Melbourne-Wuebbena combinations
 and `lambda_dual_freq` for L1 + L2 ambiguity fix.
- `rtk.rtk_fix` full LAMBDA-RTK loop: joint LSQ, LAMBDA, re-solve.

### Added — tooling

- `validate_file` QC report (header consistency, gaps, intervals).
- `concat_files` multi-file concatenation with dedup.
- `diff_datasets` first-divergence finder.
- `batch_convert` parallel directory-of-files NetCDF conversion via
 `multiprocessing.Pool`.
- `gpstime` GPS week and leap-second utilities with 10-bit rollover
 resolution.
- `to_rinex_obs` writer for RINEX 2 and RINEX 3 OBS round-trip.

### Added — performance

- Optional `numba` JIT path (`use_jit=True` or `RINEXPY_USE_JIT=1`).
 About 1.9x end-to-end on a 23-hour 15-second OBS3 file.
- Memory-mapped reader for plain-text files >= 50 MB.
- Int-ns datetime path. Bulk
 `np.asarray(int_list).view('datetime64[ns]')` is about 40x faster
 than the per-element variant.

### Added — usability and plotting

- `asyncio.aload` and `aload_many`. Thread-pool wrappers for concurrent
 multi-file loading from asyncio apps.
- `lazy.load_lazy`. Dask-backed multi-file reader.
- `plots.timeseries`, `obstimeseries`, `navtimeseries`,
 `receiver_locations`, `skyplot` (matplotlib, optional).
- Argparse `rinexpy` CLI with `read`, `times`, `info`, `convert`
 subcommands.

### Added — packaging

- `pyproject.toml` with extras for `hatanaka`, `lzw`, `netcdf`, `geo`,
 `plot`, `jit`, `zarr`, `all`.
- `uv`-managed project skeleton with a pinned `uv.lock`. Local install
 only (no PyPI publish).

### Added — documentation

- `README.md`, `CHANGELOG.md` (this file), `CONTRIBUTING.md`.
- This documentation site, built with MkDocs Material.

### Tests

- 373 pytest tests covering every public function, plus parity tests
 against installed `georinex 1.16`. Full suite finishes in under 3 s.

### Known divergences from georinex

These are deliberate, documented inline in the test that asserts each.

1. NAV3 trailing FitIntvl and required fields default to 0 per
 RINEX 3.04 §6 (georinex leaves them NaN).
2. NAV3 spare* slots stay NaN. No defined unit; zeroing them would lie
 about the data.
3. SP3 buffers are NaN-pre-filled. `georinex` uses `np.empty()`
 without `.fill(NaN)`, so SVs absent from a particular epoch read
 back as uninitialised memory.

[Unreleased]: https://github.com/ksd3/rinexpy/compare/v0.2.0...HEAD
[0.2.x]: https://github.com/ksd3/rinexpy/releases/tag/v0.2.0
[0.1.0]: https://github.com/ksd3/rinexpy/releases/tag/v0.1.0
