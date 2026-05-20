# Architecture

`rinexpy` is layered. Each layer depends only on the ones below it. The
core is small: file I/O, epoch parsing, xarray glue. Every new feature
lands as its own module rather than growing the existing ones, so the
public surface scales while the core stays at a few hundred lines.

```
                +------------------------------------------------------+
                |                       cli.py                         |
                +------------------------------------------------------+
                |                       api.py                         |
                +------------------------------------------------------+
                |   readers (obs2/3, nav2/3/4, sp3, clk, ionex, antex, |
                |            met, eop)                                 |
                |   streaming, writer, tools                           |
                |   positioning, rtk, multifreq, lambda_ar, interp,    |
                |   keplerian, ppp, kalman, kalman_multignss,          |
                |   kalman_ztd, ssr, snapshot, vrs, gnssr,             |
                |   antex_calibrate, time_transfer, network_dd, has    |
                |   geodesy, gpstime, gpt2w, tides, otl                |
                |   rtcm3, ntrip, nmea                                 |
                |   ubx, sbf, novatel, binex, rtcm2, sbas, gw10        |
                |   beidou, gps_lnav, gps_cnav, gps_cnav2,             |
                |   galileo_nav, glonass, navic                        |
                |   dcb, dcb_download, cors                            |
                |   qc, spoofing                                       |
                |   imu, imu_tight, realtime                           |
                +------------------------------------------------------+
                |   headers, netcdf, plots, zarr_io, lazy, asyncio     |
                |   nav_writer, plugins                                |
                +------------------------------------------------------+
                |   _io, _time, _common, _jit, _native, _errors        |
                +------------------------------------------------------+
                |   _version, _types                                   |
                +------------------------------------------------------+
```

## Layers in detail

### Layer 0, primitives

| Module | Role |
| --- | --- |
| `_types` | shared type aliases (`FileLike`, `TimeLimit`, ...) |
| `_version` | first-line sniff: RINEX / CRINEX / SP3 version + filetype |

### Layer 1, I/O and helpers

| Module | Role |
| --- | --- |
| `_io` | `opener(path)` context manager: gzip / bz2 / zip / Z / Hatanaka / mmap |
| `_time` | epoch parsers (positional `datetime` plus int-ns variants) and the `normalize_*` helpers |
| `_common` | Fortran float, time-system detection, RAM check, globbing |
| `_jit` | optional numba kernel (opt-in via `RINEXPY_USE_JIT=1` or `use_jit=True`) |
| `_native` | dispatcher for the C++ extension (when `[native]` is installed) |
| `_errors` | `LineCountingStream` and `format_parse_error` for file:line context in error messages |

### Layer 2, headers / NetCDF / plot / lazy

| Module | Role |
| --- | --- |
| `headers` | RINEX OBS / NAV header parsers and the public `rinexinfo` / `rinexheader` |
| `netcdf` | `write_dataset` with safe group merge |
| `plots` | matplotlib helpers (lazy import, optional) |
| `zarr_io` | Zarr writer for cloud workflows |
| `lazy` | dask-backed multi-file reader |
| `asyncio` | thread-pool `aload` / `aload_many` wrappers |
| `nav_writer` | RINEX 3 NAV writer |
| `plugins` | plugin discovery via Python entry points |

### Layer 3, readers, math, and tooling

The bulk of the library. Each module is self-contained and
depends only on the layers below.

#### Format readers

| Module | Role |
| --- | --- |
| `obs2`, `obs3` | RINEX 2 / 3 OBS readers |
| `nav2`, `nav3`, `nav4` | RINEX 2 / 3 / 4 NAV readers |
| `sp3` | SP3-a / SP3-c / SP3-d ephemeris reader and writer |
| `clk` | RINEX clock product reader plus linear interpolation |
| `ionex` | IONEX TEC map reader plus bilinear interpolation |
| `antex` | ANTEX antenna PCV reader and applicator |
| `antex_calibrate` | ANTEX PCV fit from residuals plus writer |
| `met` | RINEX MET surface met data reader |
| `eop` | IERS Bulletin A / EOP C04 reader and interpolation |
| `gpt2w` | GPT2w empirical met grid reader and evaluator |
| `streaming` | per-epoch generator for OBS 3 |
| `writer` | RINEX 2 / 3 OBS writer |
| `tools` | `validate_file`, `concat_files`, `diff_datasets` |

#### Streaming and receiver formats

| Module | Role |
| --- | --- |
| `rtcm3` | RTCM 3 framing and decoders (1004, 1005, 1006, 1019, 1020, 1029, 1033, 1230, MSM 1-7, SSR 1057-1068, SSR 1240-1263, IGS-SSR MT 4076) |
| `rtcm2` | Legacy RTCM 2.x DGPS decoder (types 1, 3, 9) |
| `ntrip` | NTRIP v1 / v2 client (sync `stream` and asyncio `astream`) |
| `nmea` | NMEA-0183 ASCII sentence decoder |
| `ubx` | u-blox UBX binary decoder |
| `sbf` | Septentrio SBF decoder |
| `novatel` | NovAtel OEM binary decoder |
| `binex` | UNAVCO BINEX framing decoder |
| `sbas` | SBAS L1 message decoder |
| `gw10` | Furuno GW-10 framed SBAS L1 extractor |
| `has` | Galileo HAS message decoder |

#### Raw nav subframes

| Module | Role |
| --- | --- |
| `gps_lnav`, `gps_cnav`, `gps_cnav2` | GPS LNAV / CNAV / CNAV-2 |
| `galileo_nav` | Galileo F-NAV / I-NAV |
| `glonass` | GLONASS L1OF / L2OF strings |
| `beidou` | BeiDou D1 / D2 |
| `navic` | NavIC subframes |

#### Math and positioning

| Module | Role |
| --- | --- |
| `interp` | Lagrange SP3 interpolation |
| `keplerian` | Keplerian to ECEF conversion |
| `geodesy` | ECEF / LLA, az / el, DOP, Klobuchar, Saastamoinen, Niell, VMF1, ECI rotation, phase wind-up |
| `gpstime` | GPS week, leap seconds, 10-bit rollover |
| `positioning` | SPP solver with RAIM, code-only PPP, static-batch PPP with AR |
| `rtk` | float RTK, `rtk_fix` with LAMBDA, `SequentialRTK` |
| `ppp` | static-or-kinematic PPP driver |
| `ppp_rtk` | PPP + RTK fusion |
| `kalman`, `kalman_ztd`, `kalman_multignss` | PPP EKFs |
| `multifreq` | wide-lane / narrow-lane / Melbourne-Wuebbena / TCAR |
| `lambda_ar` | single-frequency LAMBDA |
| `snapshot` | A-GPS code-phase-only short-data SPP |
| `vrs` | VRS synthesis |
| `network_dd` | joint multi-baseline DD solver |
| `gnssr` | GNSS reflectometry reflector height |
| `time_transfer` | P3 + common-view |
| `imu`, `imu_tight` | GNSS / IMU EKFs |
| `realtime` | NTRIP loop plus `RealtimeOrbitClock` cache |
| `cors` | IGS daily RINEX fetcher |

#### Atmosphere and corrections

| Module | Role |
| --- | --- |
| `tides` | solid Earth + pole + ocean-pole tides |
| `otl` | ocean tide loading from Scherneck BLQ files |
| `ssr` | `SSRCorrections` composer over decoded RTCM3 SSR |
| `dcb`, `dcb_download` | SINEX-BIAS reader, CODE monthly reader, IGS BKG / AIUB / CDDIS downloader |

#### Quality

| Module | Role |
| --- | --- |
| `qc` | cycle slip detectors, multipath combinations, Hatch filter |
| `spoofing` | SNR uniformity, position jumps, clock drift, AGC heuristics |

### Layer 4, public dispatch

| Module | Role |
| --- | --- |
| `api` | `load`, `rinexnav`, `rinexobs`, `gettime`, `batch_convert` |

### Layer 5, CLI

| Module | Role |
| --- | --- |
| `cli` | argparse `rinexpy` script with subcommands |
| `__main__` | `python -m rinexpy ...` shim |

## End-to-end dataflow

A `rinexpy.load("foo.rnx.gz")` call walks the layers like this:

```
api.load                                # 1. user-facing entry
 └── headers.rinexinfo                  # 2. peek at first non-blank line
      └── _io.opener(header=True)       # 3. open just the header
           └── _version.rinex_version
 └── (dispatch on rinextype + version)
 └── obs3.rinexobs3                     # 4. format-specific reader
      ├── _io.opener(header=False)      # 5. open the full data section
      ├── headers.obsheader3            # 6. parse the header
      ├── obs3._walk_epochs             # 7. SINGLE PASS over data lines
      └── obs3._assemble_obs3           # 8. ONE xarray.Dataset built here
```

Steps 7 and 8 are the main performance change against upstream:
walk and assemble are split, the assemble is linear in epoch count, and
`xarray.merge` is not called per epoch. See
[Optimizations](optimizations.md).

The streaming path (`iter_obs3_epochs`) replaces step 8 with a
yield-per-epoch loop that builds a single-time Dataset sized to the SVs
present in that epoch.

The parallel batch path (`batch_convert(workers=N)`) shards by file
across a `multiprocessing.Pool`. Each worker does the full layer 1-3
walk on its own.

## Optional dependencies

| Extra | Pulls in | Enables |
| --- | --- | --- |
| `native` | `rinexpy-native>=0.1` | C++ kernel: faster OBS3 reader + in-tree CRINEX 1 / 3 decoder |
| `hatanaka` | `hatanaka>=2.7` | legacy pure-Python CRINEX decoder |
| `lzw` | `ncompress>=1.0.1` | `.Z` reads |
| `netcdf` | `netCDF4>=1.6` | NetCDF read / write |
| `geo` | `pymap3d>=3.1` | optional ECEF helpers |
| `plot` | `matplotlib>=3.8` | `rinexpy.plots` |
| `zarr` | `zarr>=2.18` | `rinexpy.zarr_io.to_zarr` |
| `jit` | `numba>=0.60` | numba-JITed OBS3 inner loop |
| `all` | the four above (`lzw`, `netcdf`, `geo`, `plot`) | everything users typically want |

A bare `uv sync` (no extras) is enough for plain RINEX 2 / 3 NAV / OBS
and SP3 reads. Opening a `.crx` or `.Z` file without the matching extra
raises `ImportError` with an actionable message.

## Testing

The `tests/` directory has 110+ pytest modules and roughly 1000 test
cases. Unit tests cover every source module on its own; integration
tests cover the dispatch (`test_api.py`), the CLI (`test_cli.py`),
parity against an installed `georinex 1.16` (`test_parity.py`),
corrections application, RTK and ambiguity resolution, PPP, the SSR
composer, and the streaming layer.

The full suite finishes in a few seconds on a modern laptop.

```sh
uv run pytest tests/ -q                  # full suite
uv run pytest tests/test_obs3.py -v      # one module
uv run pytest tests/ -k "rtk and not real"  # by keyword
```

A handful of tests are marked `@pytest.mark.parity` and compare `rinexpy`
output against an installed `georinex` package. They are skipped when
georinex is not installed.

## Project layout

```
rinexpy/
├── pyproject.toml
├── README.md
├── docs/
│   └── ... (this docs site)
├── examples/                — 8 runnable scripts
├── benchmarks/              — bench_obs3.py + last_run.txt
├── native/                  — local C++ extension package
├── src/rinexpy/             — 80+ source modules
└── tests/                   — 110+ test modules + tests/data/ fixtures
```

## Related pages

- [Optimizations](optimizations.md): the perf changes against georinex.
- [Benchmarks](benchmarks.md): measured numbers.
- [Module index](../reference/modules.md): per-module symbol table.
