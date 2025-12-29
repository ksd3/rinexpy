# Architecture

`rinexpy` is layered. Each layer depends only on the ones below it. The
core is small: file I/O, epoch parsing, xarray glue. New features land
as their own module instead of growing existing ones.

```
                ┌──────────────────────────────────────────────────────┐
                │                       cli.py                         │
                ├──────────────────────────────────────────────────────┤
                │                       api.py                         │
                ├──────────────────────────────────────────────────────┤
                │   readers (obs2/3, nav2/3, sp3, clk, ionex, antex)   │
                │   streaming.py    writer.py    tools.py              │
                │   positioning.py  rtk.py       multifreq.py          │
                │   lambda_ar.py    interp.py    keplerian.py          │
                │   geodesy.py      gpstime.py   gpt2w.py              │
                │   rtcm3.py        ntrip.py     nmea.py               │
                │   ubx.py          sbf.py       novatel.py   binex.py │
                │   rtcm2.py        beidou.py                          │
                ├──────────────────────────────────────────────────────┤
                │   headers.py    netcdf.py     plots.py               │
                │   zarr_io.py    lazy.py       asyncio.py             │
                ├──────────────────────────────────────────────────────┤
                │   _io.py    _time.py    _common.py    _jit.py        │
                │   _errors.py                                         │
                ├──────────────────────────────────────────────────────┤
                │   _version.py            _types.py                   │
                └──────────────────────────────────────────────────────┘
```

## Layers in detail

### Layer 0, primitives

| module | role |
|---|---|
| `_types.py` | shared type aliases (`FileLike`, `TimeLimit`, ...). |
| `_version.py` | first-line sniff: RINEX/CRINEX/SP3 version + filetype. |

### Layer 1, IO + time + helpers

| module | role |
|---|---|
| `_io.py` | `opener(path)` context manager: gzip/bz2/zip/.Z/Hatanaka/mmap. |
| `_time.py` | epoch parsers (positional `datetime` + int-ns variants) and `normalize_*`. |
| `_common.py` | `fortran_float`, `check_unique_times`, RAM check, time-system, glob. |
| `_jit.py` | optional numba kernel (opt-in via `RINEXPY_USE_JIT=1` or `use_jit=True`). |
| `_errors.py` | `LineCountingStream` + `format_parse_error` for file:line context. |

### Layer 2, header / NetCDF / plot / lazy

| module | role |
|---|---|
| `headers.py` | OBS/NAV header parsers + `rinexinfo` / `rinexheader`. |
| `netcdf.py` | `write_dataset` with safe group-merge. |
| `plots.py` | matplotlib helpers (lazy, optional). |
| `zarr_io.py` | Zarr writer for cloud workflows. |
| `lazy.py` | dask-backed multi-file reader. |
| `asyncio.py` | thread-pool `aload` / `aload_many` wrappers. |

### Layer 3, readers + math + tooling

| module | role |
|---|---|
| `obs2.py`, `obs3.py` | RINEX 2/3 OBS readers. OBS3 is the rewrite that drops O(N²) merge. |
| `nav2.py`, `nav3.py` | RINEX 2/3 NAV readers. NAV3 also rewritten to drop merge-per-SV. |
| `sp3.py` | SP3-a/c/d ephemeris reader; pre-fills NaN buffers. |
| `clk.py` | RINEX clock products (.clk) + linear interp. |
| `ionex.py` | IONEX (.inx) TEC maps + bilinear/temporal interp. |
| `antex.py` | ANTEX (.atx) reader + PCV application (NOAZI or 2-D). |
| `gpt2w.py` | GPT2w empirical met-grid reader + evaluator. |
| `streaming.py` | per-epoch generator for files larger than RAM. |
| `writer.py` | RINEX 2/3 OBS writer. |
| `tools.py` | `validate_file` / `concat_files` / `diff_datasets`. |
| `interp.py` | Lagrange SP3 interpolation. |
| `keplerian.py` | Keplerian → ECEF (vectorized). |
| `geodesy.py` | ECEF/LLA, az/el, DOP, Klobuchar, Saastamoinen, Niell, VMF1. |
| `gpstime.py` | GPS week / leap seconds / 10-bit rollover. |
| `positioning.py` | iterative SPP solver. |
| `rtk.py` | float-DD RTK + `rtk_fix` (LAMBDA-RTK loop). |
| `lambda_ar.py` | single-frequency LAMBDA (LDL + bootstrap + ILS). |
| `multifreq.py` | WL/NL/MW dual-frequency LAMBDA. |
| `rtcm3.py` | RTCM3 framing + decoders for 1004/1005/1006/1019/1020/1033/MSM4/MSM7. |
| `ntrip.py` | NTRIP v1/v2 client (sourcetable + raw byte stream). |
| `nmea.py` | NMEA-0183 ASCII sentence decoder (GGA/RMC/GSA/GSV/VTG). |
| `ubx.py` | u-blox UBX binary decoder (NAV-PVT/NAV-SAT/RXM-RAWX/RXM-SFRBX). |
| `sbf.py` | Septentrio SBF binary decoder (PVTGeodetic/MeasEpoch/GPSNav). |
| `novatel.py` | NovAtel OEM binary decoder (BESTPOS/BESTXYZ/RAWEPHEM). |
| `binex.py` | UNAVCO BINEX framing decoder (forward byte order). |
| `rtcm2.py` | Legacy RTCM SC-104 v2.x DGPS decoder (Type 1/3/9). |
| `beidou.py` | BeiDou D1/D2 raw subframe decoder (clock + iono). |

### Layer 4, public dispatch

| module | role |
|---|---|
| `api.py` | `load`, `rinexnav`, `rinexobs`, `gettime`, `batch_convert` (parallel). |

### Layer 5, CLI

| module | role |
|---|---|
| `cli.py` | argparse `rinexpy` script with `read`/`times`/`info`/`convert`. |
| `__main__.py` | `python -m rinexpy ...` shim. |

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

Steps 7 and 8 are the headline performance change vs georinex: walk
and assemble are split, the assemble is linear in epoch count, and
`xarray.merge` isn't called per-epoch. See `OPTIMIZATIONS.md`.

The streaming path (`iter_obs3_epochs`) replaces step 8 with a
yield-per-epoch loop that builds a single-time `xarray.Dataset` sized
to the SVs present in that epoch.

The parallel batch path (`batch_convert(workers=N)`) shards by file
across a `multiprocessing.Pool`. Each worker does the full layer 1-3
walk on its own.

## Optional dependencies

| extra | pulls in | enables |
|---|---|---|
| `native` | `rinexpy-native>=0.1` | C++ kernels — accelerated OBS3 reader **and** in-tree CRINEX 1+3 decoder (replaces `[hatanaka]`) |
| `hatanaka` | `hatanaka>=2.7` | legacy pure-Python CRINEX decoder (fallback when `[native]` isn't installed) |
| `lzw` | `ncompress>=1.0.1` | `.Z` (LZW) reads |
| `netcdf` | `netCDF4>=1.6` | NetCDF read/write |
| `geo` | `pymap3d>=3.1` | optional ECEF helpers (NOT a hard dep) |
| `plot` | `matplotlib>=3.8` | the `rinexpy.plots` module |
| `zarr` | `zarr>=2.18` | `rinexpy.zarr_io.to_zarr` |
| `jit` | `numba>=0.60` | the optional batched OBS3 decoder |
| `all` | all of the above | everything |

A bare `uv sync` (no extras) is enough for plain RINEX 2/3 NAV/OBS and
SP3 reads. Opening a `.crx` or `.Z` file without the matching extra
raises `ImportError` with an actionable message.

## Testing

`tests/` has 315 pytest tests across 24 modules:

- 17 unit-test modules cover every src module on its own.
- `test_api.py` integrates the dispatch.
- `test_cli.py` smoke-tests the four subcommands.
- `test_parity.py` cross-checks against an installed `georinex 1.16`.
- `test_corrections.py` covers ANTEX/IONEX/CLK application.
- `test_lambda.py`, `test_multifreq.py`, `test_rtk.py` cover RTK+AR.
- `test_rtcm3.py`, `test_ntrip.py` cover the streaming layer.

The full suite finishes in under 3 seconds on a modern laptop.

## Project layout

```
rinexpy/
├── pyproject.toml
├── README.md
├── docs/
│   ├── ARCHITECTURE.md      — this file
│   ├── API.md               — per-symbol reference
│   ├── BENCHMARKS.md        — measured perf numbers
│   ├── COOKBOOK.md          — common recipes
│   ├── OPTIMIZATIONS.md     — every change vs georinex with rationale
│   └── TUTORIAL.md          — install -> RTK fix step-by-step
├── examples/                — 8 runnable .py scripts
├── benchmarks/              — bench_obs3.py + last_run.txt
├── src/rinexpy/             — 24 source modules
└── tests/                   — 24 test modules + tests/data/ fixtures
```
