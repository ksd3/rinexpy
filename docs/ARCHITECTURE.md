# Architecture

`rinexpy` is a six-layer library; each layer depends only on layers below it.

```
                ┌──────────────────────────────────────────┐
                │                  cli.py                  │
                ├──────────────────────────────────────────┤
                │                  api.py                  │
                ├──────────────────────────────────────────┤
                │  obs2 / obs3 / nav2 / nav3 / sp3         │
                ├──────────────────────────────────────────┤
                │  headers.py     keplerian.py  netcdf.py  │
                ├──────────────────────────────────────────┤
                │  _io.py     _time.py     _common.py      │
                ├──────────────────────────────────────────┤
                │  _version.py            _types.py        │
                └──────────────────────────────────────────┘
```

| Layer       | Module(s)                                                | Responsibility                                                          |
|-------------|-----------------------------------------------------------|-------------------------------------------------------------------------|
| 0  primitives | `_types.py`, `_version.py`                              | Pure-Python type aliases and version sniffing. No I/O.                   |
| 1  io+time  | `_io.py`, `_time.py`, `_common.py`                       | File-opening (gz/bz2/zip/Z/Hatanaka), epoch parsers, glue helpers.       |
| 2  headers  | `headers.py`, `keplerian.py`, `netcdf.py`               | Header parsers, Keplerian -> ECEF math, NetCDF write helper.             |
| 3  readers  | `obs2.py`, `obs3.py`, `nav2.py`, `nav3.py`, `sp3.py`    | Per-format parsers; each owns the data buffer for its format.            |
| 4  api      | `api.py`                                                 | Format-agnostic dispatch (`load`, `rinexnav`, `rinexobs`, `gettime`, `batch_convert`). |
| 5  cli      | `cli.py`                                                 | Argparse-driven `rinexpy` entry script.                                  |

## Data flow

A typical `rinexpy.load("foo.rnx.gz")` call walks the layers as follows:

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

Steps 7 and 8 are the optimization headline: the upstream
`georinex.obs3._epoch` does both at once, building a fresh `xarray.Dataset`
per epoch and `xarray.merge`-ing it into a running aggregate. That's
quadratic in the number of epochs and accounts for the bulk of the runtime
on RINEX-3 OBS files (the upstream README itself attributes "60% of time"
to it). Splitting walk + assemble lets us keep the assemble step linear.

## Public surface

The public API is small. Anything not in `__all__` is implementation detail
and may change between releases.

```python
import rinexpy as rp

# auto-dispatch
rp.load(path, *, use=..., tlim=..., meas=..., useindicators=...,
        fast=..., interval=..., out=..., overwrite=..., verbose=...)
rp.rinexnav(path, *, use=..., tlim=..., group="NAV", overwrite=...)
rp.rinexobs(path, *, use=..., tlim=..., meas=..., useindicators=...,
            group="OBS", fast=..., interval=..., overwrite=...)

# inspection
rp.rinexinfo(path)        # quick metadata (version, type, system)
rp.rinexheader(path)      # full header dict
rp.gettime(path)          # numpy.datetime64 array of epochs

# batch
rp.batch_convert(dir, "*.rnx.gz", out_dir, ...)

# math
rp.keplerian2ecef(nav.sel(sv="G07"))   # (X, Y, Z) DataArrays
rp.to_datetime(ds.time)                # xarray time coord -> datetime
```

The version-specific entry points (`rinexobs2`, `rinexobs3`, `rinexnav2`,
`rinexnav3`, `obsheader2`, `obsheader3`, `navheader2`, `navheader3`,
`obstime2`, `obstime3`, `navtime2`, `navtime3`) are also re-exported for
parity with georinex but are rarely needed in user code.

## Optional dependencies

| Extra        | Pulls in                             | Enables                                |
|--------------|--------------------------------------|----------------------------------------|
| `hatanaka`   | `hatanaka>=2.7`                      | CRINEX (`.crx*`) reads                 |
| `lzw`        | `ncompress>=1.0.1`                   | `.Z` (LZW) reads                       |
| `netcdf`     | `netCDF4>=1.6`                       | NetCDF read/write round-trip            |
| `geo`        | `pymap3d>=3.1`                       | Geodetic position attribute on OBS     |
| `plot`       | `matplotlib>=3.8`                    | (Reserved; no plotting yet built-in)   |
| `all`        | All of the above                     | Everything                             |

A bare `pip install rinexpy` is sufficient for plain RINEX 2/3 NAV/OBS and
SP3 reads. Trying to open a `.crx`/`.Z` file without the corresponding
extra raises `ImportError` with a precise actionable message.

## Testing

`tests/` contains 173 pytest tests:

- 11 modules of unit tests (`test_io.py`, `test_version.py`, `test_time.py`,
  `test_common.py`, `test_headers.py`, `test_nav2.py`, `test_nav3.py`,
  `test_obs2.py`, `test_obs3.py`, `test_sp3.py`, `test_keplerian.py`).
- API integration tests (`test_api.py`).
- CLI smoke tests (`test_cli.py`).
- Cross-implementation parity tests (`test_parity.py`) — skipped if
  `georinex` is not installed.

The full suite finishes in <2 s on a modern laptop.
