# API reference

A handful of functions in `rinexpy` cover the entire public surface.
The signatures match `georinex 1.16` for drop-in compatibility, with the
following additions:

- `tlim` accepts plain ISO-8601 strings (`"2018-07-29T12:00"`) as well
  as `datetime` instances.
- `interval` accepts plain seconds (`int` or `float`) as well as
  `timedelta` instances.
- The `out` argument may be a directory; the basename is auto-derived.

## Top-level dispatch

### `rinexpy.load(rinexfn, out=None, *, use=None, tlim=None, useindicators=False, meas=None, verbose=False, overwrite=False, fast=True, interval=None) -> xarray.Dataset | dict`

Read **any** supported file. Auto-detects RINEX 2/3 NAV/OBS, SP3-a/c/d,
and pre-converted NetCDF. For NetCDF files containing both NAV and OBS
groups, returns `{"nav": ..., "obs": ...}`.

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C
```

Filtering is cheap because filtered records are skipped without parsing:

```python
obs = rp.load("big.rnx.gz",
              use={"G", "E"},               # only GPS + Galileo
              meas=["C1C", "L1C"],          # only these measurement types
              tlim=("2018-07-29T12:00",     # only this 1-hour window
                    "2018-07-29T13:00"),
              interval=30)                  # decimate to every 30s
```

### `rinexpy.rinexnav(fn, outfn=None, *, use=None, group="NAV", tlim=None, overwrite=False)`

Read just a NAV file, or just the NAV group of a NetCDF.

### `rinexpy.rinexobs(fn, outfn=None, *, use=None, group="OBS", tlim=None, useindicators=False, meas=None, verbose=False, overwrite=False, fast=True, interval=None)`

Read just an OBS file, or just the OBS group of a NetCDF.

### `rinexpy.batch_convert(path, glob, out, *, use=None, tlim=None, useindicators=False, meas=None, verbose=False, fast=True) -> list[Path]`

Convert every file in `path` matching `glob` into NetCDF inside `out`.
Errors on individual files are logged and the conversion continues. The
list of successfully written paths is returned.

```python
written = rp.batch_convert("data/2018", "*.rnx.gz", "out/")
print(f"wrote {len(written)} files")
```

## Inspection

### `rinexpy.rinexinfo(fn) -> dict`

Cheap header peek. Returns `{"version", "filetype", "rinextype", "systems"}`
or — for NetCDF inputs — a list of `rinextype` values plus the dataset's
attributes.

### `rinexpy.rinexheader(fn) -> dict`

Full parsed header. The keys match RINEX header labels verbatim
(`"APPROX POSITION XYZ"`, `"# / TYPES OF OBSERV"`, ...) plus a handful
of derived keys (`"position"`, `"position_geodetic"`, `"t0"`, `"t1"`,
`"interval"`, `"fields"`, `"fields_ind"`, `"Nobs"`, `"Nl_sv"`).

### `rinexpy.gettime(fn) -> numpy.ndarray`

Unique sorted epoch timestamps as `datetime64[us]` (OBS) or `datetime64[ms]`
(NAV).

### `rinexpy.to_datetime(time_coord) -> datetime | numpy.ndarray`

Convert an `xarray` time coordinate (or anything with `.values.astype(...)`)
to plain `datetime` objects. Useful when you want to call APIs that don't
understand NumPy temporal types.

## Math

### `rinexpy.keplerian2ecef(sv) -> (X, Y, Z)`

Vectorized Keplerian → ECEF conversion. Pass an `xarray.Dataset` slice
for one or more satellites (`nav.sel(sv="G07")`); returns three
`xarray.DataArray`s of meters.

```python
nav = rp.load("brdc2800.15n")
g09 = nav.sel(sv="G09").dropna(dim="time", how="all")
X, Y, Z = rp.keplerian2ecef(g09)
```

GLONASS (R) and SBAS (S) records are returned unchanged — those systems
report state directly in ECEF and don't need a Keplerian conversion.

## Version-specific (parity with georinex)

For backwards-compatibility with code written against georinex, the
version-specific entry points are also exported:

| RINEX 2                                    | RINEX 3                                    |
|--------------------------------------------|--------------------------------------------|
| `rinexpy.rinexobs2(fn, ...)`               | `rinexpy.rinexobs3(fn, ...)`               |
| `rinexpy.rinexnav2(fn, ...)`               | `rinexpy.rinexnav3(fn, ...)`               |
| `rinexpy.obstime2(fn)`                     | `rinexpy.obstime3(fn)`                     |
| `rinexpy.navtime2(fn)`                     | `rinexpy.navtime3(fn)`                     |
| `rinexpy.obsheader2(fn, ...)`              | `rinexpy.obsheader3(fn, ...)`              |
| `rinexpy.navheader2(fn)`                   | `rinexpy.navheader3(fn)`                   |

These are documented in their respective module docstrings.

## SP3

### `rinexpy.load_sp3(fn, outfn=None) -> xarray.Dataset`

Read an SP3-a/c/d ephemeris. Output coords: `time`, `sv`, `ECEF=["x","y","z"]`.
Output variables: `position`, `velocity`, `clock`, `dclock`, scalar `t0`.
Velocity is NaN where no V record was provided.

## CLI

```sh
rinexpy read   <file>            # parse and print
rinexpy times  <file>            # print just epoch timestamps
rinexpy info   <file>            # parsed header
rinexpy convert <dir> <glob> --out <dir>   # batch -> NetCDF
```

All subcommands accept `-u/--use` (system selection),
`-m/--meas` (measurement selection),
`-t/--tlim START STOP`,
`--useindicators`,
`--interval`,
and `--strict` (disable speculative preallocation, slower but exact).
