# rinexpy

Modern, fast RINEX 2/3/4, CRINEX (Hatanaka), and SP3-a/c/d reader for Python.

`rinexpy` is a substantially rewritten descendant of
[`georinex`](https://github.com/geospace-code/georinex). It keeps the public
API and the same xarray-flavored output, but the OBS3 / NAV3 hot paths have
been rewritten to eliminate the O(N^2) `xarray.merge`-per-epoch pattern. On
the shared test corpus this is **13-33x faster** for RINEX-3 NAV/OBS files
and ~2-8x faster for RINEX-2 NAV (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C
```

## Install

```sh
uv add rinexpy
# or, with all optional deps (Hatanaka/CRINEX, LZW, NetCDF, geodetic, plotting)
uv add 'rinexpy[all]'
```

Python 3.11+ is required; the project itself is developed against the latest
stable CPython (3.13.x).

## Compatibility

| Format                          | Status |
|---------------------------------|--------|
| RINEX 2 OBS / NAV               | full   |
| RINEX 3 / 4 OBS / NAV           | full   |
| Hatanaka CRINEX (`.crx`)        | full*  |
| GZIP / BZ2 / ZIP / LZW          | full*  |
| SP3-a / SP3-c / SP3-d           | full   |
| StringIO input                  | full   |
| NetCDF4 / HDF5 read / write     | full*  |

`*` requires the corresponding optional extra (`hatanaka`, `lzw`, `netcdf`).

## Public API

```python
rinexpy.load(path, *, use=None, tlim=None, meas=None,
             useindicators=False, fast=True, interval=None,
             out=None, overwrite=False, verbose=False)
rinexpy.rinexnav(path, *, use=None, tlim=None, ...)
rinexpy.rinexobs(path, *, use=None, tlim=None, ...)
rinexpy.batch_convert(path, glob, out, *, workers=None, ...)  # parallel
rinexpy.iter_obs3_epochs(path, *, use=None, tlim=None, interval=None)  # streaming
rinexpy.gettime(path)         # -> np.ndarray[datetime64]
rinexpy.rinexheader(path)     # -> dict
rinexpy.rinexinfo(path)       # -> {"version", "rinextype", "filetype", "systems"}
rinexpy.load_sp3(path)
rinexpy.keplerian2ecef(nav)   # -> (X, Y, Z)
```

For files larger than RAM:

```python
for time, ds in rinexpy.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(ds)
```

For optional plots:

```python
from rinexpy.plots import timeseries  # requires the `plot` extra
timeseries(rinexpy.load("foo.18o"))
```

For the numba-jitted hot path on huge OBS3 files:

```python
rinexpy.rinexobs(fn, use_jit=True)  # requires the `jit` extra; ~1.9x faster
```

## CLI

```sh
rinexpy read myfile.18o
rinexpy times myfile.18o
rinexpy convert path/to/data "*.rnx.gz" --out converted/
rinexpy info myfile.18o
```

## Why a rewrite?

The upstream `georinex` README explicitly notes that "`xarray.concat` and
`xarray.Dataset` nested inside `concat` takes over 60% of time" for OBS3.
`rinexpy` rewrites those readers to fill a preallocated NumPy buffer in a
single pass and build the `xarray.Dataset` exactly once at the end. See
[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) for the full list and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for measured wins.

## Citation

If you use this in academic work please also cite the upstream `georinex`:
[doi:10.5281/zenodo.2580306](https://doi.org/10.5281/zenodo.2580306).

## License

MIT, like the upstream project.
