# NetCDF and Zarr output

For long-term storage and fast subsequent loads, rinexpy can write the
parsed datasets to NetCDF4 or Zarr. The same datasets can be re-read
with `rinexpy.load`, so the round trip is a free win on archive
workflows.

## NetCDF

NetCDF4 / HDF5 is the default sink. The wire format mirrors the layout
that georinex uses, so older code that reads the output keeps working.

### One file at a time

The `out=` argument on `load` triggers a NetCDF write as the file is
parsed.

```python
import rinexpy as rp

rp.load("tests/data/demo.10o", out="demo.nc")
```

The output `demo.nc` is an HDF5 file with one group (`OBS` for
observation files, `NAV` for navigation files). Re-reading is automatic
because `load` sniffs the file kind.

```python
re = rp.load("demo.nc")
print(re)
```

For NetCDF files that hold both NAV and OBS (rare but supported), the
return is a dict:

```python
out = rp.load("combined.nc")
nav = out["nav"]
obs = out["obs"]
```

### Header and data compression

Every variable is written with `zlib` compression level 1 and a
fletcher32 checksum. The compression level is conservative; level 1 is
fast and still gives 2-3x size reduction over uncompressed.

The encoding dict that the writer applies is exposed for inspection:

```python
from rinexpy.netcdf import ENC
print(ENC)
# {'zlib': True, 'complevel': 1, 'fletcher32': True}
```

### Group-merge safety

When you write multiple files to the same NetCDF (writing OBS to a path
that already contains NAV, or vice versa), the writer merges the new
group into the existing file without touching the others.

```python
rp.load("station.rnx", out="station.nc")          # writes the OBS group
rp.load("station.nav", out="station.nc")          # adds the NAV group, OBS survives

# Re-reading sees both:
out = rp.load("station.nc")
print(out["nav"], out["obs"])
```

### Direct writer entry point

For custom workflows the low-level entry point is exported.

```python
from rinexpy.netcdf import write_dataset
from pathlib import Path

write_dataset(my_dataset, Path("custom.nc"), group="OBS", overwrite=True)
```

`overwrite=` controls what happens when the target file (and group)
exists. The default is `False`, which appends a new group if the file
has room; with `True` the file is replaced.

### Parallel batch conversion

The recommended path for converting many files at once.

```python
written = rp.batch_convert(
    path="data/2024",
    glob="*.rnx.gz",
    out="out/",
    workers=0,                       # 0 = all CPUs
    use={"G", "E"},
    interval=30,
)
print(f"converted {len(written)} files")
```

Errors on individual files are logged and the conversion continues. The
returned list is the paths of the successfully written files.

The `workers=` argument:

- `None` (default) or `1`: serial in the current process.
- `> 1`: spawn `multiprocessing.Pool(workers)`.
- `0` or negative: use every CPU.

The parallel path is the natural place to use `batch_convert`: each file
is independent, so the speedup is roughly linear in the number of CPUs.
On a modern 8-core laptop, converting a directory of 100 daily RINEX 3
OBS files drops from about 30 seconds (serial) to about 5 seconds
(parallel).

## Zarr

For cloud workflows where multiple workers read partial slices of the
same dataset, Zarr is the better answer. NetCDF stores the dataset as a
single HDF5 file; Zarr stores it as a directory of compressed chunks.
Zarr is also better at incremental writes (append-as-you-go) and at
streaming reads from S3 / GCS.

### Writing

```python
from rinexpy.zarr_io import to_zarr

obs = rp.load("tests/data/obs3.01gage.10o")
to_zarr(obs, "demo.zarr")
```

The Zarr store is a directory.

```sh
ls demo.zarr/
# .zattrs .zgroup .zmetadata C1C/ C1P/ C2P/ L1C/ L2P/ S1C/ S1P/ S2P/ ...
```

### Reading

Read with the standard xarray helper:

```python
import xarray as xr

obs = xr.open_zarr("demo.zarr")
```

The result is identical to a Zarr-backed Dataset; you can index and
compute over it the same way as the NetCDF return.

### Options

```python
from rinexpy.zarr_io import to_zarr

to_zarr(
    obs,
    "demo.zarr",
    mode="w",                        # write (default), 'a' for append
    consolidated=True,                # write consolidated metadata
)
```

`consolidated=True` writes a single metadata file at the top of the
store. This dramatically speeds up subsequent reads of the store's
schema (no need to list the bucket); always leave it on unless you have
a specific reason not to.

### Cloud destination

For an S3 destination, hand `to_zarr` an `s3fs.S3Map` rather than a
local path. The `zarr` package handles the rest:

```python
import s3fs
import zarr

fs = s3fs.S3FileSystem(...)
store = s3fs.S3Map(root="my-bucket/rinex/demo.zarr", s3=fs)
to_zarr(obs, store)
```

`to_zarr` accepts a `MutableMapping` (Zarr's storage interface) wherever
a path would normally go.

## Dask-backed multi-file reads

For datasets that exceed RAM but live in many smaller files (a directory
of daily NetCDFs, for example), `lazy.load_lazy` builds a chunked,
dask-backed Dataset.

```python
from rinexpy.lazy import load_lazy
from pathlib import Path

files = sorted(Path("data/2024").glob("*.nc"))
ds = load_lazy(files, chunk_size={"time": 3600})

# All xarray operations work lazily:
mean_l1c = ds.L1C.mean(dim="time").compute()
```

The function falls back to eager `tools.concat_files` if dask is not
installed.

For a Zarr store there is no need for `load_lazy`; `xr.open_zarr` is
already lazy.

## Picking the right sink

| Use case | Sink |
| --- | --- |
| Single-machine archival | NetCDF |
| Multi-worker / cloud workflows | Zarr |
| Quick re-read on the same machine | NetCDF |
| Append-as-you-go data accumulation | Zarr |
| Compatibility with georinex / older code | NetCDF |

For the vast majority of single-machine workflows, NetCDF is the right
choice. The Zarr option exists for the cases that need it.

## Related pages

- [RINEX observation files](../formats/rinex-obs.md): the source of the parsed dataset.
- [Multi-file tools](multi-file.md): `concat_files`, `validate_file`, `diff_datasets`.
- [Streaming over RAM-sized files](streaming.md): when the file is too big to materialise.
- [Async loading](async.md): the asyncio wrappers.
