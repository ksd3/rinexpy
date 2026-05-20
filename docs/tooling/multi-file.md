# Multi-file tools

`rinexpy.tools` is the small module of high-level helpers that work
across files. Three functions sit at the top of the module:
`validate_file`, `concat_files`, and `diff_datasets`.

## validate_file

Walks the header and the first few epochs of a file, computes a small
QC report.

```python
from rinexpy.tools import validate_file

rep = validate_file("tests/data/demo.10o")
print(rep)
```

Output:

```
{
  'ok': True,
  'warnings': [],
  'info': {'version': 2.11, 'filetype': 'O', 'rinextype': 'obs', 'systems': 'M'},
  'n_epochs': 2,
  'n_sv': 14,
  'time_first': '2010-03-05T00:00:00.000000000',
  'time_last':  '2010-03-05T00:00:30.000000000',
  'interval_seconds': 30.0,
  'gap_count': 0,
}
```

`ok` is True when the file parses cleanly and contains at least one
epoch. `warnings` collects any non-fatal parser warnings: malformed
records, gaps beyond the nominal interval, header inconsistencies.

For QC at scale, walk a directory:

```python
from pathlib import Path
for p in sorted(Path("data/2024").glob("*.rnx.gz")):
    rep = validate_file(p)
    if not rep["ok"]:
        print(p.name, rep["warnings"])
```

The function is intentionally light: it does not run cycle slip
detection, multipath analysis, or carrier-phase fitting. For those, see
[QC and cycle slips](../quality/qc.md).

## concat_files

Concatenate multiple parsed RINEX files along the time axis. Handles
small header drifts between days (different antenna types, slightly
different observation labels, ...) by taking the union of fields.

```python
from rinexpy.tools import concat_files

obs_week = concat_files([
    "data/day001.18o",
    "data/day002.18o",
    "data/day003.18o",
    "data/day004.18o",
    "data/day005.18o",
    "data/day006.18o",
    "data/day007.18o",
])
print(obs_week.time.size, "epochs across all 7 days")
```

The function:

1. Loads each input through `rinexpy.load`.
2. Aligns the SV axis (union of SVs across files).
3. Aligns the variable axis (union of measurement labels).
4. Concatenates along `time` and drops duplicates on `(time, sv)`.

The `dim=` argument lets you concatenate along a different axis, but
`time` is the default and is what you want for daily-RINEX stitching.

For RINEX 2 inputs the output is in the RINEX 2 schema. For mixed RINEX 2
+ RINEX 3 inputs, the RINEX 2 labels are auto-promoted to their RINEX 3
equivalents.

The CLI `splice` subcommand wraps this for two-file inputs:

```sh
uv run rinexpy splice day001.18o day002.18o --out joined.18o
```

## diff_datasets

Find the first per-variable difference between two parsed datasets. Used
mostly for regression testing and round-trip verification.

```python
from rinexpy.tools import diff_datasets
import rinexpy as rp

a = rp.load("v1.18o")
b = rp.load("v2.18o")

delta = diff_datasets(a, b)
if not delta["equal"]:
    for d in delta["differences"]:
        print(d)
```

The return dict has:

```
{
  "equal": False,
  "differences": [
    {"variable": "C1C", "max_abs_diff": 0.005, "first_index": (3, 7),
     "a_value": 24123456.789, "b_value": 24123456.794},
    ...
  ]
}
```

`equal` is True only if every variable agrees within `(rtol, atol)`. The
defaults are `rtol=1e-6` and `atol=1e-9`. The function walks variables in
the order returned by `data_vars` and stops at the first per-variable
difference (so the list has one entry per variable that diverges, not one
per epoch).

## Round-trip verification

Pair `diff_datasets` with `to_rinex_obs` and `to_zarr` to verify that
your processing preserves the data.

```python
import tempfile
import rinexpy as rp
from rinexpy.tools import diff_datasets
from rinexpy.zarr_io import to_zarr
import xarray as xr

obs = rp.load("input.18o")

# Round-trip through RINEX 3 OBS:
with tempfile.NamedTemporaryFile(suffix=".rnx") as t:
    rp.to_rinex_obs(obs, t.name, version=3)
    obs_back = rp.load(t.name)

print(diff_datasets(obs, obs_back, atol=1e-3))   # round-trip is lossy at the mm level

# Round-trip through Zarr:
to_zarr(obs, "tmp.zarr")
obs_zarr = xr.open_zarr("tmp.zarr")
print(diff_datasets(obs, obs_zarr))              # round-trip is exact
```

Zarr and NetCDF round-trips are exact (within floating-point precision).
RINEX round-trips lose a little precision because the file format uses
fixed-decimal text representation.

## batch_convert

The parallel converter is in `rinexpy.api`, but it pairs naturally with
the tools above. The detailed page is in
[NetCDF and Zarr output](io.md#parallel-batch-conversion).

```python
written = rp.batch_convert(
    "data/", "*.18o", "out/",
    workers=0,                          # 0 = all CPUs
    use={"G"},
    interval=30,
)
```

For one-off jobs the CLI is the cleaner path:

```sh
uv run rinexpy convert data/ '*.18o' --out out/ -j 0 --use G --interval 30
```

## Related pages

- [QC and cycle slips](../quality/qc.md): deeper data-quality checks.
- [NetCDF and Zarr output](io.md): the persisted-form options.
- [Command-line interface](cli.md): the CLI subcommands that wrap these.
- [Streaming over RAM-sized files](streaming.md): when the file is too big to load.
