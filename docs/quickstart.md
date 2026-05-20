# Quickstart

Five worked examples that run against files in `tests/data/`. Paste them
straight into a REPL.

Either prefix the snippets with `uv run python`, or activate the venv first:

```sh
source .venv/bin/activate
```

## 1. Open a RINEX file

```python
import rinexpy as rp

obs = rp.load("tests/data/obs3.01gage.10o")
print(obs)
```

`load` autodetects the file kind. It accepts RINEX 2 and RINEX 3 observation
files, RINEX 2 and 3 navigation files (RINEX 4 NAV records load through the
dedicated [`rinexpy.nav4.load_nav4`](formats/rinex-nav.md) function), SP3
ephemeris files, and pre-converted NetCDF files. Compressed forms
(`.gz`, `.bz2`, `.zip`, `.Z`, `.crx`, `.crx.gz`) are decompressed
transparently as long as the matching optional extra is installed.

The return value is an `xarray.Dataset`. You can select a satellite and a
measurement code with normal xarray indexing.

```python
obs.sv.values            # array of SV labels, sorted by system then PRN
obs.time.values          # datetime64 epochs
obs["C1C"]               # one measurement type as a (time, sv) DataArray
obs.sel(sv="G07").C1C    # one SV, one observable, as a 1-D series
```

The attributes dictionary carries the header information you usually care
about.

```python
obs.attrs["filename"]            # 'obs3.01gage.10o'
obs.attrs["position"]            # ECEF station coordinates from the header
obs.attrs["position_geodetic"]   # (lat_deg, lon_deg, alt_m)
obs.attrs["interval"]            # nominal sampling interval in seconds
```

## 2. Filter while reading

Every filter you pass to `load` applies during parsing. Records that
do not match are skipped before decoding, so opening a 24-hour 1 Hz file and
keeping a five-minute window is cheap.

```python
obs = rp.load(
    "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip",
    use={"G"},                # only GPS
    meas=["C1C", "L1C"],      # only these two measurement labels
    interval=60,              # decimate to 60 s
)
print(obs.sv.size, "SVs after filtering")
```

The keyword arguments stack: `use=` reduces the SV axis, `meas=` reduces
the variables, `tlim=` reduces the time axis, and `interval=` decimates the
result by an integer factor of the input rate.

## 3. Stream a file you do not want to load whole

For files that are large enough to be uncomfortable in RAM, the
streaming iterator yields one epoch at a time. The memory usage is
constant in the file size.

```python
for i, (time, ds) in enumerate(rp.iter_obs3_epochs(
    "tests/data/obs3.01gage.10o",
    use={"G"},
)):
    print(time, ds.sv.size, "SVs visible")
    if i >= 1:
        break
```

`ds` is a single-time `xarray.Dataset` shaped to the SVs that are present
at that epoch. The iterator honours the same `use=`, `tlim=`, and
`interval=` filters as `load`.

## 4. Compute a single-point position

A receiver in WGS-84 sees a set of satellites at known ECEF positions and
records a pseudorange to each. Single-point positioning solves the
four-unknown set `(x, y, z, dt_rx)` by iterative least squares. The example
below builds a small synthetic so that you can see the solver converge
against a known truth.

```python
import numpy as np
import rinexpy as rp
from rinexpy.geodesy import lla_to_ecef

C_M_PER_S = 299_792_458.0

truth_rx = np.array(lla_to_ecef(40.0, -3.0, 100.0))

sv_radius = 2.66e7
az_el = [(0, 70), (60, 30), (120, 50), (200, 20), (260, 60), (320, 40)]
sv = []
lat, lon = np.radians(40.0), np.radians(-3.0)
sl, cl = np.sin(lon), np.cos(lon)
sp, cp = np.sin(lat), np.cos(lat)
for az, el in az_el:
    a = np.radians(az)
    elr = np.radians(el)
    e = np.cos(elr) * np.sin(a)
    n = np.cos(elr) * np.cos(a)
    u = np.sin(elr)
    x = -sl * e - sp * cl * n + cp * cl * u
    y = cl * e - sp * sl * n + cp * sl * u
    z = cp * n + sp * u
    sv.append(truth_rx + sv_radius * np.array([x, y, z]))
sv = np.array(sv)

bias_s = 1e-4
pseudoranges = np.linalg.norm(sv - truth_rx, axis=1) + C_M_PER_S * bias_s

sol = rp.spp_solve(sv, pseudoranges, max_iter=20)
print("solved ECEF:", sol["position"])
print("solved LLA: ", sol["lla"])
print("clock bias: ", sol["clock_bias"], "s")
print("iterations: ", sol["n_iter"])
```

For a noise-free input the solver converges in six iterations to the
truth (floating-point precision). With real noisy pseudoranges and
real satellite geometry you can expect 5-10 metres of horizontal error
on a single epoch of GPS-only L1 SPP, dropping to a metre or so with the
dual-frequency iono-free combination plus a tropospheric correction.

The full SPP page has the noisier real-data flow, the RAIM variant, and
the DCB / TGD corrections. See [Single-point positioning](positioning/spp.md).

## 5. Convert a directory to NetCDF

For long-term archival or fast subsequent loads, NetCDF stores the parsed
xarray dataset directly with light compression. `batch_convert` walks a
directory, applies the same filters as `load` to each file, and writes the
results in parallel.

```python
import shutil, tempfile, warnings
from pathlib import Path
import rinexpy as rp

out_dir = Path(tempfile.mkdtemp(prefix="rinexpy_out_"))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    written = rp.batch_convert(
        "tests/data",
        "*.10o",
        out_dir,
        workers=2,        # 0 means use every CPU
    )
print(f"wrote {len(written)} files")
for p in written[:3]:
    print(f"  {p.name}  {p.stat().st_size/1024:5.1f} KB")

shutil.rmtree(out_dir, ignore_errors=True)
```

The same operation from the shell:

```sh
uv run rinexpy convert tests/data "*.10o" --out out/ -j 2
```

Reading the result back is just `rp.load("out/foo.10o.nc")`.
