# Tutorial

Install to integer-fix RTK in twelve sections. Snippets are meant to
be pasted into a Python REPL.

## 1. Install

rinexpy isn't on PyPI. Clone the repo and let `uv` set up the
environment. The [README](../README.md) has the full uv install
recipe.

```sh
git clone https://github.com/ksd3/rinexpy
cd rinexpy
uv sync --all-extras       # base + every reader extra + dev tools
```

Or pick the extras you need:

```sh
uv sync                       # base only: RINEX 2/3, SP3, NetCDF
uv sync --extra hatanaka      # + CRINEX (.crx) reads
uv sync --extra lzw           # + .Z (LZW) reads
uv sync --extra plot          # + matplotlib plotting helpers
uv sync --extra jit           # + numba JIT path for huge OBS3 files
uv sync --extra zarr          # + Zarr write
uv sync --extra geo           # + pymap3d helpers
```

Python 3.11+ is required.

Run the snippets below inside the project venv: `uv run python`, or
activate `.venv/bin/activate` first.

## 2. Read your first file

```python
import rinexpy as rp

obs = rp.load("tests/data/obs3.01gage.10o")
print(obs)
```

`load` auto-detects the file kind: RINEX 2/3 NAV/OBS, SP3-a/c/d, or
an already-converted NetCDF. The result is an `xarray.Dataset`, or a
`{"nav": ..., "obs": ...}` dict for NetCDFs holding both groups.

```python
obs.attrs["filename"]    # 'obs3.01gage.10o'
obs.sv.values            # array of SV labels
obs.time.values          # datetime64[ns] epochs
obs["C1C"]               # one measurement type as a (time, sv) DataArray
```

## 3. Filter cheaply

rinexpy drops records during the parse if they fall outside the
filter. Opening a 24-hour 1 Hz file and selecting a 5-minute window
takes milliseconds.

```python
obs = rp.load(
    "big.rnx.gz",
    use={"G", "E"},                          # only GPS + Galileo
    meas=["C1C", "L1C"],                     # only these labels
    tlim=("2018-07-29T12:00",                # only this 1-hour window
          "2018-07-29T13:00"),
    interval=30,                             # decimate to 30 s
)
```

## 4. Stream a file larger than RAM

Multi-day RINEX-3 at 1 Hz with full constellation coverage can
exceed local memory. Stream one epoch at a time:

```python
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(time, ds)
```

`ds` is a single-time `xarray.Dataset` sized to the SVs present at
that epoch. Memory footprint stays constant in file size.

## 5. Convert a directory in parallel

```python
written = rp.batch_convert(
    "data/2018",
    "*.rnx.gz",
    "out/",
    workers=8,                # parallel processes; 0 = all CPUs
)
print(f"wrote {len(written)} files")
```

Or from the shell:

```sh
rinexpy convert data/2018 "*.rnx.gz" --out out/ -j 8
```

## 6. SP3 + Lagrange interpolation

```python
sp3 = rp.load_sp3("tests/data/igs19362.sp3c")
sp3.position           # (time, sv, ECEF) DataArray in km

# Interpolate one SV's position at any time within the file:
import numpy as np
queries = np.array(["2017-02-14T03:14:15"], dtype="datetime64[ns]")
interp = rp.interpolate_sp3(sp3, queries, order=10)
print(interp.position.sel(sv="G05").values)
```

## 7. Skyplot from broadcast NAV

```python
import numpy as np
from rinexpy.geodesy import azimuth_elevation, lla_to_ecef
from rinexpy.plots import skyplot

nav = rp.load("tests/data/brdc2800.15n")
rx_ecef = lla_to_ecef(40.0, -3.0, 100.0)

sv_az_el = {}
for sv_label in nav.sv.values:
    if sv_label[0] not in {"G", "E"}:
        continue                       # skip non-Keplerian systems
    sv = nav.sel(sv=sv_label).dropna(dim="time", how="all")
    if sv.time.size == 0:
        continue
    X, Y, Z = rp.keplerian2ecef(sv)
    sv_ecef = np.stack([X.values, Y.values, Z.values], axis=-1)
    az, el = azimuth_elevation(rx_ecef, sv_ecef)
    sv_az_el[sv_label] = (az, el)

skyplot(sv_az_el, title="Sky from (40, -3)")
```

## 8. Single-point positioning

```python
import numpy as np
from rinexpy.geodesy import lla_to_ecef

# In real life: SVs from interpolated SP3, pseudoranges from OBS.
truth = np.array(lla_to_ecef(40, -3, 100))
sv = ...    # (n_sv, 3) ECEF
pr = ...    # (n_sv,)  pseudoranges in meters

sol = rp.spp_solve(sv, pr, max_iter=20)
print("Solved (lat, lon, alt):", sol["lla"])
print("Receiver clock bias:    ", sol["clock_bias"], "s")
```

`examples/06_spp_positioning.py` is a runnable noise-free demo.

## 9. RTK with LAMBDA integer fix

```python
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

# Two receivers (rover + base) observing the same SVs at the same epoch.
# pr_r, pr_b   = pseudoranges (m)        (n_sv,)
# phase_r, phase_b = carrier phase (cycles) (n_sv,)
# sv          = (n_sv, 3) ECEF satellite positions
# base_ecef   = known ECEF of the base receiver

sol = rtk_fix(
    pr_r, pr_b, phase_r, phase_b, sv, base_ecef,
    wavelength=LAMBDA_L1,
    sigma_pr=1.0,        # pseudorange noise (m)
    sigma_phase=0.005,   # carrier-phase noise (m)
    ratio_threshold=3.0,
)
if sol["fixed_accepted"]:
    print("Fixed baseline:", sol["fixed"]["baseline"])
else:
    print("Float baseline:", sol["float"]["baseline"])
    print("LAMBDA ratio test:", sol["lambda"]["ratio"])
```

`examples/07_rtk_baseline.py` is a noise-free synthetic demo that
recovers a 5.4 m baseline to mm precision.

## 10. NTRIP to RTCM3

Stream live RTCM3 from an NTRIP caster and decode messages:

```python
from rinexpy.ntrip import stream
from rinexpy.rtcm3 import iter_messages
import io

bytes_iter = stream(
    "rtk2go.com", "MOUNT01",
    user="me", password="secret",
    port=2101,
)

# Buffer the bytes for the framer (or feed it incrementally with a
# more involved adapter).
buf = io.BytesIO()
for chunk in bytes_iter:
    buf.write(chunk)
    if buf.tell() > 4096:
        break
buf.seek(0)

for msg in iter_messages(buf):
    print(msg["msg_id"], msg.get("station_id"))
```

Decoded message types and their fields are listed in `API.md`.

## 11. Tropospheric correction

For cm-precision PPP/SPP you'll want a real tropo model:

```python
from rinexpy.geodesy import vmf1, saastamoinen
from rinexpy.gpt2w import gpt2w, load_gpt2w_grid

grid = load_gpt2w_grid("/path/to/gpt2_5w.grd")
met = gpt2w(grid, lat_deg=40.0, lon_deg=-3.0,
            epoch=datetime(2024, 3, 14), altitude_m=100.0)
m_h, m_w = vmf1(met["a_h"], met["a_w"],
                el_deg=15.0, lat_deg=40.0, altitude_m=100.0, doy=80)
zhd = saastamoinen(90.0, 100.0,                  # zenith dry delay
                   pressure_hpa=met["pressure_hpa"],
                   temperature_k=met["temperature_k"],
                   humidity_e_hpa=met["e_hpa"])
slant_dry = zhd * m_h
```

For coarser work the bare `saastamoinen(el_deg, alt)` (default ICAO
atmosphere) is enough. At el >= 15° it's accurate to ~1 cm.

## 12. Quality control and bookkeeping

```python
from rinexpy.tools import validate_file, concat_files, diff_datasets

rep = validate_file("data/myfile.18o")
if not rep["ok"]:
    print("WARN:", rep["warnings"])

# Stitch a week of daily files together:
combined = concat_files(["data/d001.18o", "data/d002.18o", ...])

# Make sure two runs of the same file produce the same data:
a = rp.load("a.18o")
b = rp.load("b.18o")
delta = diff_datasets(a, b)
if not delta["equal"]:
    print(delta["differences"])
```

## What next?

- `examples/` has 8 runnable scripts covering each workflow above in
  more depth.
- `COOKBOOK.md` has short recipes for common one-liners.
- `API.md` is the per-symbol reference for everything exported.
- `OPTIMIZATIONS.md` and `BENCHMARKS.md` cover the performance story
  relative to georinex.
