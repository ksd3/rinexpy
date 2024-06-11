# Cookbook

Recipes you can paste into a REPL. Each one stands alone; assume
`import rinexpy as rp` at the top.

## Read

### Load a file, any format

`rp.load` sniffs the file and dispatches. Works on NAV, OBS, SP3,
and NetCDF.

```python
data = rp.load("file.rnx.gz")           # auto-detect: NAV / OBS / SP3 / NetCDF
```

### One variable, one satellite

Pull a single observable for one SV via xarray selection.

```python
obs = rp.load("file.18o")
gps_g07_c1 = obs["C1"].sel(sv="G07")
```

### Timestamps without the data

Useful when you only need to see what's in a file before paying to
parse it.

```python
times = rp.gettime("file.18o")          # numpy.datetime64 array
```

### Just the header

Returns a plain dict keyed by RINEX header label.

```python
hdr = rp.rinexheader("file.18o")
print(hdr["APPROX POSITION XYZ"])
print(hdr["TIME OF FIRST OBS"])
```

### Restrict by time window during the read

Pushed down into the parser, so epochs outside the window aren't
materialized.

```python
obs = rp.load("big.rnx.gz", tlim=("2018-07-29T12:00", "2018-07-29T13:00"))
```

### Restrict by constellation

Pass a single letter or a set. Other systems get skipped while
reading.

```python
obs_gps_only = rp.load("mixed.rnx", use="G")            # GPS only
obs_gps_gal  = rp.load("mixed.rnx", use={"G", "E"})     # GPS + Galileo
```

### Decimate while reading

Drops epochs at parse time. Cheaper than reading everything and
slicing after.

```python
obs_30s = rp.load("1hz.rnx.gz", interval=30)
```

### Carrier-phase only

Pick the observables you care about and skip the rest.

```python
obs = rp.load("file.18o", meas=["L1", "L2"])
```

## Write

### RINEX to NetCDF

`out=` on `load` writes as it parses. Use `batch_convert` for a
whole directory.

```python
rp.load("input.18o", out="output.nc")               # one file
rp.batch_convert("data/", "*.18o", "out/", workers=4)   # whole directory
```

### OBS dataset back to RINEX 3

Round-trip a filtered slice through the RINEX 3 writer.

```python
obs = rp.load("input.18o", tlim=("2018-07-29", "2018-07-30"))
rp.to_rinex_obs(obs, "filtered.rnx", version=3)
```

### Zarr for cloud workflows

NetCDF is the default sink; Zarr is there when you need it.

```python
from rinexpy.zarr_io import to_zarr

obs = rp.load("file.18o")
to_zarr(obs, "out.zarr")
```

## Stream

### Iterate epoch by epoch

Holds one epoch in memory at a time, so file size stops mattering.

```python
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    do_something(time, ds)
```

### Stream with filters

All the `load` filters work on the iterator too.

```python
for time, ds in rp.iter_obs3_epochs(
    "huge.rnx.gz",
    use={"G"},
    tlim=("2018-07-29T12:00", "2018-07-29T13:00"),
    interval=30,
):
    do_something(time, ds)
```

## Multi-file

### Concatenate daily files into a week

Joins along the time axis. Handles small header drifts.

```python
from rinexpy.tools import concat_files

obs_week = concat_files(["data/d001.18o", "data/d002.18o", "data/d003.18o"])
```

### Diff two datasets

Returns a dict with `equal` and a list of the first differences it
finds. Handy when chasing parser regressions.

```python
from rinexpy.tools import diff_datasets

a = rp.load("v1.18o")
b = rp.load("v2.18o")
delta = diff_datasets(a, b)
if not delta["equal"]:
    for diff in delta["differences"]:
        print(diff)
```

### Parallel loads via asyncio

`aload_many` farms the reads out across the event loop. The
parsers themselves still run on threads.

```python
import asyncio
from rinexpy.asyncio import aload_many

results = asyncio.run(aload_many(["a.18o", "b.18o", "c.18o"]))
```

## Geodesy

### ECEF and geodetic

WGS-84 by default.

```python
from rinexpy.geodesy import ecef_to_lla, lla_to_ecef

x, y, z = lla_to_ecef(40.0, -3.0, 100.0)
lat, lon, alt = ecef_to_lla(x, y, z)
```

### Azimuth and elevation

Receiver position in ECEF, satellites in ECEF, angles in degrees.

```python
from rinexpy.geodesy import azimuth_elevation
import numpy as np

rx = lla_to_ecef(40.0, -3.0, 100.0)
sv_ecef = np.array([[2.66e7, 0, 0], [0, 2.66e7, 0]])
az, el = azimuth_elevation(rx, sv_ecef)        # degrees
```

### DOP

GDOP / PDOP / HDOP / VDOP / TDOP from the geometry matrix.

```python
from rinexpy.geodesy import dop

out = dop(sv_ecef, rx)
print(out["GDOP"], out["PDOP"], out["HDOP"])
```

### Tropospheric delay

Saastamoinen with default standard-atmosphere values. Pass `p`,
`t`, `e` if you have met data.

```python
from rinexpy.geodesy import saastamoinen

slant_delay_m = saastamoinen(el_deg=15.0, altitude_m=100.0)
```

### Klobuchar ionospheric correction

Coefficients come from the GPS NAV header.

```python
from rinexpy.geodesy import klobuchar

# alpha, beta from the GPS NAV header (8 coefficients)
delay_m = klobuchar(alpha, beta, (lat, lon, alt),
                    sv_az_deg=180, sv_el_deg=30, gps_sec=43200)
```

## Time

### GPS week and datetime

Round-trips through GPS week / seconds-of-week.

```python
from rinexpy.gpstime import datetime_to_gps, gps_to_datetime
from datetime import datetime

week, sow = datetime_to_gps(datetime(2018, 1, 7))
back = gps_to_datetime(week, sow)
```

### Resolve a 10-bit week number

Broadcast week numbers roll over every 1024 weeks. Pass a hint
date to disambiguate.

```python
from rinexpy.gpstime import gps_week_rollover

full_week = gps_week_rollover(0, datetime.utcnow())
```

## SP3

### Load and interpolate one satellite

Lagrange interpolation, default order 10. Returns ECEF in km.

```python
sp3 = rp.load_sp3("igs19362.sp3c")
interp = rp.interpolate_sp3(sp3, datetime(2017, 2, 14, 3, 14, 15))
g05_pos = interp.position.sel(sv="G05").values     # ECEF in km
```

## RTK / Positioning

### Single-point positioning

Iterative least-squares on pseudoranges. Returns ECEF, LLA, clock
bias, and residuals.

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m, max_iter=20)
print(sol["lla"])
```

### Float baseline

Double-difference solve without integer fixing. Use this as the
input to LAMBDA.

```python
from rinexpy.rtk import double_difference_solve
from rinexpy.multifreq import LAMBDA_L1

sol = double_difference_solve(
    pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
    wavelength=LAMBDA_L1,
)
print("Baseline (m):", sol["baseline"])
```

### Full LAMBDA fix

`rtk_fix` runs the float solve, fixes integers via LAMBDA, and
applies the ratio test. `fixed_accepted` tells you whether the
fix passed.

```python
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

sol = rtk_fix(
    pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
    wavelength=LAMBDA_L1,
    ratio_threshold=3.0,
)
if sol["fixed_accepted"]:
    print("cm-level baseline:", sol["fixed"]["baseline"])
```

### Wide-lane ambiguity resolution

For dual-frequency receivers. Fixes the WL combination first, then
backs out N1 and N2.

```python
from rinexpy.multifreq import lambda_dual_freq

out = lambda_dual_freq(a_l1_float, a_l2_float,
                       p1_m=pr_l1, p2_m=pr_l2)
print("Fixed N1:", out["N_L1"])
print("Fixed N2:", out["N_L2"])
print("Fraction fixed:", out["fraction_fixed"])
```

## Plotting

### L1 phase time series

One line per satellite. Needs the `[plot]` extra.

```python
from rinexpy.plots import obstimeseries
import matplotlib.pyplot as plt

obstimeseries(rp.load("file.18o"))
plt.show()
```

### Skyplot

See `TUTORIAL.md` § 7.

## RTCM3 / NTRIP

### Decode a captured RTCM3 stream

`iter_messages` yields one parsed message at a time. `check_crc`
trades a little speed for catching bit errors.

```python
import io
from rinexpy.rtcm3 import iter_messages

with open("capture.bin", "rb") as fp:
    for msg in iter_messages(fp, check_crc=True):
        print(msg["msg_id"])
```

### Pull a sourcetable, then stream

Sourcetables list the mountpoints a caster offers. Pick one and
open a stream.

```python
from rinexpy.ntrip import fetch_sourcetable, stream

mountpoints = fetch_sourcetable("rtk2go.com", port=2101)
for mp in mountpoints:
    if mp["type"] == "STR":
        print(mp["mountpoint"], mp["format"])

bytes_iter = stream("rtk2go.com", "MOUNT01",
                    user="me", password="x", port=2101)
for chunk in bytes_iter:
    process(chunk)
```

## Receiver-format decoders

### NMEA-0183 from a serial log

Plain text. `iter_lines` returns dicts keyed by sentence type.

```python
from rinexpy.nmea import iter_lines

with open("nmea.log") as fp:
    for msg in iter_lines(fp):
        if msg["type"] == "GGA":
            print(msg["lat"], msg["lon"], msg["altitude_m"])
```

### u-blox UBX

Match on `(msg_class, msg_id)`. The pair `(0x01, 0x07)` is NAV-PVT.

```python
from rinexpy.ubx import iter_messages

with open("ublox.ubx", "rb") as fp:
    for msg in iter_messages(fp):
        if (msg["msg_class"], msg["msg_id"]) == (0x01, 0x07):  # NAV-PVT
            print(msg["lat_deg"], msg["lon_deg"], msg["fix_type"])
```

### Septentrio SBF

Block ID 4007 is PVTGeodetic.

```python
from rinexpy.sbf import iter_blocks

with open("septentrio.sbf", "rb") as fp:
    for blk in iter_blocks(fp):
        if blk["block_id"] == 4007:                  # PVTGeodetic
            print(blk["lat_rad"], blk["lon_rad"], blk["height_m"])
```

### NovAtel OEM

Message 42 is BESTPOS.

```python
from rinexpy.novatel import iter_messages

with open("novatel.bin", "rb") as fp:
    for msg in iter_messages(fp):
        if msg["msg_id"] == 42:                      # BESTPOS
            print(msg["lat_deg"], msg["lon_deg"], msg["height_m"])
```

### Walk a BINEX archive

UNAVCO's container format. Bodies come back as raw bytes; record
IDs and lengths are parsed.

```python
from rinexpy.binex import iter_records

with open("archive.bnx", "rb") as fp:
    for rec in iter_records(fp):
        print(f"record {rec['record_id']:#04x}  {rec['length']} bytes")
```

### RTCM 2.x DGPS

Legacy format from beacon receivers. Message 1 carries the
pseudorange corrections.

```python
from rinexpy.rtcm2 import iter_messages

with open("dgps.rtcm2", "rb") as fp:
    for msg in iter_messages(fp):
        if msg["msg_type"] == 1:                  # pseudorange corrections
            for c in msg["corrections"]:
                print(f"  SV {c['sat_id']:2d}  PRC={c['prc_m']:+.2f} m"
                      f"  RRC={c['rrc_m_s']:+.4f} m/s  IODE={c['iode']}")
```

### BeiDou D1 subframe 1

Subframe 1 carries the clock parameters and iono coefficients.
Feed it ten 30-bit words from your receiver or from RTCM3 1042.

```python
from rinexpy.beidou import decode_d1_subframe1

# 10 30-bit words from your receiver / RTCM3 1042 / RXM-SFRBX
words = [...]
nav = decode_d1_subframe1(words)
print(f"BDS week {nav['week']}, t_oc {nav['t_oc_s']} s, "
      f"a0 {nav['a0_s']:e}")
```

## QC + tooling

### Quick QC report

Counts epochs, infers the nominal interval, flags gaps and bad
records.

```python
from rinexpy.tools import validate_file

rep = validate_file("file.18o")
print(rep["n_epochs"], rep["interval_seconds"], rep["warnings"])
```

### Walk a directory looking for bad files

Pair `validate_file` with `Path.glob` to triage an archive.

```python
from pathlib import Path
from rinexpy.tools import validate_file

for p in Path("data/").glob("*.rnx.gz"):
    rep = validate_file(p)
    if not rep["ok"]:
        print(p, rep["warnings"])
```

## CLI shortcuts

```sh
rinexpy info file.18o                # parsed header
rinexpy times file.18o               # epoch list
rinexpy read file.18o                # parse + print
rinexpy convert data/ '*.18o' --out out/ -j 4
```
