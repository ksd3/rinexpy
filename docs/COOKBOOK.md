# Cookbook

Short, copy-pasteable recipes for the common one-shot tasks. Each
recipe is independent; assume `import rinexpy as rp` at the top.

## Read

### Load a single file regardless of format

```python
data = rp.load("file.rnx.gz")           # auto-detect: NAV / OBS / SP3 / NetCDF
```

### Open one variable for one satellite

```python
obs = rp.load("file.18o")
gps_g07_c1 = obs["C1"].sel(sv="G07")
```

### Get just the timestamps without parsing the data

```python
times = rp.gettime("file.18o")          # numpy.datetime64 array
```

### Get just the header

```python
hdr = rp.rinexheader("file.18o")
print(hdr["APPROX POSITION XYZ"])
print(hdr["TIME OF FIRST OBS"])
```

### Cheaply restrict to a time window during the read

```python
obs = rp.load("big.rnx.gz", tlim=("2018-07-29T12:00", "2018-07-29T13:00"))
```

### Cheaply restrict to a single GNSS

```python
obs_gps_only = rp.load("mixed.rnx", use="G")            # GPS only
obs_gps_gal  = rp.load("mixed.rnx", use={"G", "E"})     # GPS + Galileo
```

### Decimate to every N seconds while reading

```python
obs_30s = rp.load("1hz.rnx.gz", interval=30)
```

### Read only the carrier-phase observations

```python
obs = rp.load("file.18o", meas=["L1", "L2"])
```

## Write

### Convert RINEX -> NetCDF

```python
rp.load("input.18o", out="output.nc")               # one file
rp.batch_convert("data/", "*.18o", "out/", workers=4)   # whole directory
```

### Convert OBS dataset back to RINEX 3

```python
obs = rp.load("input.18o", tlim=("2018-07-29", "2018-07-30"))
rp.to_rinex_obs(obs, "filtered.rnx", version=3)
```

### Write to Zarr instead of NetCDF (cloud workflows)

```python
from rinexpy.zarr_io import to_zarr

obs = rp.load("file.18o")
to_zarr(obs, "out.zarr")
```

## Stream

### Process one epoch at a time without loading the whole file

```python
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    do_something(time, ds)
```

### Stream + filter + decimate

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

### Concatenate a week of daily files

```python
from rinexpy.tools import concat_files

obs_week = concat_files(["data/d001.18o", "data/d002.18o", "data/d003.18o"])
```

### Compare two parsed datasets and find the first divergence

```python
from rinexpy.tools import diff_datasets

a = rp.load("v1.18o")
b = rp.load("v2.18o")
delta = diff_datasets(a, b)
if not delta["equal"]:
    for diff in delta["differences"]:
        print(diff)
```

### Concurrent multi-file load via asyncio

```python
import asyncio
from rinexpy.asyncio import aload_many

results = asyncio.run(aload_many(["a.18o", "b.18o", "c.18o"]))
```

## Geodesy

### Convert ECEF <-> geodetic

```python
from rinexpy.geodesy import ecef_to_lla, lla_to_ecef

x, y, z = lla_to_ecef(40.0, -3.0, 100.0)
lat, lon, alt = ecef_to_lla(x, y, z)
```

### Compute az/el from receiver to satellites

```python
from rinexpy.geodesy import azimuth_elevation
import numpy as np

rx = lla_to_ecef(40.0, -3.0, 100.0)
sv_ecef = np.array([[2.66e7, 0, 0], [0, 2.66e7, 0]])
az, el = azimuth_elevation(rx, sv_ecef)        # degrees
```

### Compute DOP

```python
from rinexpy.geodesy import dop

out = dop(sv_ecef, rx)
print(out["GDOP"], out["PDOP"], out["HDOP"])
```

### Apply a tropospheric correction

```python
from rinexpy.geodesy import saastamoinen

slant_delay_m = saastamoinen(el_deg=15.0, altitude_m=100.0)
```

### Apply Klobuchar broadcast iono correction

```python
from rinexpy.geodesy import klobuchar

# alpha, beta from the GPS NAV header (8 coefficients)
delay_m = klobuchar(alpha, beta, (lat, lon, alt),
                    sv_az_deg=180, sv_el_deg=30, gps_sec=43200)
```

## Time

### GPS week <-> datetime

```python
from rinexpy.gpstime import datetime_to_gps, gps_to_datetime
from datetime import datetime

week, sow = datetime_to_gps(datetime(2018, 1, 7))
back = gps_to_datetime(week, sow)
```

### Resolve a 10-bit broadcast week number

```python
from rinexpy.gpstime import gps_week_rollover

full_week = gps_week_rollover(0, datetime.utcnow())
```

## SP3

### Load + interpolate one satellite at one time

```python
sp3 = rp.load_sp3("igs19362.sp3c")
interp = rp.interpolate_sp3(sp3, datetime(2017, 2, 14, 3, 14, 15))
g05_pos = interp.position.sel(sv="G05").values     # ECEF in km
```

## RTK / Positioning

### Single-point positioning

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m, max_iter=20)
print(sol["lla"])
```

### Float-only RTK baseline

```python
from rinexpy.rtk import double_difference_solve
from rinexpy.multifreq import LAMBDA_L1

sol = double_difference_solve(
    pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
    wavelength=LAMBDA_L1,
)
print("Baseline (m):", sol["baseline"])
```

### Full LAMBDA RTK fix

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

### Resolve dual-frequency ambiguities via Wide-Lane

```python
from rinexpy.multifreq import lambda_dual_freq

out = lambda_dual_freq(a_l1_float, a_l2_float,
                       p1_m=pr_l1, p2_m=pr_l2)
print("Fixed N1:", out["N_L1"])
print("Fixed N2:", out["N_L2"])
print("Fraction fixed:", out["fraction_fixed"])
```

## Plotting

### Time series of L1 carrier phase per satellite

```python
from rinexpy.plots import obstimeseries
import matplotlib.pyplot as plt

obstimeseries(rp.load("file.18o"))
plt.show()
```

### Skyplot of all satellites in a NAV file

See the recipe in `TUTORIAL.md` § 7.

## RTCM3 / NTRIP

### Decode a captured RTCM3 byte stream

```python
import io
from rinexpy.rtcm3 import iter_messages

with open("capture.bin", "rb") as fp:
    for msg in iter_messages(fp, check_crc=True):
        print(msg["msg_id"])
```

### Connect to an NTRIP caster and stream

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

### Decode NMEA-0183 from a serial-port log

```python
from rinexpy.nmea import iter_lines

with open("nmea.log") as fp:
    for msg in iter_lines(fp):
        if msg["type"] == "GGA":
            print(msg["lat"], msg["lon"], msg["altitude_m"])
```

### Decode u-blox UBX from a binary capture

```python
from rinexpy.ubx import iter_messages

with open("ublox.ubx", "rb") as fp:
    for msg in iter_messages(fp):
        if (msg["msg_class"], msg["msg_id"]) == (0x01, 0x07):  # NAV-PVT
            print(msg["lat_deg"], msg["lon_deg"], msg["fix_type"])
```

### Decode Septentrio SBF blocks

```python
from rinexpy.sbf import iter_blocks

with open("septentrio.sbf", "rb") as fp:
    for blk in iter_blocks(fp):
        if blk["block_id"] == 4007:                  # PVTGeodetic
            print(blk["lat_rad"], blk["lon_rad"], blk["height_m"])
```

### Decode NovAtel OEM logs

```python
from rinexpy.novatel import iter_messages

with open("novatel.bin", "rb") as fp:
    for msg in iter_messages(fp):
        if msg["msg_id"] == 42:                      # BESTPOS
            print(msg["lat_deg"], msg["lon_deg"], msg["height_m"])
```

### Walk a UNAVCO BINEX archive

```python
from rinexpy.binex import iter_records

with open("archive.bnx", "rb") as fp:
    for rec in iter_records(fp):
        print(f"record {rec['record_id']:#04x}  {rec['length']} bytes")
```

## QC + tooling

### Quick QC report on a file

```python
from rinexpy.tools import validate_file

rep = validate_file("file.18o")
print(rep["n_epochs"], rep["interval_seconds"], rep["warnings"])
```

### Walk a directory finding bad files

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
