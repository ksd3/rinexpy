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

### SPP with RAIM fault detection

`raim=True` runs a chi-squared residual test after each fix and
excludes the worst SV until the test passes (or `max_exclusions`
runs out).

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m,
                   raim=True, sigma_pr=5.0, p_fa=1e-4)
if sol["fault_detected"]:
    print("excluded:", sol["excluded_svs"])
print("lla:", sol["lla"])
```

### SPP with DCB or broadcast TGD applied

```python
from rinexpy.dcb import read_bsx
from rinexpy.positioning import tgd_from_nav

# Either SINEX-BIAS DCBs:
records = read_bsx("CAS0MGXRAP_20231560000_01D_01D_DCB.BSX")
sol = rp.spp_solve(
    sv_ecef, pseudoranges_m,
    sv_labels=["G05", "G10", ...],
    dcb_records=records,
    dcb_obs_code="C1W",
)

# Or broadcast TGD pulled from the NAV file:
nav = rp.load("brdc2800.15n")
tgd = tgd_from_nav(nav, epoch)
sol = rp.spp_solve(sv_ecef, pseudoranges_m,
                   sv_labels=sv_labels, tgd_map=tgd, tgd_gamma=1.0)
```

### Sequential RTK with ambiguity carry-over

Re-uses the integer fix across epochs while the per-SV lock holds.
Drops the lock and re-bootstraps on detected cycle slips.

```python
from rinexpy.rtk import SequentialRTK
from rinexpy.multifreq import LAMBDA_L1

rtk = SequentialRTK(base_ecef, wavelength=LAMBDA_L1)
for epoch in epochs:
    out = rtk.update(svs, rover_pr, base_pr, rover_phase, base_phase, sv_ecef)
    record(epoch, out["baseline"], out["fixed_accepted"], out["slipped_svs"])
```

## Precise positioning (PPP)

### One-shot static PPP from SP3 + CLK

```python
from rinexpy.ppp import ppp_solve

out = ppp_solve(obs, sp3, clk,
                initial_position_ecef=tuple(approx_xyz),
                elevation_mask_deg=7.0)
print(out["position"], "+/-", out["position_sigma_m"])
```

### PPP with the full correction stack

```python
from rinexpy.antex import find_antenna, load_antex
from rinexpy.gpt2w import load_gpt2w_grid
from rinexpy.dcb_download import auto_load_dcb

ant = find_antenna(load_antex("igs20.atx"), "TRM59800.00     NONE")
gpt = load_gpt2w_grid("/path/to/gpt2_5w.grd")
dcb = auto_load_dcb(obs.time.values[0].astype("datetime64[D]").astype(object))

out = ppp_solve(
    obs, sp3, clk,
    initial_position_ecef=tuple(approx_xyz),
    antenna=ant,
    gpt2w_grid=gpt,
    dcb_records=dcb,
    apply_wind_up=True,
)
```

### PPP with SSR replacing CLK

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

with open("ssr-stream.rtcm", "rb") as fp:
    ssr = SSRCorrections(iter_messages(fp))

out = ppp_solve(obs, sp3, clk=None, ssr=ssr)
```

`SSRCorrections` consumes orbit + clock + code-bias messages across
all six constellations and exposes per-(sv, epoch, obs_code) lookups
to the driver.

## Stretch positioning

### Snapshot SPP for a < 1 s capture

For IoT receivers that have a coarse prior (~30 km, e.g. cell-tower
geolocation) and only see fractional-chip code phase. van Diggelen
A-GPS pattern.

```python
from rinexpy.snapshot import snapshot_positioning

out = snapshot_positioning(
    code_phase_chips,           # (n_sv,)  fractional 0..1023
    sv_positions_ecef,          # (n_sv, 3) at signal-emission time
    initial_position_ecef=prior,
)
print(out["lla"], out["time_bias_s"])
```

### Virtual reference station from a network

Compose observations from N base stations into a virtual base
co-located with the rover's approximate position. The rover then
runs ordinary single-baseline RTK against the synthesized base.

```python
from rinexpy.vrs import synthesize_vrs
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

vrs = synthesize_vrs(bases, rover_approx_pos, wavelength=LAMBDA_L1)
sol = rtk_fix(
    rover_pr, vrs["pr"],
    rover_phase, vrs["phase"],
    vrs["sv_positions"], vrs["base_position"],
    wavelength=LAMBDA_L1,
)
```

### GNSS-R sea-surface reflector height (Larson 2008)

SNR oscillations versus sin(elev) are detrended and a Lomb-Scargle
periodogram converts the peak frequency to reflector height.

```python
from rinexpy.gnssr import snr_to_sea_height
from rinexpy.multifreq import LAMBDA_L1

out = snr_to_sea_height(snr_db, elevation_rad,
                        wavelength_m=LAMBDA_L1)
print("reflector height:", out["height_m"], "m")
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

### GPS CNAV / CNAV-2 / Galileo F+I-NAV / GLONASS / NavIC

Same pattern: feed the message body's bytes, get a dict per ICD.

```python
from rinexpy.gps_cnav    import decode_cnav_message
from rinexpy.gps_cnav2   import decode_cnav2_subframe2
from rinexpy.galileo_nav import decode_fnav_page1, decode_inav_word1
from rinexpy.glonass     import decode_glonass_string
from rinexpy.navic       import decode_navic_subframe

print(decode_cnav_message(cnav_bytes))         # MT 10 / 11 dispatched
print(decode_cnav2_subframe2(cnav2_sf2_bytes)) # L1C subframe 2
print(decode_fnav_page1(fnav_page1_bytes))     # Galileo E5a F-NAV
print(decode_inav_word1(inav_word1_bytes))     # Galileo E1B/E5b I-NAV
print(decode_glonass_string(glo_string_bytes)) # GLONASS string 1/2/3
print(decode_navic_subframe(navic_sf_bytes))   # NavIC SF 1/2 + raw SF 3/4
```

### SBAS L1 (WAAS / EGNOS / MSAS / GAGAN)

`decode_sbas_message` walks one 250-bit SBAS L1 message and dispatches
on the 6-bit MT field. Unknown MTs return `{preamble, msg_type, raw}`.

```python
from rinexpy.sbas import decode_sbas_message

for msg_bytes in iter_sbas_messages_from_capture():
    out = decode_sbas_message(msg_bytes)
    if out["msg_type"] == 9:                         # GEO ranging
        print(out["x_m"], out["y_m"], out["z_m"], out["a_Gf0_s"])
```

### RTCM3 SSR feed

`SSRCorrections` absorbs every kind of SSR message (orbit, clock,
code-bias, combined) and exposes per-(sv, epoch) corrections.

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

with open("ssr-feed.rtcm", "rb") as fp:
    ssr = SSRCorrections(iter_messages(fp))

print("SVs with corrections:", ssr.known_satellites())
print("G05 clock at SOW 86400 s:", ssr.clock_correction_s("G05", 86400.0), "s")
print("G05 C1W code bias:", ssr.code_bias_m("G05", "C1W"), "m")
```

### RINEX 4 NAV: STO / EOP / ION records

```python
from rinexpy.nav4 import load_nav4

out = load_nav4("BRDC00WRD_S_20231560000_01D_MN.rnx")
for sto in out["STO"]:
    print(sto["sv"], sto["message_type"], sto["A0_s"])
for ion in out["ION"]:
    if ion["model"] == "KLOB":
        print("Klobuchar alpha:", ion["alpha"], "beta:", ion["beta"])
```

## Atmosphere + corrections

### Autodownload a daily DCB product

Routes by date: post-2017 dates pull the MGEX CAS Rapid SINEX-BIAS
from the IGS BKG public mirror; pre-2017 dates pull the AIUB CODE
monthly P1-P2 / P1-C1 / P2-C2 files. Both paths return records with
the same `read_bsx`-shaped schema.

```python
from datetime import datetime
from rinexpy.dcb_download import auto_load_dcb, download_dcb

records = auto_load_dcb(datetime(2024, 4, 15))  # CAS daily MGEX
records = auto_load_dcb(datetime(2010, 6, 15))  # CODE monthly P1-P2

# Or fetch only (path returned, parse later):
path = download_dcb(datetime(2024, 4, 15), product="CAS")
```

Files are cached under `~/.cache/rinexpy/dcb/`. CDDIS source is wired
but needs NASA Earthdata Login in `~/.netrc`.

### Calibrate an ANTEX PCV from residuals

Fit a 2-D phase-center grid from post-fit residuals tagged with
elevation + azimuth, and write a valid ANTEX file that round-trips
through `load_antex`.

```python
from rinexpy.antex_calibrate import calibrate_pcv, write_antex

entry = calibrate_pcv(
    residuals_m, elevation_rad, azimuth_rad,
    antenna_type="MYANT_NONE",
    serial="SN001",
)
write_antex([entry], "my-antenna.atx")
```

### Time transfer between two stations

```python
from rinexpy.time_transfer import p3_combination, common_view_difference

# Iono-free P3 combination from L1/L2 pseudoranges:
p3 = p3_combination(p1_m, p2_m)

# Common-view: stations A and B observing the same SV at the same epoch.
# After cancelling SV clock + iono, the difference is approximately the
# A-B receiver-clock difference (plus tropo gradient + multipath).
delta_dt = common_view_difference(p3_A, p3_B, rho_A, rho_B)
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
rinexpy info file.18o                          # parsed header
rinexpy times file.18o                         # epoch list
rinexpy read file.18o                          # parse + print
rinexpy convert data/ '*.18o' --out out/ -j 4  # batch -> NetCDF
rinexpy spp file.18o file.18n                  # single-point fix
rinexpy rtk rover.obs base.obs nav.nav         # RTK baseline
rinexpy ppp obs.rnx sp3 clk                    # PPP solve
rinexpy splice a.obs b.obs --out joined.obs    # concat by time
rinexpy decimate file.obs --interval 30        # interval thin
```
