# Cookbook

Every snippet stands alone and assumes
`import rinexpy as rp` at the top. You can also look at the
[tutorial](tutorial.md) oage.

## Reading

### Load any RINEX, CRINEX, SP3, or NetCDF file

```python
ds = rp.load("path/to/file.rnx.gz")
```

Suffix does not matter; gzip / bzip2 / zip /
LZW / Hatanaka decompression happens transparently.

### Open just the header

```python
hdr = rp.rinexheader("file.18o")
print(hdr["APPROX POSITION XYZ"])
print(hdr["TIME OF FIRST OBS"])
```

Keys preserve the original RINEX header labels verbatim. Derived keys such
as `position`, `position_geodetic`, `t0`, `interval`, `fields`, and `Fmax`
are also populated.

### Just the timestamp axis

```python
times = rp.gettime("file.18o")    # numpy.ndarray of datetime64
```

For OBS files the precision is `datetime64[us]`. For NAV files it is
`datetime64[ms]`.

### Probe the file kind

```python
info = rp.rinexinfo("file.18o")   # {'version', 'filetype', 'rinextype', 'systems'}
```

`rinextype` is one of `"obs"`, `"nav"`, `"sp3"`, `"meteo"`. `systems` is the
header letter (`G`, `R`, `E`, `M` for mixed, ...).

### Restrict the time window during the read

```python
obs = rp.load("big.rnx.gz",
              tlim=("2018-07-29T12:00", "2018-07-29T13:00"))
```

The time window is pushed into the parser. Records outside the window are
skipped and never become Python floats.

### Restrict to a constellation

```python
obs_gps = rp.load("mixed.rnx", use="G")
obs_gps_gal = rp.load("mixed.rnx", use={"G", "E"})
```

`use=` accepts a single letter or any iterable of letters. The canonical set
is `{G, R, E, J, C, I, S}`.

### Pick specific measurements

```python
obs = rp.load("file.18o", meas=["L1", "L2"])
```

Drops everything except the listed measurement labels. The labels are the
RINEX 3 short codes (`C1C`, `L2W`, `S1P`, ...) for RINEX 3 inputs and the
RINEX 2 short codes (`L1`, `C1`, `P2`) for RINEX 2 inputs.

### Decimate while reading

```python
obs_30s = rp.load("1hz.rnx.gz", interval=30)
```

Drops epochs at parse time. Cheaper than reading everything and slicing
afterwards because the dropped records are never decoded.

### Read straight from a stream

```python
import io
with open("file.18o") as fp:
    text = fp.read()
obs = rp.load(io.StringIO(text))
```

`load` accepts `Path`, `str`, `io.StringIO`, `io.BytesIO`, and any object
with a `.read()` method. The version detector reads ahead with `tell` /
`seek` if possible.

## Writing

### One file to NetCDF

```python
rp.load("input.18o", out="output.nc")
```

Combine read and write into one call: `out=` triggers the conversion as
records are parsed. Output is a single NetCDF4 / HDF5 file with the data
inside a group named `OBS` or `NAV`.

### A whole directory in parallel

```python
written = rp.batch_convert("data/", "*.18o", "out/", workers=4)
```

`workers=0` uses every CPU. `workers=None` (default) runs serially. Errors
on individual files are logged and the conversion continues. The list of
successfully written output paths is returned.

### Round-trip back to RINEX

```python
obs = rp.load("input.18o", tlim=("2018-07-29", "2018-07-30"))
rp.to_rinex_obs(obs, "filtered.rnx", version=3)
```

`version=` accepts 2 or 3. Header round-tripping covers the standard fields;
non-standard custom records may not survive verbatim.

### Round-trip a NAV dataset

```python
from rinexpy.nav_writer import write_nav3
write_nav3(nav, "out.rnx")
```

Currently handles the Keplerian systems: GPS (G), Galileo (E), BeiDou (C),
QZSS (J). The GLONASS / SBAS direct-ECEF format is not yet round-tripped.

### Zarr for cloud workflows

```python
from rinexpy.zarr_io import to_zarr
to_zarr(obs, "out.zarr")
```

Writes a chunked, compressed Zarr store. Needs the `zarr` extra.

## Streaming

### One epoch at a time

```python
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(time, ds)
```

`ds` is a single-time `xarray.Dataset` sized to the SVs at that epoch.
Memory footprint is constant in the file size.

### Stream with filters

```python
for time, ds in rp.iter_obs3_epochs(
    "huge.rnx.gz",
    use={"G"},
    tlim=("2018-07-29T12:00", "2018-07-29T13:00"),
    interval=30,
):
    process_one_epoch(time, ds)
```

The same `use`, `tlim`, and `interval` filters work on the iterator.

## Multi-file

### Concatenate daily files

```python
from rinexpy.tools import concat_files
obs_week = concat_files(["data/d001.18o", "data/d002.18o", "data/d003.18o"])
```

Joins along the time axis, dedups on `(time, sv)`, and tolerates minor
header drifts between days.

### Diff two datasets

```python
from rinexpy.tools import diff_datasets

delta = diff_datasets(rp.load("v1.18o"), rp.load("v2.18o"))
if not delta["equal"]:
    for d in delta["differences"]:
        print(d)
```

Handy when chasing parser regressions or verifying a round-trip.

### Concurrent loads via asyncio

```python
import asyncio
from rinexpy.asyncio import aload_many

results = asyncio.run(aload_many(["a.18o", "b.18o", "c.18o"]))
```

Runs each parse in the default thread pool. Exceptions show up as exception
instances inside the result list at the matching index, rather than aborting
the whole call.

## Geodesy

### ECEF and geodetic

```python
from rinexpy.geodesy import ecef_to_lla, lla_to_ecef

x, y, z = lla_to_ecef(40.0, -3.0, 100.0)
lat, lon, alt = ecef_to_lla(x, y, z)
```

WGS-84 by default. Degrees and metres throughout.

### Azimuth and elevation

```python
import numpy as np
from rinexpy.geodesy import azimuth_elevation, lla_to_ecef

rx = lla_to_ecef(40.0, -3.0, 100.0)
sv_ecef = np.array([[2.66e7, 0, 0], [0, 2.66e7, 0]])
az, el = azimuth_elevation(rx, sv_ecef)
```

Both outputs are NumPy arrays in degrees.

### Dilution of precision

```python
from rinexpy.geodesy import dop

out = dop(sv_ecef, rx)
print(out["GDOP"], out["PDOP"], out["HDOP"])
```

Returns a dict with `GDOP`, `PDOP`, `HDOP`, `VDOP`, and `TDOP`.

### Saastamoinen tropo

```python
from rinexpy.geodesy import saastamoinen

slant_dry_wet_m = saastamoinen(el_deg=15.0, altitude_m=100.0)
```

Without met data the function falls back to ICAO standard atmosphere
defaults. Pass `pressure_hpa=`, `temperature_k=`, `humidity_e_hpa=` if you
have them.

### Klobuchar broadcast iono

```python
from rinexpy.geodesy import klobuchar

# alpha and beta come from the GPS NAV header (8 coefficients).
delay_m = klobuchar(alpha, beta, (40.0, -3.0, 100.0),
                    sv_az_deg=180.0, sv_el_deg=30.0, gps_sec=43200)
```

Returns the L1 ionospheric slant delay in metres.

### Local east-north-up

```python
from rinexpy.geodesy import ecef_to_enu, enu_to_ecef

baseline_enu = ecef_to_enu(target_ecef, ref_ecef)
target_ecef  = enu_to_ecef(baseline_enu, ref_ecef)
```

The reference frame is the local tangent plane at `ref_ecef`.

## Time

### GPS week and datetime

```python
from datetime import datetime
from rinexpy.gpstime import datetime_to_gps, gps_to_datetime

week, sow = datetime_to_gps(datetime(2024, 1, 7))
back = gps_to_datetime(week, sow)
```

The `(week, sow)` pair is the standard ICD format. Round-trip is exact to
microseconds.

### Leap seconds

```python
from rinexpy.gpstime import leap_seconds_at
n = leap_seconds_at(datetime(2024, 1, 1))      # 37 since 2017-01-01
```

Returns TAI-UTC in seconds. The table comes from IERS Bulletin C and is
embedded in the module.

### Resolve a 10-bit week

```python
from datetime import datetime
from rinexpy.gpstime import gps_week_rollover

full_week = gps_week_rollover(0, datetime(2024, 1, 1))   # 2048
```

Broadcast week numbers from the GPS L1 C/A message are 10 bits wide and
roll over every 1024 weeks. Pass the reference date to disambiguate.

## SP3

### Load and interpolate

```python
import numpy as np
from datetime import datetime

sp3 = rp.load_sp3("tests/data/igs19362.sp3c")
t0 = sp3.time.values[5].astype("datetime64[us]").astype(datetime)
interp = rp.interpolate_sp3(sp3, np.array([t0], dtype="datetime64[ns]"))
g05 = interp.position.sel(sv="G05").values  # (1, 3) ECEF in km
```

Lagrange interpolation with the IGS-recommended order of 10. The window
straddling the file boundary is handled with a one-sided window.

### Stitch a multi-day SP3

```python
from rinexpy.sp3 import stitch_sp3
sp3 = stitch_sp3("igs01.sp3", "igs02.sp3", "igs03.sp3")
```

Joins consecutive daily SP3 files along the time axis. Useful for spanning
a job boundary mid-way through your file.

## RTK and positioning

### Single-point positioning

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m, max_iter=20)
print(sol["lla"])
```

The return dict carries `position` (ECEF), `clock_bias` (s), `n_iter`,
`residuals`, and `lla`. With `raim=True` it also carries `raim_test`,
`fault_detected`, and `excluded_svs`.

### Float baseline RTK

```python
from rinexpy.rtk import double_difference_solve
from rinexpy.multifreq import LAMBDA_L1

sol = double_difference_solve(
    pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
    wavelength=LAMBDA_L1,
)
print("baseline (m):", sol["baseline"])
```

Float-ambiguity solve. The `baseline_ecef`, `baseline_enu`, `ambiguities`,
and `residuals` fields are populated.

### Full LAMBDA fix

```python
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

sol = rtk_fix(
    pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
    wavelength=LAMBDA_L1,
    ratio_threshold=3.0,
)
if sol["fixed_accepted"]:
    print("centimetre-level baseline:", sol["fixed"]["baseline"])
```

Float solve plus LAMBDA plus ratio test. The returned dict has
`float`, `fixed`, and `lambda` sub-dicts; `fixed_accepted` is the bool.

### Wide-lane resolution

```python
from rinexpy.multifreq import lambda_dual_freq

out = lambda_dual_freq(a_l1_float, a_l2_float, p1_m=pr_l1, p2_m=pr_l2)
print("Fixed N1:", out["N_L1"])
print("Fixed N2:", out["N_L2"])
print("Fraction fixed:", out["fraction_fixed"])
```

L1 + L2 fix via the wide-lane combination with Melbourne-Wuebbena gating.

### SPP with RAIM

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m,
                   raim=True, sigma_pr=5.0, p_fa=1e-4, max_exclusions=2)
if sol["fault_detected"]:
    print("excluded:", sol["excluded_svs"])
```

Chi-squared residual test after the fix; if it fails, the worst-residual SV
is dropped and the LSQ re-runs.

### SPP with DCB or broadcast TGD

```python
from rinexpy.dcb import read_bsx
from rinexpy.positioning import tgd_from_nav

records = read_bsx("CAS0MGXRAP_20231560000_01D_01D_DCB.BSX")
sol = rp.spp_solve(
    sv_ecef, pseudoranges_m,
    sv_labels=["G05", "G10", ...],
    dcb_records=records,
    dcb_obs_code="C1W",
)

nav = rp.load("brdc2800.15n")
tgd_map = tgd_from_nav(nav, epoch)
sol = rp.spp_solve(sv_ecef, pseudoranges_m,
                   sv_labels=["G05", ...], tgd_map=tgd_map, tgd_gamma=1.0)
```

`tgd_gamma=1` for L1, `(f1/f2)**2` for L2, `0` for the iono-free combination.

### Sequential RTK

```python
from rinexpy.rtk import SequentialRTK
from rinexpy.multifreq import LAMBDA_L1

rtk = SequentialRTK(base_ecef, wavelength=LAMBDA_L1)
for epoch in epochs:
    out = rtk.update(svs, rover_pr, base_pr, rover_phase, base_phase, sv_ecef)
    record(epoch, out["baseline"], out["fixed_accepted"], out["slipped_svs"])
```

Carries the integer fix between epochs, runs partial AR on the SVs whose
lock did not survive, and detects cycle slips per SV.

## Precise point positioning

### Static PPP from SP3 + CLK

```python
from rinexpy.ppp import ppp_solve

out = ppp_solve(obs, sp3, clk,
                initial_position_ecef=tuple(approx_xyz),
                elevation_mask_deg=7.0)
print(out["position"], "+/-", out["position_sigma_m"])
```

`ppp_solve` walks each epoch through the static-or-kinematic EKF. The
return dict carries `position` (final mean), `position_sigma_m` (1-sigma
per axis), `clock_bias_s`, `n_epochs`, `trace` (per-epoch position list),
and the `filter` object itself.

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

`SSRCorrections` absorbs orbit + clock + code-bias messages from RTCM3.
The driver substitutes the SSR clock for the (missing) CLK lookup per SV.

## Specialised positioning

### Snapshot SPP for a < 1 s capture

```python
from rinexpy.snapshot import snapshot_positioning

out = snapshot_positioning(
    code_phase_chips,          # (n_sv,) fractional 0..1023
    sv_positions_ecef,         # (n_sv, 3) at signal-emission time
    initial_position_ecef=prior,
)
print(out["lla"], out["time_bias_s"])
```

For receivers that only have fractional-chip code phase, with a coarse
position prior (within roughly 150 km of truth). The van Diggelen A-GPS
pattern.

### Virtual reference station

```python
from rinexpy.vrs import synthesize_vrs
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

vrs = synthesize_vrs(bases, rover_approx_pos, wavelength=LAMBDA_L1)
sol = rtk_fix(
    rover_pr, vrs["pr"], rover_phase, vrs["phase"],
    vrs["sv_positions"], vrs["base_position"],
    wavelength=LAMBDA_L1,
)
```

Compose a virtual base from a network of physical bases and run ordinary
single-baseline RTK against it.

### GNSS reflectometry

```python
from rinexpy.gnssr import snr_to_sea_height
from rinexpy.multifreq import LAMBDA_L1

out = snr_to_sea_height(snr_db, elevation_rad, wavelength_m=LAMBDA_L1)
print("reflector height:", out["height_m"], "m")
```

SNR oscillations versus `sin(el)` are detrended, then a Lomb-Scargle
periodogram converts the peak frequency to reflector height.

### IMU + GNSS tightly coupled

```python
import numpy as np
from rinexpy.imu_tight import TightINS16

ins = TightINS16(position=np.array([0.0, 0.0, 0.0]),
                 velocity=np.array([0.0, 0.0, 0.0]))
for accel_body, gyro_body, dt in imu_samples:
    ins.predict(accel_body, gyro_body, dt)
    if gnss_epoch_now:
        ins.update_pseudoranges(sv_ecef, pseudoranges)
print(ins.position, ins.clock_bias_m)
```

16-state error-state EKF. The IMU page covers the matching loosely-coupled
filter as well.

## Plotting

### L1 phase time series

```python
from rinexpy.plots import obstimeseries
import matplotlib.pyplot as plt

obstimeseries(rp.load("file.18o"))
plt.show()
```

One line per SV.

### Skyplot

See the [tutorial section 6](tutorial.md#6-skyplot-from-broadcast-nav).

### Receiver location scatter

```python
from rinexpy.plots import receiver_locations

receiver_locations([
    {"name": "STAT01", "lat": 41.39, "lon": 2.11, "interval": 30.0},
    {"name": "STAT02", "lat": 40.42, "lon": -3.69, "interval": 1.0},
])
```

Marker size encodes the sampling interval.

## RTCM3 and NTRIP

### Decode a captured RTCM3 stream

```python
import io
from rinexpy.rtcm3 import iter_messages

with open("capture.bin", "rb") as fp:
    for msg in iter_messages(fp, check_crc=True):
        print(msg["msg_id"])
```

`check_crc=True` validates the CRC-24Q of each frame and skips invalid ones.

### Fetch the caster sourcetable then stream

```python
from rinexpy.ntrip import fetch_sourcetable, stream

mountpoints = fetch_sourcetable("rtk2go.com", port=2101)
for mp in mountpoints:
    if mp["type"] == "STR":
        print(mp["mountpoint"], mp["format"])

bytes_iter = stream("rtk2go.com", "MOUNT01", user="me", password="x", port=2101)
for chunk in bytes_iter:
    process(chunk)
```

The sourcetable is the catalogue the caster publishes; one STR; line per
mountpoint, plus optional CAS; and NET; rows.

## Receiver-format decoders

### NMEA-0183

```python
from rinexpy.nmea import iter_lines

with open("nmea.log") as fp:
    for msg in iter_lines(fp):
        if msg["type"] == "GGA":
            print(msg["lat"], msg["lon"], msg["altitude_m"])
```

Supported sentence types: GGA, RMC, GSA, GSV, VTG. Unknown sentence types
return with the raw `fields` list.

### u-blox UBX

```python
from rinexpy.ubx import iter_messages

with open("ublox.ubx", "rb") as fp:
    for msg in iter_messages(fp):
        if (msg["msg_class"], msg["msg_id"]) == (0x01, 0x07):  # NAV-PVT
            print(msg["lat_deg"], msg["lon_deg"], msg["fix_type"])
```

### Septentrio SBF

```python
from rinexpy.sbf import iter_blocks

with open("septentrio.sbf", "rb") as fp:
    for blk in iter_blocks(fp):
        if blk["block_id"] == 4007:                  # PVTGeodetic
            print(blk["lat_rad"], blk["lon_rad"], blk["height_m"])
```

### NovAtel OEM

```python
from rinexpy.novatel import iter_messages

with open("novatel.bin", "rb") as fp:
    for msg in iter_messages(fp):
        if msg["msg_id"] == 42:                      # BESTPOS
            print(msg["lat_deg"], msg["lon_deg"], msg["height_m"])
```

### BINEX

```python
from rinexpy.binex import iter_records

with open("archive.bnx", "rb") as fp:
    for rec in iter_records(fp):
        print(f"record {rec['record_id']:#04x}  {rec['length']} bytes")
```

Framing-only at the moment; record bodies come back as raw bytes.

### RTCM 2 DGPS

```python
from rinexpy.rtcm2 import iter_messages

with open("dgps.rtcm2", "rb") as fp:
    for msg in iter_messages(fp):
        if msg["msg_type"] == 1:
            for c in msg["corrections"]:
                print(f"  SV {c['sat_id']:2d}  PRC={c['prc_m']:+.2f}")
```

### Furuno GW-10 SBAS

```python
from rinexpy.gw10 import iter_sbas_messages

with open("gw10.log", "rb") as fp:
    for frame in iter_sbas_messages(fp):
        print(frame["prn"], frame["sbas_l1_bytes"][:6].hex())
```

## Raw nav-message decoders

### Walk a GPS LNAV bitstream

```python
from rinexpy.gps_lnav import decode_lnav_subframe1, decode_lnav_subframe2, decode_lnav_subframe3

clock = decode_lnav_subframe1(words_sf1)  # 10 30-bit words
eph_a = decode_lnav_subframe2(words_sf2)
eph_b = decode_lnav_subframe3(words_sf3)
```

### CNAV / CNAV-2 / Galileo / GLONASS / NavIC

```python
from rinexpy.gps_cnav    import decode_cnav_message
from rinexpy.gps_cnav2   import decode_cnav2_subframe2
from rinexpy.galileo_nav import decode_fnav_page1, decode_inav_word1
from rinexpy.glonass     import decode_glonass_string
from rinexpy.navic       import decode_navic_subframe

print(decode_cnav_message(cnav_bytes))
print(decode_cnav2_subframe2(cnav2_sf2_bytes))
print(decode_fnav_page1(fnav_page1_bytes))
print(decode_inav_word1(inav_word1_bytes))
print(decode_glonass_string(glo_string_bytes))
print(decode_navic_subframe(navic_sf_bytes))
```

### SBAS L1

```python
from rinexpy.sbas import decode_sbas_message

for msg_bytes in iter_sbas_messages_from_capture():
    out = decode_sbas_message(msg_bytes)
    if out["msg_type"] == 9:
        print(out["x_m"], out["y_m"], out["z_m"], out["a_Gf0_s"])
```

Per RTCA DO-229E §A.4. Decoded MT set: 1, 2-5, 6, 7, 9, 17, 18, 24, 25, 26.

### RTCM3 SSR feed

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

with open("ssr-feed.rtcm", "rb") as fp:
    ssr = SSRCorrections(iter_messages(fp))

print("SVs with corrections:", ssr.known_satellites())
print("G05 clock at SOW 86400 s:", ssr.clock_correction_s("G05", 86400.0), "s")
print("G05 C1W code bias:", ssr.code_bias_m("G05", "C1W"), "m")
```

### RINEX 4 NAV STO / EOP / ION

```python
from rinexpy.nav4 import load_nav4

out = load_nav4("BRDC00WRD_S_20231560000_01D_MN.rnx")
for sto in out["STO"]:
    print(sto["sv"], sto["message_type"], sto["A0_s"])
for ion in out["ION"]:
    if ion["model"] == "KLOB":
        print("Klobuchar alpha:", ion["alpha"], "beta:", ion["beta"])
```

## Atmosphere and corrections

### Auto-download a daily DCB

```python
from datetime import datetime
from rinexpy.dcb_download import auto_load_dcb, download_dcb

records = auto_load_dcb(datetime(2024, 4, 15))   # post-2017: CAS MGEX Rapid
records = auto_load_dcb(datetime(2010, 6, 15))   # pre-2017:  CODE P1-P2

# Or fetch only, parse later:
path = download_dcb(datetime(2024, 4, 15), product="CAS")
```

Files are cached under `~/.cache/rinexpy/dcb/`. The CDDIS source is wired
but requires a NASA Earthdata Login in `~/.netrc`.

### Calibrate an ANTEX entry from residuals

```python
from rinexpy.antex_calibrate import calibrate_pcv, write_antex

entry = calibrate_pcv(
    residuals_m, elevation_rad, azimuth_rad,
    antenna_type="MYANT_NONE",
    serial="SN001",
)
write_antex([entry], "my-antenna.atx")
```

Bins post-fit residuals on a 2-D azimuth × zenith grid, averages per cell,
and emits a valid ANTEX entry that round-trips through `load_antex`.

### Time transfer between two stations

```python
from rinexpy.time_transfer import p3_combination, estimate_clock_difference_s

p3_A = p3_combination(p1_A_m, p2_A_m)
p3_B = p3_combination(p1_B_m, p2_B_m)

delta_dt = estimate_clock_difference_s(
    p3_A, p3_B,
    sv_ecef=sv_positions,
    station_a_ecef=site_a,
    station_b_ecef=site_b,
)
```

Common-view P3. After cancelling SV clock and (to first order) iono and
tropo, the per-SV remainder is the receiver-clock difference times c.

## QC

### Quick QC report

```python
from rinexpy.tools import validate_file

rep = validate_file("file.18o")
print(rep["n_epochs"], rep["interval_seconds"], rep["warnings"])
```

Walks the header and the first few epochs, reports nominal interval, gap
count, header consistency, and a list of any warnings.

### Walk a directory triaging bad files

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

The full CLI documentation is on the [command-line interface page](tooling/cli.md).
