# Tutorial

This is a series of tutorials on using `rinexpy`. 

## 1. Open and inspect

Three patterns cover almost every read you will ever do.

```python
import rinexpy as rp

obs = rp.load("tests/data/obs3.01gage.10o")    # autodetect by file content
nav = rp.load("tests/data/brdc2800.15n")        # ditto for NAV
sp3 = rp.load_sp3("tests/data/igs19362.sp3c")   # SP3 has its own loader
```

The OBS and NAV returns are `xarray.Dataset` objects sized to the SVs and
epochs present in the file. SP3 returns a Dataset with separate `position`,
`velocity`, `clock`, and `dclock` variables, plus the `t0` scalar that holds
the ephemeris reference epoch.

```python
print(obs.sv.values[:5])     # ['G07' 'G09' 'G12' 'G13' 'G15']
print(obs.time.values[0])    # numpy.datetime64('2010-03-05T00:00:00.000000')
print(list(obs.data_vars))   # ['L1C', 'L2P', 'C1P', 'C2P', 'C1C', 'S1P', 'S2P', 'S1C']
```

You can index using `xarray` conventions. Boolean masks, `sel` by label, `isel`
by integer, `dropna` for missing data.

```python
g07 = obs.sel(sv="G07")          # one SV across all observables and epochs
c1c = obs.C1C                     # one observable across all SVs
mid = obs.isel(time=slice(0, 2))  # first two epochs
clean = c1c.dropna(dim="time", how="any")
```

The header information is present in the `.attrs` dict. For RINEX OBS the
useful keys are `position`, `position_geodetic`, `interval`, `fields`,
`time_system`, and `filename`.

```python
obs.attrs["position"]           # [4789028.4701, 176610.0133, 4195017.031]
obs.attrs["position_geodetic"]  # (41.3887, 2.1120, 166.25) lat, lon, alt
obs.attrs["interval"]           # 30.0 seconds
```

When you only want the header, the dedicated reader skips the data section.

```python
hdr = rp.rinexheader("tests/data/obs3.01gage.10o")
print(hdr.keys())               # every RINEX header label, verbatim
print(hdr["APPROX POSITION XYZ"])  # the raw header line text
```

And if you only want the time axis, `gettime` returns it as a `datetime64`
array without touching the observation data.

```python
times = rp.gettime("tests/data/obs3.01gage.10o")
print(times[:3])                # ['2010-03-05T00:00:00', '2010-03-05T00:00:30']
```

## 2. Filter cheaply during the read

The parser autodetects every filtering condition passed to `load` and applies it. Records outside
your filter are dropped before they are ever loaded.

```python
import datetime as dt

obs = rp.load(
    "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip",
    use={"G", "E"},                       # only GPS + Galileo
    meas=["C1C", "L1C"],                  # only those two labels
    tlim=("2018-05-13T00:00:00",          # ISO-8601 string limits
          "2018-05-13T00:05:00"),
    interval=60,                          # decimate to 60 s
)
```

The four filters are composable. `use=` accepts a single letter or a set of
letters from the canonical `{G, R, E, J, C, I, S}`. `meas=` accepts a list
of RINEX 3 observation labels (`C1C`, `L2W`, `S1P`, ...) or RINEX 2 short
codes (`L1`, `C1`, `P2`). `tlim=` accepts `(start, end)` as ISO-8601 strings,
`datetime`s, or `np.datetime64`. `interval=` accepts seconds as a float, an
integer, or a `datetime.timedelta`.

For RINEX 3 NAV the same `use=` and `tlim=` filters apply. NAV files do not
have a `meas=` analogue because every record has the same field set per
satellite system.

## 3. Stream files larger than RAM

`load` reads the whole file into memory as a single Dataset. For multi-day, 1 Hz,
multi-constellation files that can be too much. The streaming iterator yields
one epoch at a time, with constant memory usage in the file size.

```python
for time, ds in rp.iter_obs3_epochs("tests/data/obs3.01gage.10o"):
    print(time, ds.sv.size, "SVs visible at this epoch")
```

Each yielded `ds` is a single-time Dataset that holds only the SVs present at
that epoch. The same `use`, `tlim`, and `interval` filters work as on
`load`, so the iterator also drops records inside the parser.

```python
for time, ds in rp.iter_obs3_epochs(
    "tests/data/obs3.01gage.10o",
    use={"G"},
    interval=30,
):
    pseudorange = ds.C1C.values            # 1-D over the SVs at this epoch
```

## 4. SP3 and the satellite positions

SP3 files publish satellite ECEF positions every 15 minutes. Most
positioning workflows need them at every observation epoch (often 1 Hz).
`interpolate_sp3` runs an order-10 Lagrange interpolator, the same as the
IGS recommended default.

```python
import numpy as np
from datetime import datetime

sp3 = rp.load_sp3("tests/data/igs19362.sp3c")
t0 = sp3.time.values[5].astype("datetime64[us]").astype(datetime)
queries = np.array(
    [t0, t0 + np.timedelta64(60, "s"), t0 + np.timedelta64(300, "s")],
    dtype="datetime64[ns]",
)
interp = rp.interpolate_sp3(sp3, queries)
print(interp.position.sel(sv="G05").values)   # (3 queries, 3 ECEF) in km
```

Units in the returned dataset are kilometres for `position`, dm/s for
`velocity`, and microseconds for `clock`. 

If the query epoch is in the second half of the file, the Lagrange window
straddles the file boundary; `rinexpy` detects this and uses a one-sided
window so the result is still well-conditioned.

## 5. Broadcast nav and Keplerian propagation

The broadcast NAV file carries Keplerian elements per satellite per ephemeris
record. `keplerian2ecef` converts those elements into ECEF coordinates.

```python
nav = rp.load("tests/data/brdc2800.15n")
sv = nav.sel(sv="G07")          # one SV slice; keep the time dimension
X, Y, Z = rp.keplerian2ecef(sv)
print(X.shape, Y.shape, Z.shape)
print(X[:3])                    # the first three ECEF X values, in metres
```

GPS and Galileo are Keplerian. GLONASS and SBAS broadcast direct ECEF
positions per epoch; `keplerian2ecef` returns those unchanged. BeiDou D2 and
QZSS follow the GPS-style elements.

If you have the SV's broadcast clock parameters from the same NAV file,
`tgd_from_nav` pulls the broadcast group delay out for use as a single-frequency
ionospheric correction.

```python
from rinexpy.positioning import tgd_from_nav

tgd = tgd_from_nav(nav, datetime(2015, 10, 7, 12))
print(tgd)                       # {'G07': -1.4e-8, 'G09': 4.7e-9, ...}
```

## 6. Skyplot from broadcast NAV

The composition of "Keplerian elements + receiver position" gives you a
sky map for any past or future epoch. The example below renders a polar
plot for a receiver at (40°N, 3°W, 100 m).

```python
import numpy as np
from rinexpy.geodesy import azimuth_elevation, lla_to_ecef
from rinexpy.plots import skyplot
import matplotlib
matplotlib.use("Agg")           # for headless environments
import matplotlib.pyplot as plt

rx = lla_to_ecef(40.0, -3.0, 100.0)

sv_az_el = {}
for sv_label in nav.sv.values:
    if sv_label[0] not in {"G", "E"}:
        continue                  # skip non-Keplerian systems
    sv = nav.sel(sv=sv_label)
    try:
        X, Y, Z = rp.keplerian2ecef(sv)
    except (ValueError, KeyError):
        continue
    sv_ecef = np.stack([X, Y, Z], axis=-1)
    az, el = azimuth_elevation(rx, sv_ecef)
    sv_az_el[sv_label] = (az, el)

skyplot(sv_az_el, title="Sky from (40, -3)")
plt.savefig("skyplot.png", dpi=120, bbox_inches="tight")
```

The matplotlib helpers live in `rinexpy.plots` and need the `plot` extra.

## 7. Single-point positioning

The four unknowns of a code-only fix are the receiver's ECEF position and
its clock bias. `spp_solve` iterates the linearised least-squares system to
convergence.

```python
import numpy as np
from rinexpy.geodesy import lla_to_ecef

truth = np.array(lla_to_ecef(40.0, -3.0, 100.0))

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
    sv.append(truth + sv_radius * np.array([x, y, z]))
sv = np.array(sv)

bias_s = 1e-4
pr = np.linalg.norm(sv - truth, axis=1) + 299792458.0 * bias_s

sol = rp.spp_solve(sv, pr, max_iter=20)
print("ECEF:", sol["position"])
print("LLA: ", sol["lla"])
print("bias:", sol["clock_bias"], "s")
```

For noisy real data the same call accepts `raim=True`, which adds a
chi-squared residual test and excludes the worst satellite until the test
passes.

```python
sol = rp.spp_solve(sv, pr, raim=True, sigma_pr=5.0, p_fa=1e-4, max_exclusions=2)
if sol["fault_detected"]:
    print("excluded:", sol["excluded_svs"])
```

The full SPP page covers the iono-free combination, the broadcast TGD
application, and the SINEX-BIAS DCB path. See
[Single-point positioning](positioning/spp.md).

## 8. RTK with LAMBDA integer fixing

When you have two receivers tracking the same satellites, the
double-difference of their carrier-phase observations cancels the
satellite and receiver clocks, the ionospheric delay (to first order),
and the tropospheric delay (when the baseline is short). What is left is
the geometry plus an integer ambiguity per satellite pair.

`rtk_fix` runs the joint baseline-and-ambiguity LSQ, fixes the integers
through the LAMBDA algorithm, applies the ratio test, and re-solves the
baseline with the fixed integers held in.

```python
import numpy as np
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1
from rinexpy.geodesy import lla_to_ecef

rng = np.random.default_rng(2026)
base = np.array(lla_to_ecef(40.0, -3.0, 0.0))
truth_baseline = np.array([5.4, -2.1, 0.7])
rover = base + truth_baseline

sv_radius = 2.66e7
az_el = [(10, 70), (70, 30), (130, 55), (190, 20), (250, 50), (310, 40)]
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
    sv.append(base + sv_radius * np.array([x, y, z]))
sv = np.array(sv)

rho_r = np.linalg.norm(sv - rover, axis=1)
rho_b = np.linalg.norm(sv - base, axis=1)
true_amb = rng.integers(-200, 200, size=sv.shape[0])
pr_r = rho_r
pr_b = rho_b
phase_r = rho_r / LAMBDA_L1 + true_amb
phase_b = rho_b / LAMBDA_L1 + true_amb

sol = rtk_fix(
    pr_r, pr_b, phase_r, phase_b, sv, tuple(base),
    wavelength=LAMBDA_L1,
    sigma_pr=1.0,
    sigma_phase=0.005,
    ratio_threshold=3.0,
)
print("float baseline:", sol["float"]["baseline"])
print("fixed accepted:", sol["fixed_accepted"])
if sol["fixed_accepted"]:
    print("fixed baseline:", sol["fixed"]["baseline"])
print("LAMBDA ratio:", sol["lambda"]["ratio"])
```

A noise-free input lands within a few millimetres of the truth baseline.
With realistic noise on the pseudorange (around a metre per epoch) and
carrier phase (around 5 mm), the ratio test passes when the geometry is
strong enough to make the second-best integer candidate clearly worse than
the best.

The full RTK page covers `SequentialRTK` (multi-epoch with ambiguity
carry-over), partial ambiguity resolution, cycle-slip detection, and
network RTK. See [RTK and integer fixing](positioning/rtk.md).

## 9. NTRIP and real-time RTCM3

For applications that need live corrections, `rinexpy` includes an NTRIP v1 / v2
client and an RTCM3 framer.

```python
from rinexpy.ntrip import stream
from rinexpy.rtcm3 import iter_messages
import io

# rtk2go.com is a free crowd-sourced NTRIP caster with public mountpoints.
bytes_iter = stream("rtk2go.com", "your-mount", user="me", password="x", port=2101)

buf = io.BytesIO()
for chunk in bytes_iter:
    buf.write(chunk)
    if buf.tell() > 4096:
        break
buf.seek(0)

for msg in iter_messages(buf):
    print(msg["msg_id"], msg.get("station_id"))
```

`stream` is a synchronous generator. There is also `astream` for `asyncio`
use. The full set of message types decoded by `iter_messages` is in
[RTCM and NTRIP](formats/rtcm.md).

For testing without a live caster, the `tests/data/` directory does not release
RTCM3 captures, but every example in this documentation that uses RTCM3
shows how to build one on the fly with the public `PREAMBLE` and `crc24q`
helpers.

## 10. Tropospheric correction

Saastamoinen with standard-atmosphere defaults is good to about a centimetre
at elevations above 15 degrees. It is a one-line call.

```python
from rinexpy.geodesy import saastamoinen

slant_delay_m = saastamoinen(el_deg=15.0, altitude_m=100.0)
```

When you have real meteorological data, you pass it in. Pressure in hPa,
temperature in kelvin, partial water-vapour pressure in hPa.

```python
slant_delay_m = saastamoinen(
    el_deg=15.0,
    altitude_m=100.0,
    pressure_hpa=1013.25,
    temperature_k=288.15,
    humidity_e_hpa=11.7,
)
```

For PPP-class work the GPT2w empirical model plus the VMF1 mapping
function is the standard. GPT2w needs a 2 MB grid file (`gpt2_5w.grd`)
that you fetch yourself from the [Vienna VMF Data Server](https://vmf.geo.tuwien.ac.at/codes/).

```python
from rinexpy.gpt2w import gpt2w, load_gpt2w_grid
from rinexpy.geodesy import vmf1
from datetime import datetime

grid = load_gpt2w_grid("/path/to/gpt2_5w.grd")
met = gpt2w(grid, lat_deg=40.0, lon_deg=-3.0,
            epoch=datetime(2024, 3, 14), altitude_m=100.0)
m_h, m_w = vmf1(met["a_h"], met["a_w"],
                el_deg=15.0, lat_deg=40.0, altitude_m=100.0, doy=74)
```

More on the atmosphere models in [Atmospheric models](corrections/atmosphere.md).

## 11. Sequential RTK across epochs

`rtk_fix` solves one epoch at a time. For a moving rover the integer fix
should carry between epochs while the per-SV lock holds, and only re-bootstrap
on slip events. `SequentialRTK` provides that.

```python
from rinexpy.rtk import SequentialRTK
from rinexpy.multifreq import LAMBDA_L1

rtk = SequentialRTK(tuple(base), wavelength=LAMBDA_L1)

# In a real workflow you build (svs, pr, phase, sv_ecef) per epoch
# from your OBS file and SP3. Below is one synthetic epoch.
svs = ["G07", "G09", "G12", "G15", "G17", "G20"]
out = rtk.update(svs, pr_r, pr_b, phase_r, phase_b, sv)
print(f"baseline: {out['baseline']}")
print(f"fixed accepted: {out['fixed_accepted']}, ratio: {out['ratio']:.2f}")
print(f"carry-over: {out['carry_over_count']}, slipped: {out['slipped_svs']}")
```

Calling `rtk.update(...)` on the next epoch reuses the previous integer fix
where the per-SV lock still holds, runs partial AR on the remaining SVs,
and reports per-SV slip flags. `rtk.reset()` clears all cached state.

## 12. Precise point positioning

PPP closes the gap between the SP3 / CLK precision (1-2 cm in space, 100 ps
in clock) and the centimetre-class user position. The driver in `rinexpy.ppp`
combines the satellite-position interpolator, the clock interpolator, the
tropospheric model, the antenna phase-centre variations, the DCBs, and the
carrier-phase wind-up into one call.

```python
from rinexpy.ppp import ppp_solve

obs = rp.load("data/RNX/STAT00BRA_R_20231560000_24H_01S_MO.rnx.gz")
sp3 = rp.load_sp3("data/SP3/IGS0OPSFIN_20231560000_01D_15M_ORB.SP3")
clk = rp.load_clk("data/CLK/IGS0OPSFIN_20231560000_01D_30S_CLK.CLK")

approx_xyz = obs.attrs["position"]
out = ppp_solve(obs, sp3, clk, initial_position_ecef=tuple(approx_xyz))
print("position:", out["position"], "m")
print("1-sigma:", out["position_sigma_m"], "m")
print("epochs used:", out["n_epochs"])
```

The driver picks an L1/L2 obs-code quadruple from the dataset (`C1C/C2W/L1C/L2W`
on a typical IGS-class receiver, falling back through a documented priority
list), forms the iono-free combination, masks SVs below 7° elevation, and
feeds each epoch into a `StaticPPPFilter`.

For the full correction stack, supply more arguments:

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

For real-time PPP, swap `clk` for an SSR feed.

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

with open("ssr-stream.rtcm", "rb") as fp:
    ssr = SSRCorrections(iter_messages(fp))

out = ppp_solve(obs, sp3, clk=None, ssr=ssr)
```

`SSRCorrections` takes in orbit (1057, 1063, 1240, ...), clock (1058, 1064,
1241, ...), and code-bias (1059, 1065, 1242, ...) messages per system. It
rotates the RTCM radial / along-track / cross-track delta into ECEF using
each SV's instantaneous frame and extrapolates the clock polynomial off the
SSR epoch.

The full PPP page covers ZTD estimation, the multi-constellation filter,
PPP-RTK fusion, and the slip-aware EKF update. See
[Precise point positioning](positioning/ppp.md).

## 13. Raw nav message decoders

For RXM-SFRBX captures, BeiDou stream analysis, or anyone walking a raw
navigation bitstream, the decoders cover the full modernised-signal family.

```python
from rinexpy.gps_lnav import decode_lnav_subframe1, decode_lnav_subframe2, decode_lnav_subframe3
from rinexpy.gps_cnav  import decode_cnav_mt10, decode_cnav_mt11
from rinexpy.gps_cnav2 import decode_cnav2_subframe2
from rinexpy.galileo_nav import decode_fnav_page1, decode_inav_word4
from rinexpy.glonass    import decode_glonass_string1
from rinexpy.navic      import decode_navic_subframe1
from rinexpy.sbas       import decode_sbas_message
from rinexpy.beidou     import decode_d1_subframe1
from rinexpy.nav4       import load_nav4
```

Each decoder takes the raw bit-packed bytes for one message body and
returns a dict keyed by the field names in the relevant ICD. The shape
across constellations is intentionally uniform: feed bytes in, get a dict
out. The page on [raw nav subframes](formats/nav-subframes.md) walks
through every decoder.

RINEX 4 navigation files carry STO, EOP, and ION records in addition to
the modernised ephemeris set. `load_nav4` returns them all.

```python
n4 = load_nav4("BRDC00WRD_S_20231560000_01D_MN.rnx")
for sto in n4["STO"]:
    print(sto["sv"], sto["message_type"], sto["A0_s"])
for ion in n4["ION"]:
    if ion["model"] == "KLOB":
        print("Klobuchar alpha:", ion["alpha"], "beta:", ion["beta"])
```

## 14. Quality control

The `tools` module provides three short functions that cover the everyday
QC needs.

```python
from rinexpy.tools import validate_file, concat_files, diff_datasets

rep = validate_file("tests/data/demo.10o")
print(rep["n_epochs"], rep["interval_seconds"], rep["warnings"])

# Concatenate a daily series along time:
combined = concat_files(["data/d001.18o", "data/d002.18o", "data/d003.18o"])

# Diff two parses; useful for regression checks:
a = rp.load("v1.18o")
b = rp.load("v2.18o")
delta = diff_datasets(a, b)
if not delta["equal"]:
    print(delta["differences"])
```

For deeper QC, the `rinexpy.qc` module has cycle-slip detectors, multipath
combinations, and a Hatch filter. See [QC and cycle slips](quality/qc.md).

## 15. Round-trip back to disk

The complementary writers cover both RINEX and the parsed-Dataset formats.

```python
# Write a parsed dataset back to RINEX 3 OBS:
rp.to_rinex_obs(obs, "filtered.rnx", version=3)

# Save to NetCDF (xarray's default sink):
rp.load("tests/data/demo.10o", out="demo.nc")

# Save to Zarr for cloud workflows:
from rinexpy.zarr_io import to_zarr
to_zarr(obs, "demo.zarr")

# Re-emit a NAV3 dataset (round-trip a fitted ephemeris):
from rinexpy.nav_writer import write_nav3
write_nav3(nav, "out.rnx")

# Write an SP3 dataset back out:
from rinexpy.sp3 import write_sp3
write_sp3(sp3, "out.sp3", version="c")
```
