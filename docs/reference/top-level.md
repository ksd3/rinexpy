# Top-level API

The names below are exported from `rinexpy/__init__.py` and are the
recommended public surface. Anything not in this list (or anything
starting with an underscore) is internal.

## Loading

### `rinexpy.load(path, out=None, *, use=None, tlim=None, useindicators=False, meas=None, verbose=False, overwrite=False, fast=True, interval=None)`

Auto-detect the file kind and dispatch to the right reader. Handles
RINEX 2 / 3 NAV and OBS, SP3-a / SP3-c / SP3-d, and pre-converted NetCDF
files. For a NetCDF that holds both NAV and OBS groups, returns
`{"nav": ..., "obs": ...}`.

| Arg | Type | Meaning |
| --- | --- | --- |
| `path` | `Path / str / file-like` | input file or stream |
| `out` | `Path / str / None` | optional NetCDF output (directory or `.nc` path) |
| `use` | `str / set[str] / None` | restrict to GNSS systems |
| `tlim` | `(start, end) / None` | restrict to a time window |
| `useindicators` | `bool` | also load LLI / SSI as separate variables |
| `meas` | `list[str] / None` | restrict to observation labels |
| `verbose` | `bool` | INFO-level logging |
| `overwrite` | `bool` | allow `out` to be replaced |
| `fast` | `bool` | one-pass RINEX 2 parser (default True) |
| `interval` | `float / timedelta / None` | decimate to this sampling |

Returns an `xarray.Dataset` or `{"nav": ..., "obs": ...}` dict.

### `rinexpy.rinexnav(fn, outfn=None, *, use=None, group="NAV", tlim=None, overwrite=False)`

Read any RINEX 2 / 3 NAV file (or open the NAV group of a NetCDF).
Returns an `xarray.Dataset`.

### `rinexpy.rinexobs(fn, outfn=None, *, use=None, group="OBS", tlim=None, useindicators=False, meas=None, verbose=False, overwrite=False, fast=True, interval=None, use_jit=None)`

Read any RINEX 2 / 3 OBS file (or open the OBS group of a NetCDF).
`use_jit=True` opts into the numba JIT path; needs the `jit` extra.

### `rinexpy.batch_convert(path, glob, out, *, use=None, tlim=None, useindicators=False, meas=None, verbose=False, fast=True, workers=None)`

Convert every file in `path` matching `glob` to NetCDF under `out`.
`workers=0` uses every CPU; `workers=None` runs serially. Per-file errors
are logged and the run continues. Returns `list[Path]` of written files.

### `rinexpy.iter_obs3_epochs(fn, *, use=None, tlim=None, interval=None)`

Generator that yields `(datetime, xarray.Dataset)` per epoch.
Constant memory in file size. RINEX 3 OBS only.

## Inspection

### `rinexpy.rinexinfo(fn) -> dict`

First-line peek. Returns `{"version", "filetype", "rinextype", "systems"}`.

### `rinexpy.rinexheader(fn) -> dict`

Full parsed header. Keys preserve the RINEX header labels verbatim plus
derived keys: `position`, `position_geodetic`, `t0`, `t1`, `interval`,
`fields`, `fields_ind`, `Fmax`, `Nobs`, `Nl_sv`.

### `rinexpy.gettime(fn) -> numpy.ndarray`

Sorted unique epoch timestamps as `datetime64[us]` (OBS) or
`datetime64[ms]` (NAV).

### `rinexpy.to_datetime(time_coord) -> list[datetime]`

Convert an xarray time coord (or anything with `.values.astype(...)`) to
plain Python `datetime` objects.

## SP3, CLK, IONEX, ANTEX

### `rinexpy.load_sp3(fn, outfn=None) -> xarray.Dataset`

SP3-a / SP3-c / SP3-d ephemeris reader. Coords: `time`, `sv`,
`ECEF=["x", "y", "z"]`. Variables: `position` (km), `velocity` (dm/s),
`clock` (µs), `dclock` (1e-4 µs/s), scalar `t0`.

### `rinexpy.stitch_sp3(*paths) -> xarray.Dataset`

Load and concatenate consecutive SP3 files along time.

### `rinexpy.load_clk(fn) -> xarray.Dataset`

RINEX clock product. Coords `(time, sv)`; variable `bias` (s). Receiver
station IDs are in `attrs["stations"]`.

### `rinexpy.interpolate_clk(ds, sv, epoch) -> float`

Linear interpolation of `bias` for one SV at an arbitrary time.

### `rinexpy.load_ionex(fn) -> xarray.Dataset`

IONEX TEC maps. Coords `(time, lat, lon)`; variable `tec` in TECU.

### `rinexpy.interp_tec(ds, lat, lon, epoch) -> float`

Bilinear-in-space, linear-in-time TEC interpolation.

### `rinexpy.slant_tec(vertical_tec_tecu, el_deg) -> float`

Map vertical TEC to slant TEC via the 350 km thin-shell mapping.

### `rinexpy.load_antex(fn) -> list[dict]`

ANTEX antenna PCV reader. One entry per antenna with `type`, `serial`,
`valid_from`, `valid_until`, `dazi_deg`, `dzen_deg`, `zen_max_deg`, and
a `frequencies` dict.

### `rinexpy.find_antenna(entries, type_code, *, serial=None, epoch=None) -> dict | None`

Pick one ANTEX entry by model name, with optional serial and validity
epoch.

### `rinexpy.apply_antex_pcv(entry, freq_id, el_deg, *, az_deg=None) -> float`

Antenna PCV correction in metres. Bilinear on the 2-D grid when
`az_deg=` is given and the entry has `dazi_deg > 0`; NOAZI otherwise.

## Atmosphere products

### `rinexpy.load_gpt2w_grid(path) -> dict` and `rinexpy.gpt2w(grid, lat, lon, epoch, altitude_m=0.0) -> dict`

Read the GPT2w grid file (about 2 MB, fetched from the VMF Data Server),
then evaluate at `(lat, lon, day-of-year, altitude)`. Returns
`pressure_hpa`, `temperature_k`, `e_hpa`, `a_h`, `a_w`, `T_lapse`,
`undulation_m`.

### `rinexpy.load_met(fn) -> xarray.Dataset`

RINEX MET (meteorological observation) reader.

### `rinexpy.load_eop(fn) -> xarray.Dataset` and `rinexpy.interp_eop(eop, epoch) -> dict`

IERS Bulletin A / EOP C04 reader plus per-epoch linear interpolation.

## Math

### `rinexpy.keplerian2ecef(sv) -> tuple[ndarray, ndarray, ndarray]`

Vectorised Keplerian to ECEF conversion. Input is an `xarray.Dataset`
slice for one or more satellites. Returns three NumPy arrays of ECEF
metres.

### `rinexpy.interpolate_sp3(sp3, times, *, order=10) -> xarray.Dataset`

Lagrange interpolation of SP3 positions to arbitrary epochs. Default
order is 10 (the IGS recommendation).

### `rinexpy.lambda_resolve(a_float, Q, *, ratio_threshold=3.0) -> dict`

Single-frequency LAMBDA integer ambiguity resolution. Returns `a_int`,
`ratio`, `accepted`, `candidates`, `sq_errors`.

### `rinexpy.lambda_dual_freq(a_l1, a_l2, cov_block=None, *, p1_m=None, p2_m=None, sigma_threshold=0.25) -> dict`

Dual-frequency LAMBDA-style fix via WL + NL with Melbourne-Wuebbena
gating.

### `rinexpy.spp_solve(sv_ecef, pseudoranges, *, initial_guess=(0,0,0), max_iter=10, tol=1e-3, raim=False, sigma_pr=5.0, p_fa=1e-4, max_exclusions=2, sv_labels=None, dcb_records=None, dcb_obs_code="", dcb_station="", dcb_epoch=None, tgd_map=None, tgd_gamma=1.0) -> dict`

Single-point positioning by iterative LSQ. Returns `position`,
`clock_bias`, `n_iter`, `residuals`, `lla`. With `raim=True` adds the
fault detection fields.

## Writers

### `rinexpy.to_rinex_obs(obs, fn, *, version=3) -> Path`

Round-trip a parsed dataset back to a RINEX 2 or RINEX 3 OBS file.

## Submodules

The submodules below are documented in their own pages. Importing them
explicitly is the right pattern when you need their internal API.

| Module | Page |
| --- | --- |
| `rinexpy.tools` | [Multi-file tools](../tooling/multi-file.md) |
| `rinexpy.geodesy` | [Atmospheric models](../corrections/atmosphere.md), [Single-point positioning](../positioning/spp.md) |
| `rinexpy.gpstime` | [Glossary § Time scales](glossary.md) |
| `rinexpy.rtcm3` | [RTCM and NTRIP](../formats/rtcm.md) |
| `rinexpy.rtcm2` | [RTCM and NTRIP](../formats/rtcm.md) |
| `rinexpy.ntrip` | [RTCM and NTRIP](../formats/rtcm.md) |
| `rinexpy.nmea` | [Receiver binary formats](../formats/receiver-binary.md) |
| `rinexpy.ubx` | [Receiver binary formats](../formats/receiver-binary.md) |
| `rinexpy.sbf` | [Receiver binary formats](../formats/receiver-binary.md) |
| `rinexpy.novatel` | [Receiver binary formats](../formats/receiver-binary.md) |
| `rinexpy.binex` | [Receiver binary formats](../formats/receiver-binary.md) |
| `rinexpy.sbas` | [SBAS and Galileo HAS](../formats/sbas-and-has.md) |
| `rinexpy.has` | [SBAS and Galileo HAS](../formats/sbas-and-has.md) |
| `rinexpy.beidou` | [Raw nav subframes](../formats/nav-subframes.md) |
| `rinexpy.gps_lnav`, `gps_cnav`, `gps_cnav2` | [Raw nav subframes](../formats/nav-subframes.md) |
| `rinexpy.galileo_nav` | [Raw nav subframes](../formats/nav-subframes.md) |
| `rinexpy.glonass` | [Raw nav subframes](../formats/nav-subframes.md) |
| `rinexpy.navic` | [Raw nav subframes](../formats/nav-subframes.md) |
| `rinexpy.nav4` | [Raw nav subframes](../formats/nav-subframes.md) |
| `rinexpy.rtk` | [RTK and integer fixing](../positioning/rtk.md) |
| `rinexpy.ppp` | [Precise point positioning](../positioning/ppp.md) |
| `rinexpy.ppp_rtk` | [Precise point positioning](../positioning/ppp.md) |
| `rinexpy.kalman`, `kalman_ztd`, `kalman_multignss` | [Kalman filters](../positioning/kalman.md) |
| `rinexpy.lambda_ar` | [LAMBDA and ambiguity resolution](../positioning/lambda.md) |
| `rinexpy.multifreq` | [LAMBDA and ambiguity resolution](../positioning/lambda.md) |
| `rinexpy.ssr` | [SSR corrections](../corrections/ssr.md) |
| `rinexpy.snapshot` | [Snapshot positioning](../positioning/snapshot.md) |
| `rinexpy.vrs` | [Network RTK and VRS](../positioning/network.md) |
| `rinexpy.network_dd` | [Network RTK and VRS](../positioning/network.md) |
| `rinexpy.gnssr` | [GNSS reflectometry](../positioning/gnssr.md) |
| `rinexpy.imu`, `imu_tight` | [IMU and INS fusion](../positioning/imu.md) |
| `rinexpy.time_transfer` | [Time transfer](../positioning/time-transfer.md) |
| `rinexpy.realtime` | [Real-time PPP](../positioning/realtime.md) |
| `rinexpy.cors` | [Module index](modules.md) |
| `rinexpy.qc` | [QC and cycle slips](../quality/qc.md) |
| `rinexpy.spoofing` | [Spoofing and jamming heuristics](../quality/spoofing.md) |
| `rinexpy.dcb`, `dcb_download` | [DCB and code biases](../corrections/dcb.md) |
| `rinexpy.tides`, `otl` | [Tides and station displacements](../corrections/tides.md) |
| `rinexpy.eop` | [EOP and Earth orientation](../corrections/eop.md) |
| `rinexpy.gpt2w` | [Atmospheric models](../corrections/atmosphere.md) |
| `rinexpy.met` | [Atmosphere products](../formats/atmosphere-products.md) |
| `rinexpy.antex`, `antex_calibrate` | [Atmosphere products](../formats/atmosphere-products.md) |
| `rinexpy.ionex` | [Atmosphere products](../formats/atmosphere-products.md) |
| `rinexpy.clk` | [SP3 and clock products](../formats/sp3-clk.md) |
| `rinexpy.sp3` | [SP3 and clock products](../formats/sp3-clk.md) |
| `rinexpy.nav_writer` | [RINEX navigation files](../formats/rinex-nav.md) |
| `rinexpy.streaming` | [Streaming over RAM-sized files](../tooling/streaming.md) |
| `rinexpy.writer` | [RINEX observation files](../formats/rinex-obs.md) |
| `rinexpy.asyncio` | [Async loading](../tooling/async.md) |
| `rinexpy.lazy` | [NetCDF and Zarr output](../tooling/io.md) |
| `rinexpy.zarr_io` | [NetCDF and Zarr output](../tooling/io.md) |
| `rinexpy.plots` | [Plotting helpers](../tooling/plots.md) |
| `rinexpy.plugins` | [Plugin system](../tooling/plugins.md) |
| `rinexpy.cli` | [Command-line interface](../tooling/cli.md) |
| `rinexpy.gw10` | [Receiver binary formats](../formats/receiver-binary.md) |

## Version

```python
import rinexpy as rp
print(rp.__version__)            # '0.1.0'
```

## Related pages

- [Module index](modules.md): a deeper per-module symbol table.
- [Glossary](glossary.md): GNSS-specific terminology.
