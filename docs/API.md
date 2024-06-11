# API reference

The public surface fits in one file. Anything not listed below (names
starting with `_`, or anything not in `rinexpy.__all__`) is internal
and can change between releases.

The `examples/` directory has runnable scripts for each major use
case. This file is the per-symbol reference.

## Top-level dispatch

### `rinexpy.load(path, out=None, *, use=None, tlim=None, useindicators=False, meas=None, verbose=False, overwrite=False, fast=True, interval=None)`

Auto-detects the file type and dispatches to the right reader. Handles
RINEX 2/3 NAV/OBS, SP3-a/c/d, and pre-converted NetCDF. For NetCDF
files that contain both groups it returns `{"nav": ..., "obs": ...}`.

Filtering is cheap because skipped records are never parsed:

```python
obs = rinexpy.load("big.rnx.gz",
                   use={"G", "E"},               # only GPS + Galileo
                   meas=["C1C", "L1C"],          # only these labels
                   tlim=("2018-07-29T12:00",     # only this 1-hour window
                         "2018-07-29T13:00"),
                   interval=30)                  # decimate to every 30s
```

### `rinexpy.rinexnav(fn, outfn=None, *, use=None, group="NAV", tlim=None, overwrite=False)`

Read a NAV file, or the NAV group of a NetCDF.

### `rinexpy.rinexobs(fn, outfn=None, *, use=None, group="OBS", tlim=None, useindicators=False, meas=None, verbose=False, overwrite=False, fast=True, interval=None, use_jit=None)`

Read an OBS file, or the OBS group of a NetCDF. `use_jit` opts in to
the optional `numba`-jitted decoder (needs the `jit` extra; about
1.9x faster end-to-end on real OBS3 files).

### `rinexpy.batch_convert(path, glob, out, *, use=None, tlim=None, useindicators=False, meas=None, verbose=False, fast=True, workers=None)`

Convert every file in `path` matching `glob` to NetCDF under `out`.
Per-file errors are logged and the run continues.

```python
written = rinexpy.batch_convert("data/2018", "*.rnx.gz", "out/", workers=4)
```

`workers=0` uses every CPU; `workers=None` (default) runs serially.

### `rinexpy.iter_obs3_epochs(fn, *, use=None, tlim=None, interval=None)`

Generator that yields `(datetime, xarray.Dataset)` per epoch without
loading the whole file. Constant memory regardless of file size.

```python
for time, ds in rinexpy.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(time, ds)
```

## Inspection

### `rinexpy.rinexinfo(fn) -> dict`

Cheap header peek. Returns `{"version", "filetype", "rinextype",
"systems"}`. For NetCDF inputs it returns a list of `rinextype` values
and the dataset attributes.

### `rinexpy.rinexheader(fn) -> dict`

The full parsed header. Keys include the RINEX header labels verbatim
(`"APPROX POSITION XYZ"`, etc.) and derived keys (`"position"`,
`"position_geodetic"`, `"t0"`, `"t1"`, `"interval"`, `"fields"`,
`"fields_ind"`, `"Nobs"`, `"Nl_sv"`).

### `rinexpy.gettime(fn) -> numpy.ndarray`

Sorted unique epoch timestamps as `datetime64[us]` for OBS and
`datetime64[ms]` for NAV.

### `rinexpy.to_datetime(time_coord)`

Convert an `xarray` time coordinate (or anything with
`.values.astype(...)`) to plain `datetime` objects.

## Other reader formats

### `rinexpy.load_sp3(fn, outfn=None) -> xarray.Dataset`

SP3-a/c/d ephemeris reader. Coords: `time`, `sv`, `ECEF=["x","y","z"]`.
Variables: `position`, `velocity`, `clock`, `dclock`, scalar `t0`.

### `rinexpy.load_clk(fn) -> xarray.Dataset`

RINEX clock product (`.clk`). Coords `(time, sv)`; variable `bias` (s).
Receiver labels land in the `stations` attr.

### `rinexpy.interpolate_clk(ds, sv, epoch) -> float`

Linear interpolation of `bias` for one SV at an arbitrary time.

### `rinexpy.load_ionex(fn) -> xarray.Dataset`

IONEX (`.inx`) global ionospheric TEC maps. Coords `(time, lat, lon)`;
variable `tec` in TECU.

### `rinexpy.interp_tec(ds, lat, lon, epoch) -> float`

Bilinear-in-space, linear-in-time TEC interpolation.

### `rinexpy.slant_tec(vertical_tec_tecu, el_deg) -> float`

Map vertical TEC to slant TEC via the 350 km thin-shell mapping
function.

### `rinexpy.load_antex(fn) -> list[dict]`

ANTEX (`.atx`) antenna phase center variations. One entry per antenna
with `type`, `serial`, `valid_from`, `valid_until`, and `frequencies`
mapping `freq_id` to `{north, east, up, noazi[, pcv]}`.

### `rinexpy.find_antenna(entries, type_code, *, serial=None, epoch=None) -> dict | None`

Pick one ANTEX entry by model name, with optional serial and validity
epoch.

### `rinexpy.apply_antex_pcv(entry, freq_id, el_deg, *, az_deg=None) -> float`

Antenna PCV correction in meters. Bilinear on the 2-D grid when
`az_deg` is given and the entry has a `DAZI > 0`; NOAZI otherwise.

### `rinexpy.load_gpt2w_grid(path) -> dict` and `rinexpy.gpt2w(grid, lat, lon, epoch, altitude_m=0.0) -> dict`

Load a GPT2w empirical met grid (~2 MB, fetched by the user from the
[VMF Data Server](https://vmf.geo.tuwien.ac.at/codes/)) and evaluate
at a `(lat, lon, day-of-year)`. Returns `pressure_hpa`, `temperature_k`,
`e_hpa`, `a_h`, `a_w`, `T_lapse`, `undulation_m`.

## Math

### `rinexpy.keplerian2ecef(sv) -> (X, Y, Z)`

Vectorized Keplerian to ECEF conversion. Input is an `xarray.Dataset`
slice for one or more satellites. Returns three `xarray.DataArray`s in
meters.

### `rinexpy.interpolate_sp3(sp3, times, *, order=10) -> xarray.Dataset`

Lagrange interpolation of SP3 positions to arbitrary epochs. Default
order is 10, per the IGS recommendation.

### `rinexpy.spp_solve(sv_ecef, pseudoranges, *, initial_guess=(0,0,0), max_iter=10, tol=1e-3) -> dict`

Single-point positioning by iterative LSQ. Returns `position`,
`clock_bias`, `n_iter`, `residuals`, `lla`.

## RTK

### `rinexpy.rtk.double_difference_solve(...) -> dict`

Float-ambiguity RTK solution. See the module docstring for the full
signature.

### `rinexpy.rtk.rtk_fix(...) -> dict`

End-to-end RTK with LAMBDA integer fix. Returns `float`, `fixed`,
`lambda` sub-dicts and `fixed_accepted` (bool).

### `rinexpy.lambda_resolve(a_float, Q, *, ratio_threshold=3.0) -> dict`

Single-frequency LAMBDA integer ambiguity resolution. Returns
`a_int`, `ratio`, `accepted`, `candidates`, `sq_errors`.

### `rinexpy.lambda_dual_freq(a_l1_float, a_l2_float, cov_block=None, *, p1_m=None, p2_m=None, sigma_threshold=0.25) -> dict`

Dual-frequency LAMBDA-style fix via Wide-Lane / Narrow-Lane, with
Melbourne-Wübbena when pseudoranges are supplied.

## Geodesy

`rinexpy.geodesy` exports:

| function | purpose |
|---|---|
| `ecef_to_lla(x, y, z)` | WGS-84 ECEF to (lat, lon, alt) in degrees and m |
| `lla_to_ecef(lat, lon, alt)` | inverse |
| `azimuth_elevation(rx, sv)` | (az, el) in degrees from receiver to SVs |
| `dop(sv_ecef, rx_ecef)` | `{"GDOP", "PDOP", "HDOP", "VDOP", "TDOP"}` |
| `klobuchar(alpha, beta, rx_lla, az, el, gps_sec)` | broadcast L1 iono delay (m) |
| `standard_atmosphere(alt)` | ICAO standard `(T, P, e)` at altitude |
| `saastamoinen(el, alt, *, P, T, e)` | wet+dry tropo delay (m) |
| `niell_mapping(el, lat, alt, doy)` | NMF dry+wet mapping factors |
| `vmf1(a_h, a_w, el, lat, alt, doy)` | Vienna Mapping Function 1 |

## Time

`rinexpy.gpstime` exports:

| function | purpose |
|---|---|
| `datetime_to_gps(t)` | `(week, sow)` |
| `gps_to_datetime(week, sow)` | inverse |
| `datetime_to_gps_seconds(t)` / `gps_seconds_to_datetime(s)` | continuous form |
| `leap_seconds_at(t)` | TAI-UTC at a UTC datetime |
| `gps_week_rollover(week_mod_1024, reference)` | resolve 10-bit week ambiguity |
| constants: `GPS_EPOCH`, `SECONDS_PER_WEEK`, `LEAP_SECONDS` |  |

## Multi-frequency helpers

`rinexpy.multifreq` exports:

| symbol | purpose |
|---|---|
| `LAMBDA_L1`, `LAMBDA_L2`, `LAMBDA_WL`, `LAMBDA_NL`, `F1`, `F2` | constants |
| `wide_lane_phase(phi1, phi2)` | WL combination in cycles of λ_WL |
| `narrow_lane_phase(phi1, phi2)` | NL combination in cycles of λ_NL |
| `melbourne_wubbena(phi1, phi2, p1_m, p2_m)` | iono+geometry-free WL |
| `resolve_wide_lane(...)` | round MW/λ_WL with a sigma_threshold gate |
| `split_wl_into_l1_l2(n_wl, n_nl)` | recover (N1, N2) |
| `lambda_dual_freq(...)` | dual-frequency fix |

## Tooling

`rinexpy.tools` exports:

| function | purpose |
|---|---|
| `validate_file(fn) -> dict` | QC report (header consistency, gaps, intervals) |
| `concat_files(files, *, dim="time") -> xarray.Dataset` | join + dedup |
| `diff_datasets(a, b, *, rtol, atol) -> dict` | first-divergence finder |

## Plotting (optional `plot` extra)

`rinexpy.plots` exports:

| function | purpose |
|---|---|
| `obstimeseries(obs)` | L1/L1C carrier-phase time series |
| `navtimeseries(nav)` | satellite ground tracks |
| `receiver_locations(locs)` | scatter plot of receiver positions |
| `skyplot(sv_az_el, ...)` | polar trajectory plot from `{sv: (az, el)}` |
| `timeseries(data)` | dispatch on the dataset's `rinextype` attr |

## Streaming

| function | purpose |
|---|---|
| `rinexpy.iter_obs3_epochs(fn, ...)` | per-epoch generator (constant memory) |
| `rinexpy.streaming.iter_obs3_epochs` | same, reachable via the submodule |

## Async (`rinexpy.asyncio`)

| function | purpose |
|---|---|
| `aload(fn, **kwargs)` | thread-pool wrapper around `load` |
| `aload_many(files, **kwargs)` | concurrent multi-file load |

## I/O variants

| function | purpose |
|---|---|
| `rinexpy.opener(fn, *, header=False)` | the file/decompression context |
| `rinexpy.zarr_io.to_zarr(ds, store)` | write a parsed dataset as Zarr |
| `rinexpy.lazy.load_lazy(files, chunk_size=None)` | dask-backed multi-file load |

## RTCM3 / NTRIP

`rinexpy.rtcm3`:

| function / constant | purpose |
|---|---|
| `PREAMBLE = 0xD3` | RTCM3 sync byte |
| `crc24q(data)` | CRC-24Q checksum |
| `iter_messages(stream, *, check_crc=False)` | yield decoded message dicts |
| `decode_message(msg_id, body)` | dispatch one message body |

Decoded types: 1004 (extended L1/L2 RTK obs), 1005 (stationary RTK
reference station), 1006 (1005 + antenna height), 1019 (GPS ephemeris
subset), 1020 (GLONASS slot/frequency), 1033 (antenna descriptors),
MSM4 (1074-1134) and MSM7 (1077-1137) full per-cell decoders. Other
IDs come back with raw `payload_bytes`.

`rinexpy.ntrip`:

| function | purpose |
|---|---|
| `fetch_sourcetable(host, *, port=2101, timeout=30)` | parse caster STR;/CAS;/NET; |
| `stream(host, mountpoint, *, user="", password="", port=2101, ...)` | yield bytes |

## Other receiver formats

Each is a submodule with the same shape: an `iter_*` generator that
yields one decoded dict per record or message, with checksum
validation on by default.

### `rinexpy.nmea`: ASCII NMEA-0183

| function | purpose |
|---|---|
| `parse_sentence(line, *, check_crc=True)` | one sentence -> dict (or None) |
| `iter_lines(stream, *, check_crc=True)` | yield decoded sentences from a line iterator |
| `checksum(sentence)` | XOR checksum (compare against the trailing `*HH`) |

Decoded sentence types: GGA, RMC, GSA, GSV, VTG. Other types come back
with `talker`, `type`, `fields`.

### `rinexpy.ubx`: u-blox UBX binary

| function / constant | purpose |
|---|---|
| `SYNC1 = 0xB5`, `SYNC2 = 0x62` | UBX sync bytes |
| `iter_messages(stream, *, check_crc=True)` | yield decoded UBX dicts |
| `decode_message(msg_class, msg_id, payload)` | dispatch one payload |
| `fletcher_checksum(data) -> (ck_a, ck_b)` | UBX 8-bit Fletcher |

Decoded classes/IDs: NAV-PVT, NAV-SAT, RXM-RAWX, RXM-SFRBX. Others
come back with raw `payload_bytes`.

### `rinexpy.sbf`: Septentrio SBF

| function / constant | purpose |
|---|---|
| `SYNC = b"\\x24\\x40"` | SBF sync bytes (`'$@'`) |
| `iter_blocks(stream, *, check_crc=True)` | yield decoded SBF dicts |
| `crc_ccitt(data, *, init=0)` | CRC-CCITT for SBF blocks |

Decoded block IDs: PVTGeodetic (4007), MeasEpoch (4027), GPSNav (5891).

### `rinexpy.novatel`: NovAtel OEM

| function / constant | purpose |
|---|---|
| `SYNC = b"\\xaa\\x44\\x12"` | NovAtel sync sequence |
| `iter_messages(stream, *, check_crc=True)` | yield decoded NovAtel dicts |
| `crc32(data)` | IEEE-802.3 CRC32 |

Decoded message IDs: BESTPOS (42), BESTXYZ (241), RAWEPHEM (41).

### `rinexpy.binex`: UNAVCO BINEX

| function / constant | purpose |
|---|---|
| `SYNC = 0xC2` | forward byte order, normal-records sync |
| `iter_records(stream, *, check_crc=True)` | yield decoded BINEX records |
| `read_ubnxi(stream)` | variable-length unsigned int decoder |
| `encode_ubnxi(value)` | inverse |
| `xor_checksum(data)` / `crc16_ccitt(data)` | the two short-record CRCs |

Records come back with `record_id`, `length`, `body_bytes`.

### `rinexpy.rtcm2`: RTCM SC-104 v2.x (legacy DGPS)

| function / constant | purpose |
|---|---|
| `PREAMBLE = 0x66` | RTCM2 preamble byte |
| `extract_data_bits(buf)` | strip the 6-of-8 wire encoding |
| `iter_messages(stream)` | yield decoded RTCM2 message dicts |

Decoded types: 1 (DGPS pseudorange corrections per SV with
PRC/RRC/IODE), 3 (reference station ECEF in cm), 9 (high-rate
corrections, same payload as 1). Other types come back as raw
`data_words` (24-bit ints). Hamming parity is not validated.

### `rinexpy.beidou`: BeiDou D1/D2 raw subframes

| function / constant | purpose |
|---|---|
| `PREAMBLE = 0x712` | BeiDou 11-bit nav-message preamble |
| `decode_d1_subframe1(words)` | clock + ionospheric model from D1 SF1 |
| `decode_d2_page1(words)` | clock parameters from D2 page 1 |
| `encode_subframe_words(spec)` | test helper: pack data bits into 10 30-bit words |

Input is a list of 10 30-bit subframe words. Parity bits are included
but not validated; they're stripped to a 224-bit data stream and
fields are read by absolute offset per ICD-BDS-OS-200 Tables 5-3/5-4.

## Writer

### `rinexpy.to_rinex_obs(obs, fn, *, version=3) -> Path`

Round-trip a parsed dataset back to a RINEX 2 or RINEX 3 OBS file.
Good enough for the read-modify-write loop (filter, decimate, re-emit).

## CLI

```sh
rinexpy read   <file>            # parse and print
rinexpy times  <file>            # print just epoch timestamps
rinexpy info   <file>            # parsed header
rinexpy convert <dir> <glob> --out <dir> [-j WORKERS]   # batch -> NetCDF
```

All read/convert subcommands accept `-u/--use`, `-m/--meas`,
`-t/--tlim START STOP`, `--useindicators`, `--interval`, `--strict`.
