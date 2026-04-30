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

### `rinexpy.spp_solve(sv_ecef, pseudoranges, *, initial_guess=(0,0,0), max_iter=10, tol=1e-3, raim=False, sigma_pr=5.0, p_fa=1e-4, max_exclusions=2, sv_labels=None, dcb_records=None, dcb_obs_code="", dcb_station="", dcb_epoch=None, tgd_map=None, tgd_gamma=1.0) -> dict`

Single-point positioning by iterative LSQ. Returns `position`,
`clock_bias`, `n_iter`, `residuals`, `lla`.

- `raim=True` delegates to `spp_solve_raim` and adds `raim_test`,
  `raim_threshold`, `fault_detected`, `excluded_svs`, `raim_failed`.
- `dcb_records=` (from `rinexpy.dcb.read_bsx`) + `dcb_obs_code=`
  applies SINEX-BIAS OSB corrections per SV (and per station when
  `dcb_station=` is set).
- `tgd_map={sv: tgd_seconds}` (from `tgd_from_nav`) applies the
  broadcast group delay. `tgd_gamma=1` on L1, `(f1/f2)**2` on L2,
  `0` for the iono-free combination.

### `rinexpy.positioning.spp_solve_raim(sv_ecef, pseudoranges, *, sigma_pr=5.0, p_fa=1e-4, max_exclusions=2, ...) -> dict`

Direct entry point for RAIM-instrumented SPP. Same return dict as
`spp_solve(raim=True)`.

### `rinexpy.positioning.tgd_from_nav(nav, epoch, *, field="TGD") -> dict`

Pull per-SV broadcast group-delay values from a NAV dataset.
Handles GPS `TGD`, BeiDou `TGD1`/`TGD2`, Galileo `BGDe5a`/`BGDe5b`.

### `rinexpy.positioning.apply_tgd_correction(pseudoranges, sv_labels, tgd_map, *, gamma=1.0) -> ndarray`

Subtract `c * gamma * TGD` per SV from `pseudoranges`. Pair with
`tgd_from_nav` for broadcast-only single-frequency SPP.

## RTK

### `rinexpy.rtk.double_difference_solve(...) -> dict`

Float-ambiguity RTK solution. See the module docstring for the full
signature.

### `rinexpy.rtk.rtk_fix(...) -> dict`

End-to-end RTK with LAMBDA integer fix. Returns `float`, `fixed`,
`lambda` sub-dicts and `fixed_accepted` (bool).

### `rinexpy.rtk.SequentialRTK(base_position_ecef, *, wavelength, ratio_threshold=3.0, slip_threshold_cycles=0.5, sigma_pr=1.0, sigma_phase=0.005, min_lock_to_fix=2)`

Multi-epoch RTK with integer-ambiguity carry-over, per-SV cycle-slip
detection (inter-epoch SD phase vs SD code), and partial AR on the
most-precise subset when the full ratio test fails.

```python
rtk = SequentialRTK(base_ecef, wavelength=LAMBDA_L1)
out = rtk.update(svs, rover_pr, base_pr, rover_phase, base_phase, sv_ecef)
# out: baseline, rover_position, n_total, n_fixed, fixed_accepted,
#      ratio, carry_over_count, slipped_svs.
```

Call `rtk.reset()` to drop all per-SV state.

### `rinexpy.rtk.SequentialRTK.update(sv_ids, rover_pr, base_pr, rover_phase, base_phase, sv_positions_ecef, *, initial_baseline=None) -> dict`

Process one epoch. Internally calls `rtk_fix`, runs partial AR if the
full fix is rejected, and updates the cycle-slip / lock tracking
state. Returns the per-epoch result dict described above.

### `rinexpy.lambda_resolve(a_float, Q, *, ratio_threshold=3.0) -> dict`

Single-frequency LAMBDA integer ambiguity resolution. Returns
`a_int`, `ratio`, `accepted`, `candidates`, `sq_errors`.

### `rinexpy.lambda_dual_freq(a_l1_float, a_l2_float, cov_block=None, *, p1_m=None, p2_m=None, sigma_threshold=0.25) -> dict`

Dual-frequency LAMBDA-style fix via Wide-Lane / Narrow-Lane, with
Melbourne-Wübbena when pseudoranges are supplied.

## Precise Point Positioning (PPP)

### `rinexpy.ppp.ppp_solve(obs, sp3, clk=None, *, initial_position_ecef=None, obs_codes=None, sigma_code=1.0, sigma_phase=0.005, elevation_mask_deg=7.0, max_epochs=None, apply_tropo=True, antenna=None, gpt2w_grid=None, dcb_records=None, station_id="", apply_wind_up=False, ssr=None) -> dict`

Static-receiver PPP driver. Composes:

- `interp.interpolate_sp3` and `clk.interpolate_clk` per epoch.
- `geodesy.saastamoinen` (default) or `vmf1` over GPT2w-derived
  (ZHD, ZWD) when `gpt2w_grid=` is supplied.
- `antex.apply_antex_pcv` per SV per band when `antenna=` is given.
- `dcb.get_bias` per SV (and per receiver station) when `dcb_records=`
  is given.
- `geodesy.phase_wind_up_correction` when `apply_wind_up=True`.
- `kalman.GNSSFilter` (= `StaticPPPFilter`) as the per-epoch EKF.

Auto-picks an L1/L2 obs-code quadruple from the dataset
(`C1C/C2W/L1C/L2W` first, then a documented priority list). Override
with `obs_codes=(C1, C2, L1, L2)`.

`clk=None` is permitted when `ssr=` is supplied — the SSR clock
correction then replaces the CLK lookup for every SV with an SSR
entry.

Returns `{position, lla, clock_bias_s, position_sigma_m, n_epochs,
trace, obs_codes, filter}`. `trace` is a per-epoch list of
`{epoch, position, clock_bias_s}`.

### `rinexpy.kalman.GNSSFilter(n_sv, initial_position, *, sigma_code=1.0, sigma_phase=0.005, sigma_position_init=10.0, sigma_clock_init=300.0, sigma_clock_rate_m=10.0, sigma_position_rate_m=0.0, sigma_ambig_init_m=1000.0)`

The named EKF entry point — an alias for `StaticPPPFilter`. State is
`[px, py, pz, c*dt, N_1, ..., N_n_sv]`. Static when
`sigma_position_rate_m=0` (default); kinematic when > 0 (the position
variance grows by `rate^2 * dt` per `predict(dt)`).

Methods: `predict(dt)`, `update(sv_ecef, sat_clock_s, pr_if, phase_if,
tropo_m=None)`, `update_with_slip_check(...)`, `reset_ambiguity(sv_index)`,
`reset_ambiguities(sv_indices)`. Properties: `position`, `clock_bias_s`,
`ambiguities_m`, `position_sigma`.

`rinexpy.kalman_multignss.StaticPPPFilterMultiGNSS` and
`rinexpy.kalman_ztd.StaticPPPFilterZTD` are siblings with the
constellation-aware and ZTD-augmented state vectors.

### `rinexpy.ssr.SSRCorrections(messages=None)`

Composer over decoded RTCM3 SSR messages. Absorbs orbit (1057, 1063,
1240, 1246, 1252, 1258), clock (1058, 1064, 1241, ...), combined
(1060, 1066, ...), and code-bias (1059, 1065, 1242, ...) messages.

Methods:

- `add_message(msg)` — absorb one decoded message dict.
- `orbit_correction_ecef(sv, sat_pos_ecef, sat_vel_ecef, epoch_sow) ->
  ndarray` — radial / along-track / cross-track delta rotated into
  ECEF using the SV's instantaneous frame.
- `clock_correction_s(sv, epoch_sow) -> float` — c0 + c1·dt + c2·dt²
  evaluated in seconds.
- `code_bias_m(sv, obs_code) -> float` — per-(SV, RINEX-3 code) bias
  in meters. Default RINEX-3 mapping for GPS / GLO / GAL / QZS /
  BDS / SBAS signal IDs is built in.
- `known_satellites()`, `has_orbit(sv)`, `has_clock(sv)`.

## Snapshot positioning

### `rinexpy.snapshot.snapshot_positioning(code_phase_chips, sv_positions_ecef, initial_position_ecef, *, max_iter=20, tol=1.0) -> dict`

Code-phase-only short-data SPP (van Diggelen A-GPS). Resolves the
integer-millisecond ambiguity per SV from a coarse position prior,
then runs a 4-unknown LSQ for `(position, clock_bias)`. Returns
`position_ecef`, `lla`, `time_bias_s`, `pseudoranges_m`,
`K_integer_ms`, `n_iter`. Prior must be within ~150 km of truth.

## Network RTK / VRS

### `rinexpy.vrs.synthesize_vrs(bases, rover_approx_pos, *, wavelength) -> dict`

Compose observations from a network of physical bases into a virtual
base co-located with `rover_approx_pos`. For each SV the synthesized
pseudorange is the rover's geometric range plus a plane-fit of the
per-base residuals across the network. Output dict has the same shape
as the input baseline blocks for `rtk.double_difference_solve` /
`rtk_fix` (`base_position`, `sv_positions`, `pr`, `phase`).

## GNSS reflectometry

### `rinexpy.gnssr.detrend_snr(snr_db, elevation_rad, *, order=4) -> ndarray`

Subtract a low-order polynomial in `sin(elev)` from the SNR series
to isolate the multipath oscillation.

### `rinexpy.gnssr.snr_to_sea_height(snr_db, elevation_rad, *, wavelength_m, height_search_m=(0.5, 50.0), n_freqs=1024, detrend_order=4) -> dict`

SNR-based reflector-height retrieval (Larson 2008). Detrends + runs
an in-tree Lomb-Scargle periodogram, peaks the frequency, converts to
height as `H = f · lambda / 2`. Returns `height_m`, `peak_power`,
`frequencies_per_sin_elev`, `power`, `detrended`.

## Antenna PCV calibration

### `rinexpy.antex_calibrate.calibrate_pcv(residuals_m, elevation_rad, azimuth_rad, *, antenna_type, serial="", frequency="G01", valid_from=None, valid_until=None, dazi_deg=5.0, dzen_deg=5.0, zen_max_deg=90.0) -> dict`

Bin a calibration session's post-fit residuals on a 2-D
(azimuth × zenith) grid, average per cell, and emit an ANTEX-shaped
entry. NOAZI vector is the azimuth-averaged residual per zenith bin.

### `rinexpy.antex_calibrate.write_antex(entries, path) -> None`

Write a list of calibrated entries to a `.atx` file that round-trips
cleanly through `rinexpy.antex.load_antex`.

## DCB / SINEX-BIAS

### `rinexpy.dcb.read_bsx(path) -> list[dict]`

Read a SINEX-BIAS (`.BSX`) file. Each record carries `bias_type`
(`OSB` / `DSB`), `prn`, `station`, `obs1`, `obs2`, `start`, `end`,
`unit`, `value`.

### `rinexpy.dcb.read_code_dcb(path, *, year=None, month=None) -> list[dict]`

Pre-2017 AIUB CODE monthly DCB reader (`P1P2YYMM.DCB`, etc.). Returns
records in the same schema as `read_bsx` — CODE's P1/P2/C1/C2 are
translated to RINEX-3 codes (C1W, C2W, C1C, C2C).

### `rinexpy.dcb.get_bias(records, *, prn="", station="", obs1, obs2="", epoch=None) -> float | None`

Look up one bias value in meters. Returns `None` if no record matches
the (PRN/station, obs1, obs2, epoch) tuple.

### `rinexpy.dcb.correct_pseudorange(pseudorange_m, *, prn, obs_code, records, epoch=None, station="") -> float`

Apply satellite (and optional receiver) OSB to one pseudorange.

### `rinexpy.dcb_download.auto_load_dcb(date, *, cache_dir=None, timeout=60.0, source="bkg") -> list[dict]`

Date-routed convenience: pre-2017 → AIUB CODE monthly P1-P2;
2017+ → IGS BKG daily MGEX CAS Rapid. Returns records in the unified
`get_bias`-friendly schema.

### `rinexpy.dcb_download.download_dcb(date, *, product="CAS", cache_dir=None, source="bkg", timeout=60.0) -> Path`

Fetch a daily MGEX SINEX-BIAS from the IGS BKG public mirror (no
auth) or NASA CDDIS (needs Earthdata Login in `~/.netrc`). Caches the
decompressed `.BSX` under `~/.cache/rinexpy/dcb/`.

### `rinexpy.dcb_download.download_legacy_code_dcb(date, *, product="P1P2", cache_dir=None, timeout=60.0) -> Path`

Pre-2017 path: fetch one CODE monthly `.DCB.Z` from AIUB and
decompress (uses `ncompress` from the `[lzw]` extra, or `gzip -dc`
fallback).

## Modernized-signal nav decoders

Same shape across the family: feed the bit-packed message body bytes,
get a dict per ICD field.

| module | symbols | signal |
|---|---|---|
| `rinexpy.gps_lnav` | `decode_lnav_subframe1/2/3`, `encode_lnav_words` | GPS L1 C/A LNAV |
| `rinexpy.gps_cnav` | `decode_cnav_mt10`, `decode_cnav_mt11`, `decode_cnav_message` | GPS L2C / L5 CNAV |
| `rinexpy.gps_cnav2` | `decode_cnav2_subframe2` | GPS L1C CNAV-2 |
| `rinexpy.galileo_nav` | `decode_fnav_page1/2`, `decode_inav_word1/4` | Galileo F-NAV (E5a) + I-NAV (E1B/E5b) |
| `rinexpy.glonass` | `decode_glonass_string1/2/3`, `decode_glonass_string` | GLONASS L1OF/L2OF strings (sign-magnitude per ICD §4.4) |
| `rinexpy.navic` | `decode_navic_subframe1/2`, `decode_navic_subframe34`, `decode_navic_subframe` | NavIC / IRNSS L5 + S |
| `rinexpy.beidou` | `decode_d1_subframe1`, `decode_d2_page1`, `encode_subframe_words` | BeiDou D1 + D2 |
| `rinexpy.sbas` | `decode_sbas_mt1/2_5/6/7/9/17/18/24/25/26`, `decode_sbas_message` | SBAS L1 (WAAS / EGNOS / MSAS / GAGAN) |

### `rinexpy.nav4.load_nav4(fn) -> dict[str, list[dict]]`

RINEX 4 NAV reader for the structured record types. Returns a dict
keyed by `"EPH"`, `"STO"`, `"EOP"`, `"ION"`. ION dispatches by model
(`KLOB`, `NEQG`, `BDGIM`).

## Tides + EOP

### `rinexpy.tides.solid_earth_tide_displacement(epoch, station_ecef) -> ndarray`

IERS Conventions 2010 step-1 + step-2 solid-earth tide ECEF
displacement (m) at one station, one epoch.

### `rinexpy.tides.pole_tide_displacement(epoch, station_ecef, eop) -> ndarray`

Solid pole-tide displacement (m). Needs polar-motion values from
`rinexpy.eop`.

### `rinexpy.tides.ocean_pole_tide_displacement(epoch, station_ecef, eop) -> ndarray`

IERS 2010 ocean-pole-tide model.

### `rinexpy.otl.read_blq(path) -> dict` and `rinexpy.otl.ocean_tide_loading_displacement(coeffs, epoch) -> ndarray`

BLQ-format ocean-tide-loading reader and 11-constituent displacement
evaluator.

### `rinexpy.geodesy.phase_wind_up_correction(sat_xhat, sat_yhat, rx_xhat, rx_yhat, los_rx_to_sat, *, previous_cycles=0.0) -> float`

Wu et al. 1993 carrier-phase wind-up. Accumulator-style: pass the
previous epoch's value to keep continuity across 2π wraps.

### `rinexpy.eop.load_eop(fn) -> xarray.Dataset` and `rinexpy.eop.interp_eop(eop, epoch) -> dict`

IERS Bulletin A / C04 reader and `(x, y, ut1_utc, lod, dx, dy)`
interpolation at one epoch. Pair with `geodesy.ecef_to_eci`.

## Time transfer

### `rinexpy.time_transfer.p3_combination(p1_m, p2_m) -> ndarray`

Iono-free P3 combination per SV.

### `rinexpy.time_transfer.common_view_difference(p3_A, p3_B, rho_A, rho_B) -> ndarray`

Same-SV common-view difference between two stations.

### `rinexpy.time_transfer.estimate_clock_difference_s(differences) -> float`

Robust median estimate of the receiver-clock difference (s).

## QC / multipath / cycle slips

`rinexpy.qc`:

| symbol | purpose |
|---|---|
| `detect_slips(obs)` | dispatch slip detection per SV via the best available method |
| `detect_slips_phase_only`, `_geometry_free`, `_mw` | individual detectors |
| `mp1(...)`, `mp2(...)` | TEQC-style multipath combinations |
| `multipath_rms(mp_m)` | per-arc RMS |
| `hatch_filter(pr, phi, slips, window)` | carrier-smoothed code |

## Spoofing / jamming heuristics

`rinexpy.spoofing`:

| function | purpose |
|---|---|
| `check_snr_uniformity(snr_db_by_sv)` | flag suspicious uniform SNR across SVs |
| `check_position_jumps(positions, *, max_speed_m_s)` | inter-epoch position jumps |
| `check_clock_drift(times, clock_bias_s, *, max_drift_s_per_s)` | clock-rate sanity |
| `check_agc(agc, *, baseline)` | AGC level vs baseline |

## RINEX MET

### `rinexpy.load_met(fn) -> xarray.Dataset`

Reader for RINEX MET (`.m`) files (surface met data: pressure,
temperature, humidity, plus optional sensor metadata).

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

Decoded types:

- 1004 (extended L1/L2 RTK obs).
- 1005, 1006 (stationary RTK reference station, with optional antenna
  height).
- 1019 (GPS ephemeris subset), 1020 (GLONASS slot / frequency).
- 1029 (Unicode text string).
- 1033 (antenna descriptors).
- 1230 (GLONASS L1/L2 C/A + P code-phase biases).
- MSM 1/2/3/4/5/6/7 (1074-1134, 1077-1137, ...) — full per-cell
  decoders covering pseudorange-only, phase-only, mixed, and signal
  extension variants.
- SSR family (RTCM 10403.3 §3.5.10): 1057-1068 (GPS + GLONASS) and
  1240-1263 (Galileo / QZSS / SBAS / BeiDou) for orbit / clock /
  combined / URA / high-rate-clock / code-bias.
- IGS-SSR MT 4076 (subtype-dispatched).

Other IDs come back with raw `payload_bytes`.

## SBAS L1

`rinexpy.sbas`:

| function | purpose |
|---|---|
| `decode_sbas_message(payload)` | dispatch by MT |
| `decode_sbas_mt1(payload)` | PRN mask + IODP |
| `decode_sbas_mt2_5(payload)` | fast pseudorange corrections (13 PRC + UDREI) |
| `decode_sbas_mt6(payload)` | integrity (51 UDREI + 4 IODF) |
| `decode_sbas_mt7(payload)` | fast-correction degradation factors |
| `decode_sbas_mt9(payload)` | GEO ranging (ECEF pos/vel/acc + a_Gf0/1) |
| `decode_sbas_mt17(payload)` | GEO almanacs (up to 3 entries) |
| `decode_sbas_mt18(payload)` | iono grid point mask |
| `decode_sbas_mt24(payload)` | mixed fast + long-term half |
| `decode_sbas_mt25(payload)` | long-term satellite corrections |
| `decode_sbas_mt26(payload)` | iono delays for 15 grid points |

Reference: RTCA DO-229E §A.4. Each L1 SBAS message is 250 bits.

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
rinexpy read   <file>                                   # parse and print
rinexpy times  <file>                                   # epoch timestamps
rinexpy info   <file>                                   # parsed header
rinexpy convert <dir> <glob> --out <dir> [-j WORKERS]   # batch -> NetCDF
rinexpy spp <obs> <nav>                                 # single-point fix
rinexpy rtk <rover.obs> <base.obs> <nav>                # RTK baseline
rinexpy ppp <obs> <sp3> <clk>                           # PPP solve
rinexpy splice <a.obs> <b.obs> --out <out.obs>          # concat by time
rinexpy decimate <file.obs> --interval <s>              # interval thin
```

All read/convert subcommands accept `-u/--use`, `-m/--meas`,
`-t/--tlim START STOP`, `--useindicators`, `--interval`, `--strict`.
