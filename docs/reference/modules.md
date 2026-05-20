# Module index

A per-module table of contents. Modules are grouped by layer in the
architecture. For the layered diagram, see
[Architecture](../internals/architecture.md).

## Layer 0 and 1 (primitives)

The lowest two layers are internal. The names start with an underscore
and are not part of the public surface.

| Module | Role |
| --- | --- |
| `_io` | file opener, gzip / bz2 / zip / Hatanaka / mmap |
| `_time` | epoch parsers and time-axis normalisation |
| `_common` | Fortran float, time-system sniffer, globbing |
| `_jit` | optional numba JIT path |
| `_errors` | line-counting stream and error-context helpers |
| `_native` | dispatcher for the optional C++ extension |
| `_types` | type aliases (`FileLike`, `TimeLimit`, ...) |
| `_version` | first-line version detector |

## Readers

### `rinexpy.obs2`

RINEX 2 observation file reader.

| Symbol | Type | Notes |
| --- | --- | --- |
| `rinexobs2(fn, use=None, ...)` | function | the main reader |
| `rinexsystem2(fn, system, ...)` | function | one-constellation reader |
| `obstime2(fn)` | function | the time axis only |

### `rinexpy.obs3`

RINEX 3 observation file reader. The main performance win is here.

| Symbol | Type | Notes |
| --- | --- | --- |
| `rinexobs3(fn, use=None, ...)` | function | the main reader |
| `obstime3(fn)` | function | the time axis only |

### `rinexpy.nav2`

RINEX 2 navigation file reader.

| Symbol | Type | Notes |
| --- | --- | --- |
| `rinexnav2(fn, *, tlim=None)` | function | the main reader |
| `navtime2(fn)` | function | the time axis only |

### `rinexpy.nav3`

RINEX 3 navigation file reader.

| Symbol | Type | Notes |
| --- | --- | --- |
| `rinexnav3(fn, *, use=None, tlim=None)` | function | the main reader |
| `navtime3(fn)` | function | the time axis only |

### `rinexpy.nav4`

RINEX 4 navigation file reader (STO / EOP / ION / EPH).

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_nav4(fn)` | function | returns `{"EPH": [...], "STO": [...], "EOP": [...], "ION": [...]}` |

### `rinexpy.nav_writer`

RINEX 3 NAV writer.

| Symbol | Type | Notes |
| --- | --- | --- |
| `write_nav3(ds, outpath)` | function | round-trip Keplerian ephemerides (G, E, C, J) |

### `rinexpy.sp3`

SP3-a / SP3-c / SP3-d reader and writer.

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_sp3(fn, outfn=None)` | function | reader |
| `write_sp3(ds, outpath, *, version="c")` | function | writer |
| `stitch_sp3(*paths)` | function | concatenate consecutive SP3 files |

### `rinexpy.clk`

RINEX clock product reader.

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_clk(fn)` | function | reader |
| `interpolate_clk(ds, sv, epoch)` | function | linear interpolation per SV |

### `rinexpy.ionex`

IONEX TEC map reader.

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_ionex(fn)` | function | reader |
| `interp_tec(ds, lat_deg, lon_deg, epoch)` | function | bilinear interp |
| `slant_tec(vtec_tecu, el_deg)` | function | 350 km thin-shell mapping |

### `rinexpy.antex`, `rinexpy.antex_calibrate`

ANTEX antenna PCV reader plus calibration tool.

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_antex(fn)` | function | reader |
| `find_antenna(entries, type_code, *, serial=None, epoch=None)` | function | lookup |
| `apply_antex_pcv(entry, freq_id, el_deg, *, az_deg=None)` | function | per-observation correction |
| `pcv_corrections_for_observations(entry, freq_id, sv_ecef, station_ecef)` | function | vectorised version |
| `calibrate_pcv(residuals_m, elevation_rad, azimuth_rad, *, antenna_type, serial="", ...)` | function | fit a new ANTEX entry |
| `write_antex(entries, path)` | function | round-trip the calibrator output |

### `rinexpy.met`

RINEX MET (meteorological observation) reader.

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_met(fn)` | function | reader |

### `rinexpy.eop`

IERS EOP C04 reader.

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_eop(fn)` | function | reader |
| `interp_eop(eop, epoch)` | function | linear interpolation at one epoch |

### `rinexpy.gpt2w`

GPT2w empirical met / VMF1 grid evaluator.

| Symbol | Type | Notes |
| --- | --- | --- |
| `load_gpt2w_grid(path)` | function | reader |
| `gpt2w(grid, lat_deg, lon_deg, epoch, altitude_m=0.0)` | function | evaluate |

## Streaming readers

### `rinexpy.streaming`

Per-epoch generator for RINEX 3 OBS.

| Symbol | Type | Notes |
| --- | --- | --- |
| `iter_obs3_epochs(fn, *, use=None, tlim=None, interval=None)` | generator | yields `(time, ds)` |

### `rinexpy.rtcm3`

RTCM 3 framer and decoder.

| Symbol | Type | Notes |
| --- | --- | --- |
| `PREAMBLE = 0xD3` | constant | sync byte |
| `crc24q(data)` | function | CRC-24Q |
| `iter_messages(stream, *, check_crc=False)` | generator | decoded messages |
| `decode_message(msg_id, body)` | function | dispatch one message |

### `rinexpy.rtcm2`

Legacy RTCM 2.x DGPS framer.

| Symbol | Type | Notes |
| --- | --- | --- |
| `PREAMBLE = 0x66` | constant | sync byte |
| `extract_data_bits(buf)` | function | strip the 6-of-8 wire encoding |
| `iter_messages(stream)` | generator | decoded messages |

### `rinexpy.ntrip`

NTRIP v1 / v2 client (sync and asyncio).

| Symbol | Type | Notes |
| --- | --- | --- |
| `fetch_sourcetable(host, *, port=2101, timeout=30.0)` | function | sync sourcetable |
| `stream(host, mountpoint, *, ...)` | generator | sync byte stream |
| `afetch_sourcetable(host, ...)` | coroutine | async sourcetable |
| `astream(host, mountpoint, ...)` | async generator | async byte stream |

## Receiver binary decoders

### `rinexpy.nmea`

NMEA-0183 ASCII sentence decoder.

| Symbol | Type | Notes |
| --- | --- | --- |
| `checksum(sentence)` | function | XOR checksum |
| `parse_sentence(line, *, check_crc=True)` | function | one line |
| `iter_lines(stream, *, check_crc=True)` | generator | many lines |

### `rinexpy.ubx`

u-blox UBX binary decoder.

| Symbol | Type | Notes |
| --- | --- | --- |
| `SYNC1 = 0xB5`, `SYNC2 = 0x62` | constants | sync bytes |
| `fletcher_checksum(data)` | function | 8-bit Fletcher |
| `iter_messages(stream, *, check_crc=True)` | generator | decoded messages |
| `decode_message(msg_class, msg_id, payload)` | function | dispatch |

### `rinexpy.sbf`

Septentrio SBF decoder.

| Symbol | Type | Notes |
| --- | --- | --- |
| `SYNC = b"$@"` | constant | sync bytes |
| `crc_ccitt(data, *, init=0)` | function | CRC-CCITT |
| `iter_blocks(stream, *, check_crc=True)` | generator | decoded blocks |

### `rinexpy.novatel`

NovAtel OEM binary decoder.

| Symbol | Type | Notes |
| --- | --- | --- |
| `SYNC = b"\xaa\x44\x12"` | constant | sync bytes |
| `crc32(data)` | function | IEEE-802.3 CRC32 |
| `iter_messages(stream, *, check_crc=True)` | generator | decoded messages |

### `rinexpy.binex`

UNAVCO BINEX framing decoder.

| Symbol | Type | Notes |
| --- | --- | --- |
| `SYNC = 0xC2` | constant | sync byte |
| `read_ubnxi(stream)` / `encode_ubnxi(value)` | functions | ubnxi codec |
| `xor_checksum(data)` / `crc16_ccitt(data)` | functions | small / medium body checksums |
| `iter_records(stream, *, check_crc=True)` | generator | decoded records |

### `rinexpy.gw10`

Furuno GW-10 framed SBAS L1 extractor.

| Symbol | Type | Notes |
| --- | --- | --- |
| `SYNC = 0x8B` | constant | sync byte |
| `iter_frames(stream, *, check_checksum=True)` | generator | all frames |
| `decode_sbas(payload)` | function | ID 0x03 only |
| `iter_sbas_messages(stream, *, check_checksum=True)` | generator | SBAS L1 frames only |

## Raw nav subframe decoders

### `rinexpy.gps_lnav`, `rinexpy.gps_cnav`, `rinexpy.gps_cnav2`

GPS LNAV (L1 C/A), CNAV (L2C / L5), CNAV-2 (L1C).

### `rinexpy.galileo_nav`

Galileo F-NAV (E5a), I-NAV (E1B / E5b).

### `rinexpy.glonass`

GLONASS L1OF / L2OF strings 1-3 plus per-channel frequency helpers.

| Constant | Value |
| --- | --- |
| `C_M_PER_S` | 299_792_458.0 |
| `F_L1OF_BASE_HZ` | 1_602_000_000.0 |
| `F_L2OF_BASE_HZ` | 1_246_000_000.0 |
| `F_L1OF_STEP_HZ` | 562_500.0 |
| `F_L2OF_STEP_HZ` | 437_500.0 |
| `CHANNEL_MIN` | -7 |
| `CHANNEL_MAX` | 6 |
| `KM_TO_M` | 1000.0 |

### `rinexpy.navic`

NavIC / IRNSS subframes 1-2 (decoded) and 3-4 (raw).

### `rinexpy.beidou`

BeiDou D1 / D2 subframe / page 1 decoder.

| Constant | Value |
| --- | --- |
| `PREAMBLE` | 0x712 (11 bits) |

### `rinexpy.sbas`

SBAS L1 message decoder.

| Constant | Value |
| --- | --- |
| `PREAMBLES` | (0x53, 0x9A, 0xC6) |

### `rinexpy.has`

Galileo HAS message decoder.

| Symbol | Type | Notes |
| --- | --- | --- |
| `HAS_GNSS_NAMES` | dict | system index → name |
| `HAS_VALIDITY_S` | dict | validity index → seconds |
| `decode_has_header`, `decode_has_mask`, `decode_has_orbit`, `decode_has_clock_full`, `decode_has_clock_subset`, `decode_has_code_bias`, `decode_has_phase_bias`, `decode_has_ura`, `decode_has_message` | functions | per-MT |

## Math and helpers

### `rinexpy.geodesy`

| Symbol | Notes |
| --- | --- |
| `ecef_to_lla`, `lla_to_ecef` | WGS-84 |
| `ecef_to_enu`, `enu_to_ecef` | local frame |
| `ecef_to_eci`, `eci_to_ecef` | ECEF / ECI rotation |
| `azimuth_elevation(rx, sv)` | rx → sv angles |
| `dop(sv_ecef, rx_ecef)` | GDOP / PDOP / HDOP / VDOP / TDOP |
| `klobuchar(alpha, beta, rx_lla, az, el, gps_sec)` | broadcast L1 iono |
| `standard_atmosphere(alt)` | ICAO standard |
| `saastamoinen(el_deg, alt, *, P, T, e)` | tropo slant delay |
| `niell_mapping(el, lat, alt, doy)` | NMF |
| `vmf1(a_h, a_w, el, lat, alt, doy)` | Vienna mapping |
| `phase_wind_up_correction(...)` | Wu 1993 |

### `rinexpy.gpstime`

| Symbol | Notes |
| --- | --- |
| `GPS_EPOCH` | datetime(1980, 1, 6) |
| `SECONDS_PER_WEEK` | 604_800 |
| `LEAP_SECONDS` | table |
| `leap_seconds_at(t)` | TAI - UTC in seconds |
| `datetime_to_gps(t)` | `(week, sow)` |
| `gps_to_datetime(week, sow)` | inverse |
| `datetime_to_gps_seconds(t)` / `gps_seconds_to_datetime(s)` | continuous form |
| `gps_week_rollover(week_mod_1024, reference)` | 10-bit week resolver |

### `rinexpy.interp`

Lagrange interpolation of SP3 positions.

| Symbol | Notes |
| --- | --- |
| `interpolate_sp3(sp3, times, *, order=10)` | the main interpolator |

### `rinexpy.keplerian`

Keplerian to ECEF conversion.

| Symbol | Notes |
| --- | --- |
| `keplerian2ecef(sv)` | the main converter |

### `rinexpy.multifreq`

Wide-lane / narrow-lane / Melbourne-Wuebbena / TCAR.

| Constant | Value |
| --- | --- |
| `F1`, `F2`, `F5` | GPS L1 / L2 / L5 frequencies in Hz |
| `LAMBDA_L1`, `LAMBDA_L2`, `LAMBDA_L5` | wavelengths in metres |
| `LAMBDA_WL`, `LAMBDA_NL` | wide-lane / narrow-lane |
| `LAMBDA_EWL_15`, `LAMBDA_EWL_25` | extra-wide-lane combinations |

| Function | Notes |
| --- | --- |
| `wide_lane_phase(phi1, phi2)` | |
| `narrow_lane_phase(phi1, phi2)` | |
| `melbourne_wubbena(phi1, phi2, p1_m, p2_m)` | |
| `resolve_wide_lane(...)` | round-and-gate WL fix |
| `split_wl_into_l1_l2(n_wl, n_nl)` | recover (N1, N2) |
| `fix_iono_free_ambiguity(...)` | iono-free integer fix |
| `lambda_dual_freq(...)` | dual-freq L1+L2 fix |
| `extra_wide_lane_phase(phi2, phi5)` | EWL combination |
| `melbourne_wubbena_ewl(phi2, phi5, p2_m, p5_m)` | EWL MW |
| `resolve_extra_wide_lane(mw_ewl)` | EWL integer fix |
| `tcar_resolve(phi1, phi2, phi5, p1, p2, p5)` | EWL → WL → L1 chain |

### `rinexpy.lambda_ar`

LAMBDA integer ambiguity resolution.

| Symbol | Notes |
| --- | --- |
| `ldl(Q)` | LDL decomposition |
| `bootstrap(L, a_float)` | quick integer estimate |
| `integer_least_squares(a_float, Q, *, n_cands=2, max_nodes=100_000, max_seconds=None)` | full ILS |
| `lambda_resolve(a_float, Q, *, ratio_threshold=3.0)` | high-level wrapper |
| `ILSAborted` | exception carrying partial candidates |

## Positioning

### `rinexpy.positioning`

| Symbol | Notes |
| --- | --- |
| `spp_solve(...)` | iterative LSQ SPP |
| `spp_solve_raim(...)` | with chi-squared fault detection |
| `tgd_from_nav(nav, epoch, *, field="TGD")` | broadcast TGD extractor |
| `apply_tgd_correction(...)` | apply TGD to pseudoranges |
| `apply_klobuchar_correction(...)` | apply Klobuchar to pseudoranges |
| `apply_light_time_and_earth_rotation(...)` | fixed-point SV emission time |
| `iono_free_pseudorange(p1, p2)` | iono-free P |
| `iono_free_phase(l1, l2)` | iono-free L |
| `ppp_solve_code_only(...)` | code-only PPP |
| `ppp_solve_static_batch(...)` | static carrier-phase PPP batch |
| `ppp_solve_static_batch_with_ar(...)` | + integer ambiguity fixing |

### `rinexpy.rtk`

| Symbol | Notes |
| --- | --- |
| `double_difference_solve(...)` | float DD baseline |
| `rtk_fix(...)` | float + LAMBDA + ratio test |
| `SequentialRTK(...)` | multi-epoch with carry-over |

### `rinexpy.ppp`, `rinexpy.ppp_rtk`

| Symbol | Notes |
| --- | --- |
| `ppp_solve(obs, sp3, clk, ...)` | high-level PPP driver |
| `PPPRTKFusion` | PPP + RTK blended estimator |

### `rinexpy.kalman`, `kalman_ztd`, `kalman_multignss`

| Symbol | Notes |
| --- | --- |
| `StaticPPPFilter` / `GNSSFilter` | basic EKF (state: position + clock + ambiguities) |
| `StaticPPPFilterZTD` | + ZWD state |
| `StaticPPPFilterMultiGNSS` | + per-constellation ISBs |

### `rinexpy.snapshot`

| Symbol | Notes |
| --- | --- |
| `snapshot_positioning(code_phase_chips, sv_positions_ecef, initial_position_ecef, ...)` | van Diggelen A-GPS |

### `rinexpy.vrs`, `rinexpy.network_dd`

| Symbol | Notes |
| --- | --- |
| `synthesize_vrs(bases, rover_approx_pos, *, wavelength)` | VRS composer |
| `network_dd_solve(...)` | float network DD |
| `network_dd_solve_ar(...)` | + LAMBDA per baseline |

### `rinexpy.gnssr`

| Symbol | Notes |
| --- | --- |
| `detrend_snr(snr_db, elevation_rad, *, order=4)` | polynomial detrend |
| `snr_to_sea_height(...)` | Larson 2008 |

### `rinexpy.imu`, `rinexpy.imu_tight`

| Symbol | Notes |
| --- | --- |
| `LooseINS15` | loosely-coupled 15-state EKF |
| `TightINS16` | tightly-coupled 16-state EKF |
| `quat_normalize`, `quat_to_matrix`, `quat_mul`, `quat_from_axis_angle` | quaternion helpers |
| `GRAVITY_M_PER_S2 = 9.80665` | constant |

### `rinexpy.time_transfer`

| Symbol | Notes |
| --- | --- |
| `p3_combination(p1, p2)` | iono-free P |
| `common_view_difference(...)` | per-SV CV difference |
| `estimate_clock_difference_s(...)` | aggregated estimate |

### `rinexpy.ssr`

| Symbol | Notes |
| --- | --- |
| `SSRCorrections(messages=None)` | the composer |

### `rinexpy.realtime`

| Symbol | Notes |
| --- | --- |
| `RealtimeOrbitClock` | dataclass cache |
| `ntrip_message_loop(...)` | NTRIP byte stream → decoded messages |

### `rinexpy.cors`

| Symbol | Notes |
| --- | --- |
| `igs_daily_url(station, when, source="cddis")` | URL builder |
| `fetch_igs_daily(station, when, source="cddis", *, timeout=60.0, overwrite=False)` | downloader |

## Tides and EOP

### `rinexpy.tides`

| Symbol | Notes |
| --- | --- |
| `GM_SUN`, `GM_MOON`, `GM_EARTH`, `R_EARTH`, `H2_LOVE`, `L2_SHIDA` | constants |
| `sun_position_ecef(epoch)` / `moon_position_ecef(epoch)` | low-precision ephemeris |
| `solid_earth_tide_displacement(...)` | IERS 2010 step 1 |
| `step2_diurnal_displacement(...)` / `step2_long_period_displacement(...)` / `step2_displacement(...)` | IERS 2010 step 2 |
| `pole_tide_displacement(station_ecef, eop, epoch)` | IERS 2010 §7.1.4 |
| `ocean_pole_tide_displacement(station_ecef, eop, epoch)` | IERS 2010 §7.1.5 |

### `rinexpy.otl`

| Symbol | Notes |
| --- | --- |
| `read_blq(path)` | Scherneck BLQ reader |
| `ocean_tide_loading_displacement(blq_entry, epoch)` | local east, north, up |
| `ocean_tide_loading_ecef(blq_entry, station_ecef, epoch)` | ECEF |

## QC

### `rinexpy.qc`

| Symbol | Notes |
| --- | --- |
| `detect_slips_phase_only(phi, *, threshold_cycles=1.0)` | |
| `detect_slips_geometry_free(phi1, phi2, *, ...)` | |
| `detect_slips_mw(phi1, phi2, p1, p2, *, ...)` | |
| `detect_slips(obs, *, ...)` | dispatcher |
| `mp1(p1, l1, l2)`, `mp2(p2, l1, l2)` | TEQC-style multipath |
| `multipath_rms(mp_m, slips=None)` | per-arc RMS |
| `hatch_filter(pr_m, phase_m, *, window=100, slips=None)` | carrier-smoothed code |
| `repair_slips(phase_cycles, slips, *, fit_window=5)` | |
| `repair_slips_dual(phi1, phi2, p1, p2, slips, ...)` | |

### `rinexpy.spoofing`

| Symbol | Notes |
| --- | --- |
| `check_snr_uniformity(snr_db, elevation_deg, *, sigma_threshold=1.5)` | |
| `check_position_jumps(positions_ecef, times_s, *, max_speed_m_per_s=300.0)` | |
| `check_clock_drift(clock_bias_s, times_s, *, max_drift_rate=1e-6, max_jump_s=1e-5)` | |
| `check_agc(agc_db, *, max_jump_db=6.0)` | |

## DCB

### `rinexpy.dcb`

| Symbol | Notes |
| --- | --- |
| `C_M_PER_S` | 299_792_458.0 |
| `read_bsx(path)` | SINEX-BIAS reader |
| `read_code_dcb(path, *, year=None, month=None)` | legacy CODE monthly reader |
| `get_bias(records, *, prn="", station="", obs1, obs2="", epoch=None)` | lookup |
| `correct_pseudorange(pr_m, *, prn, obs_code, records, epoch=None, station="")` | apply OSB |

### `rinexpy.dcb_download`

| Symbol | Notes |
| --- | --- |
| `BKG_MGEX_BASE`, `CDDIS_BIAS_BASE`, `AIUB_CODE_BASE` | URL constants |
| `MGEX_FIRST_YEAR = 2017` | dispatch boundary |
| `download_dcb(date, *, product="CAS", ...)` | fetch + cache |
| `download_legacy_code_dcb(date, *, product="P1P2", ...)` | pre-2017 fetch |
| `load_daily_dcb(date, *, ...)` | fetch + parse |
| `load_monthly_code_dcb(date, *, product="P1P2", ...)` | pre-2017 fetch + parse |
| `auto_load_dcb(date, *, ...)` | date-dispatched |

## Tooling

### `rinexpy.tools`

| Symbol | Notes |
| --- | --- |
| `validate_file(fn)` | QC report |
| `concat_files(files, *, dim="time")` | along-time concat |
| `diff_datasets(a, b, *, rtol=1e-6, atol=1e-9)` | first-divergence finder |

### `rinexpy.api`

| Symbol | Notes |
| --- | --- |
| `load`, `rinexnav`, `rinexobs`, `gettime`, `batch_convert` | top-level dispatch |

### `rinexpy.headers`

| Symbol | Notes |
| --- | --- |
| `rinexinfo`, `rinexheader`, `obsheader2`, `obsheader3`, `navheader2`, `navheader3` | header readers |

### `rinexpy.netcdf`, `rinexpy.zarr_io`, `rinexpy.lazy`, `rinexpy.asyncio`

| Symbol | Notes |
| --- | --- |
| `netcdf.ENC` | default compression encoding |
| `netcdf.write_dataset(ds, path, *, group, overwrite=False)` | low-level writer |
| `zarr_io.to_zarr(ds, store, *, mode="w", consolidated=True)` | Zarr writer |
| `lazy.load_lazy(files, *, chunk_size=None)` | dask-backed multi-file |
| `asyncio.aload(fn, **kwargs)` | thread-pool wrapper |
| `asyncio.aload_many(files, **kwargs)` | concurrent multi-file |

### `rinexpy.plots`

| Symbol | Notes |
| --- | --- |
| `timeseries(data)` | dispatcher |
| `obstimeseries(obs)` | L1 phase per SV |
| `navtimeseries(nav)` | satellite ground tracks |
| `receiver_locations(locs)` | scatter map |
| `skyplot(sv_az_el, *, elevation_mask_deg=5.0, title="")` | polar plot |

### `rinexpy.plugins`

| Symbol | Notes |
| --- | --- |
| `discover_plugins()` | enumerate registered readers |
| `load_with_plugins(path, ...)` | built-in load + plugin fallback |

### `rinexpy.cli`

| Symbol | Notes |
| --- | --- |
| `build_parser()` | construct the argparse tree |
| `main(argv=None)` | the `rinexpy` console script entry |

### `rinexpy.writer`

| Symbol | Notes |
| --- | --- |
| `to_rinex_obs(obs, fn, *, version=3)` | round-trip back to RINEX |

### `rinexpy.streaming`

| Symbol | Notes |
| --- | --- |
| `iter_obs3_epochs(fn, *, ...)` | per-epoch generator |

## Related pages

- [Top-level API](top-level.md): the main functions.
- [Glossary](glossary.md): GNSS-specific terminology.
- [Architecture](../internals/architecture.md): how the modules layer.
