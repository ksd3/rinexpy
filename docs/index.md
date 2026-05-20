# `rinexpy`

`rinexpy` is a Python toolkit for working with GNSS data. It started as a
faster fork of [georinex](https://github.com/geospace-code/georinex) and grew
from a RINEX reader into a complete pipeline that covers everything from the
raw bytes of a receiver log to a centimetre-accurate position solution.

The library is built around a small, layered core. The core functions are file readers that turn raw bytes into NumPy buffers and `xarray.Dataset` objects.
The math layers: geodesy, GPS time, Keplerian orbit
propagation, Lagrange interpolation of SP3 ephemerides manipulate the results of these file readers.
On top of those come
the positioning engines: single-point positioning with optional RAIM, a real
double-difference RTK solver with LAMBDA integer fixing, a sequential RTK
loop that carries ambiguities across epochs, a static-or-kinematic precise
point positioning driver, a tightly-coupled GNSS/IMU filter, plus a snapshot
solver for assisted GPS. Every layer is documented in this site.

## Why should you use `rinexpy`?

1. You have a directory of RINEX 2 or RINEX 3 observation files, possibly
gzip-compressed or in Hatanaka CRINEX form, and you want them in memory as
labelled tables instead of fixed-width text. 
2. You have an SP3 file from the
IGS and need satellite positions at a 1 Hz cadence rather than the published
15 minute interval. 
3. You want to listen to an NTRIP caster, decode the live
RTCM3 stream, and feed those corrections into a PPP filter. 
4. You want a tool
that does not require a Fortran compiler, a five-step install, or a paid
license.

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C  # one observable, one satellite, as an xarray DataArray
```

The same `load` call recognises a RINEX 2 file, a RINEX 3 file, a RINEX 4 NAV
file, an SP3 ephemeris file, or a pre-converted NetCDF, and calls the matching
reader. Filters like `tlim=`, `meas=`, `use=`, and `interval=` apply during
parsing, so records outside your window are skipped before they get decoded.

## Where to start

If you have never used `rinexpy` before, the [installation guide](installation.md)
walks through the `uv` setup and the optional extras. You can then look at
[quickstart](quickstart.md) for some straightforward workflows.

If you are looking for a particular reader, head straight to the
[file formats](formats/rinex-obs.md) section. Every supported format has its
own page with the full schema of the returned dataset and a runnable code
sample.

If you want the per-symbol reference, the [top-level API](reference/top-level.md)
documents every public function, kwarg, and return type. The
[module index](reference/modules.md) maps the source tree against what gets
re-exported.

## What is included


**Readers:** RINEX 2 / 3 / 4 observation files and navigation files. SP3-a,
SP3-c, and SP3-d satellite ephemerides. RINEX clock products. IONEX TEC
maps. ANTEX antenna phase-centre variations. RINEX MET surface met data.
IERS EOP C04 Earth-orientation series. GPT2w gridded empirical met for
hydrostatic and wet zenith delays. SINEX-BIAS for differential and observable
biases, plus the legacy AIUB monthly format. NetCDF4 and Zarr round-trips of
the parsed datasets. [File formats →](formats/rinex-obs.md)

**Streaming inputs:** RTCM 2.x DGPS frames. RTCM 3.x with full MSM 1-7
decoding, the GPS LNAV broadcast ephemeris messages 1019 / 1020, the IGS
SSR family 1057-1068 and 1240-1263 covering orbit, clock, code-bias, and
URA, plus IGS-SSR message 4076. NTRIP v1 and v2 clients with both a
synchronous generator and an `asyncio` variant. NMEA-0183 sentences. u-blox
UBX. Septentrio SBF. NovAtel OEM. UNAVCO BINEX. Furuno GW-10 framed SBAS.
[RTCM and NTRIP →](formats/rtcm.md)

**Raw navigation message decoders:** GPS LNAV subframes 1-3 on L1 C/A, GPS
CNAV messages 10 and 11 on L2C/L5, GPS CNAV-2 subframe 2 on L1C. Galileo
F-NAV pages 1 and 2 on E5a, Galileo I-NAV words 1 and 4 on E1B/E5b. GLONASS
L1OF/L2OF strings 1 to 3 with sign-magnitude decoding per the GLONASS ICD.
BeiDou D1 subframe 1 and D2 page 1. NavIC subframes 1 and 2, with raw
subframes 3 and 4 returned for downstream dispatch. SBAS L1 message types
1, 2-5, 6, 7, 9, 17, 18, 24, 25, 26 per RTCA DO-229E.
[Raw subframes →](formats/nav-subframes.md)

**Positioning:** Iterative single-point positioning with optional RAIM
fault detection and Klobuchar / DCB / TGD corrections. Double-difference
RTK with LAMBDA integer fixing, a ratio test, and partial ambiguity
resolution. A `SequentialRTK` class that carries the integer fix across
epochs and detects per-SV cycle slips. A static-or-kinematic precise
point positioning driver that combines SP3, CLK, ANTEX, GPT2w, DCB,
and carrier-phase wind-up into one call. PPP-RTK fusion. Tightly-coupled
GNSS/IMU. Snapshot positioning for assisted GPS. Network double-difference.
VRS synthesis. GNSS reflectometry. [Positioning →](positioning/spp.md)

**Atmosphere and tides:** Klobuchar broadcast ionospheric model. Niell
NMF and Vienna VMF1 mapping functions. Saastamoinen zenith hydrostatic
delay. GPT2w empirical surface met and mapping coefficients. Solid-Earth
tides per IERS Conventions 2010 step 1 and step 2. Pole tide and ocean
pole tide. Ocean tide loading via Scherneck BLQ files.
[Corrections →](corrections/atmosphere.md)

**Quality and integrity:** Three independent cycle-slip detectors
(phase-only, geometry-free, Melbourne-Wuebbena) plus repair helpers. TEQC-style
multipath combinations. Hatch filter for carrier-smoothed code. Spoofing
heuristics on SNR uniformity, position-rate, clock-rate, and AGC.
[Quality and integrity →](quality/qc.md)

**Tooling:** A `rinexpy` argparse CLI covering read / times / info /
convert / spp / rtk / ppp / splice / decimate. Parallel batch conversion to
NetCDF. Streaming iterator for files larger than RAM. Async wrappers.
Plotting helpers (matplotlib, optional). Validation and diff tools.
[Tooling →](tooling/cli.md)

## Project status

The project is on its v0.2 series. The names exported from
`rinexpy/__init__.py` will not move; new functions get added on top of
the existing ones. The [changelog](project/changelog.md) tracks every
release. Anything not in `__init__.py` or anything that starts with an
underscore is internal.

The library is local-install only. There is no PyPI release; you clone the
repository and let `uv` build the project venv. The
[installation guide](installation.md) walks through the options.

## Compatibility matrix

| What | Status | Notes |
| --- | --- | --- |
| RINEX 2 / 3 OBS | full | gzip, bz2, zip, LZW `.Z`, Hatanaka CRINEX 1 and 3 |
| RINEX 2 / 3 NAV | full | GPS, Galileo, GLONASS, BeiDou, QZSS, SBAS, NavIC |
| RINEX 4 NAV | partial | STO / EOP / ION records plus the modernized EPH set |
| SP3-a / SP3-c / SP3-d | full | |
| RINEX clock products `.clk` | full | |
| IONEX `.inx` | full | |
| ANTEX `.atx` | full | NOAZI plus the 2-D azimuth-dependent path |
| RINEX MET | full | |
| IERS EOP C04 | full | |
| RTCM 2.x | partial | types 1, 3, 9 fully decoded |
| RTCM 3.x | partial | extensive; see [RTCM →](formats/rtcm.md) for the matrix |
| NTRIP v1 / v2 | full | sync and async |
| NMEA-0183 | partial | GGA, RMC, GSA, GSV, VTG |
| u-blox UBX | partial | NAV-PVT, NAV-SAT, RXM-RAWX, RXM-SFRBX |
| Septentrio SBF | partial | PVTGeodetic, MeasEpoch, GPSNav |
| NovAtel OEM | partial | BESTPOS, BESTXYZ, RAWEPHEM |
| UNAVCO BINEX | framing-only | forward byte order |
| SBAS L1 | partial | MT 1, 2-5, 6, 7, 9, 17, 18, 24-26 |
| BeiDou D1 / D2 | partial | clock plus ionospheric model from SF 1 / page 1 |
| GPS LNAV / CNAV / CNAV-2 | partial | clock plus full ephemeris |
| Galileo F-NAV / I-NAV | partial | clock plus part of the ephemeris |
| GLONASS L1OF / L2OF | partial | strings 1 to 3 |
| NavIC | partial | subframes 1 and 2 |

## Citation

If you use `rinexpy` in academic work, please also cite the upstream georinex
project for the original reader design:
[doi:10.5281/zenodo.2580306](https://doi.org/10.5281/zenodo.2580306).

## License

MIT.
