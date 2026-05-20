# Glossary

GNSS work is heavy on acronyms. This page is a short reference for the
terms used elsewhere in the documentation. Cross-references point to
the page where the term shows up most prominently.

## Reference frames

**ECEF.** Earth-Centred Earth-Fixed. The frame the satellite positions
in RINEX NAV, SP3, and ANTEX live in. The x-axis points to the prime
meridian at the equator, the z-axis points to the geographic north pole,
the y-axis completes the right-handed frame. Units are metres.

**ECI.** Earth-Centred Inertial. The frame that does not rotate with the
Earth. Satellite orbital dynamics are naturally expressed in ECI. The
ECEF/ECI rotation uses the Earth-orientation parameters (EOP).

**ITRF.** International Terrestrial Reference Frame. The convention that
fixes the location of the ECEF origin, axes, and scale. The current
version is ITRF2020.

**WGS-84.** The reference ellipsoid that geodetic coordinates (lat, lon,
altitude) are defined against. Semi-major axis 6378137 m, flattening
1/298.257223563.

**Local ENU.** East-North-Up. A local tangent-plane frame at a station.
Used for baseline displays and antenna axes.

## Time scales

**UTC.** Coordinated Universal Time. The civilian time scale with leap
seconds.

**UT1.** Earth-rotation time. The difference UT1 - UTC is one of the EOP
parameters; it stays within ±0.9 s of UTC.

**TAI.** International Atomic Time. The continuous time scale with no
leap seconds. TAI - UTC was 37 seconds as of 2017.

**GPS time.** GPS uses an atomic time scale that started at 1980-01-06
00:00:00 UTC and does not insert leap seconds. GPS time - UTC = TAI -
UTC - 19 = 18 s as of 2017 onwards.

**Galileo System Time (GST).** Same definition as GPS time, with a small
offset published in the RINEX 4 STO record.

**GLONASS time.** UTC(SU) + 3 hours. The leap second is applied,
unlike GPS.

**BeiDou time.** Started at 2006-01-01 00:00:00 UTC. No leap seconds.

**GPS week, seconds of week.** The pair `(week, sow)` is the ICD-200
format for an instant in GPS time. The week resets every 604 800
seconds; the broadcast week field is only 10 bits wide and rolls over
every 1024 weeks.

See [`rinexpy.gpstime`](modules.md#rinexpygpstime) for the conversion
helpers.

## Measurements

**Pseudorange.** The receiver's measurement of the time between signal
transmission and reception, multiplied by `c`. Carries the geometric
range plus the satellite and receiver clock biases plus atmospheric
delays plus hardware biases plus noise.

**Carrier phase.** The receiver's measurement of the phase of the
carrier wave, expressed in cycles. Equivalent in scale to the
pseudorange (with a per-satellite integer cycle ambiguity), but with
1 mm noise instead of the pseudorange's 1 m noise.

**Doppler.** The frequency shift of the received carrier wave relative
to the nominal carrier. Proportional to the line-of-sight velocity.

**SNR / C/N0.** Signal-to-noise ratio of the received signal, in dB-Hz.
A typical clear-sky GNSS observation is 40-50 dB-Hz.

**LLI.** Loss-of-lock indicator. A per-observation flag in the RINEX
file that the receiver had a tracking discontinuity.

**SSI.** Signal-strength indicator. A coarse 1-9 scale that RINEX uses
for some legacy signal-strength reporting.

## Frequencies and signals

**L1.** GPS C/A signal at 1575.42 MHz.

**L2.** GPS P / L2C signal at 1227.60 MHz.

**L5.** GPS L5 signal at 1176.45 MHz.

**L1C.** GPS L1C civil signal at 1575.42 MHz (a separate code from L1
C/A, sharing the carrier).

**E1.** Galileo E1 signal at 1575.42 MHz (same carrier as GPS L1).

**E5a, E5b.** Galileo E5a / E5b at 1176.45 MHz / 1207.14 MHz.

**E6.** Galileo E6 at 1278.75 MHz; carries the High Accuracy Service
(HAS).

**L1OF / L2OF.** GLONASS open civil signals at 1602.0 + k*0.5625 MHz
(L1OF) and 1246.0 + k*0.4375 MHz (L2OF). `k` is the per-satellite
channel number, -7 to +6.

**B1I, B2I, B3I.** BeiDou B1 / B2 / B3 signals.

**B1C, B2a.** BeiDou modernised signals.

## Combinations

**Iono-free P3 / IF.** The combination that cancels the first-order
ionospheric delay. `(f1² * P1 - f2² * P2) / (f1² - f2²)`. See
[Atmospheric models](../corrections/atmosphere.md).

**Geometry-free L1-L2.** The combination that cancels geometry and
keeps the ionospheric delay. Used for slip detection.

**Wide-lane (WL).** `L1 - L2` in cycles. Wavelength is 86 cm.
Used for ambiguity resolution.

**Narrow-lane (NL).** `L1 + L2` in cycles. Wavelength is 10.7 cm.

**Melbourne-Wuebbena (MW).** Combination that cancels both geometry and
ionosphere, leaving the wide-lane integer. Standard for WL fixing.

**Extra-wide-lane (EWL).** `L2 - L5` in cycles. Wavelength is about
5.86 m. Used for TCAR.

## File formats

**RINEX.** Receiver Independent Exchange Format. The plain-text file
format for GNSS observations and broadcast navigation messages.
Versions 2, 3, and 4 are in use.

**CRINEX.** Hatanaka-compressed RINEX. About 2x smaller than gzip on
RINEX 2 files.

**SP3.** Standard Product 3. The IGS-published precise satellite orbit
format. Versions a, c, d are in use.

**CLK.** RINEX clock product. Per-satellite and per-station clock biases
at 30 s or 5 min sampling.

**IONEX.** Ionosphere Map Exchange Format. Global TEC maps.

**ANTEX.** Antenna Exchange Format. Per-antenna phase-centre offsets
and variations.

**EOP C04.** IERS-published Earth-orientation parameters (polar motion,
UT1-UTC, length-of-day, celestial-pole offsets).

**SINEX-BIAS.** The modern format for satellite and receiver
differential and observable-specific signal biases.

**BLQ.** Scherneck ocean-tide-loading parameters.

**RTCM.** Radio Technical Commission for Maritime Services. The wire
format for real-time GNSS corrections.

**MSM.** Multiple Signal Message. The RTCM 3 family for raw observations
across all constellations.

**SSR.** State-Space Representation. The RTCM 3 family for precise
satellite orbit and clock corrections.

**HAS.** Galileo High Accuracy Service. SSR-style corrections on E6-B.

**NTRIP.** Networked Transport of RTCM via Internet Protocol. The
HTTP-styled streaming protocol for RTCM 3.

**NMEA-0183.** ASCII protocol for marine instruments, used by many
consumer GNSS receivers.

**UBX.** u-blox binary protocol.

**SBF.** Septentrio Binary Format.

**BINEX.** UNAVCO Binary Exchange archive format.

## Positioning

**SPP.** Single-Point Positioning. The simplest fix: pseudorange-only,
broadcast NAV.

**RTK.** Real-Time Kinematic. Carrier-phase positioning against a nearby
base.

**PPP.** Precise Point Positioning. Carrier-phase positioning with IGS
precise orbits and clocks.

**PPP-RTK.** Fusion of PPP and RTK with inverse-variance blending.

**Network RTK / VRS.** RTK with a virtual reference station synthesised
from a network of physical bases.

**A-GPS / Snapshot.** Code-phase-only short-data fix with a coarse
position prior.

**GNSS-R.** GNSS Reflectometry. Extracting environmental parameters
from reflected signals.

**LAMBDA.** Least-squares AMBiguity Decorrelation Adjustment. The
algorithm for fixing carrier-phase ambiguities to integers.

**Ratio test.** The acceptance criterion for the LAMBDA fix: the
second-best squared residual divided by the best squared residual.

**Float ambiguity.** The continuous-valued estimate of the cycle
ambiguity, before integer fixing.

**Integer fix.** The integer-rounded ambiguity that passes the LAMBDA
ratio test.

**RAIM.** Receiver Autonomous Integrity Monitoring. The chi-squared
test that flags an outlier satellite in an SPP fix.

**EKF.** Extended Kalman Filter. The sequential estimator used by all
of `rinexpy`'s positioning filters.

## Atmosphere

**TEC.** Total Electron Content. The ionospheric observable, in
TECU = 10^16 electrons / m².

**ZHD, ZWD, ZTD.** Zenith Hydrostatic Delay, Zenith Wet Delay, Zenith
Total Delay. The decomposition of the tropospheric delay at zenith.

**STD.** Slant Total Delay. The tropospheric delay along the
line-of-sight to a satellite.

**Mapping function.** A function that maps the zenith delay to a slant
delay as a function of elevation. NMF and VMF1 are the common ones.

**Klobuchar.** The broadcast ionospheric model on GPS L1. 8 coefficients
in the GPS NAV header.

**Saastamoinen.** The closed-form tropospheric model.

**Niell (NMF).** The 1996 Niell mapping function.

**VMF1.** Vienna Mapping Function 1.

**GPT2w.** Global Pressure and Temperature 2 + water vapour. Empirical
surface met grid with built-in mapping coefficients.

**DCB.** Differential Code Bias. The per-satellite hardware bias
between two observation codes.

**OSB.** Observable-Specific signal Bias. The per-satellite absolute
bias for one observation code.

**TGD.** Timing Group Delay. The broadcast group delay in the NAV file.

## Tides

**Solid Earth tide.** Crustal deformation under Sun + Moon gravity.

**Pole tide.** Crustal deformation from polar motion.

**Ocean pole tide.** Crustal deformation from the ocean's response to
polar motion.

**Ocean tide loading (OTL).** Crustal deformation from the sea-surface
mass redistribution at tidal frequencies.

**Love numbers.** The dimensionless coefficients (`h_2`, `l_2`) that
parameterise the elastic Earth's response to a tidal forcing.

**Doodson arguments.** The astronomical arguments that drive the tidal
frequencies.

## Receivers and antennas

**PCO.** Phase Centre Offset. The fixed offset of the antenna's
electrical phase centre from its mechanical reference point.

**PCV.** Phase Centre Variation. The elevation- and azimuth-dependent
correction to the PCO.

**Antenna calibration.** The process of measuring the PCO and PCV for a
given antenna model.

**Multipath.** Signal reflections off nearby surfaces (buildings,
ground, vegetation) that interfere with the direct line-of-sight signal.

**Cycle slip.** A sudden jump in the integer carrier-phase ambiguity
because the receiver lost tracking lock for a moment.

**RAW vs PVT.** The receiver's RAW output is the per-satellite
observation (pseudorange, phase, Doppler, SNR). The PVT output is the
solved position, velocity, and time.

## Sources

**IGS.** International GNSS Service. Publishes the precise satellite
orbit (SP3) and clock (CLK) products plus DCBs, ANTEX, IONEX, and
hosting for the global station network.

**MGEX.** Multi-GNSS Experiment. The IGS branch that covers Galileo,
BeiDou, QZSS, and NavIC alongside GPS and GLONASS.

**IERS.** International Earth Rotation and Reference Systems Service.
Publishes the EOP series and the Conventions document.

**BKG.** Bundesamt für Kartographie und Geodäsie. Runs the public BKG
mirror of MGEX SINEX-BIAS DCB files.

**CDDIS.** NASA's Crustal Dynamics Data Information System. Mirror of
the IGS products; requires NASA Earthdata Login.

**AIUB.** Astronomical Institute of the University of Bern. Publishes
the legacy monthly CODE DCB files.

## Related pages

- [Top-level API](top-level.md)
- [Module index](modules.md)
- [Architecture](../internals/architecture.md)
