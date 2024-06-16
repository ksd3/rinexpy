# Roadmap

Wishlist of features that would push rinexpy from a fast reader plus
decimeter-grade SPP/RTK toward a real cm/mm GNSS toolkit. Rough
order is "what's most likely to land first" within each section.
None of this is committed; it's a working document.

If you want to pick something up, open an Issue first so we can
sync on scope.

## Positioning

- **PPP (precise point positioning).** Single-receiver cm-level
  using IGS final SP3 + CLK, ionosphere-free L1/L2 combination,
  Saastamoinen/VMF1 troposphere, dual-freq LAMBDA. Acceptance:
  `rinexpy.ppp.ppp_solve(obs, sp3, clk, ...)` recovers a fixed
  station to within 2 cm on a 24h dataset.
- **EKF / Kalman framework.** State carries (x, y, z, dt, ZWD,
  ambiguities). Replaces per-epoch SPP and per-epoch rtk_fix as the
  default. Acceptance: `rinexpy.kalman.GNSSFilter` matches the
  existing SPP/RTK accuracy on a static fixture and beats it on a
  kinematic one.
- **Sequential RTK with ambiguity carry-over.** Re-use the integer
  fix across epochs while the lock counter holds; partial
  ambiguity resolution when the full fix fails the ratio test.
  Acceptance: a moving-baseline replay produces a continuous fixed
  solution without rejecting > 5% of epochs.
- **RAIM.** Chi-squared residual test on the SPP solution; isolate
  the worst SV and re-solve. Acceptance:
  `spp_solve(..., raim=True)` flags an injected 50 m bias on one
  SV.

## Corrections and augmentation

- **SSR decoding** (RTCM3 1057-1068 GPS, 1240-1264 Galileo, the
  IGS-SSR 4076 sub-types). Real-time PPP corrections.
  Acceptance: `rtcm3.iter_messages` returns decoded
  orbit/clock/bias dicts; PPP can consume them in place of CLK.
- **SBAS L1.** WAAS, EGNOS, MSAS, GAGAN message decoder. Same
  shape as the existing RTCM2 module.
- **DCB / TGD.** Broadcast TGD from NAV plus the IGS / CAS DCB
  products. Acceptance: SPP applies the right per-SV bias and
  shows the ~ns absolute-time correction.
- **Solid-earth, ocean, and pole tides; phase wind-up.** IERS
  Conventions 2010 models. Required for sub-cm PPP. Acceptance:
  tide displacement matches the IERS test vectors within 0.1 mm.
- **EOP files.** IERS Bulletin A and C04 readers. Acceptance:
  `ecef_to_eci(epoch, eop)` matches an Astropy round-trip within
  a milliarcsecond.

## More decoders

- **GPS LNAV / CNAV / CNAV2 raw subframes.** Same shape as the
  existing BeiDou D1/D2 module. Acceptance: decode the clock and
  ephemeris from a captured `RXM-SFRBX` stream.
- **Galileo F/I-NAV, GLONASS strings, NavIC subframes.** Cheap to
  add once GPS subframes exist.
- **RTCM3 holes.** 1029 (text), 1230 (GLONASS code-phase biases),
  MSM1/2/3/5/6.
- **Wider UBX / SBF / NovAtel coverage.** Currently a handful of
  records each; the protocols have many more useful ones (e.g.
  UBX NAV-CLOCK, NAV-DOP, SBF GALNav, GLONav, NovAtel TRACKSTAT).
  Driven by user requests rather than a list.

## Quality and analysis

- **Cycle slip detection.** Melbourne-Wübbena, geometry-free
  combination, time-differenced phase. Prerequisite for usable
  long-baseline RTK and PPP. Acceptance:
  `rinexpy.qc.detect_slips(obs)` flags an injected 1-cycle slip on
  a synthetic dataset.
- **Multipath metrics.** MP1 / MP2 combinations, code-minus-
  carrier RMS, SNR maps. TEQC-style QC report.
- **Carrier-smoothed code (Hatch filter).** Adjustable smoothing
  window per SV with reset on detected cycle slips.
- **Spoofing / jamming heuristics.** Power consistency, position
  jumps, clock-drift sanity. Out-of-the-box checks; users plug in
  their own thresholds.

## Ergonomics and plumbing

- **AsyncIO NTRIP client.** `async for chunk in ntrip.astream(...)`
  alongside the existing sync version.
- **CLI growth.** `rinexpy spp`, `rinexpy rtk`, `rinexpy ppp`,
  `rinexpy splice`, `rinexpy decimate`. The last two cover the
  TEQC editing surface most users miss.
- **SP3 and NAV writers from fitted solutions.** Round-trip a
  fitted orbit back to SP3, a fitted ephemeris back to NAV.
- **Plugin entry points.** `rinexpy.readers` group so third-party
  packages can register new format readers without forking.
- **Time-transfer mode.** GPS common-view, P3 combination.

## Stretch

- **GNSS reflectometry (GNSS-R).** SNR-based altimetry / soil
  moisture / sea-state retrieval. Pure analysis on top of the
  obs readers.
- **Snapshot positioning.** Short-data positioning for IoT and
  asset tracking. Different solver, different fixture set.
- **Network RTK / VRS composition.** Virtual reference stations
  from a network of base receivers.
- **Antenna calibration tool.** Generate ANTEX PCV entries from a
  calibration session.

## Definitely not

- A C++ rewrite of the whole stack. The pure-Python path is already
  I/O bound; the optional `[native]` extra handles the OBS3 hot
  loop and that's enough.
- A Conda recipe. uv is the supported install path.
- A GUI. Out of scope; downstream tools can sit on the API.
