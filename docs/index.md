# rinexpy

A fast RINEX reader for Python that grew into a GNSS toolkit.

rinexpy is a fork of [georinex](https://github.com/geospace-code/georinex)
with the OBS3 and NAV3 readers rewritten. Upstream builds an
`xarray.Dataset` per epoch and merges them, which is O(N²) in epoch
count. rinexpy fills a NumPy buffer once and builds the Dataset at
the end. On the shared corpus that's 13-33× faster on RINEX-3
NAV/OBS files.

## Quick start

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C
```

## Where to start

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **New here?**

    ---

    Install to RTK fix in 12 sections.

    [:octicons-arrow-right-24: Tutorial](TUTORIAL.md)

-   :material-clipboard-text:{ .lg .middle } **Need a one-liner?**

    ---

    Short recipes grouped by topic.

    [:octicons-arrow-right-24: Cookbook](COOKBOOK.md)

-   :material-book-open-variant:{ .lg .middle } **Want the reference?**

    ---

    Per-symbol docs, 43 entries.

    [:octicons-arrow-right-24: API reference](API.md)

-   :material-graph:{ .lg .middle } **Curious how it's built?**

    ---

    Module map and dataflow.

    [:octicons-arrow-right-24: Architecture](ARCHITECTURE.md)

</div>

## Performance

| Path | Time (23-h 15-s OBS3 file) | vs georinex |
|---|---|---|
| `georinex` baseline | ~1100 ms | 1.0× |
| `rinexpy` pure Python | 75 ms | 15× |
| `rinexpy` + numba JIT | 38 ms | 29× |
| `rinexpy` + C++ extension | 39 ms | 28× |

See [Benchmarks](BENCHMARKS.md) for the per-file breakdown and
[Optimizations](OPTIMIZATIONS.md) for what changed.

## Install

rinexpy isn't on PyPI. Clone the repo, let `uv` set it up.

```sh
git clone https://github.com/ksd3/rinexpy
cd rinexpy
uv sync --all-extras
```

Python 3.11 or newer. The [README](https://github.com/ksd3/rinexpy#install)
has the uv install recipe for macOS, Linux, and Windows.

## Coverage

ASCII formats: RINEX 2, 3, 4 NAV/OBS (incl. RINEX 4 STO/EOP/ION),
NMEA-0183, IGS clock/iono/antex, SINEX-BIAS (DCB / OSB).

Binary: SP3, BINEX, NetCDF, Zarr.

Streaming: RTCM 2.x DGPS, RTCM 3.x with NTRIP (sync + asyncio
`astream`). RTCM3 covers the SSR family (1057-1068, 1240-1263,
IGS-SSR MT 4076), MSM 1-7, plus the usual base-station + ephemeris
types. SBAS L1 message types 1, 2-5, 6, 7, 9, 17, 18, 24-26.

Vendor binary: u-blox UBX, Septentrio SBF, NovAtel OEM.

Raw subframes: GPS LNAV / CNAV (MT 10, 11) / CNAV-2 (subframe 2),
Galileo F-NAV + I-NAV, GLONASS strings 1-3, NavIC subframes 1+2,
BeiDou D1/D2.

Positioning: SPP with optional RAIM, RTK with LAMBDA integer fix,
sequential RTK with ambiguity carry-over, snapshot SPP (code-phase
only), VRS synthesis for network RTK, and a static-or-kinematic
PPP driver that consumes SP3+CLK or RTCM-SSR and wires ANTEX PCV,
GPT2w+VMF1 troposphere, DCB, and carrier-phase wind-up.

Atmosphere + tides: Klobuchar, Saastamoinen, Niell, VMF1, GPT2w
empirical met grid; solid-earth, ocean (OTL via BLQ), pole, and
ocean-pole tides; ECEF/ECI rotation with IERS Bulletin A/C04 EOP.

Stretch: GNSS-R reflector-height retrieval (Larson 2008), antenna
PCV calibration writer, time-transfer (P3 + common-view), DCB
autodownload from IGS BKG (post-2017) or AIUB CODE (pre-2017).

The [compatibility table in the README](https://github.com/ksd3/rinexpy#compatibility)
has the full matrix and notes about what's full vs partial.

## Project info

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [GitHub repository](https://github.com/ksd3/rinexpy)

## License

MIT.
