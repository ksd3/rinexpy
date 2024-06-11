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

ASCII formats: RINEX 2, 3, 4 NAV/OBS, NMEA-0183, IGS clock/iono/antex.

Binary: SP3, BINEX, NetCDF, Zarr.

Streaming: RTCM 2.x DGPS, RTCM 3.x with NTRIP.

Vendor binary: u-blox UBX, Septentrio SBF, NovAtel OEM.

Raw subframes: BeiDou D1/D2 (clock + iono only).

Math: Keplerian to ECEF, SP3 Lagrange interpolation, single-point
positioning, RTK with LAMBDA integer fixing (single and dual
frequency), Klobuchar, Saastamoinen, Niell, VMF1, GPT2w empirical
met grid.

The [compatibility table in the README](https://github.com/ksd3/rinexpy#compatibility)
has the full matrix and notes about what's full vs partial.

## Project info

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [GitHub repository](https://github.com/ksd3/rinexpy)

## License

MIT.
