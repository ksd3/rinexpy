# rinexpy

Modern, fast GNSS toolkit for Python.

`rinexpy` started as a substantially rewritten descendant of
[`georinex`](https://github.com/geospace-code/georinex) — same
xarray-flavored output, same public API names, but the OBS3 / NAV3
hot paths drop the O(N²) `xarray.merge`-per-epoch pattern. On the
shared test corpus this is **13–33× faster** for RINEX-3 NAV/OBS
files. From there the project grew the rest of a real GNSS stack.

## Quick start

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C
```

## Where to start

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **New to rinexpy?**

    ---

    Walk through install -> RTK fix in 12 sections.

    [:octicons-arrow-right-24: Tutorial](TUTORIAL.md)

-   :material-clipboard-text:{ .lg .middle } **Looking for a one-liner?**

    ---

    Bite-sized recipes grouped by topic.

    [:octicons-arrow-right-24: Cookbook](COOKBOOK.md)

-   :material-book-open-variant:{ .lg .middle } **Need the reference?**

    ---

    Per-symbol docs for all 43 public entries.

    [:octicons-arrow-right-24: API reference](API.md)

-   :material-graph:{ .lg .middle } **Curious how it's built?**

    ---

    Six-layer module map and end-to-end dataflow.

    [:octicons-arrow-right-24: Architecture](ARCHITECTURE.md)

</div>

## Performance

| Path | Time (23-h 15-s OBS3 file) | vs georinex |
|---|---|---|
| `georinex` baseline | ~1100 ms | 1.0× |
| `rinexpy` pure Python | 75 ms | **15×** |
| `rinexpy[jit]` numba | 38 ms | **29×** |
| `rinexpy[native]` C++ | 39 ms | **28×** |

See [Benchmarks](BENCHMARKS.md) for the full per-file breakdown and
[Optimizations](OPTIMIZATIONS.md) for what changed under the hood.

## Install

```sh
uv add rinexpy                       # pure Python
uv add 'rinexpy[all]'                # + every optional reader extra
uv add 'rinexpy[native]'             # + the optional C++ extension
```

Python 3.11+ is required.

## Coverage at a glance

- **Open ASCII**: RINEX 2/3/4 NAV/OBS, NMEA-0183, IGS clock/iono/antex
- **Open binary**: SP3, BINEX, NetCDF, Zarr
- **Streaming**: RTCM 2.x DGPS, RTCM 3.x + NTRIP
- **Vendor binary**: u-blox UBX, Septentrio SBF, NovAtel OEM
- **Raw subframes**: BeiDou D1/D2
- **Math**: Keplerian -> ECEF, SP3 Lagrange interpolation, SPP, RTK with
  full LAMBDA loop (single + dual freq), Klobuchar / Saastamoinen /
  Niell / VMF1, GPT2w empirical met grid

See [Compatibility table in README](https://github.com/ksd3/rinexpy#compatibility)
for the full matrix with footnotes.

## Project info

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Engineering log](SCRATCHPAD.md) — dated walkthrough of how this got built
- [GitHub repository](https://github.com/ksd3/rinexpy)

## License

MIT.
