# rinexpy

Modern, fast GNSS toolkit for Python: RINEX 2/3/4 readers, SP3 / CLK /
IONEX / ANTEX, RTCM3 + NTRIP streaming, NMEA-0183 + UBX + SBF +
NovAtel + BINEX receiver formats, single-point and RTK positioning
with LAMBDA integer fixing, full troposphere/ionosphere correction
stack.

`rinexpy` started as a substantially rewritten descendant of
[`georinex`](https://github.com/geospace-code/georinex) — same
xarray-flavored output, same public API names, but with the OBS3 /
NAV3 hot paths rewritten to drop the O(N²) `xarray.merge`-per-epoch
pattern. On the shared test corpus this is **13-33× faster** for
RINEX-3 NAV/OBS files (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
Then it grew the rest of a real GNSS stack on top.

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C
```

## Install

```sh
uv add rinexpy

# Or with optional extras (CRINEX, LZW, NetCDF, plotting, JIT, Zarr, ...):
uv add 'rinexpy[all]'
```

Python 3.11+ is required; the project itself is developed against
the latest stable CPython (3.13.x).

## Documentation

| | |
|---|---|
| [TUTORIAL.md](docs/TUTORIAL.md) | Step-by-step walk-through from install to RTK fix (12 sections). |
| [COOKBOOK.md](docs/COOKBOOK.md) | Bite-sized recipes for common one-shot tasks. |
| [API.md](docs/API.md) | Per-symbol reference of the public surface (43 entries). |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout, dependency layers, dataflow. |
| [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) | Every change vs georinex with rationale. |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Measured perf numbers vs georinex. |
| [SCRATCHPAD.md](SCRATCHPAD.md) | Dated engineering log of the build. |
| [examples/](examples/) | 8 runnable scripts covering the major workflows. |

## Compatibility

| Format                          | Status |
|---------------------------------|--------|
| RINEX 2 OBS / NAV               | full   |
| RINEX 3 / 4 OBS / NAV           | full   |
| Hatanaka CRINEX (`.crx`)        | full*  |
| GZIP / BZ2 / ZIP / LZW          | full*  |
| SP3-a / SP3-c / SP3-d           | full   |
| RINEX clock products (`.clk`)   | full   |
| IONEX TEC maps (`.inx`)         | full   |
| ANTEX antenna PCV (`.atx`)      | full   |
| GPT2w empirical met grid        | full*  |
| RTCM 3.x framing + decoders     | full†  |
| NTRIP v1/v2 client              | full   |
| NMEA-0183 sentences             | full‡  |
| u-blox UBX binary               | partial§ |
| Septentrio SBF binary           | partial§ |
| NovAtel OEM binary              | partial§ |
| UNAVCO BINEX                    | framing only¶ |
| StringIO input                  | full   |
| NetCDF4 / HDF5 read / write     | full*  |
| Zarr write                      | full*  |

`*` requires the corresponding optional extra (`hatanaka`, `lzw`,
`netcdf`, `zarr`, `jit`, ...). `†` decoded RTCM3 message types: 1004
(extended L1/L2 RTK obs), 1005, 1006, 1019, 1020, 1033, MSM4
(1074-1134), MSM7 (1077-1137). Other IDs come back with raw payload
bytes. `‡` decoded NMEA sentences: GGA, RMC, GSA, GSV, VTG. `§`
binary framing + checksum + a small set of high-value messages
(NAV-PVT/NAV-SAT/RXM-RAWX/RXM-SFRBX for UBX; PVTGeodetic/MeasEpoch/
GPSNav for SBF; BESTPOS/BESTXYZ/RAWEPHEM for NovAtel). Other IDs come
back with raw payload bytes for caller dispatch. `¶` BINEX framing
+ ubnxi + checksum; record bodies returned as raw bytes.

## At a glance

### Read any file (auto-detect)

```python
obs = rp.load("file.rnx.gz")            # RINEX 2/3, NAV/OBS, NetCDF, SP3, ...
obs.sel(sv="G07").C1C                   # one variable for one satellite
```

### Stream a multi-GB file

```python
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(time, ds)         # constant memory
```

### Convert a directory in parallel

```python
rp.batch_convert("data/", "*.rnx.gz", "out/", workers=8)
```

### Single-point positioning

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m)
print(sol["lla"])
```

### RTK with LAMBDA integer fix

```python
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

sol = rtk_fix(pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
              wavelength=LAMBDA_L1, ratio_threshold=3.0)
print(sol["fixed"]["baseline"])         # cm-level accuracy
```

### Stream RTCM3 from an NTRIP caster

```python
from rinexpy.ntrip import stream
from rinexpy.rtcm3 import iter_messages
import io

buf = io.BytesIO()
for chunk in stream("rtk2go.com", "MOUNT01", port=2101):
    buf.write(chunk); break              # collect a window
buf.seek(0)
for msg in iter_messages(buf):
    print(msg["msg_id"], msg.get("station_id"))
```

## CLI

```sh
rinexpy read myfile.18o
rinexpy times myfile.18o
rinexpy info myfile.18o
rinexpy convert path/to/data "*.rnx.gz" --out converted/ -j 4
```

## Why a rewrite?

The upstream `georinex` README explicitly notes that "`xarray.concat`
and `xarray.Dataset` nested inside `concat` takes over 60% of time"
for OBS3. `rinexpy` rewrites those readers to fill a preallocated
NumPy buffer in a single pass and build the `xarray.Dataset` exactly
once at the end. From there the project grew the rest of a real GNSS
stack: SP3 interpolation, Klobuchar / Saastamoinen / Niell / VMF1
tropo+iono, single-point positioning, single- and dual-frequency
LAMBDA, full RTK loop, RTCM3 + NTRIP, IONEX / ANTEX / CLK / GPT2w
correction layers.

See [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) for the full list of
performance changes and [SCRATCHPAD.md](SCRATCHPAD.md) for the
build log.

## Tests + CI

`uv run pytest tests/` runs 315 tests in <3 s. CI matrix on GitHub
Actions covers Linux + macOS + Windows × Python 3.11 / 3.12 / 3.13,
plus a separate parity-against-georinex job and a benchmark-publishing
job.

## Citation

If you use this in academic work please also cite the upstream
`georinex` for the original readers:
[doi:10.5281/zenodo.2580306](https://doi.org/10.5281/zenodo.2580306).

## License

MIT, like the upstream project.
