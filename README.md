# rinexpy

Modern, fast GNSS toolkit for Python: RINEX 2/3/4 readers, SP3 / CLK /
IONEX / ANTEX, RTCM2 + RTCM3 + NTRIP streaming, NMEA-0183 + UBX +
SBF + NovAtel + BINEX receiver formats, BeiDou D1/D2 raw subframes,
single-point and RTK positioning with LAMBDA integer fixing, full
troposphere/ionosphere correction stack.

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

`rinexpy` is not published to PyPI — it builds and runs from a local
checkout. Python 3.11+ is required (developed against 3.13.x); the
base install is **pure Python**, no compiler needed.

### 1. Install [`uv`](https://github.com/astral-sh/uv)

`uv` is the package and project manager this repo is configured for.
It installs Python interpreters, manages the virtualenv, resolves
the lockfile, and runs the test/bench scripts.

```sh
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew / pipx if you prefer:
brew install uv          # macOS
pipx install uv          # any OS
```

Verify: `uv --version`.

### 2. Clone and set up

```sh
git clone https://github.com/ksd3/rinexpy
cd rinexpy
uv sync --all-extras       # creates .venv, installs every reader extra + dev deps
```

That's it. The `--all-extras` flag pulls in CRINEX, LZW, NetCDF,
plotting, JIT, Zarr, geo helpers, and the local-path C++ extension
(`./native/`) in one shot. For a minimal install:

```sh
uv sync                    # base package only
uv sync --extra hatanaka   # + a specific reader extra
```

### 3. Use it

Run anything inside the project's virtualenv via `uv run`:

```sh
uv run python -c "import rinexpy; print(rinexpy.__version__)"
uv run pytest tests/ -q
uv run rinexpy read tests/data/demo.21o
```

Or `source .venv/bin/activate` once and call `python` / `pytest`
directly.

## Documentation

| | |
|---|---|
| [TUTORIAL.md](docs/TUTORIAL.md) | Step-by-step walk-through from install to RTK fix (12 sections). |
| [COOKBOOK.md](docs/COOKBOOK.md) | Bite-sized recipes for common one-shot tasks. |
| [API.md](docs/API.md) | Per-symbol reference of the public surface (43 entries). |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout, dependency layers, dataflow. |
| [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) | Every change vs georinex with rationale. |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Measured perf numbers vs georinex. |
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
| RTCM SC-104 v2.x (legacy DGPS)  | full‖  |
| NTRIP v1/v2 client              | full   |
| NMEA-0183 sentences             | full‡  |
| u-blox UBX binary               | partial§ |
| Septentrio SBF binary           | partial§ |
| NovAtel OEM binary              | partial§ |
| UNAVCO BINEX                    | framing only¶ |
| BeiDou D1/D2 raw subframes      | clock + iono only# |
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
+ ubnxi + checksum; record bodies returned as raw bytes. `‖` decoded
RTCM2 message types: 1 (DGPS pseudorange corrections), 3 (reference
station ECEF), 9 (high-rate corrections); other types come back as
raw 24-bit data words; Hamming parity NOT validated. `#` decoded
BeiDou D1 subframe 1 (clock + iono coefs) and D2 page 1 (clock);
ephemeris subframes 2/3 and almanac 4/5 not yet decoded.

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
performance changes.

## Tests

```sh
uv run pytest tests/ -q              # full suite, ~3 s
uv run pytest tests/test_obs3.py -v  # one module
uv run pytest tests/ -k native -v    # filter by keyword

# Cross-check against an installed georinex:
uv pip install georinex
uv run pytest tests/test_parity.py -q

# Benchmarks (writes results under benchmarks/):
uv run python benchmarks/bench_obs3.py
```

Lint and format:

```sh
uv run ruff check src/ tests/ benchmarks/
uv run ruff format src/ tests/ benchmarks/
```

There is no hosted CI — testing is local-only. If you want to verify
across multiple Python versions, install them through `uv` and pass
`--python`:

```sh
uv python install 3.11 3.12 3.13
for v in 3.11 3.12 3.13; do
  uv sync --all-extras --python "$v"
  uv run --python "$v" pytest tests/ -q
done
```

## Build the docs locally

```sh
uv run mkdocs serve        # live preview at http://127.0.0.1:8000
uv run mkdocs build        # static site under site/
```

## Citation

If you use this in academic work please also cite the upstream
`georinex` for the original readers:
[doi:10.5281/zenodo.2580306](https://doi.org/10.5281/zenodo.2580306).

## License

MIT, like the upstream project.
