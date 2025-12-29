# rinexpy

A fast RINEX reader for Python that grew into a GNSS toolkit.

Started as a fork of [georinex](https://github.com/geospace-code/georinex)
with the OBS3 and NAV3 readers rewritten. Upstream builds an
`xarray.Dataset` per epoch and merges them, which is O(N²) in epoch
count. rinexpy fills a NumPy buffer in one pass and builds the
Dataset once at the end. On the shared corpus this works out to
13-33× faster on RINEX-3 NAV/OBS (see [BENCHMARKS.md](docs/BENCHMARKS.md)).

Other things landed on top: SP3, CLK, IONEX, ANTEX readers; RTCM 2
and 3 with NTRIP; vendor binary formats (UBX, SBF, NovAtel, BINEX,
NMEA-0183); single-point positioning; an RTK loop with LAMBDA
integer fixing.

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C
```

## Install

rinexpy isn't on PyPI. Clone it and let `uv` handle the rest.

First, install `uv` if you don't have it:

```sh
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via a package manager
brew install uv     # macOS
pipx install uv     # any OS
```

Then:

```sh
git clone https://github.com/ksd3/rinexpy
cd rinexpy
uv sync --all-extras
```

`--all-extras` pulls in every reader extension (CRINEX, LZW, NetCDF,
matplotlib, Zarr, numba JIT, pymap3d) plus the local C++ extension
under `native/`. Drop the flag for a minimal install:

```sh
uv sync                    # base: RINEX 2/3 + SP3 + NetCDF
uv sync --extra native     # + CRINEX support (preferred) + ~40x OBS3 reader
uv sync --extra hatanaka   # + CRINEX support (legacy pure-Python alt)
uv sync --extra plot       # + matplotlib helpers
```

Python 3.11 or newer. Day-to-day development is on 3.13.

Run anything inside the project venv via `uv run`:

```sh
uv run python -c "import rinexpy; print(rinexpy.__version__)"
uv run pytest tests/ -q
uv run rinexpy read tests/data/demo.21o
```

Or `source .venv/bin/activate` once and call `python` and `pytest`
directly.

## Documentation

| | |
|---|---|
| [TUTORIAL.md](docs/TUTORIAL.md) | Install to RTK fix in 12 sections. |
| [COOKBOOK.md](docs/COOKBOOK.md) | Short recipes. |
| [API.md](docs/API.md) | Per-symbol reference, 43 entries. |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map and dataflow. |
| [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) | What changed vs georinex. |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Measured perf numbers. |
| [examples/](examples/) | 8 runnable scripts. |

## Compatibility

Full support:

- RINEX 2, 3, and 4 OBS/NAV
- SP3-a/c/d
- RINEX clock products (`.clk`)
- IONEX (`.inx`) TEC maps
- ANTEX (`.atx`) phase-center variation
- GZIP, BZ2, ZIP archives. LZW needs the `[lzw]` extra; Hatanaka
  CRINEX (versions 1 and 3) ships with the `[native]` C++ extension
  byte-for-byte against the reference decoder (the legacy `[hatanaka]`
  Python package extra is still supported as a pure-Python fallback).
- NTRIP v1/v2 client
- NetCDF4/HDF5 read and write; Zarr write
- StringIO input

Partial:

- RTCM 3.x. Decoders for 1004 (extended L1/L2 RTK obs), 1005, 1006,
  1019, 1020, 1033, MSM4 (1074-1134), MSM7 (1077-1137). Other IDs
  come back with raw payload bytes.
- RTCM 2.x. Decoders for messages 1, 3, 9 (DGPS pseudorange / ECEF /
  high-rate corrections). Other types come back as raw 24-bit data
  words. Hamming parity is NOT validated.
- NMEA-0183. GGA, RMC, GSA, GSV, VTG.
- UBX, SBF, NovAtel OEM. Binary framing plus checksum plus a small
  set of high-value records (NAV-PVT, NAV-SAT, RXM-RAWX, RXM-SFRBX
  for UBX; PVTGeodetic, MeasEpoch, GPSNav for SBF; BESTPOS, BESTXYZ,
  RAWEPHEM for NovAtel). Other IDs return raw payload bytes.
- BINEX. Framing, ubnxi, and checksum; bodies returned as raw bytes.
- BeiDou. D1 subframe 1 (clock + iono coefficients) and D2 page 1
  (clock). Ephemeris subframes 2/3 and almanac 4/5 not yet decoded.
- GPT2w empirical met grid. The grid file (`gpt2_5w.grd`) is user-
  supplied from the VMF Data Server; not shipped here.

## Examples

Auto-detect any file:

```python
obs = rp.load("file.rnx.gz")
obs.sel(sv="G07").C1C
```

Stream a multi-GB file at constant memory:

```python
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(time, ds)
```

Convert a directory in parallel:

```python
rp.batch_convert("data/", "*.rnx.gz", "out/", workers=8)
```

Single-point positioning:

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m)
print(sol["lla"])
```

RTK with LAMBDA integer fix:

```python
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

sol = rtk_fix(pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
              wavelength=LAMBDA_L1, ratio_threshold=3.0)
print(sol["fixed"]["baseline"])
```

Read RTCM3 from an NTRIP caster:

```python
from rinexpy.ntrip import stream
from rinexpy.rtcm3 import iter_messages
import io

buf = io.BytesIO()
for chunk in stream("rtk2go.com", "MOUNT01", port=2101):
    buf.write(chunk); break
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

## Tests

```sh
uv run pytest tests/ -q              # full suite, about 3 s
uv run pytest tests/test_obs3.py -v  # one module
uv run pytest tests/ -k native -v    # by keyword
```

Cross-check against an installed georinex:

```sh
uv pip install georinex
uv run pytest tests/test_parity.py -q
```

Benchmarks:

```sh
uv run python benchmarks/bench_obs3.py
```

Lint and format:

```sh
uv run ruff check src/ tests/ benchmarks/
uv run ruff format src/ tests/ benchmarks/
```

No hosted CI. To verify across Python versions locally:

```sh
uv python install 3.11 3.12 3.13
for v in 3.11 3.12 3.13; do
  uv sync --all-extras --python "$v"
  uv run --python "$v" pytest tests/ -q
done
```

## Build the docs

```sh
uv run mkdocs serve    # live preview at http://127.0.0.1:8000
uv run mkdocs build    # static site under site/
```

## Citation

If you use rinexpy in academic work please also cite the upstream
georinex for the original readers:
[doi:10.5281/zenodo.2580306](https://doi.org/10.5281/zenodo.2580306).

## License

MIT.
