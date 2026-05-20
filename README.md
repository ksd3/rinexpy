# rinexpy

A fast RINEX reader for Python that grew into a GNSS toolkit.

`rinexpy` is a fork of [georinex](https://github.com/geospace-code/georinex)
with the OBS3 and NAV3 readers rewritten. Upstream builds an
`xarray.Dataset` per epoch and merges them, which is O(N²) in epoch
count. `rinexpy` fills a NumPy buffer in one pass and builds the
Dataset once at the end. On the shared corpus that is 13-33× faster
on RINEX-3 NAV/OBS (see [docs/internals/benchmarks.md](docs/internals/benchmarks.md)).

On top of the readers, `rinexpy` includes:

- SP3, CLK, IONEX, ANTEX readers.
- RTCM 2 and 3 (including the full SSR family) with NTRIP, sync and async.
- Vendor binary formats: UBX, SBF, NovAtel, BINEX, NMEA-0183.
- Raw nav-message decoders for GPS LNAV / CNAV / CNAV-2, Galileo F-NAV /
  I-NAV, GLONASS strings, NavIC, BeiDou D1 / D2, and the SBAS L1 stream.
- SPP with RAIM, RTK with LAMBDA integer fixing plus a sequential variant
  with ambiguity carry-over, a static-or-kinematic EKF, and a PPP driver
  that combines SP3+CLK (or RTCM-SSR), ANTEX PCV, GPT2w+VMF1 troposphere,
  DCB, and carrier-phase wind-up into one call.

```python
import rinexpy as rp

obs = rp.load("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
obs.sel(sv="G07").C1C
```

## Install

`rinexpy` isn't on PyPI. Clone it and let `uv` handle the rest.

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
  1019, 1020, 1029 (text), 1033, 1230 (GLONASS code-phase biases),
  the full MSM 1-7 family (1074-1134 + 1077-1137 etc.), the SSR
  family (1057-1068 GPS+GLONASS, 1240-1263 Galileo/QZSS/SBAS/BeiDou),
  and IGS-SSR MT 4076. Other IDs come back with raw payload bytes.
- RTCM 2.x. Decoders for messages 1, 3, 9 (DGPS pseudorange / ECEF /
  high-rate corrections). Other types come back as raw 24-bit data
  words. Hamming parity is NOT validated.
- SBAS L1. Message types 1, 2-5, 6, 7, 9, 17, 18, 24, 25, 26
  (PRN mask, fast corrections, integrity, GEO ranging, almanacs,
  iono grid, long-term + iono delays). RTCA DO-229E §A.4.
- NMEA-0183. GGA, RMC, GSA, GSV, VTG.
- UBX, SBF, NovAtel OEM. Binary framing plus checksum plus a small
  set of high-value records (NAV-PVT, NAV-SAT, RXM-RAWX, RXM-SFRBX
  for UBX; PVTGeodetic, MeasEpoch, GPSNav for SBF; BESTPOS, BESTXYZ,
  RAWEPHEM for NovAtel). Other IDs return raw payload bytes.
- BINEX. Framing, ubnxi, and checksum; bodies returned as raw bytes.
- BeiDou D1 subframe 1 + D2 page 1 (clock + iono).
- GPS LNAV (subframes 1-3), CNAV (MT 10, 11), CNAV-2 (subframe 2).
- Galileo F-NAV (page 1, 2), I-NAV (word 1, 4).
- GLONASS L1OF/L2OF strings 1, 2, 3 (X/Y/Z + velocity +
  acceleration, sign-magnitude encoding per ICD §4.4).
- NavIC / IRNSS subframes 1 and 2; subframes 3/4 returned with
  message-id + raw payload for downstream dispatch.
- RINEX 4 NAV. STO, EOP, ION (Klobuchar / NeQuick-G / BDGIM) per
  RINEX 4.01 §5.
- GPT2w empirical met grid. The grid file (`gpt2_5w.grd`) is user-
  supplied from the VMF Data Server; not shipped here.
- DCB autodownload. Daily SINEX-BIAS from IGS BKG (MGEX CAS rapid)
  for dates from 2017 on, and monthly CODE P1-P2 / P1-C1 / P2-C2
  from AIUB for earlier dates. CDDIS path is wired but requires
  NASA Earthdata Login in `~/.netrc`.

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

Single-point positioning, with optional RAIM fault detection:

```python
sol = rp.spp_solve(sv_ecef, pseudoranges_m, raim=True)
print(sol["lla"], sol["excluded_svs"])
```

RTK with LAMBDA integer fix:

```python
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

sol = rtk_fix(pr_r, pr_b, phase_r, phase_b, sv_ecef, base_ecef,
              wavelength=LAMBDA_L1, ratio_threshold=3.0)
print(sol["fixed"]["baseline"])
```

Sequential RTK that carries the integer fix across epochs:

```python
from rinexpy.rtk import SequentialRTK
rtk = SequentialRTK(base_ecef, wavelength=LAMBDA_L1)
for epoch in epochs:
    result = rtk.update(svs, rover_pr, base_pr, rover_phase, base_phase, sv_ecef)
    if result["fixed_accepted"]:
        print(result["baseline"], "carried:", result["carry_over_count"])
```

Static-receiver PPP with SP3 + CLK and the full correction stack:

```python
from rinexpy.ppp import ppp_solve
from rinexpy.dcb_download import auto_load_dcb
from rinexpy.antex import find_antenna, load_antex

ant = find_antenna(load_antex("igs20.atx"), "TRM59800.00     NONE")
dcb = auto_load_dcb(obs.time.values[0].astype("datetime64[D]").astype(object))
out = ppp_solve(
    obs, sp3, clk,
    antenna=ant,
    dcb_records=dcb,
    apply_wind_up=True,
)
print(out["lla"], "1-sigma:", out["position_sigma_m"])
```

Same driver, but with real-time SSR replacing CLK:

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

ssr = SSRCorrections(iter_messages(open("ssr-stream.rtcm", "rb")))
out = ppp_solve(obs, sp3, clk=None, ssr=ssr)
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
rinexpy spp myfile.18o myfile.18n
rinexpy rtk rover.obs base.obs nav.nav
rinexpy ppp obs.rnx sp3 clk
rinexpy splice a.obs b.obs --out joined.obs
rinexpy decimate myfile.obs --interval 30
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

If you use `rinexpy` in academic work please also cite the upstream
georinex for the original readers:
[doi:10.5281/zenodo.2580306](https://doi.org/10.5281/zenodo.2580306).

## License

MIT.
