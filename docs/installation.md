# Installation

rinexpy is not published on PyPI. The project is installed from the GitHub
repository under [`uv`](https://docs.astral.sh/uv/), which manages the Python
version, the virtual environment, the lock file, and the optional native
extension as one workflow.

## Install uv

If you do not have `uv` on your system, install it first. The instructions
below mirror the official `uv` documentation; pick whichever one fits your
platform.

```sh
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via your package manager
brew install uv      # macOS Homebrew
pipx install uv      # any OS that has pipx
```

The installer drops the `uv` binary into `~/.local/bin` (or a comparable
location on Windows) and prints the line you need to add to your shell
profile if it is not already on `$PATH`.

## Clone and sync

Once `uv` is on your `$PATH`, the rest is two commands.

```sh
git clone https://github.com/ksd3/rinexpy
cd rinexpy
uv sync --all-extras
```

The `--all-extras` flag pulls every optional reader, the matplotlib plotting
helpers, the numba JIT path, the Zarr writer, and the local C++ extension
that lives under `native/`. The project venv lands in `.venv/`. The lock
file is `uv.lock`.

If you would rather have a thinner install, drop the flag and pick the
extras you actually need.

```sh
uv sync                       # base: RINEX 2 / 3 OBS / NAV, SP3, NetCDF
uv sync --extra native        # base + CRINEX decoder + ~40x faster OBS3 reader
uv sync --extra hatanaka      # base + legacy pure-Python CRINEX decoder
uv sync --extra lzw           # base + Unix LZW (.Z) decompression
uv sync --extra plot          # base + matplotlib helpers
uv sync --extra jit           # base + numba JIT path for huge OBS3 files
uv sync --extra zarr          # base + Zarr writer
uv sync --extra geo           # base + pymap3d helpers
```

`uv sync --extra <name>` is cumulative across invocations; the lock file
tracks what has been selected. If you change your mind, just call `uv sync`
again with the new flags.

## Python version

rinexpy needs Python 3.11 or newer. Day-to-day development is on 3.13.
`uv` will pick the right interpreter from your system automatically;
if you want a specific one, `uv python install 3.12` (or 3.11 or 3.13) and
then `uv sync --python 3.12` will pin the project to that version.

## Running anything

Anything you run against rinexpy goes through `uv run`, which handles
activating the venv for you. There is no need to source `.venv/bin/activate`
yourself.

```sh
uv run python -c "import rinexpy; print(rinexpy.__version__)"
uv run pytest tests/ -q
uv run rinexpy info tests/data/demo.10o
```

If you prefer the classic workflow, `source .venv/bin/activate` once and
the bare `python` / `pytest` / `rinexpy` commands behave normally inside that
shell.

## Optional extras explained

The choices below map to the bracket extras in `pyproject.toml`. They are
independent; pick any combination.

### `native`

Builds the rinexpy C++17 extension that lives under `native/` in the
repository. The extension provides two things. First, a CRINEX 1 and CRINEX 3
decoder that matches the upstream `hatanaka` package byte-for-byte and is
several times faster. Second, an in-place RINEX 3 OBS decoder that drops
the parse time of a 24-hour 30-second file from about 70 ms to about 40 ms.

When `native` is installed, rinexpy uses it automatically for CRINEX reads
and for the OBS3 parse kernel. You do not need to opt in. If the native
extension is not present, the pure-Python decoder runs instead.

### `hatanaka`

Pulls in the [hatanaka](https://pypi.org/project/hatanaka/) package, the
pure-Python implementation that the original georinex used. It is slower
than the C++ extension but ships as a single wheel with no compiler step,
which makes it useful on platforms where building a C++ extension is
inconvenient.

If both `native` and `hatanaka` are installed, the native path wins.

### `lzw`

Pulls in [ncompress](https://pypi.org/project/ncompress/) so that
rinexpy can decompress old Unix `.Z` files. The standard library does not
ship an LZW decoder, so without this extra a `.Z` file raises `ImportError`
with an actionable message.

### `netcdf`

Pulls in `netCDF4`, which is the engine xarray uses for the NetCDF round
trips. Most platforms have this preinstalled because xarray itself depends
on it, but the explicit extra is there so that minimal installs that lean on
the bare xarray Zarr / HDF backends still pick up NetCDF support.

### `geo`

Pulls in [pymap3d](https://pypi.org/project/pymap3d/), an alternative
implementation of the ECEF / geodetic / ENU conversions. rinexpy does not
require pymap3d, but a couple of plotting helpers can use it for cartopy-style
projections.

### `plot`

Pulls in matplotlib so that `rinexpy.plots` can render time series, ground
tracks, skyplots, and receiver maps. Without matplotlib, the module raises
`ImportError` only when you try to call one of its functions; importing
rinexpy itself never touches matplotlib.

### `jit`

Pulls in `numba`. When you opt in with `use_jit=True` on a call to
`rinexobs3`, or by setting `RINEXPY_USE_JIT=1` in the environment, rinexpy
runs the OBS3 inner loop through numba. The end-to-end speedup is about
1.9x on a 23-hour 15-second file. The trade-off is the one-shot JIT compile
cost on the first call and an extra dependency tree.

If `native` is also installed, the native path wins over the JIT path. Both
end up at roughly the same wall-clock time, but `native` does not need
`numba` and `llvmlite`.

### `zarr`

Pulls in [zarr](https://pypi.org/project/zarr/) so that `rinexpy.zarr_io.to_zarr`
can write a parsed dataset out as a Zarr store. Handy for cloud workflows
where multiple workers want to read partial slices of the same dataset.

### `all`

Convenience extra that turns on `lzw`, `netcdf`, `geo`, and `plot` in one
shot. Equivalent to the four `--extra` flags called together.

## Verifying the install

Open a Python REPL inside the project and run the lines below. Each one
exercises a different layer of the stack; if any fails, the error message
should point at the missing dependency.

```python
import rinexpy as rp
print(rp.__version__)               # 0.1.0

# Layer 0: top-level reader on a tiny shipped file.
obs = rp.load("tests/data/demo.10o")
print(obs.sv.size, "SVs,", obs.time.size, "epochs")

# Layer 1: NAV reader on a richer file.
nav = rp.load("tests/data/brdc2800.15n")
print(nav.sv.size, "SVs, vars:", list(nav.data_vars)[:5])

# Layer 2: SP3 reader.
sp3 = rp.load_sp3("tests/data/igs19362.sp3c")
print(sp3.sv.size, "SVs,", sp3.time.size, "epochs")
```

Expected output is roughly:

```
0.1.0
14 SVs, 2 epochs
32 SVs, vars: ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE', 'Crs']
32 SVs, 96 epochs
```

If you opted into `plot`, render a skyplot from the broadcast nav file to
confirm matplotlib works:

```python
import matplotlib
matplotlib.use("Agg")   # headless mode, no display required
import matplotlib.pyplot as plt
import numpy as np
from rinexpy.geodesy import azimuth_elevation, lla_to_ecef
from rinexpy.plots import skyplot

rx = lla_to_ecef(40.0, -3.0, 100.0)
sv_az_el = {}
for sv_label in nav.sv.values:
    if sv_label[0] not in {"G", "E"}:
        continue
    sv = nav.sel(sv=sv_label)
    try:
        X, Y, Z = rp.keplerian2ecef(sv)
    except (ValueError, KeyError):
        continue
    sv_ecef = np.stack([X, Y, Z], axis=-1)
    az, el = azimuth_elevation(rx, sv_ecef)
    sv_az_el[sv_label] = (az, el)

skyplot(sv_az_el, title="Sky from (40, -3)")
plt.savefig("skyplot.png", dpi=120, bbox_inches="tight")
```

A PNG file appears in your current directory. If matplotlib raises an
import error, you forgot to install the `plot` extra.

## Running the tests

A full pytest run takes a few seconds on a modern laptop. The
`tests/data/` directory carries enough fixtures that the entire suite runs
without any network access.

```sh
uv run pytest tests/ -q
uv run pytest tests/test_obs3.py -v        # one module
uv run pytest tests/ -k "rtk and not real"  # by keyword
```

A handful of tests are marked `@pytest.mark.parity` and compare rinexpy
output against an installed `georinex` package. If you want them to run,
install `georinex` into the project venv first.

```sh
uv pip install georinex
uv run pytest tests/test_parity.py -q
```

A few `..._real.py` tests load real RINEX files that are not shipped with
the repository. They are skipped automatically when the data is missing.

## Building the docs locally

The docs you are reading right now are built with MkDocs Material. The
configuration lives in `mkdocs.yml` at the repository root, the pages live
under `docs/`, and the build itself runs inside the project venv.

```sh
uv run mkdocs serve     # live preview at http://127.0.0.1:8000
uv run mkdocs build     # static site under site/
```

Read the Docs uses the same `mkdocs.yml` file. The RTD configuration in
`.readthedocs.yaml` sets the build image, the Python version, and the
dependencies it needs.

## Cross-version verification

If you maintain a downstream project that needs rinexpy to work across
several Python versions, `uv` makes the loop tight.

```sh
uv python install 3.11 3.12 3.13
for v in 3.11 3.12 3.13; do
  uv sync --all-extras --python "$v"
  uv run --python "$v" pytest tests/ -q
done
```

Each iteration switches the project venv to a different interpreter, syncs
the lock file under that interpreter, and runs the suite. The whole loop
takes about a minute on a modern laptop.
