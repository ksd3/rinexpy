# Command-line interface

The `rinexpy` package installs a console script called `rinexpy` that
exposes the most common workflows behind a small argparse interface.
Each subcommand is a thin wrapper around the matching Python function.

```sh
uv run rinexpy --help
```

The high-level layout is:

```
rinexpy <subcommand> <args>

subcommands:
  read       parse + print a RINEX, SP3, NetCDF, or CRINEX file
  times      list every epoch timestamp in a RINEX file
  info       print the parsed header
  convert    batch-convert a directory of RINEX files to NetCDF
  spp        single-point positioning from OBS + NAV
  rtk        RTK baseline from two OBS files + a NAV file
  ppp        precise point positioning from OBS + SP3 + CLK
  splice     concatenate two RINEX OBS files along time
  decimate   decimate a RINEX OBS file by interval
```

All read/convert subcommands accept the common filter flags:

| Flag | Meaning |
| --- | --- |
| `-u/--use` | restrict to GNSS systems (e.g. `G,E` for GPS + Galileo) |
| `-m/--meas` | restrict to observation labels |
| `-t/--tlim START STOP` | restrict to a time window |
| `--useindicators` | also load loss-of-lock and signal-strength indicators |
| `--interval` | decimate to one epoch per interval (seconds) |
| `--strict` | error on parser warnings instead of warning |
| `-v/--verbose` | INFO logging |
| `-q/--quiet` | suppress per-line progress |

## `read`

Parse a file and print the resulting dataset.

```sh
uv run rinexpy read tests/data/demo.10o
```

Output is the xarray `repr` of the parsed dataset. For NetCDF files that
hold both NAV and OBS groups, two datasets are printed.

```sh
uv run rinexpy read tests/data/r3all.nc
```

For RINEX 3 OBS, the filters apply:

```sh
uv run rinexpy read tests/data/obs3.01gage.10o --use G --interval 30
```

## `times`

Print the timestamp axis. Useful when you want to see what is in a file
without paying the data-parse cost.

```sh
uv run rinexpy times tests/data/demo.10o
```

Output:

```
tests/data/demo.10o: 2 epochs
first: 2010-03-05T00:00:00.000000
last:  2010-03-05T00:00:30.000000
```

## `info`

Print the parsed header.

```sh
uv run rinexpy info tests/data/demo.10o
```

Output is the parsed header dict; standard RINEX header labels appear
verbatim plus the derived keys (`position`, `position_geodetic`, `t0`,
`interval`, `fields`, ...).

## `convert`

Convert a directory of RINEX files to NetCDF, in parallel.

```sh
uv run rinexpy convert data/2024 '*.rnx.gz' --out out/ -j 4
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--out PATH` | output directory |
| `-j N`, `--workers N` | number of worker processes; `0` = all CPUs |

Combined with the common filter flags:

```sh
uv run rinexpy convert data/2024 '*.rnx.gz' --out out/ -j 4 \
    --use G --interval 30 -t 2024-06-01 2024-06-02
```

The output files mirror the input names with a `.nc` suffix.

## `spp`

Single-point positioning from one OBS file plus one NAV file.

```sh
uv run rinexpy spp tests/data/demo.10o tests/data/demo.10n
```

The command reads both files, evaluates the satellite positions per
epoch from the NAV file (Keplerian propagation), picks the C1 or C1C
pseudorange from the OBS file, and runs `spp_solve` per epoch. Output
is one line per epoch with the resolved position and clock bias.

The default elevation mask is 7°.

## `rtk`

RTK baseline from a rover OBS, a base OBS, and a NAV file.

```sh
uv run rinexpy rtk rover.obs base.obs nav.nav
```

The command computes the common epochs between the two OBS files, picks
the C1 / L1 pair, runs `rtk_fix` per epoch with the LAMBDA integer fix,
and prints one baseline per accepted epoch.

The default ratio threshold is 3.0 and the wavelength is `LAMBDA_L1`.

## `ppp`

Precise point positioning from an OBS, an SP3, and a CLK file.

```sh
uv run rinexpy ppp obs.rnx sp3.sp3 clk.clk
```

The command runs `ppp_solve` with default options: 7° elevation mask, no
ANTEX, no DCB, no wind-up, no GPT2w. It is intended for spot checks; for
serious PPP work drive the solver from Python so you can configure the
corrections.

## `splice`

Concatenate two OBS files along the time axis.

```sh
uv run rinexpy splice day001.18o day002.18o --out week.18o
```

Equivalent to:

```python
from rinexpy.tools import concat_files
import rinexpy as rp
obs = concat_files(["day001.18o", "day002.18o"])
rp.to_rinex_obs(obs, "week.18o", version=3)
```

## `decimate`

Decimate a RINEX OBS file by interval.

```sh
uv run rinexpy decimate myfile.18o --interval 30 --out decimated.18o
```

Reads the file with `interval=30`, then writes the result back out via
`to_rinex_obs`.

## Programmatic equivalents

Every subcommand has a Python equivalent.

| CLI | Python |
| --- | --- |
| `rinexpy read FILE` | `rinexpy.load(file)` |
| `rinexpy times FILE` | `rinexpy.gettime(file)` |
| `rinexpy info FILE` | `rinexpy.rinexheader(file)` |
| `rinexpy convert DIR GLOB --out OUT -j N` | `rinexpy.batch_convert(dir, glob, out, workers=N)` |
| `rinexpy spp OBS NAV` | `rinexpy.spp_solve(...)` per epoch |
| `rinexpy rtk ROVER BASE NAV` | `rinexpy.rtk.rtk_fix(...)` per epoch |
| `rinexpy ppp OBS SP3 CLK` | `rinexpy.ppp.ppp_solve(obs, sp3, clk)` |
| `rinexpy splice A B --out C` | `concat_files([A, B])` + `to_rinex_obs(...)` |
| `rinexpy decimate F --interval 30` | `rinexpy.load(F, interval=30)` + `to_rinex_obs(...)` |

For anything beyond the standard recipes (custom DCB application,
real-time PPP, IMU fusion), use the Python API.

## Running from `python -m`

The `__main__.py` shim makes the CLI available as a module too:

```sh
uv run python -m rinexpy info myfile.18o
```

This is identical to `uv run rinexpy info myfile.18o`; it is useful when
you want to bypass the script wrapper.

## Exit codes

| Exit code | Meaning |
| --- | --- |
| 0 | success |
| 1 | parser error or unrecognised input |
| 2 | argparse error (invalid flag combination) |
| 130 | interrupted by Ctrl+C |

## Related pages

- [Multi-file tools](multi-file.md): the `tools` module that backs `splice` and `decimate`.
- [Streaming over RAM-sized files](streaming.md): the streaming iterator behind large reads.
- [NetCDF and Zarr output](io.md): the persisted-form pages.
