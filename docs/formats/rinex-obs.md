# RINEX observation files

RINEX (Receiver Independent Exchange Format) observation files are the
canonical way to share GNSS observables. Each file holds time-tagged
measurements at a fixed sampling interval. A typical RINEX 3 file from an
IGS station contains pseudorange, carrier phase, Doppler, and signal-strength
observations for every satellite of every constellation that the receiver
tracked over the file's time span.

rinexpy reads RINEX 2.x, RINEX 3.x, and (for header-only inspection) RINEX 4
observation files. The output is always an `xarray.Dataset`.

## The dataset schema

A parsed observation Dataset has two indexed dimensions and one nested
dictionary of observation labels.

| Coord | Type | Notes |
| --- | --- | --- |
| `time` | `datetime64[us]` | epoch timestamps |
| `sv` | `<U3` | satellite labels like `G07`, `R12`, `E15` |

Each measurement label is a separate `DataArray` indexed on `(time, sv)`.
The labels follow the RINEX 3 convention even when the source is RINEX 2:
`C1C`, `L2W`, `S1P`, and so on. For RINEX 2 inputs the labels are the
short two-character codes (`L1`, `C1`, `P2`, `L2`).

If you ask for indicators (`useindicators=True`) the parser keeps the
loss-of-lock indicator and the signal-strength indicator on each phase
observation as separate `<code>_lli` and `<code>_ssi` variables.

The attributes dict carries the header information.

| Key | Type | Notes |
| --- | --- | --- |
| `version` | `float` | RINEX version (2.11, 3.04, etc.) |
| `rinextype` | `str` | always `"obs"` here |
| `filetype` | `str` | RINEX filetype letter (`O`) |
| `systems` | `str` | header letter (`G`, `R`, `E`, `M`, ...) |
| `interval` | `float` | nominal sampling interval in seconds |
| `position` | `list[float]` | ECEF approximate station position |
| `position_geodetic` | `tuple[float, float, float]` | (lat, lon, alt) derived from `position` |
| `time_system` | `str` | usually `"GPS"`, `"GAL"`, `"GLO"`, `"BDS"` |
| `filename` | `str` | the input filename (when known) |
| `t0`, `t1` | `datetime` | first / last epoch from the header |
| `fields` | `dict[str, list[str]]` | per-system measurement labels |
| `Fmax` | `int` | maximum measurement count across systems |

## The basic read

```python
import rinexpy as rp

obs = rp.load("tests/data/obs3.01gage.10o")
print(obs)
```

For the small file that ships with the repository, the output is:

```
<xarray.Dataset>
Dimensions:  (time: 2, sv: 14)
Coordinates:
  * time     (time) datetime64[us] 2010-03-05 ... 2010-03-05T00:00:30
  * sv       (sv) <U3 'G07' 'G09' 'G12' 'G13' 'G15' 'G17' 'G19' 'G27' ...
Data variables:
    L1C      (time, sv) float64 ...
    L2P      (time, sv) float64 ...
    C1P      (time, sv) float64 ...
    C2P      (time, sv) float64 ...
    C1C      (time, sv) float64 ...
    S1P      (time, sv) float64 ...
    S2P      (time, sv) float64 ...
    S1C      (time, sv) float64 ...
Attributes:
    version:    3.01
    interval:   30.0
    rinextype:  obs
    ...
```

## Compression

The opener decompresses on the fly. The mapping from file suffix to
backend is:

| Suffix | Backend | Extra | Notes |
| --- | --- | --- | --- |
| `.gz` | stdlib `gzip` | none | always available |
| `.bz2` | stdlib `bz2` | none | always available |
| `.zip` | stdlib `zipfile` | none | reads the first file in the archive |
| `.Z` | `ncompress` | `lzw` | LZW compression from the Unix `compress` tool |
| `.crx`, `.crx.gz` | native C++ or `hatanaka` | `native` or `hatanaka` | Hatanaka compressed RINEX |
| `.rnx`, `.18o`, `.YYn`, ... | plain text | none | the usual suffixes |

If the matching extra is not installed, opening the file raises a clean
`ImportError` rather than silently truncating.

## RINEX 2 vs RINEX 3

RINEX 2 has a single observation-type header (`# / TYPES OF OBSERV`) that
applies to every satellite. RINEX 3 has one such header per satellite system
(`SYS / # / OBS TYPES`), so the per-system field lists can differ. The
parser tracks both layouts. The returned Dataset always uses the union of
all per-system labels; cells where the system did not record a particular
observable are NaN.

The signal-naming convention also differs. RINEX 2 uses two characters
(`L1`, `C1`, `P2`, ...). RINEX 3 uses three (`L1C`, `C2W`, `S1P`, ...): a
band number, a tracking-mode letter, and the optional attribute. rinexpy
keeps the original labels in both directions.

If you need version-aware code, the `version` attribute on the parsed
dataset and on the header tells you which side of 3.0 you are on.

```python
hdr = rp.rinexheader("tests/data/demo.10o")     # RINEX 2.11 → version is 2.11
hdr_v3 = rp.rinexheader("tests/data/obs3.01gage.10o")    # version is 3.0
```

## Filtering during the read

Every filter pushes down into the parser; records that do not match are
skipped before they ever turn into Python floats.

```python
obs = rp.load(
    "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip",
    use={"G"},                    # only GPS
    meas=["C1C", "L1C"],          # only these labels
    interval=60,                  # decimate to every 60 s
)
```

The keyword arguments are:

`use=` — a single letter or any iterable of letters from
`{"G", "R", "E", "J", "C", "I", "S"}`.

`meas=` — a list of measurement labels. For RINEX 3 these are the
three-character codes; for RINEX 2 the two-character codes.

`tlim=` — `(start, end)` as ISO-8601 strings, plain `datetime`s, or
`np.datetime64` values. Records outside the window are dropped.

`interval=` — seconds (`float` or `int`) or a `timedelta`. The parser
keeps one epoch every `interval` seconds, snapping to the file's own
sampling grid.

`useindicators=` — when `True`, keep loss-of-lock and signal-strength
indicators as `<code>_lli` and `<code>_ssi` variables.

`fast=` — `True` (default) uses a one-pass speculative-preallocation
parse for RINEX 2 OBS. `False` does a clean two-pass scan and is slower
but always safe on pathological inputs.

`verbose=` — `True` enables INFO-level logging of the parse progress.

`use_jit=` — `True` to opt into the numba-JITed inner loop for RINEX 3
OBS. Needs the `jit` extra. Off by default; the native C++ extension
(installed via `[native]`) is preferred when available.

`use_native=` — `True` to force the C++ kernel; `False` to force pure
Python. Defaults to autodetect.

## Reading just the header

Three header readers cover the common cases.

```python
info = rp.rinexinfo("tests/data/obs3.01gage.10o")
# {'version': 3.0, 'filetype': 'O', 'rinextype': 'obs', 'systems': 'M'}

hdr = rp.rinexheader("tests/data/obs3.01gage.10o")
# Full parsed header dict

# Or directly:
from rinexpy.headers import obsheader2, obsheader3
hdr_v3 = obsheader3(open("tests/data/obs3.01gage.10o"))
```

`rinexinfo` reads only the first non-blank line and is the cheapest. It is
what `rp.load` calls internally before dispatch. `rinexheader` reads the
whole header but stops at `END OF HEADER`, so it skips the data section
entirely.

The header dict is keyed by the RINEX header label verbatim plus a handful
of derived keys.

```python
hdr["APPROX POSITION XYZ"]   # raw header line, padded to 60 chars
hdr["TIME OF FIRST OBS"]
hdr["INTERVAL"]
hdr["# / TYPES OF OBSERV"]   # RINEX 2 only
hdr["SYS / # / OBS TYPES"]   # RINEX 3 only

# Derived:
hdr["position"]              # [x, y, z] in metres
hdr["position_geodetic"]     # (lat_deg, lon_deg, alt_m)
hdr["interval"]              # nominal interval in seconds
hdr["t0"]                    # datetime of TIME OF FIRST OBS
hdr["t1"]                    # datetime of TIME OF LAST OBS, if present
hdr["fields"]                # {"G": [...], "R": [...], ...}
hdr["fields_ind"]            # same shape, indicator labels
hdr["Fmax"]                  # max len(fields[sys]) across systems
hdr["Nobs"]                  # RINEX 2 only: number of obs types
hdr["Nl_sv"]                 # RINEX 2 only: header lines per epoch
```

## Iterating epoch by epoch

For files larger than RAM (multi-day, 1 Hz, multi-constellation OBS3
files are the usual offenders), the streaming iterator yields one epoch
at a time.

```python
for time, ds in rp.iter_obs3_epochs("huge.rnx.gz"):
    process_one_epoch(time, ds)
```

`ds` is a single-time `xarray.Dataset` sized to the SVs visible at that
epoch. Memory footprint is constant regardless of how long the file is.

The same `use`, `tlim`, and `interval` filters work on the iterator.

The streaming reader only supports RINEX 3. RINEX 2 files load into memory
with `load`.

## NetCDF round-trip

The `out=` argument on `load` triggers a NetCDF write as the file is
parsed. The data lands in an HDF5 group named after the file kind.

```python
rp.load("tests/data/demo.10o", out="demo.nc")
# Re-read:
re = rp.load("demo.nc")
```

`batch_convert` is the parallel form.

```python
written = rp.batch_convert("data/", "*.18o", "out/", workers=0)
```

`workers=0` uses all CPUs. `workers=1` (or `None`) is serial. Per-file
errors are logged and the conversion continues with the next file.

## Writing back to RINEX

`to_rinex_obs` emits a parsed dataset back to disk.

```python
obs = rp.load("input.18o", tlim=("2018-07-29", "2018-07-30"))
rp.to_rinex_obs(obs, "filtered.rnx", version=3)
```

`version=` is 2 or 3. The writer emits the standard header records; if
your input carried non-standard comments or custom records, they may not
round-trip verbatim.

The typical use case is "filter / decimate / re-emit" workflows, where
the goal is a smaller RINEX file that another tool can read.

## Performance

A worked benchmark is on the [benchmarks page](../internals/benchmarks.md).
Headline numbers on the bundled test corpus:

| Path | Time | Speedup vs georinex |
| --- | --- | --- |
| RINEX 2 OBS, pure Python | 16-50 ms per file | 1.04-1.11x |
| RINEX 3 OBS, pure Python | 1-10 ms per file | 13-18x |
| RINEX 3 OBS, native C++ | 40 ms on a 23h 15s file | 28x |
| RINEX 3 OBS, numba JIT | 38 ms on a 23h 15s file | 29x |

The headline win is on RINEX 3 OBS, where dropping the per-epoch
`xarray.merge` from the upstream implementation removes an O(N²) factor.
The page on [optimizations](../internals/optimizations.md) walks through
every change.

## Bundled fixtures

The `tests/data/` directory ships a corpus large enough to exercise every
combination of the reader. The headline files are:

| File | What it covers |
| --- | --- |
| `demo.10o` | minimal RINEX 2.11 OBS, GPS + GLONASS, 2 epochs |
| `obs3.01gage.10o` | minimal RINEX 3.0 OBS, mixed constellation, 2 epochs |
| `ABMF00GLP_R_20181330000_01D_30S_MO.zip` | full-day RINEX 3 OBS inside a zip |
| `CEDA00USA_R_20182100000_23H_15S_MO.rnx.gz` | gzipped 23-hour 15-second file |
| `york0440.zip` | RINEX 2 OBS as a zip archive |
| `ac660270.18o.Z` | RINEX 2 OBS as a Unix LZW archive (needs `lzw` extra) |
| `york0440.15d` | RINEX 2 Hatanaka CRINEX (needs `native` or `hatanaka` extra) |
| `P43300USA_R_20190012056_17M_15S_MO.crx.bz2` | RINEX 3 CRINEX inside bzip2 |
| `r3all.nc` | NetCDF round-trip of `obs3.01gage.10o` |
| `wrong_obs2_count.10o` | malformed file used to exercise the warnings path |

The full list is `ls tests/data/`.

## Edge cases the parser handles

The library has been hammered against the upstream georinex test corpus and
inherits its handling of the awkward cases.

**Trailing whitespace on epoch headers.** Some receivers pad the epoch
line with spaces. The parser strips them.

**SVs in the epoch header that have no data rows.** Old RINEX 2 files
sometimes list more satellites in the epoch header than they actually
record. The parser pads with NaN.

**Junk timestamps.** A handful of receivers emit epoch lines that are off
by leap seconds or off by year. The fast path detects the gap and resyncs.
The `tests/data/junk_time_obs3.10o` and `badtime.10o` fixtures exercise
this.

**Receiver clock offset already applied.** When the `RCV CLOCK OFFS APPL`
header field is 1, the observations have been corrected. The parser
records the flag in the `receiver_clock_offset_applied` attribute so
downstream code can avoid double-correcting.

**System / DCB / PCV applied flags.** Carried into the attrs.

**Time system inference.** Mixed-constellation RINEX 3 files often declare
no explicit `TIME SYSTEM CORR` records. The parser infers the system
letter from the file's header signature and falls back to GPS time.

## RINEX 4 OBS

RINEX 4 extends RINEX 3 mostly in the NAV layer; OBS files are
backward-compatible with RINEX 3. The `headers.rinexinfo` helper reports a
RINEX 4 OBS file with `version >= 4.00`, and `rinexobs3` reads them as
RINEX 3.

## Related pages

- [RINEX navigation files](rinex-nav.md): broadcast ephemerides.
- [SP3 and clock products](sp3-clk.md): precise orbits and clocks.
- [Streaming over RAM-sized files](../tooling/streaming.md): the iterator.
- [NetCDF and Zarr output](../tooling/io.md): for persisted forms.
- [Architecture](../internals/architecture.md): how the reader is layered.
- [Optimizations](../internals/optimizations.md): why it is fast.
