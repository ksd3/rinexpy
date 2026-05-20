# SP3 ephemerides and clock products

SP3 (Standard Product 3) is the IGS-published precise satellite-orbit
format. Each SP3 file gives ECEF position, optionally velocity, and the
satellite clock offset for every satellite in the constellations it
covers, sampled every 15 minutes. The clock RMS in an IGS final SP3 is on
the order of 75 picoseconds, which is good enough for centimetre-class
positioning when combined with the matching RINEX clock product.

`rinexpy` reads SP3-a, SP3-c, and SP3-d. The matching `.clk` clock products
go through `load_clk`.

## SP3

### Loading

```python
import rinexpy as rp

sp3 = rp.load_sp3("tests/data/igs19362.sp3c")
print(sp3)
```

The output is an `xarray.Dataset`:

```
<xarray.Dataset>
Dimensions:    (time: 96, sv: 32, ECEF: 3)
Coordinates:
  * time       (time) datetime64[ns] 2017-02-14 ... 2017-02-14T23:45:00
  * sv         (sv) <U3 'G01' 'G02' 'G03' 'G04' 'G05' ...
  * ECEF       (ECEF) <U1 'x' 'y' 'z'
Data variables:
    position   (time, sv, ECEF) float64 ...   # km
    velocity   (time, sv, ECEF) float64 ...   # dm/s
    clock      (time, sv)       float64 ...   # microseconds
    dclock     (time, sv)       float64 ...   # rate, 1e-4 µs/s
    t0         ()               datetime64[ns]
Attributes:
    version:        c
    coord_system:   IGS14
    orbit_type:     FIT
    rinextype:      sp3
```

The units inside the Dataset match the SP3 wire format. `position` is in
kilometres, `velocity` is in decimetres per second (the SP3 convention),
`clock` is in microseconds, `dclock` is the clock rate in 1e-4 microseconds
per second.

To work in metres directly, convert on the fly:

```python
position_m = sp3.position * 1000.0
clock_s = sp3.clock * 1e-6
```

### Filtering

`load_sp3` does not push filters down into the parser, because SP3 files
are small. Filter the resulting Dataset:

```python
sp3_g = sp3.where(sp3.sv.str[0] == "G", drop=True)
sp3_window = sp3.sel(time=slice("2017-02-14T06:00", "2017-02-14T12:00"))
```

### Interpolating

SP3 files are sampled every 15 minutes. For positioning workflows that
need satellite positions at every observation epoch (typically 1 Hz),
`interpolate_sp3` runs an order-10 Lagrange interpolator.

```python
import numpy as np
from datetime import datetime

sp3 = rp.load_sp3("tests/data/igs19362.sp3c")

# Interpolate G05 at three explicit epochs.
t0 = sp3.time.values[5].astype("datetime64[us]").astype(datetime)
queries = np.array(
    [t0, t0 + np.timedelta64(60, "s"), t0 + np.timedelta64(300, "s")],
    dtype="datetime64[ns]",
)
interp = rp.interpolate_sp3(sp3, queries)
print(interp.position.sel(sv="G05").values)   # (3 queries, 3 ECEF), km
```

The default order is 10, per the IGS recommendation. To override:

```python
interp = rp.interpolate_sp3(sp3, queries, order=8)
```

Order-10 is the standard choice for almost every workflow. The IGS file is
sampled at 900 seconds, so a 10-point window spans 2.5 hours. The
Lagrange polynomial is stable in the middle of that window and degrades
at the edges; near the file boundary the function uses a one-sided window
so the interpolation is still well-conditioned.

If your query times fall outside the file's time range, the function
raises `ValueError`. Stitch consecutive SP3 files first.

### Stitching multiple SP3 files

```python
from rinexpy.sp3 import stitch_sp3

sp3 = stitch_sp3("igs01.sp3", "igs02.sp3", "igs03.sp3")
```

The resulting Dataset has the same shape as a single-file load, just with
a longer time axis. Duplicate epochs at file boundaries are dropped.

### Writing SP3

The matching writer round-trips a parsed dataset back to disk in SP3-c or
SP3-d.

```python
from rinexpy.sp3 import write_sp3
write_sp3(sp3, "out.sp3", version="c")
```

Useful when you want to filter, re-sample, or merge SP3 files in a script
and feed the result to another tool. Headers preserve the agency,
coordinate system, and orbit type fields.

### NetCDF round-trip

SP3 also goes through `rp.load(..., out="out.nc")`. The NetCDF group is
named `SP3`.

```python
sp3 = rp.load("tests/data/igs19362.sp3c", out="igs19362.nc")
re = rp.load("igs19362.nc")
```

## RINEX clock products

The `.clk` file is the IGS clock product. Per-satellite clock biases at
30-second sampling (final) or 5-minute sampling (rapid). Receiver clock
estimates per station follow the satellite clock records when present.

### Loading

```python
from rinexpy.clk import load_clk

# A real .clk would be loaded with the line below; the bundled fixtures
# do not ship one, so the example below uses an in-line minimal file.
import tempfile
sample = """\
     3.00           CLOCK              GPS                 RINEX VERSION / TYPE
COD                                                         ANALYSIS CENTER
                                                            END OF HEADER
AR ALGO        2024 03 14 00 00  0.000000  2    1.234D-09  3.0D-12
AS G01         2024 03 14 00 00  0.000000  2   -1.000D-04  1.0D-12
AS G02         2024 03 14 00 00  0.000000  2   -2.500D-04  2.0D-12
AS G01         2024 03 14 00 05  0.000000  2   -1.001D-04  1.0D-12
AS G02         2024 03 14 00 05  0.000000  2   -2.501D-04  2.0D-12
"""
p = tempfile.NamedTemporaryFile(suffix=".clk", mode="w", delete=False)
p.write(sample); p.close()

clk = load_clk(p.name)
print(clk)
print(clk.attrs["stations"])           # ['ALGO']
```

The Dataset schema is:

```
Dimensions:  (time: 2, sv: 2)
Coordinates:
  * time  (time) datetime64[ns] 2024-03-14 ... 2024-03-14T00:05:00
  * sv    (sv)   <U3 'G01' 'G02'
Data variables:
    bias  (time, sv) float64 -0.0001 -0.00025 -0.0001001 -0.0002501
Attributes:
    rinextype:  clk
    stations:   ['ALGO']
```

`bias` is in seconds. Station clock biases (the `AR` records) are stored
in the `stations` attribute as a sorted list of identifiers; their per-epoch
biases come back in a sibling Dataset if you go through the lower-level
`clk` parser directly, but for most positioning workflows the satellite
clocks are what you want.

### Interpolating

For positioning at sub-30-second epochs, `interpolate_clk` does a linear
interpolation per SV. Linear is appropriate because the published clock
products are themselves smoothed.

```python
from datetime import datetime
from rinexpy.clk import interpolate_clk

bias_s = interpolate_clk(clk, "G01", datetime(2024, 3, 14, 0, 2, 30))
print(bias_s)             # -0.00010005 seconds
```

If you need bulk interpolation, vectorise it yourself with
`np.interp(...)` against `clk.bias.sel(sv=...)`.

## Using SP3 + CLK in positioning

The combination of an SP3 file and a matching CLK file is what controls a PPP
solution. The PPP driver in `rinexpy.ppp` takes both directly:

```python
from rinexpy.ppp import ppp_solve

obs = rp.load("data/RNX/STAT00BRA_R_20231560000_24H_01S_MO.rnx.gz")
sp3 = rp.load_sp3("data/SP3/IGS0OPSFIN_20231560000_01D_15M_ORB.SP3")
clk = load_clk("data/CLK/IGS0OPSFIN_20231560000_01D_30S_CLK.CLK")

out = ppp_solve(obs, sp3, clk, initial_position_ecef=tuple(obs.attrs["position"]))
print(out["position"], "+/-", out["position_sigma_m"])
```

The driver interpolates both per epoch and forms the iono-free combination
from the observation set. See [Precise point positioning](../positioning/ppp.md).

## Choosing an SP3 / CLK source

The IGS publishes three product latencies, each in its own combined file.

| Product | Latency | Typical filename | Orbit / clock RMS |
| --- | --- | --- | --- |
| Ultra-rapid (predicted half) | a few hours into the future | `IGU` or `IGS0OPSULT_*_ULT.SP3` | ~5 cm orbit, 3 ns clock |
| Ultra-rapid (observed half) | 3 hours after the day | same files, second half | ~3 cm, 150 ps |
| Rapid | 17 hours | `IGR` or `IGS0OPSRAP` | ~2.5 cm, 75 ps |
| Final | 12-18 days | `IGS` or `IGS0OPSFIN` | ~2.5 cm, 75 ps |

For real-time work the `IGS-SSR` corrections (RTCM3 messages 1057-1068 and
1240-1263, plus 4076) refine a recent broadcast or rapid solution. The
`SSRCorrections` class in `rinexpy.ssr` takes in those messages and the
`ppp_solve` driver accepts `ssr=` in place of `clk=`.

## Performance

The SP3 reader pre-allocates a NaN-filled buffer sized to the maximum
satellite count from the header, then writes positions and clocks into it
in a single pass. This is a slight win over the upstream `np.empty()`
followed by per-SV writes, and is also more correct: SVs that are listed in
the header but absent from a given epoch read back as NaN, not as
uninitialised memory.

main numbers from the [benchmarks page](../internals/benchmarks.md):

| File | georinex | `rinexpy` | Speedup |
| --- | --- | --- | --- |
| `example1.sp3a` (3 KB) | 1.2 ms | 0.7 ms | 1.6x |
| `igs19362.sp3c` (225 KB) | 3.0 ms | 2.5 ms | 1.2x |
| `minimal.sp3d` (11 KB) | 1.2 ms | 0.8 ms | 1.6x |

## Bundled fixtures

| File | Format | What it covers |
| --- | --- | --- |
| `example1.sp3a`, `example1.sp3a.gz` | SP3-a | three-epoch test file |
| `example2.sp3a` | SP3-a | second variant |
| `header.sp3` | SP3-c | header-only test |
| `igs19362.sp3c` | SP3-c | full-day GPS+GLO IGS file |
| `minimal.sp3c` | SP3-c | minimum-record fixture |
| `minimal.sp3d` | SP3-d | SP3 version d test |
| `truncated.sp3` | SP3-c | exercises the trunc/EOF path |
| `blank.sp3` | SP3-c | empty file |

## Related pages

- [RINEX navigation files](rinex-nav.md): broadcast alternative to SP3.
- [Precise point positioning](../positioning/ppp.md): the use case.
- [SSR corrections](../corrections/ssr.md): real-time alternative to CLK.
- [Module index](../reference/modules.md): full module API surface.
