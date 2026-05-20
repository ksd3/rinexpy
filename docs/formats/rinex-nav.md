# RINEX navigation files

A RINEX navigation file is the broadcast ephemeris: the satellite-by-satellite
clock parameters and orbit elements that the satellite itself transmits in
the GNSS data message. There is one record per satellite per ephemeris
update, typically every two hours for GPS, or every thirty minutes for
Galileo. The file is enough for a code-only fix at the ten-metre level if
you have nothing else.

`rinexpy` reads RINEX 2.x, RINEX 3.x, and the structured records of RINEX 4
NAV files.

## The dataset schema

```python
import rinexpy as rp
nav = rp.load("tests/data/brdc2800.15n")
print(nav)
```

For the bundled file, the output is:

```
<xarray.Dataset>
Dimensions:  (time: 25, sv: 32)
Coordinates:
  * time     (time) datetime64[ms] 2015-10-07 ... 2015-10-08
  * sv       (sv) <U3 'G01' 'G02' 'G03' 'G05' 'G06' 'G07' 'G08' ...
Data variables:
    SVclockBias    (time, sv) float64 ...
    SVclockDrift   (time, sv) float64 ...
    ...
    sqrtA          (time, sv) float64 ...
    Eccentricity   (time, sv) float64 ...
    Omega0         (time, sv) float64 ...
    M0             (time, sv) float64 ...
    omega          (time, sv) float64 ...
    Toe            (time, sv) float64 ...
Attributes:
    version:        2.1
    filetype:       N
    rinextype:      nav
    ionospheric_corrections: ...
```

The variable set differs by satellite system, because each system has its
own ICD. GPS, Galileo, QZSS, BeiDou, and NavIC broadcast Keplerian
elements with extra harmonic correction coefficients. GLONASS broadcasts
direct ECEF position, velocity, and acceleration. SBAS broadcasts a
similar ECEF triple.

The canonical Keplerian field set is:

| Field | Meaning | Units |
| --- | --- | --- |
| `SVclockBias` | clock bias `a_f0` | seconds |
| `SVclockDrift` | clock drift `a_f1` | s/s |
| `SVclockDriftRate` | clock drift rate `a_f2` | s/s² |
| `IODE` | issue of data, ephemeris | dimensionless |
| `Crs`, `Crc` | sine and cosine orbital-radius correction | metres |
| `DeltaN` | mean motion correction `Delta n` | radians/s |
| `M0` | mean anomaly at ref epoch | radians |
| `Cuc`, `Cus` | sine and cosine argument-of-latitude correction | radians |
| `Eccentricity` | orbit eccentricity `e` | dimensionless |
| `sqrtA` | square root of semi-major axis | sqrt(m) |
| `Toe` | time of ephemeris (sec of week) | seconds |
| `Cic`, `Cis` | sine and cosine inclination correction | radians |
| `Omega0` | RAAN at start of week | radians |
| `Io` | inclination at `Toe` | radians |
| `omega` | argument of perigee | radians |
| `OmegaDot` | RAAN rate | radians/s |
| `IDOT` | inclination rate | radians/s |
| `CodesL2`, `L2Pflag`, `health`, `accuracy`, `TGD`, `IODC` | misc | various |

The GLONASS field set is:

| Field | Meaning | Units |
| --- | --- | --- |
| `X`, `Y`, `Z` | ECEF position | metres |
| `dX`, `dY`, `dZ` | ECEF velocity | metres / s |
| `dX2`, `dY2`, `dZ2` | ECEF acceleration (luni-solar perturbations) | metres / s² |
| `SVclockBias`, `SVrelFreqBias` | satellite clock parameters | s, s/s |
| `FreqNum` | GLONASS channel number, integer | dimensionless |
| `MessageFrameTime`, `health` | misc | various |

The original RINEX 2 GLONASS broadcasts position in km, velocity in km/s,
acceleration in km/s². `rinexpy` converts to SI metres in the read path.

## Reading

The same `load` dispatcher handles every variant.

```python
nav2 = rp.load("tests/data/brdc2800.15n")              # RINEX 2 GPS
nav3 = rp.load("tests/data/CEDA00USA_R_20182100000_01D_MN.rnx.gz")  # RINEX 3 mixed
```

If you want the version-specific entry point, the underlying functions are
exported.

```python
from rinexpy.nav2 import rinexnav2, navtime2
from rinexpy.nav3 import rinexnav3, navtime3

ds = rinexnav2("tests/data/brdc2800.15n")
ds = rinexnav3("tests/data/CEDA00USA_R_20182100000_01D_MN.rnx.gz", use={"G", "E"})
```

The RINEX 2 NAV reader does not honour the `use=` filter because a RINEX 2
NAV file holds only one constellation per file.

`navtime2` / `navtime3` return the sorted unique epoch times as
`datetime64[ms]` without loading the data section.

## Filtering

`tlim=` drops records outside a window during the parse. For RINEX 3 NAV,
`use=` drops constellations.

```python
nav = rp.load(
    "tests/data/CEDA00USA_R_20182100000_01D_MN.rnx.gz",
    use={"E"},                # only Galileo (this file is Galileo-only)
    tlim=("2018-07-29T00:00", "2018-07-29T06:00"),
)
```

## Duplicate ephemerides

A receiver may receive multiple copies of an ephemeris for the same
satellite at almost the same Toe. The reader keeps each one as a separate
row with a numeric suffix on the SV label: `E04`, `E04_1`, `E04_2`, and so
on. The bookkeeping mirrors the upstream georinex convention so existing
analysis code keeps working.

```python
print([sv for sv in nav.sv.values if sv.startswith("E04")])
# ['E04', 'E04_1', 'E04_2']
```

The numeric suffix is per-Toe; if you want to deduplicate, group by Toe and
keep the row with the highest IODE or the latest reception time.

## Keplerian to ECEF

`keplerian2ecef` converts the broadcast elements to ECEF coordinates. The
output X, Y, Z values are NumPy arrays in metres.

```python
import numpy as np
nav = rp.load("tests/data/brdc2800.15n")
sv = nav.sel(sv="G07")              # one SV slice
X, Y, Z = rp.keplerian2ecef(sv)
print(X[:3])                          # first three values, metres
```

The function expects a Dataset slice that still has the `time` dimension.
GPS and Galileo are full Keplerian. GLONASS and SBAS broadcast direct ECEF
and are returned unchanged. BeiDou D2 (GEO) and QZSS follow the GPS-style
elements with a small correction in the orbit-propagation step.

If you need the position at a specific epoch rather than at the broadcast
Toe, evaluate the elements at the epoch you want. The standard procedure
(propagate the mean anomaly forward by `(t - Toe)`, apply the harmonic
corrections, rotate by `OmegaDot * (t - Toe)`) is what `keplerian2ecef`
does for the broadcast Toe. For arbitrary `t`, build the inputs yourself or
use the SP3 interpolator if you have one.

## Reading just the header

```python
hdr = rp.rinexheader("tests/data/brdc2800.15n")
print(hdr.keys())                     # RINEX header labels
print(hdr["LEAP SECONDS"])
print(hdr["ION ALPHA"])               # Klobuchar coefficients for GPS NAV2
```

For RINEX 3 NAV the equivalent records are:

```python
hdr = rp.rinexheader(
    "tests/data/CEDA00USA_R_20182100000_01D_MN.rnx.gz",
)
print(hdr["IONOSPHERIC CORR"])        # iono coefficients per system
print(hdr["TIME SYSTEM CORR"])        # system-to-system offsets
```

These come back as parsed sub-dicts when possible.

## RINEX 4 NAV

RINEX 4 navigation files extend the format with structured records for
system time offsets, Earth orientation parameters, and ionospheric models
(Klobuchar, NeQuick-G, BDGIM). The dedicated reader returns each record
type in its own list.

```python
from rinexpy.nav4 import load_nav4

out = load_nav4("BRDC00WRD_S_20231560000_01D_MN.rnx")
print(out.keys())              # dict_keys(['EPH', 'STO', 'EOP', 'ION'])
```

The structure of each record type:

```python
for sto in out["STO"]:
    # System time offset: STO type, A0/A1/A2 polynomial, ref time
    print(sto["sv"], sto["message_type"], sto["A0_s"])

for eop in out["EOP"]:
    # Earth orientation: pole motion, UT1-UTC offsets
    print(eop["sv"], eop["ref_time"], eop["PM_X"], eop["PM_Y"], eop["delta_UT1"])

for ion in out["ION"]:
    if ion["model"] == "KLOB":
        # Klobuchar coefficients
        print(ion["alpha"], ion["beta"])
    elif ion["model"] == "NEQG":
        # NeQuick-G coefficients for Galileo
        print(ion["a0"], ion["a1"], ion["a2"])
    elif ion["model"] == "BDGIM":
        # BeiDou Global Ionospheric Map
        print(ion["coefficients"])

for eph in out["EPH"]:
    # The standard broadcast ephemeris records
    print(eph["sv"], eph["toe_s"], eph["sqrt_A"])
```

The matching round-trip writer is in `rinexpy.nav_writer`.

```python
from rinexpy.nav_writer import write_nav3
write_nav3(nav, "out.rnx")
```

Currently supports the Keplerian systems (G, E, C, J). GLONASS / SBAS
round-trip is on the roadmap.

## Broadcast group delay (TGD)

For single-frequency SPP the broadcast TGD removes the satellite
inter-frequency hardware bias on the L1 measurement. Pull it from a NAV
dataset and feed it to `spp_solve`.

```python
from rinexpy.positioning import tgd_from_nav, apply_tgd_correction
from datetime import datetime

nav = rp.load("tests/data/brdc2800.15n")
tgd_map = tgd_from_nav(nav, datetime(2015, 10, 7, 12))
# {'G01': 8.4e-9, 'G02': -1.2e-8, ...}

# Pass to spp_solve directly:
sol = rp.spp_solve(sv_ecef, pseudoranges_m,
                   sv_labels=svs, tgd_map=tgd_map, tgd_gamma=1.0)
# Or apply manually:
corrected = apply_tgd_correction(pseudoranges_m, svs, tgd_map, gamma=1.0)
```

The `tgd_gamma` factor depends on which frequency the pseudorange is on:
1.0 for L1, `(f1/f2)**2 ≈ 1.647` for L2, 0.0 for the iono-free combination.

For BeiDou the broadcast TGD is in two parts (`TGD1` for B1I, `TGD2` for
B2I); for Galileo it is the BGD on E5a or E5b. Pass `field="TGD1"` etc.

## NAV from RINEX 4

If you need RINEX 4 NAV records in the same Dataset shape as RINEX 3, use
`rinexnav3` against the file directly. The RINEX 4 EPH record set is
backward-compatible.

## Performance

The RINEX 3 NAV reader is the largest speedup over the upstream
implementation; the per-SV `xarray.merge` pattern is gone, and the parser
collects flat tuples into a single Dataset at the end. main numbers
from the [benchmarks page](../internals/benchmarks.md):

| File | georinex | `rinexpy` | Speedup |
| --- | --- | --- | --- |
| `demo_nav3.17n` (RINEX 3 NAV, small) | 18 ms | 0.5 ms | 33x |
| `ELKO...MN.rnx.gz` (RINEX 3 NAV, 188 KB) | 1003 ms | 31 ms | 33x |
| `demo.10n` (RINEX 2 NAV, small) | 3.8 ms | 0.5 ms | 8x |
| `brdc2800.15n` (RINEX 2 NAV, 263 KB) | 6.7 ms | 3.2 ms | 2x |

## Bundled fixtures

| File | What it covers |
| --- | --- |
| `demo.10n` | minimal RINEX 2.11 NAV, GPS, 1 epoch |
| `demo_nav3.17n` | minimal RINEX 3.04 NAV, GPS, 1 epoch |
| `brdc2800.15n` | full-day RINEX 2 GPS NAV, 263 KB |
| `brdc2420.18n.gz` | gzipped RINEX 2 GPS NAV |
| `CEDA00USA_R_20182100000_01D_MN.rnx.gz` | RINEX 3 multi-system NAV |
| `galileo3.15n` | RINEX 3 Galileo NAV |
| `qzss_nav3.14n` | RINEX 3 QZSS NAV |
| `BRDM00DLR_R_20130010000_01D_MN.rnx` | RINEX 3 multi-system NAV, large |
| `BRDC00IGS_R_20201360000_01D_MN.rnx` | RINEX 3 IGS combined NAV |

## Related pages

- [RINEX observation files](rinex-obs.md): the matching observation reader.
- [SP3 and clock products](sp3-clk.md): precise alternative to broadcast.
- [Raw nav subframes](nav-subframes.md): the underlying bit-level decoders.
- [Atmospheric models](../corrections/atmosphere.md): Klobuchar from NAV2 / 3 / 4.
- [Single-point positioning](../positioning/spp.md): using NAV in a fix.
