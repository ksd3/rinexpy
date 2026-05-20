# Atmosphere products

`rinexpy` reads the five standard atmospheric-correction file formats from
the IGS: IONEX (global ionospheric maps), ANTEX (antenna phase-centre
variations), RINEX MET (surface meteorology), IERS EOP C04 (Earth-orientation
parameters), and GPT2w (empirical surface met grid).

Each one feeds into the positioning layer in a different place.

| Format | Purpose | Used by |
| --- | --- | --- |
| IONEX `.inx` | global TEC maps | `slant_tec` for L1 single-freq SPP |
| ANTEX `.atx` | antenna phase-centre variation | `ppp_solve` for PCV correction |
| RINEX MET `.m` | site surface met | Saastamoinen with real `(P, T, e)` |
| IERS EOP C04 | Earth orientation | ECEF/ECI rotation for tides |
| GPT2w grid | empirical met for VMF1 | `ppp_solve` for ZHD/ZWD |
| SINEX-BIAS `.BSX` | DCBs / OSBs | `ppp_solve` for code bias |

## IONEX

IONEX (Ionosphere Map Exchange Format) is the IGS file format for global
total electron content (TEC) maps. One TEC map per epoch, sampled on a
regular lat/lon grid (typically 5° longitude × 2.5° latitude × 2 hours
in time).

### Loading

```python
from rinexpy.ionex import load_ionex

# A real IONEX file would go here. The bundled fixture is built from a
# tiny synthetic sample.
ds = load_ionex("path/to/CODG2860.24I")
print(ds)
```

The Dataset has the schema:

| Coord | Type | Notes |
| --- | --- | --- |
| `time` | `datetime64[ns]` | epoch of each TEC map |
| `lat` | `float64` | latitude grid points in degrees |
| `lon` | `float64` | longitude grid points in degrees |

The single data variable `tec` is shaped `(time, lat, lon)` and reports the
vertical TEC in TECU (1 TECU = 10^16 electrons / m²).

### Interpolating

```python
from datetime import datetime
from rinexpy.ionex import interp_tec, slant_tec

vtec = interp_tec(ds, lat_deg=40.0, lon_deg=-3.0,
                  epoch=datetime(2024, 10, 12, 12, 0, 0))
# Vertical TEC at the (lat, lon, time) sample, in TECU.

stec_l1 = slant_tec(vtec, el_deg=30.0)
# Slant TEC mapped through the 350 km thin-shell mapping function.
```

For an L1 single-frequency receiver, the ionospheric delay in metres is
`40.3 / f^2 * STEC_in_electrons_per_m2`, which works out to roughly
`0.16 m per TECU`. So a 30 TECU slant value is about 5 m of delay on L1.

### IONEX schema details

The header tracks the grid spacing, the time spacing, the height of the
single-layer model (default 350 km), the mapping function, and the
exponent used to scale the integer cell values. The reader honours each
field; if the file declares a non-standard mapping height, the resulting
Dataset records it in `ds.attrs["height_m"]`.

## ANTEX

ANTEX (Antenna Exchange Format) is the IGS file for antenna phase-centre
offsets (PCO) and phase-centre variations (PCV) per frequency. Without
ANTEX, the carrier-phase observable at a fixed elevation is biased by the
antenna's elevation-dependent gain pattern, which limits PPP to roughly
decimetre level.

### Loading

```python
from rinexpy.antex import load_antex, find_antenna, apply_antex_pcv

entries = load_antex("igs20.atx")
print(len(entries))             # number of antennas in the file

# Find the entry for one antenna by model name (and optionally serial):
ant = find_antenna(entries, "TRM59800.00     NONE")
print(ant["type"])
print(list(ant["frequencies"]))   # ['G01', 'G02', 'R01', ...]
```

Each entry is a dict:

| Key | Type | Notes |
| --- | --- | --- |
| `type` | `str` | antenna model + radome name |
| `serial` | `str` | serial number; can be empty |
| `frequencies` | `dict[str, dict]` | per-frequency entries keyed by RINEX 3 freq ID |

Each per-frequency dict has:

| Key | Type | Notes |
| --- | --- | --- |
| `north`, `east`, `up` | `float` | PCO offsets in millimetres |
| `noazi` | `ndarray` | NOAZI PCV vector |
| `pcv` | `ndarray` (optional) | 2-D PCV grid (azi, zen), when the entry has DAZI > 0 |

### Applying a correction

```python
correction_m = apply_antex_pcv(ant, freq_id="G01", el_deg=30.0, az_deg=120.0)
```

When `dazi > 0` and `az_deg=` is provided, the function does a bilinear
interpolation on the 2-D grid. Otherwise it interpolates the NOAZI vector.
The result is the per-observation PCV correction in metres.

The `apply_antex_pcv` call is what the PPP driver does internally per
satellite per band. You only need to call it directly if you are running a
custom positioning workflow.

### Calibration

If you have residuals from a calibrated antenna, you can generate a valid
ANTEX entry with `calibrate_pcv`. The output round-trips through
`load_antex`.

```python
from rinexpy.antex_calibrate import calibrate_pcv, write_antex

entry = calibrate_pcv(
    residuals_m, elevation_rad, azimuth_rad,
    antenna_type="MYANT_NONE",
    serial="SN001",
    frequency="G01",
    dazi_deg=5.0,
    dzen_deg=5.0,
)
write_antex([entry], "my-antenna.atx")
```

The calibrator bins the residuals on the standard ANTEX 5°/5° grid, averages
per cell, and back-fills sparse cells with the azimuth-averaged NOAZI
value.

## RINEX MET

RINEX MET (Meteorological Observation File) is the per-site surface met
record. Pressure, temperature, humidity, and any optional sensor metadata.
Most permanent stations log a MET file alongside their OBS file.

```python
from rinexpy.met import load_met

ds = load_met("path/to/STAT00BRA_R_20231560000_24H_01H_MM.rnx")
print(ds)
```

The Dataset is indexed on `time` and has data variables `PR` (pressure,
hPa), `TD` (dry-bulb temperature, °C), `HR` (relative humidity, %), and
optionally `WD`/`WS` (wind direction and speed) and `RI`/`HI`/`ZW`/`ZD`/`ZT`
depending on which sensors were configured.

The output is suitable for feeding into Saastamoinen with real met instead
of the standard-atmosphere defaults.

## IERS Earth orientation parameters

The EOP C04 file is published by the International Earth Rotation and
Reference Systems Service (IERS). It carries daily values of polar motion
(x, y), UT1-UTC, length of day, and the celestial pole offsets (dX, dY)
in the IAU 2006 convention.

```python
from rinexpy.eop import load_eop, interp_eop
from datetime import datetime

eop = load_eop("/path/to/EOP_C04_14.62-NOW.IAU2000A.txt")
out = interp_eop(eop, datetime(2024, 3, 14, 12))
print(out)
# {'PM_X_arcsec': 0.115, 'PM_Y_arcsec': 0.302, 'delta_UT1_s': -0.029, ...}
```

The fields are:

| Key | Units | Notes |
| --- | --- | --- |
| `PM_X_arcsec`, `PM_Y_arcsec` | arcseconds | polar motion in the CIO frame |
| `delta_UT1_s` | seconds | UT1 minus UTC |
| `LOD_s` | seconds | length of day excess |
| `dX_arcsec`, `dY_arcsec` | arcseconds | celestial-pole nutation offsets |

These feed into the ECEF/ECI rotation used by the solid-Earth tide model
and the satellite-position transform inside the PPP pipeline.

## GPT2w

GPT2w is the empirical surface-meteorology grid published by the Vienna
group (Boehm et al., 2014). One gridded set per day-of-year: pressure,
temperature, water-vapour pressure, lapse rate, geoid undulation, and the
hydrostatic / wet mapping coefficients `a_h` / `a_w` for VMF1. The grid
file (`gpt2_5w.grd`, about 2 MB) is fetched from the
[Vienna VMF Data Server](https://vmf.geo.tuwien.ac.at/codes/) and supplied
by the user (not included with rinexpy).

### Loading

```python
from rinexpy.gpt2w import load_gpt2w_grid, gpt2w
from datetime import datetime

grid = load_gpt2w_grid("/path/to/gpt2_5w.grd")
met = gpt2w(grid,
            lat_deg=40.0,
            lon_deg=-3.0,
            epoch=datetime(2024, 3, 14),
            altitude_m=100.0)
print(met)
# {'pressure_hpa': 1011.5, 'temperature_k': 286.4, 'e_hpa': 7.8,
#  'a_h': 0.001253, 'a_w': 0.000601, 'T_lapse': -6.5, 'undulation_m': 47.2}
```

The `(pressure, temperature, e)` triple feeds straight into `saastamoinen`.
The `a_h`/`a_w` values feed into `vmf1` as the elevation mapping
coefficients.

### Using it with the troposphere chain

```python
from rinexpy.geodesy import saastamoinen, vmf1

zhd_m = saastamoinen(
    el_deg=90.0,             # zenith
    altitude_m=100.0,
    pressure_hpa=met["pressure_hpa"],
    temperature_k=met["temperature_k"],
    humidity_e_hpa=met["e_hpa"],
)
m_h, m_w = vmf1(met["a_h"], met["a_w"],
                el_deg=15.0, lat_deg=40.0, altitude_m=100.0, doy=74)
slant_dry = zhd_m * m_h
```

`m_h` and `m_w` are the hydrostatic and wet mapping factors for the
elevation. The full zenith delay is the sum of zenith hydrostatic delay
(ZHD, from Saastamoinen) and zenith wet delay (ZWD, from your filter).

The PPP driver does this composition internally when you pass
`gpt2w_grid=` to `ppp_solve`. See [Precise point positioning](../positioning/ppp.md).

## SINEX-BIAS (DCB / OSB)

SINEX-BIAS is the file format the IGS uses for differential code biases
(DCBs) and observable-specific signal biases (OSBs). These corrections are
the per-satellite hardware bias between the L1 and L2 code observables (or
between any two RINEX 3 observation codes). Without DCB correction, the
iono-free combination in PPP has a few-decimetre satellite-dependent
bias.

### Loading

```python
from rinexpy.dcb import read_bsx, get_bias, correct_pseudorange

records = read_bsx("CAS0MGXRAP_20231560000_01D_01D_DCB.BSX")
print(records[0].keys())
# dict_keys(['bias_type', 'prn', 'station', 'obs1', 'obs2', 'start',
#            'end', 'unit', 'value'])

# Apply one satellite OSB to a pseudorange:
corrected = correct_pseudorange(
    pseudorange_m=24123456.789,
    prn="G05",
    obs_code="C1W",
    records=records,
    epoch=datetime(2023, 6, 5, 12),
)

# Or look up one bias directly:
bias_m = get_bias(records, prn="G05", obs1="C1W", obs2="C2W",
                  epoch=datetime(2023, 6, 5, 12))
```

The records are:

| Key | Type | Notes |
| --- | --- | --- |
| `bias_type` | `"OSB"` or `"DSB"` | observable-specific or differential |
| `prn` | `str` | satellite PRN like `"G05"`, or empty for receiver biases |
| `station` | `str` | 9-character station identifier, or empty |
| `obs1`, `obs2` | `str` | RINEX 3 observation codes |
| `start`, `end` | `datetime` | validity window |
| `unit` | `"ns"` or `"cyc"` | wire unit (always converted to metres in `value`) |
| `value` | `float` | bias in metres |

`get_bias` matches on the tuple `(prn or station, obs1, obs2, epoch)` and
returns `None` if no record fits.

### Auto-download

`dcb_download.auto_load_dcb` picks the right source for the date:

| Date | Source | Format |
| --- | --- | --- |
| 2017 onwards | IGS BKG mirror | daily SINEX-BIAS from CAS or DLR MGEX |
| pre-2017 | AIUB FTP mirror | monthly CODE P1-P2, P1-C1, P2-C2 |

```python
from datetime import datetime
from rinexpy.dcb_download import auto_load_dcb

records = auto_load_dcb(datetime(2024, 4, 15))   # CAS daily MGEX
records = auto_load_dcb(datetime(2010, 6, 15))   # CODE monthly P1-P2
```

Files are cached under `~/.cache/rinexpy/dcb/`. The CDDIS source is connected
but requires a NASA Earthdata Login in `~/.netrc`. The AIUB mirror is the
default for the pre-2017 path and is anonymous HTTP.

### Legacy CODE format

```python
from rinexpy.dcb import read_code_dcb

records = read_code_dcb("P1P22406.DCB", year=2024, month=6)
```

The reader translates the CODE-style P1/P2/C1/C2 codes into RINEX 3 codes
(C1W, C2W, C1C, C2C) so that the returned records share the schema with
`read_bsx`.

## Related pages

- [Atmospheric models](../corrections/atmosphere.md): Klobuchar, NMF, VMF1, Saastamoinen.
- [Tides and station displacements](../corrections/tides.md): solid Earth, pole, ocean pole.
- [DCB and code biases](../corrections/dcb.md): the broader application path.
- [Precise point positioning](../positioning/ppp.md): the PPP driver that uses these.
