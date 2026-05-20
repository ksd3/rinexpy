# Tides and station displacements

The Earth deforms under the gravitational pull of the Sun and Moon and
under its own rotation. For PPP-class positioning these tidal
displacements matter at the centimetre level. The IERS Conventions 2010
list four contributions: solid Earth tides, the pole tide, the ocean
pole tide, and ocean tide loading from the sea-surface mass redistribution.

`rinexpy` implements all four. The implementations live in
`rinexpy.tides` and `rinexpy.otl`.

## Solid Earth tides

The dominant displacement is from the solid Earth tide: the elastic
response of the Earth's crust to the Sun-Moon gravity field. Peak
amplitude is roughly 50 cm radial and 20 cm horizontal.

The IERS 2010 model has two steps. Step 1 is the nominal degree-2 response.
Step 2 is a frequency-dependent correction in the diurnal and long-period
bands.

### Step 1

```python
import numpy as np
from rinexpy.tides import (
    sun_position_ecef,
    moon_position_ecef,
    solid_earth_tide_displacement,
    H2_LOVE,
    L2_SHIDA,
)
from datetime import datetime

epoch = datetime(2024, 3, 14, 12)
station_ecef = np.array([4789028.4701, 176610.0133, 4195017.031])

sun = sun_position_ecef(epoch)            # ECEF of the Sun, in metres
moon = moon_position_ecef(epoch)

displacement = solid_earth_tide_displacement(
    station_ecef,
    sun_ecef=sun,
    moon_ecef=moon,
    epoch=epoch,
    h2=H2_LOVE,                          # 0.6078 (default)
    l2=L2_SHIDA,                          # 0.0847 (default)
)
print(displacement)                       # (dx, dy, dz) in ECEF, metres
```

`solid_earth_tide_displacement` evaluates the degree-2 elastic response
using the Love (`h2`) and Shida (`l2`) numbers. The constants exposed at
module level are the IERS 2010 recommended values.

`sun_position_ecef` and `moon_position_ecef` are approximate low-precision
ECEF positions (Vallado low-precision formulas). The sun is good to
about 1 arcminute; the moon is good to a few hundred km, which is enough
for the tide model.

For higher precision sun/moon ECEF positions, supply them yourself from
an ephemeris like JPL DE440. The tide-displacement function does not
care where the positions come from.

### Step 2 (frequency-dependent)

The diurnal-band correction has 31 tidal lines; the long-period band has
5 lines. Together they add a few millimetres to the step-1 displacement.

```python
from rinexpy.tides import (
    step2_diurnal_displacement,
    step2_long_period_displacement,
    step2_displacement,
)

diurnal = step2_diurnal_displacement(station_ecef, epoch)
long_period = step2_long_period_displacement(station_ecef, epoch)
total_step2 = step2_displacement(station_ecef, epoch)
```

The full step-1 + step-2 displacement is

```python
total = solid_earth_tide_displacement(station_ecef, epoch=epoch) + step2_displacement(station_ecef, epoch)
```

For most PPP workflows step 2 is below the relevant precision and step 1
alone is sufficient.

## Pole tide

As the Earth wobbles on its rotation axis (the "polar motion"), the
deformation of the rotating ellipsoid shifts. The pole tide displacement
is roughly 5 mm at mid-latitudes, with most of the power at the Chandler
period (about 433 days) and the annual period.

```python
from rinexpy.tides import pole_tide_displacement
from rinexpy.eop import load_eop, interp_eop

eop = load_eop("/path/to/EOP_C04_14.62-NOW.IAU2000A.txt")
displacement = pole_tide_displacement(station_ecef, eop, epoch)
```

`displacement` is the ECEF vector in metres. The function uses the
IERS 2010 §7.1.4 model.

## Ocean pole tide

The ocean responds to polar motion separately, redistributing mass that
loads the seafloor. The resulting deformation at coastal stations is at
the millimetre level.

```python
from rinexpy.tides import ocean_pole_tide_displacement

ocean_pole = ocean_pole_tide_displacement(station_ecef, eop, epoch)
```

The model is from IERS 2010 §7.1.5.

## Ocean tide loading (OTL)

The ocean tide itself loads the seafloor. The resulting crustal
displacement is the largest non-solid-Earth tidal signal at coastal
stations: up to 5 cm in the diurnal and semi-diurnal bands.

OTL parameters are site-specific. The standard format is the Scherneck
BLQ file, available from the
[Onsala Space Observatory's online OTL service](http://holt.oso.chalmers.se/loading/).

### Reading a BLQ file

```python
from rinexpy.otl import read_blq

coeffs = read_blq("station.blq")
print(list(coeffs.keys()))            # one entry per station in the file
print(coeffs["STAT01"].keys())        # 'amplitudes_m', 'phases_deg'
```

Each station's coefficients carry:

| Key | Type | Notes |
| --- | --- | --- |
| `amplitudes_m` | `ndarray` shape `(3, 11)` | per-component (radial / NS / EW), per-constituent amplitude in metres |
| `phases_deg` | `ndarray` shape `(3, 11)` | matching phases in degrees |

The 11 constituents are the IERS short-list `M2, S2, N2, K2, K1, O1, P1,
Q1, Mf, Mm, Ssa`.

### Computing the displacement

```python
from rinexpy.otl import ocean_tide_loading_displacement, ocean_tide_loading_ecef

# In the local east-north-up frame:
enu = ocean_tide_loading_displacement(coeffs["STAT01"], epoch)
print(enu)            # (east_m, north_m, up_m)

# In ECEF, rotated to the station's local frame:
ecef_disp = ocean_tide_loading_ecef(coeffs["STAT01"], station_ecef, epoch)
```

The model evaluates each constituent's amplitude and phase at the
requested epoch using the standard Doodson argument tables.

For PPP at coastal stations OTL must be applied. For inland stations it
is typically negligible.

## Putting it together

The full IERS-2010 station-displacement chain for PPP is:

1. Apply the solid Earth tide displacement (step 1 + step 2).
2. Apply the pole tide displacement.
3. Apply the ocean pole tide displacement.
4. Apply the ocean tide loading displacement.

Each correction is added to the station's nominal a priori ECEF position
before forming the iono-free combinations.

```python
total_displacement = (
    solid_earth_tide_displacement(station_ecef, epoch=epoch)
    + step2_displacement(station_ecef, epoch)
    + pole_tide_displacement(station_ecef, eop, epoch)
    + ocean_pole_tide_displacement(station_ecef, eop, epoch)
    + ocean_tide_loading_ecef(blq_entry, station_ecef, epoch)
)
corrected_station_ecef = station_ecef + total_displacement
```

The `ppp_solve` driver does not currently apply the tide corrections
automatically (since the BLQ file is operator-specific). For cm-class PPP
apply them in your own pre-processing step, before calling the driver.

## Constants

```python
from rinexpy.tides import (
    GM_SUN, GM_MOON, GM_EARTH, R_EARTH,
    H2_LOVE, L2_SHIDA,
)

print(GM_SUN)       # 1.32712442099e20 m^3/s^2
print(GM_MOON)      # 4.9048695e12 m^3/s^2
print(GM_EARTH)     # 3.986004418e14 m^3/s^2
print(R_EARTH)      # 6378137.0 m
print(H2_LOVE)      # 0.6078
print(L2_SHIDA)     # 0.0847
```

## Related pages

- [EOP and Earth orientation](eop.md): the EOP series used by the pole tide.
- [Atmosphere products](../formats/atmosphere-products.md): the EOP file reader.
- [Precise point positioning](../positioning/ppp.md): where these corrections feed in.
