# EOP and Earth orientation

Earth Orientation Parameters (EOP) describe the difference between the
Earth-fixed reference frame (ITRF) and the Earth-rotating-with-stars
frame (ICRF). They are the small (few arcseconds) wobble that the IERS
publishes daily, derived from VLBI, SLR, GNSS, and DORIS observations.

For sub-centimetre PPP, EOP are needed by:

- The ECEF/ECI rotation used by the solid-Earth tide model.
- The polar motion that drives the pole tide and ocean pole tide.
- The UT1-UTC offset for any computation that needs sidereal time.

rinexpy's EOP reader is in `rinexpy.eop`.

## The IERS EOP C04 file

The IERS publishes the C04 series at daily cadence:
[IERS EOP](https://hpiers.obspm.fr/iers/eop/eopc04/). The current
filename is `EOP_C04_14.62-NOW.IAU2000A.txt`, updated continuously.

Each line in the file carries one day's EOP values:

| Field | Units | Notes |
| --- | --- | --- |
| Date | year, month, day | UTC midnight |
| `PM_X` | arcseconds | polar motion x in the CIO frame |
| `PM_Y` | arcseconds | polar motion y in the CIO frame |
| `delta_UT1` | seconds | UT1 - UTC |
| `LOD` | seconds | excess length of day |
| `dX` | arcseconds | celestial-pole nutation offset |
| `dY` | arcseconds | celestial-pole nutation offset |

## Loading

```python
from rinexpy.eop import load_eop, interp_eop

eop = load_eop("/path/to/EOP_C04_14.62-NOW.IAU2000A.txt")
print(eop)
```

The result is an `xarray.Dataset`:

```
Dimensions:    (time: N)
Coordinates:
  * time       (time) datetime64[ns]
Data variables:
    PM_X       (time) float64        arcseconds
    PM_Y       (time) float64        arcseconds
    delta_UT1  (time) float64        seconds
    LOD        (time) float64        seconds
    dX         (time) float64        arcseconds
    dY         (time) float64        arcseconds
```

## Interpolation

The published values are daily. For PPP at 1 Hz cadence you interpolate
linearly between days.

```python
from datetime import datetime

out = interp_eop(eop, epoch=datetime(2024, 3, 14, 12, 0, 0))
print(out)
# {
#   'PM_X_arcsec': 0.115,
#   'PM_Y_arcsec': 0.302,
#   'delta_UT1_s': -0.029,
#   'LOD_s': 0.0008,
#   'dX_arcsec': 0.0002,
#   'dY_arcsec': -0.0003,
# }
```

The function does a linear interpolation per field between the bracketing
days.

## Use in the tide chain

The pole tide and ocean pole tide need polar motion values; the
`pole_tide_displacement` and `ocean_pole_tide_displacement` functions in
`rinexpy.tides` take an EOP Dataset and an epoch.

```python
from rinexpy.tides import pole_tide_displacement, ocean_pole_tide_displacement

epoch = datetime(2024, 3, 14, 12)
station_ecef = np.array([4789028.4701, 176610.0133, 4195017.031])

pole = pole_tide_displacement(station_ecef, eop, epoch)
ocean_pole = ocean_pole_tide_displacement(station_ecef, eop, epoch)
```

For the full IERS-2010 tide chain see [Tides and station displacements](tides.md).

## Use in the ECEF/ECI transform

The geodesy module exposes a low-precision ECEF/ECI rotation that uses the
EOP polar motion and UT1-UTC offsets.

```python
from rinexpy.geodesy import ecef_to_eci, eci_to_ecef
import numpy as np

pos_eci = ecef_to_eci(
    pos_ecef=np.array([sx, sy, sz]),
    epoch=datetime(2024, 3, 14, 12, 0, 0),
    eop=eop,                         # optional; high-precision rotation
)
pos_ecef = eci_to_ecef(pos_eci, epoch, eop=eop)
```

Without `eop=`, the transformation uses a low-precision sidereal time
calculation that is accurate to about 1 second. With `eop=`, it uses the
proper UT1-UTC offset and polar motion, accurate to milliseconds.

For most GNSS workflows the ECEF/ECI transform is needed only inside the
tide model. For relativistic-class work (light-time correction with
satellite-clock parameters expressed in TAI / GPS time), the
high-precision rotation is required.

## Typical magnitudes

Per-component EOP magnitudes:

| Quantity | Range | Period |
| --- | --- | --- |
| Polar motion x, y | ±300 mas (~9 m at the surface) | Chandler 433 days, annual |
| UT1-UTC | ±0.9 s (kept within bounds by leap seconds) | secular drift + variations |
| LOD | ±2 ms | seasonal |
| dX, dY | ±0.5 mas | secular plus annual |

For PPP the only EOP value that matters at the centimetre level on the
position is the polar motion (through the pole tide). UT1-UTC matters for
absolute time tagging but cancels in differential workflows.

## Update cadence

The IERS publishes:

- **Bulletin A:** daily, with predictions out 1 year. Accuracy at the
  prediction horizon is roughly 0.1 mas in polar motion and 0.0001 s in
  UT1-UTC.
- **EOP 14 C04:** the smoothed solution, updated every 5 days, ~7-10
  days lagged. Accuracy is roughly 0.05 mas in polar motion.

For real-time PPP, Bulletin A is the right source. For final PPP, EOP
C04 is the standard.

## Related pages

- [Tides and station displacements](tides.md): the pole-tide application.
- [Atmosphere products](../formats/atmosphere-products.md): the EOP reader.
- [GPS time](../reference/glossary.md): related time-scale glossary.
