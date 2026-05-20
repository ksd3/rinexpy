# Carrier-phase wind-up

GNSS antennas are circularly polarised. As the satellite moves across the
sky, the relative orientation of the satellite antenna and the receiver
antenna rotates. The resulting phase angle change accumulates as an
apparent carrier-phase shift, the "wind-up". It is up to one full cycle
per satellite orbit, which is roughly 20 cm on L1.

The wind-up shows up in carrier-phase observations only. Pseudoranges are
not polarisation-sensitive. For PPP-class positioning the correction is
applied as a per-satellite per-epoch accumulator.

The model is Wu et al. (1993). `rinexpy` exposes the per-epoch wind-up
calculation in `rinexpy.geodesy.phase_wind_up_correction`.

## The model

The wind-up angle for one satellite at one epoch is computed from four
direction vectors:

- `sat_xhat`, `sat_yhat`: the satellite antenna's body-frame x and y axes,
 in ECEF.
- `rx_xhat`, `rx_yhat`: the receiver antenna's body-frame x and y axes,
 in ECEF.
- `los_rx_to_sat`: the unit vector from the receiver to the satellite.

The wind-up is the angle between the satellite's transmitting axes and
the receiver's receiving axes, projected onto a plane perpendicular to
the line of sight.

```python
from rinexpy.geodesy import phase_wind_up_correction
import numpy as np

# All vectors in ECEF, unit-length.
phi_cycles = phase_wind_up_correction(
    sat_xhat=np.array([1, 0, 0]),         # satellite x-axis in ECEF
    sat_yhat=np.array([0, 1, 0]),
    rx_xhat=np.array([1, 0, 0]),
    rx_yhat=np.array([0, 1, 0]),
    los_rx_to_sat=np.array([0, 0, 1]),
    previous_cycles=0.0,                   # the previous epoch's accumulated value
)
```

The function returns the accumulated phase wind-up in cycles. To convert
to metres, multiply by the carrier wavelength.

## The accumulator

The Wu et al. recipe is a per-satellite accumulator. The naive
calculation gives an angle modulo 2π, but the cumulative value needs to
track the 2π wraps continuously so the carrier-phase model remains
consistent.

The `previous_cycles` argument is the previous epoch's accumulated value.
The function unwraps the new angle against the previous one and returns
the continuous count.

```python
phi_prev = 0.0
for epoch in epochs:
    phi_new = phase_wind_up_correction(
        sat_xhat_at_epoch, sat_yhat_at_epoch,
        rx_xhat, rx_yhat,
        los_at_epoch,
        previous_cycles=phi_prev,
    )
    phi_prev = phi_new
    # Apply phi_new * wavelength to the carrier-phase observation.
```

## Computing the antenna axes

The satellite antenna axes follow a yaw-steering model. For most GNSS
satellites (GPS Block II, Galileo IOV / FOC, GLONASS), the satellite
keeps its solar panels facing the Sun, which fixes the body-frame y-axis.
The body x-axis points in the orbit direction.

```python
def sat_xhat_yhat(sat_pos_ecef, sat_vel_ecef, sun_ecef):
    """Compute the satellite body axes from sun-pointing yaw steering."""
    import numpy as np
    # z-axis: from satellite to Earth centre (nadir-pointing)
    z = -sat_pos_ecef / np.linalg.norm(sat_pos_ecef)
    # y-axis: cross product of nadir and sun direction, normalised
    sun_dir = sun_ecef - sat_pos_ecef
    sun_dir /= np.linalg.norm(sun_dir)
    y = np.cross(z, sun_dir)
    y /= np.linalg.norm(y)
    # x-axis: completes the right-handed frame
    x = np.cross(y, z)
    return x, y
```

The receiver antenna axes typically follow the local east-north-up frame
of the station: x = east, y = north.

```python
def rx_xhat_yhat(station_ecef):
    """Local east/north basis vectors at the station, in ECEF."""
    import numpy as np
    lat = np.arctan2(station_ecef[2],
                     np.linalg.norm(station_ecef[:2]))
    lon = np.arctan2(station_ecef[1], station_ecef[0])
    east  = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.array([-np.sin(lat)*np.cos(lon),
                      -np.sin(lat)*np.sin(lon),
                       np.cos(lat)])
    return east, north
```

For permanently-mounted GNSS antennas these are constants; for a
kinematic rover you recompute the local frame per epoch.

## Use in PPP

The `ppp_solve` driver applies wind-up automatically when
`apply_wind_up=True` is passed.

```python
out = ppp_solve(
    obs, sp3, clk,
    initial_position_ecef=tuple(approx_xyz),
    apply_wind_up=True,
)
```

The driver maintains the accumulator state per satellite and computes the
satellite-frame axes from the sun-pointing yaw model. The receiver-frame
axes are computed from the current best estimate of the receiver position.

## Magnitude and impact

Typical wind-up magnitudes per satellite:

- Per epoch: a few mm to tens of mm, depending on geometry.
- Per pass (rising to setting): up to about 30 cm.
- Across multiple satellites: when the geometry is symmetric, much of
 the wind-up cancels in the common-mode receiver clock estimate; what
 is left biases the position by 5-10 cm at the worst.

For RTK over short baselines the wind-up cancels in the double
difference (both receivers see the same satellite from nearly the same
direction). For long-baseline RTK and for PPP, the correction matters.

For SPP with code-only fixes the wind-up is irrelevant because pseudorange
measurements are not polarisation-sensitive.

## Eclipse seasons

When the satellite enters Earth's shadow (the "eclipse season"), the
sun-pointing yaw model breaks down. The satellite applies a yaw
manoeuvre to keep itself pointed correctly when the sun reappears. The
manoeuvre takes a few minutes, during which the body axes do not follow
the standard formula and the wind-up calculation is wrong.

Operationally, PPP processors flag observations during eclipse manoeuvres
and either down-weight them or drop them. `rinexpy` does not currently
expose eclipse-season flags; for cm-class PPP across eclipse seasons,
implement an eclipse mask in your pre-processing.

## Related pages

- [Precise point positioning](../positioning/ppp.md): where `apply_wind_up=True` plugs in.
- [Atmosphere products](../formats/atmosphere-products.md): the ANTEX reader, which carries the satellite antenna PCO/PCV (a separate per-satellite correction that is in the same area of the pipeline).
- [Tides and station displacements](tides.md): the matching site displacement chain.
