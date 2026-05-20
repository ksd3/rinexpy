# Plotting helpers

The optional `rinexpy.plots` module provides matplotlib helpers for the
most common GNSS visualisations: per-satellite observation time series,
satellite ground tracks, receiver location maps, and polar skyplots.

The module needs the `plot` extra (`uv sync --extra plot`). Without
matplotlib, importing the module itself works, but calling any of its
functions raises `ImportError` with an actionable message.

## obstimeseries

The L1 (or L1C) carrier-phase time series, one line per satellite.

```python
import rinexpy as rp
from rinexpy.plots import obstimeseries
import matplotlib.pyplot as plt

obs = rp.load("tests/data/obs3.01gage.10o")
obstimeseries(obs)
plt.show()
```

For headless environments:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

obstimeseries(obs)
plt.savefig("obs.png", dpi=120, bbox_inches="tight")
```

The function picks the first phase observable it finds (`L1C` in RINEX 3,
`L1` in RINEX 2) and plots one line per satellite against time. Satellites
that have no data in the file are dropped.

## navtimeseries

Satellite ground tracks from a NAV dataset.

```python
from rinexpy.plots import navtimeseries

nav = rp.load("tests/data/brdc2800.15n")
navtimeseries(nav)
plt.show()
```

The function evaluates the Keplerian elements at each broadcast `Toe`,
projects to (lat, lon), and connects the per-SV trajectory points.

For NAV files with GLONASS satellites (which broadcast direct ECEF rather
than Keplerian elements), the function uses the ECEF coordinates
directly.

## receiver_locations

Scatter a set of receiver positions on a world map. Marker size encodes
the sampling interval.

```python
from rinexpy.plots import receiver_locations

receiver_locations([
    {"name": "ALGO", "lat": 45.95, "lon": -78.07, "interval": 30.0},
    {"name": "STAT00BRA", "lat": -22.40, "lon": -54.62, "interval": 1.0},
    {"name": "GODE", "lat": 39.02, "lon": -76.83, "interval": 30.0},
])
plt.show()
```

Each entry is a dict with at least `lat`, `lon`. Optional fields are
`name` (label) and `interval` (sampling cadence). With `interval`, large
markers indicate high-cadence stations.

If [cartopy](https://scitools.org.uk/cartopy/docs/) is installed, the
plot uses a proper map projection. Without cartopy, it falls back to a
plain matplotlib scatter on a lat / lon grid.

## skyplot

Polar plot of satellite trajectories above a receiver.

```python
import numpy as np
from rinexpy.geodesy import azimuth_elevation, lla_to_ecef
from rinexpy.plots import skyplot

nav = rp.load("tests/data/brdc2800.15n")
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
plt.show()
```

The input is a dict `{sv_label: (az_deg, el_deg)}`. The function draws
a polar projection with azimuth as the angle and (90 - elevation) as the
radius, so the centre of the plot is the zenith.

Optional kwargs:

| Kwarg | Default | Meaning |
| --- | --- | --- |
| `elevation_mask_deg` | 5.0 | satellites below this elevation are clipped |
| `title` | "" | plot title |

Constellation-specific colours are picked automatically from the SV
label's first character.

## timeseries dispatch

For convenience, `timeseries(data)` calls either `obstimeseries`
or `navtimeseries` based on the `rinextype` attribute of the input
dataset.

```python
from rinexpy.plots import timeseries

timeseries(obs)   # -> obstimeseries
timeseries(nav)   # -> navtimeseries
```

When you pass a tuple `(nav, obs)`, the function plots the navtimeseries
ground tracks with the OBS receiver location overlaid.

## Customisation

Each function returns the matplotlib `Axes` object so you can customise
afterwards.

```python
import matplotlib.pyplot as plt

ax = skyplot(sv_az_el, title="Sky from (40, -3)")
ax.set_facecolor("black")
plt.savefig("sky.png")
```

For more involved customisations, the recommended pattern is to copy the
function's body (the `plots.py` module is short) and adapt it.

## Picking a backend

`matplotlib` is the only dependency. Cartopy is optional and only enables
the proper map projection in `receiver_locations`.

For headless rendering (no display, no GUI), set the backend to `Agg`
before importing `matplotlib.pyplot`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

For interactive use, the default backend works.

## Related pages

- [RINEX observation files](../formats/rinex-obs.md): the source of `obstimeseries`.
- [RINEX navigation files](../formats/rinex-nav.md): the source of `navtimeseries` and `skyplot`.
- [Installation](../installation.md): the `plot` extra setup.
